#!/usr/bin/env python3

# Copyright (C) 2026 Daisuke Takago
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Callable

import httpx
import pymupdf as fitz
import yaml
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from nicegui import app, run, ui
from PIL import Image, ImageDraw

from slide_lecture import (
    generate_single_alignment,
    generate_single_explanation,
    generate_single_tts,
    load_project_json,
    make_client,
    parse_page_ranges,
    run_vlm_element_detection,
    VLM_BBOX_COORD_MAX,
    VLM_BBOX_FORMAT,
    save_project_json,
)
from tts_filter import (
    build_system_prompt as build_tts_filter_prompt,
    load_config as load_tts_filter_config,
    make_client as make_tts_filter_client,
    transform_text as tts_filter_transform,
)


# ----------------------------------------------------------------------
# Concurrency / Global Execution Lock (排他制御)
# ----------------------------------------------------------------------

class ExecutionLockManager:
    """複数のユーザーやセッションが同時に重い推論・レンダリング処理を行わないよう排他制御するマネージャ"""
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.current_user: str | None = None
        self.task_name: str | None = None

    def is_locked(self) -> bool:
        return self.lock.locked()

    async def acquire(self, username: str, task_name: str) -> bool:
        """ロックを即座に試行取得。他が実行中の場合はキューイングせず即座に False を返す"""
        if self.lock.locked():
            return False
        await self.lock.acquire()
        self.current_user = username
        self.task_name = task_name
        return True

    def release(self) -> None:
        if self.lock.locked():
            self.lock.release()
        self.current_user = None
        self.task_name = None


GLOBAL_EXECUTION_LOCK = ExecutionLockManager()


# ----------------------------------------------------------------------
# Password Hashing & Verification (PBKDF2-HMAC-SHA256)
# ----------------------------------------------------------------------

def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 によるソルト付きストレッチングハッシュ生成"""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        iterations=100_000,
    )
    return f"pbkdf2:sha256:100000${salt}${key.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """平文パスワードと保存されたハッシュ値の照合（平文との互換移行対応）"""
    if not stored_hash:
        return False

    # 平文からハッシュ化への過渡期互換（平文で完全一致すればOK）
    if not stored_hash.startswith('pbkdf2:sha256:'):
        return password == stored_hash

    try:
        header, salt, key_hex = stored_hash.split('$')
        _, _, iter_str = header.split(':')
        iterations = int(iter_str)
        calc_key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=iterations,
        )
        return secrets.compare_digest(calc_key.hex(), key_hex)
    except Exception:
        return False


# ----------------------------------------------------------------------
# User Management & Persistence (users.yaml)
# ----------------------------------------------------------------------

USERS_FILE = Path('users.yaml')

DEFAULT_USERS_DATA = {
    'users': {
        'admin': {
            'password': hash_password('secret'),
            'is_admin': True,
            'valid_from': '',
            'valid_until': '',
        },
        'guest': {
            'password': hash_password('guest'),
            'is_admin': False,
            'valid_from': '',
            'valid_until': '',
        },
    }
}


def load_users() -> dict[str, Any]:
    if not USERS_FILE.exists():
        save_users(DEFAULT_USERS_DATA)
        return DEFAULT_USERS_DATA
    try:
        data = yaml.safe_load(USERS_FILE.read_text(encoding='utf-8'))
        if not data or 'users' not in data:
            return DEFAULT_USERS_DATA
        return data
    except Exception:
        return DEFAULT_USERS_DATA


def save_users(data: dict[str, Any]) -> None:
    USERS_FILE.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )


def is_user_within_allowed_period(user_info: dict[str, Any]) -> tuple[bool, str]:
    """利用可能日時制限の判定 (YYYY-MM-DD HH:MM または YYYY-MM-DD 形式)"""
    # 管理者権限を持つユーザーは事故防止のため常に利用可能とする
    if user_info.get('is_admin', False):
        return True, ''

    v_from = str(user_info.get('valid_from') or '').strip()
    v_until = str(user_info.get('valid_until') or '').strip()

    if not v_from and not v_until:
        return True, ''

    now = datetime.now()

    def parse_dt(s: str, is_end: bool = False) -> datetime | None:
        for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
            try:
                dt = datetime.strptime(s, fmt)
                if fmt == '%Y-%m-%d' and is_end:
                    dt = dt.replace(hour=23, minute=59, second=59)
                return dt
            except ValueError:
                pass
        return None

    if v_from:
        dt_from = parse_dt(v_from, is_end=False)
        if dt_from and now < dt_from:
            return False, f'このアカウントは利用開始日時（{v_from}）前です．'

    if v_until:
        dt_until = parse_dt(v_until, is_end=True)
        if dt_until and now > dt_until:
            return False, f'このアカウントは利用可能期限（{v_until}）を過ぎています．'

    return True, ''


# ----------------------------------------------------------------------
# Configuration / filesystem helpers
# ----------------------------------------------------------------------

CONFIG_PATH = Path('config.yaml')
TTS_FILTER_PATH = Path('tts_filter.yaml')
BASE_UPLOAD_DIR = Path('webui_uploads')
BASE_UPLOAD_DIR.mkdir(exist_ok=True)


def get_user_workspace(username: str) -> Path:
    """ユーザーごとの個別作業ディレクトリ"""
    safe_name = "".join(c for c in username if c.isalnum() or c in ('_', '-')).strip() or 'unknown'
    p = BASE_UPLOAD_DIR / safe_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def save_config(path: Path, cfg: dict) -> None:
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )


def run_command(args: list[str]) -> tuple[int, str]:
    p = subprocess.run(args, capture_output=True, text=True)
    return p.returncode, p.stdout + '\n' + p.stderr


# ----------------------------------------------------------------------
# Project Paths Manager
# ----------------------------------------------------------------------

class ProjectPaths:
    """プロジェクト配下の各種ファイルパスを一元管理するヘルパークラス"""

    def __init__(self, pdf: Path) -> None:
        self.pdf = pdf
        self.root = pdf.with_name(pdf.stem + '_lecture')
        self.pages_dir = self.root / 'pages'
        self.explanations_dir = self.root / 'explanations'
        self.audio_dir = self.root / 'audio'
        self.video_dir = self.root / 'video'

    def explanation(self, page: int) -> Path:
        return self.explanations_dir / f'{page:03d}.txt'

    def alignment(self, page: int) -> Path:
        return self.explanations_dir / f'{page:03d}_align.json'

    def page_image(self, page: int) -> Path:
        return self.pages_dir / f'{page:03d}.png'

    def page_preview(self, page: int) -> Path:
        return self.pages_dir / f'{page:03d}_preview.png'

    def audio(self, page: int) -> Path:
        return self.audio_dir / f'{page:03d}.mp3'

    def video(self, page: int) -> Path:
        return self.video_dir / f'{page:03d}.mp4'

    @property
    def final_video(self) -> Path:
        return self.root / f'{self.pdf.stem}.mp4'

    @property
    def final_ja_srt(self) -> Path:
        return self.root / f'{self.pdf.stem}_ja.srt'

    @property
    def final_en_srt(self) -> Path:
        return self.root / f'{self.pdf.stem}_en.srt'

    @staticmethod
    def _unlink(path: Path) -> None:
        if path.exists():
            path.unlink()

    def cleanup_downstream(self, page: int | None = None, include_alignment: bool = False) -> None:
        """下流のメディアファイル（動画・音声・字幕）を削除します．"""
        for path in (self.final_video, self.final_ja_srt, self.final_en_srt):
            self._unlink(path)

        if page is not None:
            if include_alignment:
                self._unlink(self.alignment(page))
            for path in (self.audio(page), self.video(page)):
                self._unlink(path)
            return

        if include_alignment and self.explanations_dir.exists():
            for path in self.explanations_dir.glob('*_align.json'):
                self._unlink(path)

        for directory in (self.audio_dir, self.video_dir):
            if directory.exists():
                for path in directory.iterdir():
                    if path.is_file():
                        self._unlink(path)


def ensure_page_images(paths: ProjectPaths, dpi: int = 120) -> list[Path]:
    paths.pages_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(paths.pdf)
    result: list[Path] = []
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    for i, page in enumerate(doc, 1):
        out = paths.page_image(i)
        result.append(out)
        if not out.exists():
            page.get_pixmap(matrix=matrix, alpha=False).save(out)
    return result


def draw_block_preview(img_path: Path, blocks: list[dict]) -> Image.Image:
    im = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(im)

    color_map = {
        'image_subpart': ('#ef4444', 3),  # 赤色: VLM検出要素（太枠）
        'image': ('#f59e0b', 2),          # 橙色: 画像領域全体
        'text': ('#3b82f6', 2),           # 青色: 通常テキスト
        'code_line': ('#06b6d4', 2),      # 水色: コード行
    }

    for b in blocks:
        box = b.get('bbox')
        bid = b.get('block_id')
        btype = b.get('type', 'text')
        if box:
            color, width = color_map.get(btype, ('#3b82f6', 2))
            draw.rectangle(box, outline=color, width=width)
            badge_width = max(35, 12 + len(str(bid)) * 10)
            draw.rectangle([box[0], max(0, box[1] - 20), box[0] + badge_width, box[1]], fill=color)
            draw.text((box[0] + 4, max(0, box[1] - 18)), f'#{bid}', fill='white')
    return im


def format_page_ranges(pages: list[int]) -> str:
    """整数のリストから '1-5,7,9-10' のような範囲文字列を生成します．"""
    if not pages:
        return ''
    sorted_pages = sorted(set(pages))
    ranges = []
    start = sorted_pages[0]
    prev = sorted_pages[0]

    for p in sorted_pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f'{start}-{prev}')
            start = p
            prev = p

    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f'{start}-{prev}')

    return ','.join(ranges)


def file_url(path: Path) -> str:
    """ファイルの最終更新日時(mtime)をクエリに付与し、ブラウザキャッシュを安全に回避します．"""
    rel = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    ts = int(path.stat().st_mtime) if path.exists() else int(time.time())
    return f'/files/{rel}?t={ts}'


def _resolve_file(path: str) -> Path | None:
    root = Path.cwd().resolve()
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        return None
    return target


@app.get('/files/{path:path}')
async def serve_file(path: str):
    target = _resolve_file(path)
    if target is None:
        return PlainTextResponse('Forbidden', status_code=403)
    if not target.is_file():
        return PlainTextResponse('Not found', status_code=404)
    response = FileResponse(target)
    response.headers['Cache-Control'] = 'no-cache, must-revalidate'
    return response


@app.get('/download/{path:path}')
async def download_file(path: str):
    target = _resolve_file(path)
    if target is None:
        return PlainTextResponse('Forbidden', status_code=403)
    if not target.is_file():
        return PlainTextResponse('Not found', status_code=404)
    return FileResponse(target, filename=target.name)


# ----------------------------------------------------------------------
# Application state
# ----------------------------------------------------------------------

class SlideNarratorApp:
    def __init__(self, username: str) -> None:
        self.username = username
        self.cfg = load_config(CONFIG_PATH)

        # ユーザー固有の作業ディレクトリ
        self.user_dir = get_user_workspace(self.username)
        self.test_audio_dir = self.user_dir / 'test_audio'
        self.test_audio_dir.mkdir(exist_ok=True)
        self.test_vlm_dir = self.user_dir / 'test_vlm'
        self.test_vlm_dir.mkdir(exist_ok=True)

        self.pdf: Path | None = None
        self.paths: ProjectPaths | None = None
        self.images: list[Path] = []
        self.proj_cfg: dict[str, Any] = {}
        self.mode_code = 'lecture'
        self.lang_code = 'ja'
        self.default_use_vlm: bool = True
        self.slide_visual_modes: dict[str, str] = {}
        self.force_run = False
        self.edit_page: int | None = None
        self.log = ''
        self.processing = False

        self._selected_pages: set[int] = set()

        self.current_process: asyncio.subprocess.Process | None = None
        self.current_task: asyncio.Task | None = None
        self.cancellation_requested = False

        self.uploader = None
        self.active_count_label = None
        self.history_log_widget = None
        self.tabs = None
        self.gallery = None
        self.simple_edit_container = None
        self.edit_container = None
        self.slide_video_gallery = None
        self.final_video_container = None
        self.settings_container = None
        self.user_manage_container = None
        self.mode_select = None
        self.lang_select = None
        self.default_vlm_switch = None
        self.pages_input = None
        self.force_checkbox = None
        self.pipeline_buttons = []

    @property
    def total_slides(self) -> int:
        return len(self.images)

    @property
    def active_pages(self) -> list[int]:
        return sorted(self._selected_pages)

    @property
    def pages_spec(self) -> str:
        if not self.pdf or self.total_slides == 0:
            return ''
        if not self._selected_pages:
            return 'none'
        if len(self._selected_pages) == self.total_slides:
            return ''
        return format_page_ranges(self.active_pages)

    def get_slide_order(self, page_num: int) -> int:
        if page_num in self._selected_pages:
            return self.active_pages.index(page_num) + 1
        return 0

    def apply_pages_spec(self, spec: str, save_and_refresh: bool = True) -> None:
        if not self.pdf or self.total_slides == 0:
            self._selected_pages = set()
        else:
            raw = spec.strip()
            if raw == 'none':
                self._selected_pages = set()
            elif raw:
                self._selected_pages = parse_page_ranges(raw) & set(range(1, self.total_slides + 1))
            else:
                self._selected_pages = set(range(1, self.total_slides + 1))

        self._on_pages_updated(sync_input=(self.pages_input and self.pages_input.value != self.pages_spec),
                               save=save_and_refresh, refresh=save_and_refresh)

    def _on_pages_updated(self, sync_input: bool = True, save: bool = True, refresh: bool = True) -> None:
        if self.edit_page not in self._selected_pages:
            self.edit_page = self.active_pages[0] if self.active_pages else None

        if sync_input and self.pages_input:
            self.pages_input.value = self.pages_spec

        if self.active_count_label:
            self.active_count_label.text = f'対象スライド: {len(self._selected_pages)} / {self.total_slides} スライド'

        if save:
            self.save_project_settings()

        if refresh:
            self.request_views_refresh()

    def request_views_refresh(self) -> None:
        asyncio.create_task(self.refresh_views())

    async def refresh_views(self) -> None:
        await self.refresh_gallery()
        await self.refresh_simple_editor()
        await self.refresh_editor()
        await self.refresh_slide_videos()

    async def load_pdf(self, e) -> None:
        filename = Path(e.file.name).name
        target_pdf = self.user_dir / filename
        target_paths = ProjectPaths(target_pdf)

        temp_pdf = self.user_dir / f".upload_{int(time.time())}_{filename}"
        await e.file.save(temp_pdf)

        has_existing_project = target_paths.root.exists() or target_pdf.exists()

        async def finalize_loading(delete_existing: bool) -> None:
            if delete_existing:
                if target_paths.root.exists():
                    shutil.rmtree(target_paths.root, ignore_errors=True)
                if target_pdf.exists():
                    target_pdf.unlink()
                ui.notify(f'既存のプロジェクトデータを削除しました: {target_paths.root.name}', type='info')

            temp_pdf.replace(target_pdf)

            self.pdf = target_pdf
            self.paths = target_paths
            self.proj_cfg = load_project_json(self.paths.root)
            self.mode_code = self.proj_cfg.get('mode') or self.cfg.get('mode', 'lecture')
            self.lang_code = self.proj_cfg.get('language') or self.cfg.get('language', 'ja')

            saved_vmode = self.proj_cfg.get('visual_mode') or self.cfg.get('visual_mode', 'vlm')
            self.default_use_vlm = saved_vmode in ('vlm', 'auto', True, 'true')
            self.slide_visual_modes = self.proj_cfg.get('slide_visual_modes', {})

            self.images = await run.io_bound(
                ensure_page_images, self.paths, int(self.cfg.get('pdf', {}).get('dpi', 120))
            )

            saved_pages_spec = self.proj_cfg.get('pages', '')
            self.apply_pages_spec(saved_pages_spec, save_and_refresh=False)

            self.refresh_project_widgets()
            await self.refresh_all()
            ui.notify(f'プレゼンテーションを読み込みました: {target_pdf.name}', type='positive')

        if has_existing_project:
            with ui.dialog() as dialog, ui.card().classes('p-5 gap-4 max-w-md'):
                dialog.props('persistent')
                with ui.row().classes('items-center gap-2 text-warning'):
                    ui.icon('warning', size='md').classes('text-amber-500')
                    ui.label('既存プロジェクトが見つかりました').classes('text-base font-bold text-zinc-100')

                ui.label(
                    f'「{filename}」に対応する既存フォルダ（{target_paths.root.name}）が既に存在します。'
                    'フォルダ一式（生成済みのナレーション原稿・音声・動画など）をすべて削除して新しくやり直しますか？'
                ).classes('text-sm text-zinc-300 leading-relaxed')

                with ui.row().classes('w-full justify-end gap-3 pt-2'):
                    async def on_keep():
                        dialog.close()
                        await finalize_loading(delete_existing=False)

                    async def on_delete():
                        dialog.close()
                        await finalize_loading(delete_existing=True)

                    ui.button('既存データを引き継ぐ', on_click=on_keep).props('outline color=grey')
                    ui.button('一式を削除して初期化', on_click=on_delete).props('color=negative')

            dialog.open()
        else:
            await finalize_loading(delete_existing=False)

    def refresh_project_widgets(self) -> None:
        if not self.pdf:
            return
        if self.mode_select:
            self.mode_select.value = self.mode_code
        if self.lang_select:
            self.lang_select.value = self.lang_code
        if self.default_vlm_switch:
            self.default_vlm_switch.value = self.default_use_vlm
        if self.pages_input:
            self.pages_input.value = self.pages_spec
        if self.active_count_label:
            self.active_count_label.text = f'対象スライド: {len(self._selected_pages)} / {self.total_slides} スライド'

    def save_project_settings(self) -> None:
        if not self.paths:
            return
        self.proj_cfg.update({
            'mode': self.mode_code,
            'language': self.lang_code,
            'pages': self.pages_spec,
            'skip_pages': '',
            'visual_mode': 'vlm' if self.default_use_vlm else 'pdf',
            'slide_visual_modes': self.slide_visual_modes,
        })
        save_project_json(self.paths.root, self.proj_cfg)

    def toggle_slide_active(self, page_num: int, active: bool) -> None:
        if active:
            self._selected_pages.add(page_num)
        else:
            self._selected_pages.discard(page_num)
        self._on_pages_updated(sync_input=True, save=True, refresh=True)

    def select_all_slides(self) -> None:
        if self.pdf and self.total_slides > 0:
            self._selected_pages = set(range(1, self.total_slides + 1))
        else:
            self._selected_pages = set()
        self._on_pages_updated(sync_input=True, save=True, refresh=True)

    def clear_all_slides(self) -> None:
        self._selected_pages = set()
        self._on_pages_updated(sync_input=True, save=True, refresh=True)

    def open_processing_dialog(self, initial_title: str) -> tuple[ui.dialog, Callable[[str, float | None, str | None, int | None], None], Callable[[str], None]]:
        dialog = ui.dialog()
        dialog.props('persistent')
        self.cancellation_requested = False

        with dialog, ui.card().classes('items-center p-6 gap-3 min-w-[620px] max-w-[760px]'):
            title_label = ui.label(initial_title).classes('text-base font-bold text-center text-zinc-100')

            slide_preview_row = ui.row().classes('w-full items-end justify-center gap-3 py-2')
            with slide_preview_row:
                with ui.column().classes('items-center w-28 opacity-45'):
                    prev_img_box = ui.column().classes('w-28 aspect-video items-center justify-center')
                    with prev_img_box:
                        prev_image = ui.image('').props('fit=contain').classes('w-full h-full rounded').style('display: none')
                        prev_placeholder = ui.label('-').classes('text-xs text-zinc-500')

                with ui.column().classes('items-center w-52 scale-105 transition-all'):
                    curr_img_box = ui.column().classes('w-52 aspect-video items-center justify-center')
                    with curr_img_box:
                        curr_image = ui.image('').props('fit=contain').classes('w-full h-full rounded').style('display: none')
                        curr_placeholder = ui.label('スライド待機中').classes('text-xs text-zinc-400')

                with ui.column().classes('items-center w-28 opacity-45'):
                    next_img_box = ui.column().classes('w-28 aspect-video items-center justify-center')
                    with next_img_box:
                        next_image = ui.image('').props('fit=contain').classes('w-full h-full rounded').style('display: none')
                        next_placeholder = ui.label('-').classes('text-xs text-zinc-500')

            status_label = ui.label('準備中…').classes('text-sm text-zinc-400 text-center')

            with ui.row().classes('w-full items-center gap-2'):
                progress_bar = ui.linear_progress(value=0.0, show_value=False).props('rounded size=14px').classes('grow')

            with ui.expansion('詳細ログを表示', icon='terminal').classes('w-full border border-zinc-700 rounded-lg text-xs mt-1'):
                dialog_log = ui.log(max_lines=300).classes('w-full h-40 font-mono text-xs bg-zinc-900 text-zinc-300 p-2')

            ui.spinner(size='md')

            with ui.row().classes('w-full justify-center pt-2'):
                ui.button('処理を中断', on_click=self.request_cancel, color='negative').props('text-color=white')

        dialog.open()

        def update_slide_preview(current_page: int | None) -> None:
            if not self.paths or current_page is None:
                return
            pages_list = self.active_pages if self.active_pages else list(range(1, self.total_slides + 1))
            if current_page not in pages_list:
                return

            idx = pages_list.index(current_page)
            prev_p = pages_list[idx - 1] if idx > 0 else None
            next_p = pages_list[idx + 1] if idx + 1 < len(pages_list) else None

            def set_preview(image_widget, placeholder, page: int | None) -> None:
                if page is not None:
                    image_path = self.paths.page_image(page)
                    if image_path.exists():
                        image_widget.set_source(file_url(image_path))
                        image_widget.style('display: block')
                        placeholder.style('display: none')
                        return
                image_widget.style('display: none')
                placeholder.style('display: block')

            set_preview(prev_image, prev_placeholder, prev_p)
            set_preview(curr_image, curr_placeholder, current_page)
            set_preview(next_image, next_placeholder, next_p)

        def update_status(text: str, frac: float | None = None, title: str | None = None, current_page: int | None = None) -> None:
            if title is not None:
                title_label.text = title
            status_label.text = text
            if frac is None:
                progress_bar.value = 0.0
            else:
                clamped = max(0.0, min(1.0, float(frac)))
                progress_bar.value = clamped

            if current_page is not None:
                update_slide_preview(current_page)

        return dialog, update_status, dialog_log.push

    async def request_cancel(self) -> None:
        if self.cancellation_requested:
            return
        self.cancellation_requested = True
        ui.notify('処理の中断を要求しました．停止中…', type='warning')

        if self.current_process and self.current_process.returncode is None:
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
                else:
                    self.current_process.terminate()
            except Exception:
                try:
                    self.current_process.terminate()
                except Exception:
                    pass

        if self.current_task and not self.current_task.done():
            self.current_task.cancel()

    def set_processing(self, value: bool) -> None:
        self.processing = value
        for button in self.pipeline_buttons:
            button.disable() if value else button.enable()

    def base_args(self) -> list[str]:
        args = [
            '--mode', self.mode_code,
            '--lang', self.lang_code,
            '--visual-mode', 'vlm' if self.default_use_vlm else 'pdf',
        ]
        if self.force_run:
            args.append('--force')
        spec = self.pages_spec.strip()
        if spec and spec != 'none':
            args.extend(['--pages', spec])
        return args

    async def pipeline(self, stage: str, initial_title: str) -> None:
        if not self.pdf:
            ui.notify('先にプレゼンテーションPDFを選択してください．', type='warning')
            return
        if not self.active_pages:
            ui.notify('処理対象となるスライドが1枚も選択されていません．', type='warning')
            return
        if self.processing:
            ui.notify('別の処理が実行中です．処理が終わるまでお待ちください．', type='warning')
            return

        # --- 追加: ナレーション原稿の存在・空チェック ---
        if stage in ('align', 'tts', 'video') and self.paths:
            empty_pages = []
            for p in self.active_pages:
                txt_p = self.paths.explanation(p)
                if not txt_p.exists() or not txt_p.read_text(encoding='utf-8').strip():
                    empty_pages.append(p)

            if empty_pages:
                pages_str = ", ".join(f"スライド {p}" for p in empty_pages)
                ui.notify(
                    f'{pages_str} のナレーション原稿が空です．確認・生成してください．',
                    type='warning',
                    duration=6.0,
                )
                return
        # --------------------------------------------------

        # 排他制御チェック（キューイングなし）
        lock_acquired = await GLOBAL_EXECUTION_LOCK.acquire(self.username, f"パイプライン ({stage})")
        if not lock_acquired:
            msg = f'他のユーザー（{GLOBAL_EXECUTION_LOCK.current_user or "誰か"}）が「{GLOBAL_EXECUTION_LOCK.task_name or "処理"}」を実行中です．完了するまでリクエストは受け付けられません．'
            ui.notify(msg, type='negative', duration=5)
            return

        self.save_project_settings()
        dialog, update_status, push_log = self.open_processing_dialog(initial_title)
        self.set_processing(True)

        total_active_slides = len(self.active_pages)
        cmd = [sys.executable, '-u', 'slide_lecture.py', str(self.pdf), '--from', stage, *self.base_args()]

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        phase_titles = {
            'explain': '① ナレーション原稿を作成中…',
            'align': '② 字幕・ポインタを解析中…',
            'tts': f'③ 音声を合成中…',
            'video': '④ スライド動画をレンダリング中…',
            'concat': '④ 完成動画を結合・生成中…',
        }

        try:
            update_status('処理を開始しています…', 0.0, current_page=self.active_pages[0])
            await asyncio.sleep(0.05)

            create_group_kwargs = {}
            if sys.platform != "win32":
                create_group_kwargs['preexec_fn'] = os.setsid

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                **create_group_kwargs,
            )
            self.current_process = proc

            full_logs: list[str] = []

            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode('utf-8', errors='replace').rstrip()
                full_logs.append(line)
                push_log(line)
                if self.history_log_widget:
                    self.history_log_widget.push(line)

                if line.startswith('[PROGRESS]'):
                    try:
                        p_data = json.loads(line[10:].strip())
                        phase = p_data.get('phase')
                        cur = p_data.get('current', 1)
                        tot = p_data.get('total', total_active_slides)
                        p_num = p_data.get('page')
                        msg = p_data.get('message', '')

                        title = phase_titles.get(phase, initial_title)
                        weight = 0.9 if phase == 'video' else 1.0
                        frac = (cur / max(1, tot)) * weight if tot else 0.5
                        if phase == 'concat':
                            frac = 0.95
                            status_label_text = msg or '完成動画を結合・生成中…'
                        elif p_num is not None:
                            status_label_text = f'スライド {p_num}（{cur}/{tot}）'
                        else:
                            status_label_text = msg or f'スライド {cur}（{cur}/{tot}）'

                        update_status(status_label_text, frac, title=title, current_page=p_num)
                    except Exception:
                        pass

                await asyncio.sleep(0.01)

            rc = await proc.wait()
            self.log = '\n'.join(full_logs)

            if self.cancellation_requested:
                ui.notify('処理を中断しました．完了したスライドは保存されています．', type='warning')
            elif rc == 0:
                update_status('完了しました！', 1.0, title='処理が完了しました')
                await asyncio.sleep(0.5)
                ui.notify('パイプライン処理が完了しました．', type='positive')
            else:
                ui.notify(f'処理が終了コード {rc} で終了しました．ログタブを確認してください．', type='negative')

            await self.refresh_all()

        except asyncio.CancelledError:
            ui.notify('処理がキャンセルされました．', type='warning')
        except Exception as exc:
            self.log = str(exc)
            if self.history_log_widget:
                self.history_log_widget.push(str(exc))
            ui.notify(f'処理に失敗しました: {exc}', type='negative')
        finally:
            self.current_process = None
            self.set_processing(False)
            GLOBAL_EXECUTION_LOCK.release()
            dialog.close()

    def alignment_data(self, page: int) -> tuple[dict, list[dict], list[dict]]:
        if not self.paths:
            return {}, [], []
        p = self.paths.alignment(page)
        if not p.exists():
            return {}, [], []
        data = json.loads(p.read_text(encoding='utf-8'))
        return data, data.get('blocks', []), data.get('alignments', [])

    async def select_edit_page(self, page: int) -> None:
        self.edit_page = int(page)
        await self.refresh_editor()

    async def prev_edit(self) -> None:
        if self.edit_page in self._selected_pages:
            i = self.active_pages.index(self.edit_page)
            if i > 0:
                await self.select_edit_page(self.active_pages[i - 1])

    async def next_edit(self) -> None:
        if self.edit_page in self._selected_pages:
            i = self.active_pages.index(self.edit_page)
            if i + 1 < len(self.active_pages):
                await self.select_edit_page(self.active_pages[i + 1])

    async def regenerate_narration(self, page: int, text_area) -> None:
        if not self.pdf or not self.paths:
            return
        if self.processing:
            ui.notify('別の処理が実行中です．処理が終わるまでお待ちください．', type='warning')
            return

        lock_acquired = await GLOBAL_EXECUTION_LOCK.acquire(self.username, f"スライド {page} ナレーション再生成")
        if not lock_acquired:
            msg = f'他のユーザー（{GLOBAL_EXECUTION_LOCK.current_user or "誰か"}）が「{GLOBAL_EXECUTION_LOCK.task_name or "処理"}」を実行中です．完了するまでリクエストは受け付けられません．'
            ui.notify(msg, type='negative', duration=5)
            return

        dialog, update_status, push_log = self.open_processing_dialog(f'スライド {page} のナレーションをLLMで再作成中…')
        self.set_processing(True)
        self.current_task = asyncio.current_task()
        try:
            update_status('全体概要と前後文脈をロード中…', 0.2, current_page=page)
            push_log(f'[REGEN] スライド {page} ナレーション再生成開始')
            await asyncio.sleep(0.01)

            cfg = self.cfg
            client = await run.io_bound(make_client, cfg['llm'])

            update_status(f'LLMでスライド {page} の解説文を推論中…', 0.7, current_page=page)
            push_log(f'[REGEN] LLM推論中...')
            await asyncio.sleep(0.01)

            new_narration = await run.io_bound(
                generate_single_explanation,
                client=client,
                cfg=cfg['llm'],
                pdf=self.pdf,
                images=self.images,
                out_dir=self.paths.explanations_dir,
                page=page,
                active_pages=self.active_pages,
                mode=self.mode_code,
                lang=self.lang_code,
            )

            if not self.cancellation_requested:
                self.paths.cleanup_downstream(page, include_alignment=True)
                text_area.value = new_narration.strip()
                update_status('完了しました！', 1.0, current_page=page)
                push_log(f'[REGEN] 完了: 新しいナレーションを保存しました')
                await asyncio.sleep(0.3)
                ui.notify('ナレーションを再生成しました（古い字幕・音声・動画を初期化しました）．', type='positive')
                await self.refresh_simple_editor()
                await self.refresh_editor()
        except asyncio.CancelledError:
            ui.notify('ナレーション再生成を中断しました．', type='warning')
        except Exception as exc:
            ui.notify(f'再生成に失敗しました: {exc}', type='negative')
        finally:
            self.current_task = None
            self.set_processing(False)
            GLOBAL_EXECUTION_LOCK.release()
            dialog.close()

    async def save_and_realign(self, page: int, text: str, page_use_vlm: bool | None = None) -> None:
        if not self.pdf or not self.paths:
            return
        ep = self.paths.explanation(page)
        ep.parent.mkdir(parents=True, exist_ok=True)
        temp_ep = ep.with_name(f".{ep.name}.tmp")
        temp_ep.write_text(text.rstrip() + '\n', encoding='utf-8')
        temp_ep.replace(ep)

        # ページ固有のVLM併用フラグを保存
        if page_use_vlm is not None:
            mode_str = 'vlm' if page_use_vlm else 'pdf'
        else:
            mode_str = self.slide_visual_modes.get(str(page), 'vlm' if self.default_use_vlm else 'pdf')

        self.slide_visual_modes[str(page)] = mode_str
        self.save_project_settings()

        if self.processing:
            ui.notify('別の処理が実行中です．処理が終わるまでお待ちください．', type='warning')
            return

        lock_acquired = await GLOBAL_EXECUTION_LOCK.acquire(self.username, f"スライド {page} 字幕・ポインタ再解析")
        if not lock_acquired:
            msg = f'他のユーザー（{GLOBAL_EXECUTION_LOCK.current_user or "誰か"}）が「{GLOBAL_EXECUTION_LOCK.task_name or "処理"}」を実行中です．完了するまでリクエストは受け付けられません．'
            ui.notify(msg, type='negative', duration=5)
            return

        mode_label = "VLM併用" if mode_str == 'vlm' else "PDF基準"
        dialog, update_status, push_log = self.open_processing_dialog(f'スライド {page} の要素抽出と視線誘導を再解析中…')
        self.set_processing(True)
        self.current_task = asyncio.current_task()
        try:
            update_status(f'要素抽出（{mode_label}）と対訳・視線誘導を再計算中…', 0.5, current_page=page)
            push_log(f'[ALIGN] スライド {page} の要素抽出 ({mode_label}) と視線誘導を再計算中...')
            await asyncio.sleep(0.01)
            client = await run.io_bound(make_client, self.cfg['llm'])
            dpi = int(self.cfg.get('pdf', {}).get('dpi', 150))
            data = await run.io_bound(
                generate_single_alignment,
                client, self.cfg['llm'], self.pdf, page,
                self.paths.page_image(page), text.strip(), dpi, self.lang_code,
                visual_mode=mode_str,
            )

            if not self.cancellation_requested:
                alp = self.paths.alignment(page)
                alp.parent.mkdir(parents=True, exist_ok=True)
                temp_alp = alp.with_name(f".{alp.name}.tmp")
                temp_alp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
                temp_alp.replace(alp)

                self.paths.cleanup_downstream(page, include_alignment=False)
                update_status('完了しました！', 1.0, current_page=page)
                push_log(f'[ALIGN] 完了: 字幕・ポインタアライメントを保存しました')
                await asyncio.sleep(0.3)
                ui.notify(f'スライド {page} の字幕とポインタを再生成しました（{mode_label}）．', type='positive')
                await self.refresh_simple_editor()
                await self.refresh_editor()
        except asyncio.CancelledError:
            ui.notify('解析を中断しました．', type='warning')
        except Exception as exc:
            ui.notify(f'再解析に失敗しました: {exc}', type='negative')
        finally:
            self.current_task = None
            self.set_processing(False)
            GLOBAL_EXECUTION_LOCK.release()
            dialog.close()

    async def save_alignment(self, page: int, rows: list[dict], align_data: dict) -> None:
        if not self.paths:
            return
        alp = self.paths.alignment(page)
        align_data['alignments'] = rows
        temp_alp = alp.with_name(f".{alp.name}.tmp")
        temp_alp.write_text(json.dumps(align_data, ensure_ascii=False, indent=2), encoding='utf-8')
        temp_alp.replace(alp)

        for f in [self.paths.video(page), self.paths.final_video, self.paths.final_ja_srt, self.paths.final_en_srt]:
            if f.exists():
                f.unlink()
        ui.notify('字幕とポインタの修正を保存しました．', type='positive')
        await self.refresh_editor()

    async def generate_slide_tts(self, page: int) -> None:
        if not self.pdf or not self.paths:
            return
        txt_path = self.paths.explanation(page)
        if not txt_path.exists() or not txt_path.read_text(encoding='utf-8').strip():
            ui.notify(f'スライド {page} のナレーション原稿がありません．先に作成してください．', type='warning')
            return
        if self.processing:
            ui.notify('別の処理が実行中です．処理が終わるまでお待ちください．', type='warning')
            return

        lock_acquired = await GLOBAL_EXECUTION_LOCK.acquire(self.username, f"スライド {page} 音声合成")
        if not lock_acquired:
            msg = f'他のユーザー（{GLOBAL_EXECUTION_LOCK.current_user or "誰か"}）が「{GLOBAL_EXECUTION_LOCK.task_name or "処理"}」を実行中です．完了するまでリクエストは受け付けられません．'
            ui.notify(msg, type='negative', duration=5)
            return

        dialog, update_status, push_log = self.open_processing_dialog(f'スライド {page} の音声を合成中 ({self.lang_code})…')
        self.set_processing(True)
        self.current_task = asyncio.current_task()
        try:
            update_status(f'音声合成中…', 0.5, current_page=page)
            push_log(f'[TTS] スライド {page} の音声合成を開始 (lang={self.lang_code})')
            await asyncio.sleep(0.01)

            tts_all = self.cfg.get('tts', {})
            tts_cfg = tts_all.get(self.lang_code, tts_all)
            out_mp3 = self.paths.audio(page)

            await run.io_bound(
                generate_single_tts,
                text_path=txt_path,
                out_path=out_mp3,
                tts_cfg=tts_cfg,
                filter_config_path=TTS_FILTER_PATH,
                force=True,
                lang=self.lang_code,
            )

            if not self.cancellation_requested:
                for f in [self.paths.video(page), self.paths.final_video, self.paths.final_ja_srt, self.paths.final_en_srt]:
                    if f.exists():
                        f.unlink()

                update_status('完了しました！', 1.0, current_page=page)
                push_log(f'[TTS] スライド {page} の音声を保存しました: {out_mp3.name}')
                await asyncio.sleep(0.3)
                ui.notify(f'スライド {page} の音声を合成しました．', type='positive')
                await self.refresh_editor()
                await self.refresh_slide_videos()
        except asyncio.CancelledError:
            ui.notify('音声合成を中断しました．', type='warning')
        except Exception as exc:
            ui.notify(f'音声合成に失敗しました: {exc}', type='negative')
        finally:
            self.current_task = None
            self.set_processing(False)
            GLOBAL_EXECUTION_LOCK.release()
            dialog.close()

    async def refresh_gallery(self) -> None:
        if not self.gallery:
            return
        self.gallery.clear()
        if not self.images:
            self.gallery.add(ui.label('プレゼンテーションPDFを選択してください．'))
            return
        with self.gallery:
            with ui.row().classes('w-full items-center justify-between pb-2'):
                ui.label(f'全 {self.total_slides} スライド（カードをクリックまたはチェックボックスで対象を切り替えられます）').classes('text-sm text-gray-500 dark:text-gray-400')
                with ui.row().classes('gap-2'):
                    ui.button('全スライドを選択', on_click=self.select_all_slides).props('dense outline size=sm')
                    ui.button('すべて解除', on_click=self.clear_all_slides).props('dense outline size=sm color=negative')

            with ui.grid(columns=5).classes('w-full gap-4'):
                for n, img in enumerate(self.images, 1):
                    is_active = n in self._selected_pages
                    card_classes = 'w-full cursor-pointer transition-all border p-3 rounded-lg gap-2 bg-zinc-800 border-zinc-700 shadow-sm '
                    img_classes = 'w-full rounded shadow-sm'

                    if not is_active:
                        card_classes += 'opacity-40 hover:border-zinc-600'
                        img_classes += ' grayscale'

                    card = ui.card().classes(card_classes)
                    with card:
                        card.on('click', lambda _, page=n, cur=is_active: self.toggle_slide_active(page, not cur))
                        ui.image(file_url(img)).classes(img_classes)

                        with ui.row().classes('w-full items-center justify-between pt-1'):
                            ui.checkbox(
                                f'スライド {n}',
                                value=is_active,
                                on_change=lambda e, page=n: self.toggle_slide_active(page, e.value),
                            ).props('dense dark color=blue').classes('text-white font-medium text-sm')

    async def refresh_simple_editor(self) -> None:
        if not self.simple_edit_container:
            return
        self.simple_edit_container.clear()
        if not self.active_pages or not self.pdf or not self.paths:
            with self.simple_edit_container:
                ui.label('⚠ 処理対象となるスライドがありません．「スライドデッキ」タブで対象スライドを選択してください．').classes('text-orange-500 dark:text-orange-400')
            return

        with self.simple_edit_container:
            ui.label(f'対象スライド一覧（全 {len(self.active_pages)} スライド）').classes('text-sm text-gray-400 pb-2')

            with ui.column().classes('w-full gap-6'):
                for p in self.active_pages:
                    img_path = self.paths.page_image(p)
                    txt_path = self.paths.explanation(p)
                    curr_txt = txt_path.read_text(encoding='utf-8') if txt_path.exists() else ''

                    with ui.card().classes('w-full p-4 bg-zinc-900 border border-zinc-800 rounded-lg shadow-sm'):
                        with ui.row().classes('w-full items-start gap-4 flex-nowrap'):
                            with ui.column().classes('w-72 shrink-0 items-center gap-1.5'):
                                if img_path.exists():
                                    ui.image(file_url(img_path)).classes('w-full rounded border border-zinc-700 shadow-sm')
                                else:
                                    ui.label('（スライド画像なし）').classes('text-xs text-zinc-500')
                                ui.label(f'スライド {p}').classes('text-sm font-bold text-zinc-300 self-center')

                            with ui.column().classes('grow gap-2'):
                                ta = ui.textarea(
                                    label=f'ナレーション原稿（スライド {p}）',
                                    value=curr_txt,
                                ).props('outlined').classes('w-full h-[180px]').style(
                                    'height: 180px; min-height: 180px;'
                                )
                                ta.props('input-style="height: 140px; resize: vertical;"')

                                with ui.row().classes('w-full items-center justify-end gap-3 pt-1'):
                                    ui.button(
                                        '✨ ナレーションを再生成',
                                        on_click=lambda _, page=p, area=ta: self.regenerate_narration(page, area),
                                    ).props('outline dense')

                                    ui.button(
                                        '🔄 保存して字幕・ポインタを再解析',
                                        on_click=lambda _, page=p, area=ta: self.save_and_realign(page, area.value),
                                    ).props('dense color=primary')

    async def refresh_editor(self) -> None:
        if not self.edit_container:
            return
        self.edit_container.clear()
        if not self.active_pages or not self.pdf or not self.paths:
            with self.edit_container:
                ui.label('⚠ 処理対象となるスライドがありません．「スライドデッキ」タブで対象スライドを選択してください．').classes('text-orange-500 dark:text-orange-400')
            return
        page = self.edit_page or self.active_pages[0]
        self.edit_page = page
        idx = self.active_pages.index(page)
        align_data, blocks, alignments = self.alignment_data(page)
        img = self.paths.page_image(page)
        current_text_path = self.paths.explanation(page)
        current_text = current_text_path.read_text(encoding='utf-8') if current_text_path.exists() else ''

        page_vmode = self.slide_visual_modes.get(str(page), 'vlm' if self.default_use_vlm else 'pdf')
        page_use_vlm_init = (page_vmode == 'vlm')

        with self.edit_container:
            with ui.row().classes('w-full items-center justify-between pb-2'):
                with ui.row().classes('items-center gap-3'):
                    sel = ui.select(
                        {p: f'スライド {p} ({self.get_slide_order(p)} / {len(self.active_pages)} スライド)' for p in self.active_pages},
                        value=page,
                        label='編集スライド選択',
                    ).classes('min-w-[280px] max-w-[340px]').props('outlined dense')
                    sel.on_value_change(lambda e: asyncio.create_task(self.select_edit_page(e.value)))

                    ui.button('◀ 前', on_click=self.prev_edit).props(f'disable={idx == 0} outlined dense')
                    ui.button('次 ▶', on_click=self.next_edit).props(f'disable={idx == len(self.active_pages)-1} outlined dense')

                # スライド個別トグル
                with ui.row().classes('items-center gap-2 p-1 px-3 bg-zinc-900 border border-zinc-800 rounded-lg'):
                    page_vlm_switch = ui.switch(
                        'VLMを活用してポインタ配置を決定する',
                        value=page_use_vlm_init,
                    ).props('dense color=primary')
                    page_vlm_switch.tooltip('ONにすると図やグラフ・数式の内部要素まで細かくポインティングします．OFFにするとPDFテキストのみを使用します．')

            with ui.row().classes('w-full items-start gap-6 flex-nowrap'):
                with ui.column().classes('flex-[2] min-w-0 gap-3'):
                    if img.exists():
                        preview = draw_block_preview(img, blocks) if blocks else None
                        if preview is not None:
                            preview_path = self.paths.page_preview(page)
                            preview.save(preview_path)
                            ts = int(time.time() * 1000)
                            ui.image(f"{file_url(preview_path)}&ts={ts}").classes('w-full rounded border border-zinc-700 shadow-sm')
                        else:
                            ui.image(file_url(img)).classes('w-full rounded border border-zinc-700 shadow-sm')

                        with ui.row().classes('w-full items-center justify-center gap-3 text-xs pt-1'):
                            with ui.row().classes('items-center gap-1'):
                                ui.element('div').classes('w-3 h-3 bg-red-500 rounded-sm')
                                ui.label('赤: VLM画像内要素').classes('text-zinc-300')
                            with ui.row().classes('items-center gap-1'):
                                ui.element('div').classes('w-3 h-3 bg-amber-500 rounded-sm')
                                ui.label('橙: 図全体').classes('text-zinc-300')
                            with ui.row().classes('items-center gap-1'):
                                ui.element('div').classes('w-3 h-3 bg-blue-500 rounded-sm')
                                ui.label('青: テキスト/行').classes('text-zinc-300')

                    with ui.card().classes('w-full p-3 bg-zinc-900 border border-zinc-800 rounded-lg gap-2'):
                        ui.label('🔊 ナレーション音声').classes('text-sm font-bold text-zinc-200')
                        ap = self.paths.audio(page)
                        if ap.exists():
                            ui.audio(file_url(ap)).classes('w-full')
                            ui.button(
                                '🔄 このスライドの音声を再生成',
                                on_click=lambda _, p=page: self.generate_slide_tts(p),
                            ).props('dense outline size=sm').classes('w-full mt-1')
                        else:
                            ui.label('（音声未生成）').classes('text-xs text-zinc-500 py-1')
                            ui.button(
                                '🔊 このスライドの音声を生成',
                                on_click=lambda _, p=page: self.generate_slide_tts(p),
                            ).props('dense color=primary size=sm').classes('w-full')

                with ui.column().classes('flex-[3] min-w-0 gap-3'):
                    ui.label('🎯 検出された要素一覧（バッジ対応）').classes('text-xs font-semibold text-zinc-400')
                    badge_colors = {
                        'image_subpart': ('bg-red-500/20 text-red-300 border-red-500/40', '注目要素'),
                        'image': ('bg-amber-500/20 text-amber-300 border-amber-500/40', '画像・領域'),
                        'text': ('bg-blue-500/20 text-blue-300 border-blue-500/40', 'テキスト'),
                        'code_line': ('bg-cyan-500/20 text-cyan-300 border-cyan-500/40', 'コード/数式'),
                    }
                    if not blocks:
                        ui.label('（検出された要素はありません）').classes('text-xs text-zinc-500 italic')
                    else:
                        with ui.column().classes('w-full gap-1.5 max-h-[450px] overflow-y-auto pr-1'):
                            for b in blocks:
                                bid = b.get('block_id')
                                label = b.get('label') or '名称なし'
                                btype = b.get('type', 'image_subpart')
                                style_cls, type_name = badge_colors.get(btype, badge_colors['image_subpart'])

                                with ui.row().classes(
                                    'w-full items-center justify-between p-2 rounded bg-zinc-900/90 '
                                    'border border-zinc-800 hover:border-zinc-700 transition-colors'
                                ):
                                    with ui.row().classes('items-center gap-2 grow'):
                                        ui.label(f'#{bid}').classes(
                                            f'text-xs font-bold font-mono px-2 py-0.5 rounded border {style_cls}'
                                        )
                                        ui.label(label).classes('text-xs text-zinc-200 font-medium break-all')

                                    ui.badge(type_name).props('outline').classes('text-[10px] text-zinc-400 shrink-0')

            with ui.column().classes('w-full gap-3 pt-2'):
                main_lang = '日本語' if self.lang_code == 'ja' else '英語'
                text_area = ui.textarea(
                    f'主言語ナレーション原稿（{main_lang}）',
                    value=current_text,
                ).props('outlined').classes('w-full').style('min-height: 180px')
                with ui.row().classes('w-full gap-2'):
                    ui.button('✨ ナレーションを再生成',
                              on_click=lambda: self.regenerate_narration(page, text_area)).classes('grow').props('outline')
                    ui.button('🔄 保存して字幕・ポインタを再解析',
                              on_click=lambda: self.save_and_realign(page, text_area.value, page_vlm_switch.value)).classes('grow').props('color=primary')

                if alignments:
                    ui.separator()
                    ui.label('🎯 文ごとのポインタ先 & 対訳字幕の微調整').classes('text-h6')
                    block_options = [None] + [b['block_id'] for b in blocks]
                    rows = []
                    for item in alignments:
                        rows.append({
                            'sentence': item.get('sentence', ''),
                            'block_id': item.get('block_id'),
                            'translation': item.get('en_sentence', '') if self.lang_code == 'ja' else item.get('ja_sentence', ''),
                        })

                    for i, row in enumerate(rows, 1):
                        with ui.card().classes('w-full p-4 bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg gap-2 shadow-xs'):
                            with ui.row().classes('w-full items-start gap-2'):
                                ui.label(f'文 {i}').classes('text-xs font-bold text-white bg-blue-600 dark:bg-blue-500 px-2 py-0.5 rounded shrink-0 mt-0.5')
                                ui.label(row['sentence']).classes('text-sm font-medium text-gray-900 dark:text-gray-100 grow leading-relaxed')

                            with ui.row().classes('w-full items-center gap-3 pt-1'):
                                select = ui.select(block_options, value=row['block_id'], label='ポインタ先').props('dense outlined').classes('w-44 shrink-0')
                                trans = ui.input('対訳字幕', value=row['translation']).props('dense outlined').classes('grow')
                                row['_select'] = select
                                row['_trans'] = trans

                    async def save_rows() -> None:
                        out = []
                        for row in rows:
                            bid = row['_select'].value
                            tr = row['_trans'].value or ''
                            if self.lang_code == 'ja':
                                out.append({'sentence': row['sentence'], 'block_id': bid, 'ja_sentence': row['sentence'], 'en_sentence': tr})
                            else:
                                out.append({'sentence': row['sentence'], 'block_id': bid, 'en_sentence': row['sentence'], 'ja_sentence': tr})
                        await self.save_alignment(page, out, align_data)

                    ui.button('💾 字幕・ポインタ修正を保存', on_click=save_rows).classes('w-full mt-2')

    async def refresh_slide_videos(self) -> None:
        if not self.slide_video_gallery:
            return
        self.slide_video_gallery.clear()
        if not self.paths:
            with self.slide_video_gallery:
                ui.label('プレゼンテーションPDFを選択してください．').classes('text-blue-500 dark:text-blue-400')
            return

        with self.slide_video_gallery:
            with ui.grid(columns=4).classes('w-full gap-4'):
                for p in self.active_pages:
                    vp = self.paths.video(p)
                    img = self.paths.page_image(p)
                    with ui.card().classes('w-full p-3 gap-2 rounded-lg bg-zinc-800 border border-zinc-700 shadow-sm'):
                        if vp.exists():
                            ui.video(file_url(vp)).classes('w-full rounded shadow-sm')
                        else:
                            if img.exists():
                                ui.image(file_url(img)).classes('w-full opacity-60 rounded shadow-sm')
                            ui.label('（単体動画 未生成）').classes('text-xs text-orange-400 font-medium')
                        ui.label(f'スライド {p}').classes('text-sm font-bold text-white self-center pt-1')

    async def refresh_final_video(self) -> None:
        if not self.final_video_container:
            return
        self.final_video_container.clear()
        if not self.paths:
            with self.final_video_container:
                ui.label('PDFを選択してください．').classes('text-xs text-zinc-500')
            return

        final = self.paths.final_video

        with self.final_video_container:
            if final.exists():
                ui.video(file_url(final)).classes('w-full rounded shadow-md')
                with ui.row().classes('w-full items-center justify-between pt-1'):
                    ui.link(
                        '🎬 動画 (MP4) をダウンロード',
                        '/download/' + final.resolve().relative_to(Path.cwd().resolve()).as_posix(),
                        new_tab=True,
                    ).classes('text-primary font-bold text-xs')
            else:
                ui.label('動画未生成').classes('text-xs text-zinc-500')

    def _build_llm_settings(self) -> tuple[ui.input, ui.input, ui.select, ui.number]:
        with ui.card().classes('w-full p-5 bg-zinc-900 border border-zinc-800 rounded-xl gap-4 shadow-sm'):
            with ui.row().classes('w-full items-center justify-between border-b border-zinc-800 pb-2'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('psychology', size='sm').classes('text-blue-400')
                    ui.label('メイン LLM / VLM 設定 (`config.yaml: llm`)').classes('text-lg font-bold text-zinc-100')
                ui.label('ナレーション原稿作成・字幕対訳・VLM画像アライメント解析').classes('text-xs text-zinc-400')

            llm = self.cfg.setdefault('llm', {})
            current_model = llm.get('model', '')

            with ui.row().classes('w-full items-center gap-4'):
                llm_base = ui.input('LLM Base URL', value=llm.get('base_url', '')).classes('grow')
                llm_key = ui.input('LLM API Key', value=llm.get('api_key', 'dummy'), password=True, password_toggle_button=True).classes('w-72')

            with ui.row().classes('w-full items-center gap-4'):
                initial_options = [current_model] if current_model else []
                llm_model_select = ui.select(
                    options=initial_options,
                    value=current_model,
                    label='LLM Model (選択または直接入力)',
                ).props('use-input new-value-mode="add-unique" outlined dense').classes('grow')

                fetch_btn = ui.button('🔄 モデル一覧を取得').props('dense outline')
                llm_temp = ui.number('Temperature', value=float(llm.get('temperature', 0.3)), min=0, max=2, step=0.1).classes('w-36')

            async def fetch_llm_models() -> None:
                base = (llm_base.value or '').strip()
                key = (llm_key.value or '').strip() or 'dummy'
                if not base:
                    ui.notify('先に Base URL を入力してください．', type='warning')
                    return
                fetch_btn.disable()
                try:
                    client = await run.io_bound(make_client, {'base_url': base, 'api_key': key})
                    models_resp = await run.io_bound(client.models.list)
                    model_ids = sorted([m.id for m in models_resp.data])
                    if not model_ids:
                        ui.notify('モデルが見つかりませんでした．', type='warning')
                        return
                    llm_model_select.options = model_ids
                    if llm_model_select.value not in model_ids and model_ids:
                        llm_model_select.value = model_ids[0]
                    llm_model_select.update()
                    ui.notify(f'{len(model_ids)} 個のモデルを取得しました．', type='positive')
                except Exception as err:
                    ui.notify(f'モデル一覧取得に失敗しました: {err}', type='negative')
                finally:
                    fetch_btn.enable()

            fetch_btn.on_click(fetch_llm_models)

            # テキスト接続テスト
            with ui.card().classes('w-full p-3 bg-zinc-950 border border-zinc-800 rounded-lg gap-2'):
                ui.label('🧪 LLM 接続テスト（テキスト応答）').classes('text-xs font-bold text-zinc-300')
                with ui.row().classes('w-full items-center gap-2'):
                    llm_test_input = ui.input(
                        'テストプロンプト',
                        value='こんにちは！自己紹介を1文でしてください．',
                    ).props('dense outlined').classes('grow')
                    llm_test_btn = ui.button('💬 テスト送信').props('dense outline')

                llm_test_result = ui.label('').classes('text-xs text-zinc-300 font-mono p-2 bg-zinc-900 border border-zinc-800 rounded min-h-[36px] w-full whitespace-pre-wrap')

                async def run_llm_test() -> None:
                    prompt = (llm_test_input.value or '').strip()
                    if not prompt:
                        ui.notify('プロンプトを入力してください．', type='warning')
                        return
                    selected_model = (str(llm_model_select.value) if llm_model_select.value is not None else '').strip()
                    if not llm_base.value or not selected_model:
                        ui.notify('Base URL と Model を入力してください．', type='warning')
                        return

                    llm_test_btn.disable()
                    llm_test_result.text = 'LLMにリクエスト中…'
                    try:
                        test_cfg = {
                            'base_url': llm_base.value.strip(),
                            'api_key': (llm_key.value or '').strip() or 'dummy',
                        }
                        client = await run.io_bound(make_client, test_cfg)

                        def call_llm() -> str:
                            res = client.chat.completions.create(
                                model=selected_model,
                                temperature=float(llm_temp.value or 0.3),
                                max_tokens=5000,
                                messages=[{'role': 'user', 'content': prompt}],
                                extra_body={"reasoning_effort": "none"},
                            )
                            return res.choices[0].message.content or '（空の応答でした）'

                        answer = await run.io_bound(call_llm)
                        llm_test_result.text = answer
                        ui.notify('LLMからの応答を受信しました．', type='positive')
                    except Exception as err:
                        llm_test_result.text = f'【エラー】\n{err}'
                        ui.notify(f'LLM接続テストに失敗しました: {err}', type='negative')
                    finally:
                        llm_test_btn.enable()

                llm_test_btn.on_click(run_llm_test)

            # --- VLM マルチモーダル接続テスト ---
            with ui.card().classes('w-full p-3 bg-zinc-950 border border-zinc-800 rounded-lg gap-3 mt-2'):
                with ui.row().classes('items-center justify-between w-full'):
                    ui.label('👁️ VLM 画像認識 & アライメントテスト').classes('text-xs font-bold text-zinc-300')
                    ui.label('任意の画像（写真・イラスト・図表など何でも可）を放り込んで認識とバウンディングボックス抽出をテスト').classes('text-xs text-zinc-500')

                uploaded_vlm_img = {'path': None}

                with ui.row().classes('w-full items-start gap-4'):
                    with ui.column().classes('w-80 shrink-0 gap-2'):
                        vlm_uploader = ui.upload(
                            label='画像をアップロード（何でも可）',
                            auto_upload=True,
                            max_files=1,
                        ).props('accept="image/*" dense').classes('w-full')

                        vlm_run_btn = ui.button('🔍 VLM解析を実行', color='primary').props('dense outline').classes('w-full')
                        vlm_run_btn.disable()

                        async def handle_vlm_upload(e):
                            ext = Path(e.file.name).suffix or '.png'
                            saved_path = self.test_vlm_dir / f'vlm_test_input{ext}'
                            await e.file.save(saved_path)
                            uploaded_vlm_img['path'] = saved_path
                            vlm_run_btn.enable()
                            vlm_preview_container.clear()
                            vlm_blocks_container.clear()
                            with vlm_preview_container:
                                ui.image(file_url(saved_path)).classes('w-full rounded border border-zinc-700 shadow-sm')
                            ui.notify('テスト画像を読み込みました．「VLM解析を実行」を押してください．', type='info')

                        vlm_uploader.on_upload(handle_vlm_upload)

                    with ui.column().classes('grow gap-2'):
                        vlm_status_label = ui.label('画像をアップロードしてテストを開始してください．').classes('text-xs text-zinc-400')
                        vlm_description_box = ui.label('').classes('text-xs text-zinc-300 font-sans p-3 bg-zinc-900 border border-zinc-800 rounded min-h-[48px] w-full whitespace-pre-wrap leading-relaxed')

                vlm_preview_container = ui.column().classes('w-full')
                vlm_blocks_container = ui.column().classes('w-full')

                async def run_vlm_test() -> None:
                    img_p: Path | None = uploaded_vlm_img['path']
                    if not img_p or not img_p.exists():
                        ui.notify('テスト画像をアップロードしてください．', type='warning')
                        return
                    selected_model = (str(llm_model_select.value) if llm_model_select.value is not None else '').strip()
                    if not llm_base.value or not selected_model:
                        ui.notify('Base URL と Model を設定してください．', type='warning')
                        return

                    vlm_run_btn.disable()
                    vlm_status_label.text = 'VLMで画像認識・要素検出中…'
                    vlm_description_box.text = '解析中…'

                    try:
                        test_cfg = {
                            'base_url': llm_base.value.strip(),
                            'api_key': (llm_key.value or '').strip() or 'dummy',
                        }
                        client = await run.io_bound(make_client, test_cfg)

                        prompt = (
                            'この画像について以下の2つを行ってください。\n'
                            '1. 何が写っているか、状況や内容を日本語で2〜3文で簡潔に説明してください。\n'
                            '2. 画像内の主要な物体、被写体、人物、テキスト、アイコン、図表要素などを検出し、'
                            'そのバウンディングボックスをJSON形式（```json ... ```）で出力してください。\n'
                            'フォーマット仕様:\n'
                            '{\n'
                            '  "summary": "画像の説明文",\n'
                            '  "blocks": [\n'
                            f'    {{"block_id": 1, "type": "image_subpart", "label": "要素名（例: 猫, 人物, タイトル文字列など）", "bbox": {VLM_BBOX_FORMAT}}}\n'
                            '  ]\n'
                            '}\n'
                            f"※ bbox は {VLM_BBOX_FORMAT} の順で、0〜{VLM_BBOX_COORD_MAX:g} の正規化座標（横の割合がxmin/xmax、縦の割合がymin/ymax）で出力してください。\n"
                            "※ bbox は必ず4個の数値を指定してください。値を省略したり、末尾に余分なカンマを付けたりしないでください。\n"
                            "※ type は 'image_subpart'（注目物体・図形）, 'text'（文字領域）, 'image'（大きな領域）, 'code_line'（コードや数式）のいずれかを指定してください。"
                        )

                        summary_text, parsed_blocks = await run.io_bound(
                            run_vlm_element_detection,
                            client=client,
                            model=selected_model,
                            image_path=img_p,
                            prompt=prompt,
                            temperature=float(llm_temp.value or 0.2),
                            max_tokens=1500,
                        )

                        vlm_description_box.text = summary_text.strip()
                        vlm_status_label.text = f'解析完了: {len(parsed_blocks)} 個の要素を検出しました．'

                        preview_path = self.test_vlm_dir / 'vlm_preview_result.png'
                        if parsed_blocks:
                            preview_im = draw_block_preview(img_p, parsed_blocks)
                            preview_im.save(preview_path)
                        else:
                            shutil.copy(img_p, preview_path)

                        ts = int(time.time() * 1000)
                        vlm_preview_container.clear()

                        badge_colors = {
                            'image_subpart': ('bg-red-500/20 text-red-300 border-red-500/40', '注目要素'),
                            'image': ('bg-amber-500/20 text-amber-300 border-amber-500/40', '画像・領域'),
                            'text': ('bg-blue-500/20 text-blue-300 border-blue-500/40', 'テキスト'),
                            'code_line': ('bg-cyan-500/20 text-cyan-300 border-cyan-500/40', 'コード/数式'),
                        }

                        with vlm_preview_container:
                            ui.label('🎯 アライメントプレビュー & 検出要素一覧').classes('text-xs font-bold text-zinc-300 mt-2')

                            with ui.row().classes('w-full items-start gap-4'):
                                ui.image(f'{file_url(preview_path)}&ts={ts}').classes(
                                    'w-full max-w-xl rounded-lg border border-zinc-700 shadow-md'
                                )

                                with ui.column().classes('grow min-w-[280px] max-w-md gap-2'):
                                    ui.label('検出された要素一覧（バッジ対応）').classes('text-xs font-semibold text-zinc-400')

                                    if not parsed_blocks:
                                        ui.label('（バウンディングボックスは検出されませんでした）').classes('text-xs text-zinc-500 italic')
                                    else:
                                        with ui.column().classes('w-full gap-1.5 max-h-[450px] overflow-y-auto pr-1'):
                                            for b in parsed_blocks:
                                                bid = b.get('block_id')
                                                label = b.get('label') or '名称なし'
                                                btype = b.get('type', 'image_subpart')
                                                style_cls, type_name = badge_colors.get(btype, badge_colors['image_subpart'])

                                                with ui.row().classes(
                                                    'w-full items-center justify-between p-2 rounded bg-zinc-900/90 '
                                                    'border border-zinc-800 hover:border-zinc-700 transition-colors'
                                                ):
                                                    with ui.row().classes('items-center gap-2 grow'):
                                                        ui.label(f'#{bid}').classes(
                                                            f'text-xs font-bold font-mono px-2 py-0.5 rounded border {style_cls}'
                                                        )
                                                        ui.label(label).classes('text-xs text-zinc-200 font-medium break-all')

                                                    ui.badge(type_name).props('outline').classes('text-[10px] text-zinc-400 shrink-0')

                        vlm_blocks_container.clear()
                        if parsed_blocks:
                            with vlm_blocks_container:
                                with ui.expansion('検出要素の生データ (JSON)', icon='code').classes('w-full max-w-3xl border border-zinc-800 rounded text-xs'):
                                    ui.code(json.dumps(parsed_blocks, ensure_ascii=False, indent=2), language='json').classes('w-full bg-zinc-950')

                        ui.notify('VLM 解析とプレビュー生成が完了しました．', type='positive')

                    except Exception as err:
                        vlm_status_label.text = '解析エラーが発生しました．'
                        vlm_description_box.text = f'【エラー】\n{err}'
                        ui.notify(f'VLM テストに失敗しました: {err}', type='negative')
                    finally:
                        vlm_run_btn.enable()

                vlm_run_btn.on_click(run_vlm_test)

        return llm_base, llm_key, llm_model_select, llm_temp

    @staticmethod
    def _fetch_voices_from_server(base_url: str, api_key: str) -> list[str]:
        url_clean = base_url.rstrip('/')
        candidate_urls = [
            f"{url_clean}/audio/voices",
            f"{url_clean}/voices",
        ]
        if url_clean.endswith('/v1'):
            root_url = url_clean[:-3]
            candidate_urls.extend([
                f"{root_url}/audio/voices",
                f"{root_url}/voices",
            ])

        headers = {'Authorization': f'Bearer {api_key}'} if api_key and api_key != 'dummy' else {}

        with httpx.Client(trust_env=False, verify=False, timeout=8.0) as client:
            for endpoint in candidate_urls:
                try:
                    resp = client.get(endpoint, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        voices: list[str] = []
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, str):
                                    voices.append(item)
                                elif isinstance(item, dict):
                                    v_id = item.get('id') or item.get('voice_id') or item.get('name')
                                    if v_id:
                                        voices.append(str(v_id))
                        elif isinstance(data, dict):
                            v_list = data.get('voices') or data.get('data') or []
                            if isinstance(v_list, list):
                                for item in v_list:
                                    if isinstance(item, str):
                                        voices.append(item)
                                    elif isinstance(item, dict):
                                        v_id = item.get('id') or item.get('voice_id') or item.get('name')
                                        if v_id:
                                            voices.append(str(v_id))
                        if voices:
                            return sorted(set(voices))
                except Exception:
                    continue
        return []

    def _build_tts_settings(self) -> tuple[tuple[ui.input, ui.input, ui.select, ui.select], tuple[ui.input, ui.input, ui.select, ui.select]]:
        tts = self.cfg.setdefault('tts', {})
        ja = tts.setdefault('ja', {})
        en = tts.setdefault('en', {})

        with ui.card().classes('w-full p-5 bg-zinc-900 border border-zinc-800 rounded-xl gap-4 shadow-sm mt-2'):
            with ui.row().classes('w-full items-center justify-between border-b border-zinc-800 pb-2'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('🇯🇵').classes('text-xl')
                    ui.label('日本語 TTS 設定 (`config.yaml: tts.ja`)').classes('text-lg font-bold text-zinc-100')
                ui.label('※互換API未実装のTTSサーバでは一覧取得できません（直接入力可）').classes('text-xs text-zinc-400')

            with ui.row().classes('w-full items-center gap-4'):
                ja_base = ui.input('日本語 Base URL', value=ja.get('base_url', '')).classes('grow')
                ja_key = ui.input('日本語 API Key', value=ja.get('api_key', 'dummy'), password=True, password_toggle_button=True).classes('w-72')

            cur_ja_model = ja.get('model', '')
            cur_ja_voice = ja.get('voice', '')

            with ui.row().classes('w-full items-center gap-4'):
                ja_model_select = ui.select(
                    options=[cur_ja_model] if cur_ja_model else [],
                    value=cur_ja_model,
                    label='日本語 Model',
                ).props('use-input new-value-mode="add-unique" outlined dense').classes('grow')

                ja_model_fetch_btn = ui.button('🔄 モデル取得').props('dense outline')

                ja_voice_select = ui.select(
                    options=[cur_ja_voice] if cur_ja_voice else [],
                    value=cur_ja_voice,
                    label='日本語 Voice / 話者',
                ).props('use-input new-value-mode="add-unique" outlined dense').classes('grow')

                ja_voice_fetch_btn = ui.button('🗣 ボイス取得').props('dense outline')

            async def fetch_ja_models() -> None:
                base = (ja_base.value or '').strip()
                key = (ja_key.value or '').strip() or 'dummy'
                if not base:
                    ui.notify('先に日本語 TTS Base URL を入力してください．', type='warning')
                    return
                ja_model_fetch_btn.disable()
                try:
                    client = await run.io_bound(make_client, {'base_url': base, 'api_key': key})
                    models_resp = await run.io_bound(client.models.list)
                    model_ids = sorted([m.id for m in models_resp.data])
                    if not model_ids:
                        ui.notify('モデルが見つかりませんでした．', type='warning')
                        return
                    ja_model_select.options = model_ids
                    if ja_model_select.value not in model_ids:
                        ja_model_select.value = model_ids[0]
                    ja_model_select.update()
                    ui.notify(f'{len(model_ids)} 個のモデルを取得しました．', type='positive')
                except Exception as err:
                    ui.notify(f'モデル一覧取得に失敗しました（/v1/models 未対応の可能性）: {err}', type='negative')
                finally:
                    ja_model_fetch_btn.enable()

            async def fetch_ja_voices() -> None:
                base = (ja_base.value or '').strip()
                key = (ja_key.value or '').strip() or 'dummy'
                if not base:
                    ui.notify('先に日本語 TTS Base URL を入力してください．', type='warning')
                    return
                ja_voice_fetch_btn.disable()
                try:
                    voices = await run.io_bound(self._fetch_voices_from_server, base, key)
                    if not voices:
                        ui.notify('ボイス一覧を取得できませんでした（一覧API未対応の可能性）．', type='warning')
                        return
                    ja_voice_select.options = voices
                    if ja_voice_select.value not in voices:
                        ja_voice_select.value = voices[0]
                    ja_voice_select.update()
                    ui.notify(f'{len(voices)} 件のボイスを取得しました．', type='positive')
                except Exception as err:
                    ui.notify(f'ボイス一覧取得に失敗しました: {err}', type='negative')
                finally:
                    ja_voice_fetch_btn.enable()

            ja_model_fetch_btn.on_click(fetch_ja_models)
            ja_voice_fetch_btn.on_click(fetch_ja_voices)

            with ui.card().classes('w-full p-3 bg-zinc-950 border border-zinc-800 rounded-lg gap-2'):
                ui.label('🧪 日本語 TTS 接続・音声再生テスト').classes('text-xs font-bold text-zinc-300')
                with ui.row().classes('w-full items-center gap-2'):
                    ja_test_text = ui.input(
                        '読み上げテキスト',
                        value='こんにちは。日本語の音声合成テストです。正常に聞こえますか？',
                    ).props('dense outlined').classes('grow')
                    ja_test_btn = ui.button('🔊 音声を生成・再生').props('dense outline')

                ja_audio_container = ui.column().classes('w-full')

                async def run_ja_tts_test() -> None:
                    txt = (ja_test_text.value or '').strip()
                    if not txt:
                        ui.notify('読み上げテキストを入力してください．', type='warning')
                        return
                    m = (str(ja_model_select.value) if ja_model_select.value is not None else '').strip()
                    v = (str(ja_voice_select.value) if ja_voice_select.value is not None else '').strip()
                    if not ja_base.value or not m or not v:
                        ui.notify('日本語 TTS の Base URL, Model, Voice を指定してください．', type='warning')
                        return

                    ja_test_btn.disable()
                    ja_audio_container.clear()
                    with ja_audio_container:
                        ui.label('音声を合成中…').classes('text-xs text-zinc-400')
                    try:
                        tts_cfg = {
                            'base_url': ja_base.value.strip(),
                            'api_key': (ja_key.value or '').strip() or 'dummy',
                            'model': m,
                            'voice': v,
                            'response_format': 'mp3',
                        }
                        out_path = self.test_audio_dir / 'test_ja.mp3'

                        def call_tts():
                            cl = make_client(tts_cfg)
                            resp = cl.audio.speech.create(
                                model=tts_cfg['model'],
                                voice=tts_cfg['voice'],
                                input=txt,
                                response_format='mp3',
                            )
                            resp.write_to_file(out_path)

                        await run.io_bound(call_tts)
                        ja_audio_container.clear()
                        ts = int(time.time() * 1000)
                        cached_url = f"{file_url(out_path)}?t={ts}"
                        with ja_audio_container:
                            ui.audio(cached_url).props('autoplay').classes('w-full max-w-lg mt-1')
                            ui.label(f'Model: {m} / Voice: {v} で生成完了').classes('text-xs text-zinc-400')
                        ui.notify('日本語音声を合成しました．', type='positive')
                    except Exception as err:
                        ja_audio_container.clear()
                        with ja_audio_container:
                            ui.label(f'【TTS合成失敗】: {err}').classes('text-xs text-red-400')
                        ui.notify(f'日本語 TTS テストに失敗しました: {err}', type='negative')
                    finally:
                        ja_test_btn.enable()

            ja_test_btn.on_click(run_ja_tts_test)

        with ui.card().classes('w-full p-5 bg-zinc-900 border border-zinc-800 rounded-xl gap-4 shadow-sm mt-2'):
            with ui.row().classes('w-full items-center justify-between border-b border-zinc-800 pb-2'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('🇺🇸').classes('text-xl')
                    ui.label('英語 TTS 設定 (`config.yaml: tts.en`)').classes('text-lg font-bold text-zinc-100')
                ui.label('※互換API未実装のTTSサーバでは一覧取得できません（直接入力可）').classes('text-xs text-zinc-400')

            with ui.row().classes('w-full items-center gap-4'):
                en_base = ui.input('英語 Base URL', value=en.get('base_url', '')).classes('grow')
                en_key = ui.input('英語 API Key', value=en.get('api_key', 'dummy'), password=True, password_toggle_button=True).classes('w-72')

            cur_en_model = en.get('model', '')
            cur_en_voice = en.get('voice', '')

            with ui.row().classes('w-full items-center gap-4'):
                en_model_select = ui.select(
                    options=[cur_en_model] if cur_en_model else [],
                    value=cur_en_model,
                    label='英語 Model',
                ).props('use-input new-value-mode="add-unique" outlined dense').classes('grow')

                en_model_fetch_btn = ui.button('🔄 モデル取得').props('dense outline')

                en_voice_select = ui.select(
                    options=[cur_en_voice] if cur_en_voice else [],
                    value=cur_en_voice,
                    label='英語 Voice / 話者',
                ).props('use-input new-value-mode="add-unique" outlined dense').classes('grow')

                en_voice_fetch_btn = ui.button('🗣 ボイス取得').props('dense outline')

            async def fetch_en_models() -> None:
                base = (en_base.value or '').strip()
                key = (en_key.value or '').strip() or 'dummy'
                if not base:
                    ui.notify('先に英語 TTS Base URL を入力してください．', type='warning')
                    return
                en_model_fetch_btn.disable()
                try:
                    client = await run.io_bound(make_client, {'base_url': base, 'api_key': key})
                    models_resp = await run.io_bound(client.models.list)
                    model_ids = sorted([m.id for m in models_resp.data])
                    if not model_ids:
                        ui.notify('モデルが見つかりませんでした．', type='warning')
                        return
                    en_model_select.options = model_ids
                    if en_model_select.value not in model_ids:
                        en_model_select.value = model_ids[0]
                    en_model_select.update()
                    ui.notify(f'{len(model_ids)} 個のモデルを取得しました．', type='positive')
                except Exception as err:
                    ui.notify(f'モデル一覧取得に失敗しました: {err}', type='negative')
                finally:
                    en_model_fetch_btn.enable()

            async def fetch_en_voices() -> None:
                base = (en_base.value or '').strip()
                key = (en_key.value or '').strip() or 'dummy'
                if not base:
                    ui.notify('先に英語 TTS Base URL を入力してください．', type='warning')
                    return
                en_voice_fetch_btn.disable()
                try:
                    voices = await run.io_bound(self._fetch_voices_from_server, base, key)
                    if not voices:
                        ui.notify('ボイス一覧を取得できませんでした（一覧API未対応の可能性）．', type='warning')
                        return
                    en_voice_select.options = voices
                    if en_voice_select.value not in voices:
                        en_voice_select.value = voices[0]
                    en_voice_select.update()
                    ui.notify(f'{len(voices)} 件のボイスを取得しました．', type='positive')
                except Exception as err:
                    ui.notify(f'ボイス一覧取得に失敗しました: {err}', type='negative')
                finally:
                    en_voice_fetch_btn.enable()

            en_model_fetch_btn.on_click(fetch_en_models)
            en_voice_fetch_btn.on_click(fetch_en_voices)

            with ui.card().classes('w-full p-3 bg-zinc-950 border border-zinc-800 rounded-lg gap-2'):
                ui.label('🧪 英語 TTS 接続・音声再生テスト').classes('text-xs font-bold text-zinc-300')
                with ui.row().classes('w-full items-center gap-2'):
                    en_test_text = ui.input(
                        '読み上げテキスト (English)',
                        value='Hello! This is a test for English text-to-speech synthesis.',
                    ).props('dense outlined').classes('grow')
                    en_test_btn = ui.button('🔊 音声を生成・再生').props('dense outline')

                en_audio_container = ui.column().classes('w-full')

                async def run_en_tts_test() -> None:
                    txt = (en_test_text.value or '').strip()
                    if not txt:
                        ui.notify('Text is required.', type='warning')
                        return
                    m = (str(en_model_select.value) if en_model_select.value is not None else '').strip()
                    v = (str(en_voice_select.value) if en_voice_select.value is not None else '').strip()
                    if not en_base.value or not m or not v:
                        ui.notify('英語 TTS の Base URL, Model, Voice を指定してください．', type='warning')
                        return

                    en_test_btn.disable()
                    en_audio_container.clear()
                    with en_audio_container:
                        ui.label('Synthesizing speech...').classes('text-xs text-zinc-400')
                    try:
                        tts_cfg = {
                            'base_url': en_base.value.strip(),
                            'api_key': (en_key.value or '').strip() or 'dummy',
                            'model': m,
                            'voice': v,
                            'response_format': 'mp3',
                        }
                        out_path = self.test_audio_dir / 'test_en.mp3'

                        def call_tts():
                            cl = make_client(tts_cfg)
                            resp = cl.audio.speech.create(
                                model=tts_cfg['model'],
                                voice=tts_cfg['voice'],
                                input=txt,
                                response_format='mp3',
                            )
                            resp.write_to_file(out_path)

                        await run.io_bound(call_tts)
                        en_audio_container.clear()
                        with en_audio_container:
                            ui.audio(file_url(out_path)).props('autoplay').classes('w-full max-w-lg mt-1')
                            ui.label(f'Model: {m} / Voice: {v} done').classes('text-xs text-zinc-400')
                        ui.notify('英語音声を合成しました．', type='positive')
                    except Exception as err:
                        en_audio_container.clear()
                        with en_audio_container:
                            ui.label(f'【TTS合成失敗】: {err}').classes('text-xs text-red-400')
                        ui.notify(f'英語 TTS テストに失敗しました: {err}', type='negative')
                    finally:
                        en_test_btn.enable()

            en_test_btn.on_click(run_en_tts_test)

        return (ja_base, ja_key, ja_model_select, ja_voice_select), (en_base, en_key, en_model_select, en_voice_select)

    def _build_dict_editor(self) -> None:
        filter_cfg = load_tts_filter_config(TTS_FILTER_PATH) if TTS_FILTER_PATH.exists() else {}
        server_cfg = filter_cfg.setdefault('server', {})
        gen_cfg = filter_cfg.setdefault('generation', {})
        filter_dict = filter_cfg.setdefault('dictionary', {})

        with ui.card().classes('w-full p-5 bg-zinc-900 border border-zinc-800 rounded-xl gap-4 shadow-sm mt-4'):
            with ui.row().classes('w-full items-center justify-between border-b border-zinc-800 pb-2'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('translate', size='sm').classes('text-emerald-400')
                    ui.label('日本語TTS用ヨミ変換フィルタ (`tts_filter.yaml`)').classes('text-lg font-bold text-zinc-100')
                ui.label('技術用語・識別子・コマンド等の自動ヨミ変換').classes('text-xs text-zinc-400')

            current_filter_model = server_cfg.get('model', '')
            with ui.row().classes('w-full items-center gap-4'):
                filter_base = ui.input('フィルタ用 LLM Base URL', value=server_cfg.get('base_url', '')).classes('grow')
                filter_key = ui.input('フィルタ用 API Key', value=server_cfg.get('api_key', 'dummy'), password=True, password_toggle_button=True).classes('w-72')

            with ui.row().classes('w-full items-center gap-4'):
                initial_filter_options = [current_filter_model] if current_filter_model else []
                filter_model_select = ui.select(
                    options=initial_filter_options,
                    value=current_filter_model,
                    label='フィルタ用 LLM Model (選択または直接入力)',
                ).props('use-input new-value-mode="add-unique" outlined dense').classes('grow')

                filter_fetch_btn = ui.button('🔄 モデル一覧を取得').props('dense outline')
                filter_temp = ui.number('Temperature', value=float(gen_cfg.get('temperature', 0)), min=0, max=2, step=0.1).classes('w-36')

            async def fetch_filter_models() -> None:
                base = (filter_base.value or '').strip()
                key = (filter_key.value or '').strip() or 'dummy'
                if not base:
                    ui.notify('先にフィルタ用 Base URL を入力してください．', type='warning')
                    return
                filter_fetch_btn.disable()
                try:
                    client = await run.io_bound(make_tts_filter_client, {'server': {'base_url': base, 'api_key': key}})
                    models_resp = await run.io_bound(client.models.list)
                    model_ids = sorted([m.id for m in models_resp.data])
                    if not model_ids:
                        ui.notify('モデルが見つかりませんでした．', type='warning')
                        return
                    filter_model_select.options = model_ids
                    if filter_model_select.value not in model_ids and model_ids:
                        filter_model_select.value = model_ids[0]
                    filter_model_select.update()
                    ui.notify(f'{len(model_ids)} 個のモデルを取得しました．', type='positive')
                except Exception as err:
                    ui.notify(f'モデル一覧取得に失敗しました: {err}', type='negative')
                finally:
                    filter_fetch_btn.enable()

            filter_fetch_btn.on_click(fetch_filter_models)

            with ui.card().classes('w-full p-3 bg-zinc-950 border border-zinc-800 rounded-lg gap-2'):
                ui.label('🧪 ヨミ変換フィルタ リアルタイムテスト').classes('text-xs font-bold text-zinc-300')
                with ui.row().classes('w-full items-center gap-2'):
                    filter_test_input = ui.input(
                        'テスト入力文（技術文書・プログラムなど）',
                        value='argc と argv を確認し、/usr/bin/python で実行します。cnt++ でカウンタを増やします。',
                    ).props('dense outlined').classes('grow')
                    filter_test_btn = ui.button('🔄 ヨミ変換テスト実行').props('dense outline')

                filter_test_result = ui.label('').classes('text-xs text-zinc-300 font-mono p-2 bg-zinc-900 border border-zinc-800 rounded min-h-[36px] w-full whitespace-pre-wrap')

                async def run_filter_test() -> None:
                    src_txt = (filter_test_input.value or '').strip()
                    if not src_txt:
                        ui.notify('テスト対象の文章を入力してください．', type='warning')
                        return
                    m = (str(filter_model_select.value) if filter_model_select.value is not None else '').strip()
                    b = (filter_base.value or '').strip()
                    k = (filter_key.value or '').strip() or 'dummy'
                    if not b or not m:
                        ui.notify('フィルタ用の Base URL と Model を設定してください．', type='warning')
                        return

                    filter_test_btn.disable()
                    filter_test_result.text = 'ヨミ変換中…'
                    try:
                        test_filter_cfg = {
                            'server': {'base_url': b, 'api_key': k, 'model': m},
                            'generation': {'temperature': float(filter_temp.value or 0)},
                            'dictionary': filter_dict,
                            'prompt': filter_cfg.get('prompt'),
                        }
                        client = await run.io_bound(make_tts_filter_client, test_filter_cfg)
                        system_prompt = build_tts_filter_prompt(test_filter_cfg)

                        transformed = await run.io_bound(
                            tts_filter_transform,
                            client=client,
                            model=m,
                            system_prompt=system_prompt,
                            text=src_txt,
                            generation=test_filter_cfg['generation'],
                        )
                        filter_test_result.text = transformed
                        ui.notify('ヨミ変換フィルタを適用しました．', type='positive')
                    except Exception as err:
                        filter_test_result.text = f'【エラー】\n{err}'
                        ui.notify(f'ヨミ変換テストに失敗しました: {err}', type='negative')
                    finally:
                        filter_test_btn.enable()

                filter_test_btn.on_click(run_filter_test)

            ui.separator().classes('my-2 border-zinc-800')
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('📖 読み仮名辞書の編集').classes('text-base font-bold text-zinc-200')
                ui.label(f'登録済み単語: {len(filter_dict)} 件').classes('text-xs text-zinc-400')

            with ui.row().classes('w-full items-end gap-3'):
                new_word = ui.input('単語・識別子（例: argc）').classes('grow')
                new_reading = ui.input('読みの目安（例: アーギューシー）').classes('grow')

                def add_word() -> None:
                    if new_word.value and new_reading.value:
                        filter_dict[new_word.value.strip()] = new_reading.value.strip()
                        save_config(TTS_FILTER_PATH, filter_cfg)
                        ui.notify(f'「{new_word.value.strip()} → {new_reading.value.strip()}」を追加しました．', type='positive')
                        self.refresh_settings()

                ui.button('辞書に追加', on_click=add_word).props('dense outline')

            rows = [{'単語 / 識別子': k, '読みの目安': v} for k, v in filter_dict.items()]
            grid = ui.aggrid({
                'columnDefs': [
                    {'headerName': '単語 / 識別子', 'field': '単語 / 識別子', 'editable': True},
                    {'headerName': '読みの目安', 'field': '読みの目安', 'editable': True},
                ],
                'rowData': rows,
                ':getRowId': '(params) => params.data[\"単語 / 識別子\"]',
                'defaultColDef': {'flex': 1, 'resizable': True},
                'animateRows': False,
                'stopEditingWhenCellsLoseFocus': True,
            }).classes('w-full h-80')

            async def save_filter_configuration() -> None:
                await grid.load_client_data()
                data = grid.options.get('rowData', [])
                new_dictionary = {}
                for row in data or []:
                    w = str(row.get('単語 / 識別子', '')).strip()
                    r = str(row.get('読みの目安', '')).strip()
                    if w and r and w != 'nan' and r != 'nan':
                        new_dictionary[w] = r

                filter_cfg['server'] = {
                    'base_url': (filter_base.value or '').strip(),
                    'api_key': (filter_key.value or '').strip() or 'dummy',
                    'model': (str(filter_model_select.value) if filter_model_select.value is not None else '').strip(),
                }
                filter_cfg['generation'] = {
                    'temperature': float(filter_temp.value or 0),
                }
                filter_cfg['dictionary'] = new_dictionary
                save_config(TTS_FILTER_PATH, filter_cfg)
                ui.notify(f'tts_filter.yaml を更新しました（辞書全 {len(new_dictionary)} 件）．', type='positive')

            ui.button('💾 ヨミ変換設定・辞書 (tts_filter.yaml) を保存', on_click=save_filter_configuration).props('dense outline color=emerald').classes('w-full')

    def refresh_settings(self) -> None:
        if not self.settings_container:
            return
        self.settings_container.clear()

        with self.settings_container:
            with ui.row().classes('w-full items-center justify-between pb-2'):
                ui.label('システム設定').classes('text-h4 font-bold')

            llm_base, llm_key, llm_model, llm_temp = self._build_llm_settings()
            (ja_base, ja_key, ja_model, ja_voice), (en_base, en_key, en_model, en_voice) = self._build_tts_settings()

            def save_main_config() -> None:
                self.cfg['llm']['base_url'] = (llm_base.value or '').strip()
                self.cfg['llm']['api_key'] = (llm_key.value or '').strip() or 'dummy'
                self.cfg['llm']['model'] = (str(llm_model.value) if llm_model.value is not None else '').strip()
                self.cfg['llm']['temperature'] = llm_temp.value
                self.cfg['tts']['ja'] = {
                    'base_url': (ja_base.value or '').strip(),
                    'api_key': (ja_key.value or '').strip() or 'dummy',
                    'model': (str(ja_model.value) if ja_model.value is not None else '').strip(),
                    'voice': (str(ja_voice.value) if ja_voice.value is not None else '').strip(),
                    'response_format': 'mp3',
                }
                self.cfg['tts']['en'] = {
                    'base_url': (en_base.value or '').strip(),
                    'api_key': (en_key.value or '').strip() or 'dummy',
                    'model': (str(en_model.value) if en_model.value is not None else '').strip(),
                    'voice': (str(en_voice.value) if en_voice.value is not None else '').strip(),
                    'response_format': 'mp3',
                }
                save_config(CONFIG_PATH, self.cfg)
                ui.notify('config.yaml を保存しました．', type='positive')

            ui.button('💾 基本設定 (config.yaml) を一括保存', on_click=save_main_config).props('color=primary size=lg').classes('w-full mt-2 mb-2')

            self._build_dict_editor()

    def refresh_user_management(self) -> None:
        """(2) 管理者用ユーザー管理タブの構築（ハッシュ化・方針B: 管理者は利用制限入力不可・連動制御）"""
        if not self.user_manage_container:
            return
        self.user_manage_container.clear()

        users_data = load_users()
        users_dict = users_data.get('users', {})

        def calc_future_dt(days: float = 0, hours: float = 0) -> str:
            target = datetime.now() + timedelta(days=days, hours=hours)
            return target.strftime('%Y-%m-%d %H:%M')

        with self.user_manage_container:
            with ui.row().classes('w-full items-center justify-between pb-2'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('manage_accounts', size='md').classes('text-blue-400')
                    ui.label('ユーザーアカウント管理 (`users.yaml`)').classes('text-h4 font-bold')
                ui.label('パスワードはソルト付きハッシュで安全に保護されます．一般ユーザーの利用可能日時を制限できます（管理者は常時無制限）').classes('text-xs text-zinc-400')

            # 新規ユーザー追加カード
            with ui.card().classes('w-full p-5 bg-zinc-900 border border-zinc-800 rounded-xl gap-3'):
                ui.label('➕ 新規ユーザーの追加').classes('text-lg font-bold text-zinc-100')
                with ui.row().classes('w-full items-center gap-3'):
                    add_name = ui.input('ユーザー名 (英数字)').props('outlined dense').classes('w-40')
                    add_pass = ui.input('パスワード', password=True, password_toggle_button=True).props('outlined dense').classes('w-40')
                    add_admin = ui.checkbox('管理者権限').classes('text-zinc-300')
                    add_from = ui.input('利用開始日時', placeholder='例: 2026-10-09 13:00').props('outlined dense').classes('grow')
                    add_until = ui.input('利用終了日時', placeholder='例: 2026-10-16 13:00').props('outlined dense').classes('grow')

                # 新規追加用のクイック加算ツールバー
                with ui.row().classes('w-full items-center gap-2 p-2 bg-zinc-950/60 rounded border border-zinc-800/80 text-xs') as add_quick_bar:
                    ui.label('⏱ 終了日時のクイック設定:').classes('text-zinc-400 font-semibold')
                    btn_24h = ui.button('+24時間', on_click=lambda: add_until.set_value(calc_future_dt(hours=24))).props('dense outline size=xs')
                    btn_3d = ui.button('+3日', on_click=lambda: add_until.set_value(calc_future_dt(days=3))).props('dense outline size=xs')
                    btn_7d = ui.button('+7日 (1週間)', on_click=lambda: add_until.set_value(calc_future_dt(days=7))).props('dense outline size=xs')
                    btn_30d = ui.button('+30日 (1ヶ月)', on_click=lambda: add_until.set_value(calc_future_dt(days=30))).props('dense outline size=xs')

                    ui.label('│ 任意加算:').classes('text-zinc-500 mx-1')
                    add_custom_days = ui.number('日', value=1, min=0, max=365).props('dense outlined size=xs').classes('w-16')
                    add_custom_hours = ui.number('時間', value=0, min=0, max=23).props('dense outlined size=xs').classes('w-16')

                    def apply_custom_to_add():
                        d = float(add_custom_days.value or 0)
                        h = float(add_custom_hours.value or 0)
                        add_until.set_value(calc_future_dt(days=d, hours=h))

                    btn_apply = ui.button('セット', on_click=apply_custom_to_add).props('dense outline color=primary size=xs')
                    btn_clear = ui.button('クリア', on_click=lambda: add_until.set_value('')).props('dense outline color=grey size=xs')

                # 管理者フラグによる新規入力欄の有効/無効連動
                def update_add_fields(is_adm: bool) -> None:
                    if is_adm:
                        add_from.disable()
                        add_until.disable()
                        add_from.set_value('')
                        add_until.set_value('')
                        for w in (btn_24h, btn_3d, btn_7d, btn_30d, add_custom_days, add_custom_hours, btn_apply, btn_clear):
                            w.disable()
                        add_quick_bar.classes(add='opacity-40 pointer-events-none')
                    else:
                        add_from.enable()
                        add_until.enable()
                        for w in (btn_24h, btn_3d, btn_7d, btn_30d, add_custom_days, add_custom_hours, btn_apply, btn_clear):
                            w.enable()
                        add_quick_bar.classes(remove='opacity-40 pointer-events-none')

                add_admin.on_value_change(lambda e: update_add_fields(bool(e.value)))

                async def handle_add_user() -> None:
                    u_name = (add_name.value or '').strip()
                    u_pass = (add_pass.value or '').strip()
                    if not u_name or not u_pass:
                        ui.notify('ユーザー名とパスワードを入力してください．', type='warning')
                        return
                    if u_name in users_dict:
                        ui.notify(f'ユーザー「{u_name}」は既に存在します．', type='negative')
                        return

                    is_adm = bool(add_admin.value)
                    users_dict[u_name] = {
                        'password': hash_password(u_pass),
                        'is_admin': is_adm,
                        'valid_from': '' if is_adm else (add_from.value or '').strip(),
                        'valid_until': '' if is_adm else (add_until.value or '').strip(),
                    }
                    save_users(users_data)
                    ui.notify(f'ユーザー「{u_name}」を追加しました．', type='positive')
                    self.refresh_user_management()

                with ui.row().classes('w-full justify-end pt-1'):
                    ui.button('ユーザーを登録', on_click=handle_add_user).props('color=primary dense').classes('w-44')

            # 既存ユーザー一覧カード
            ui.label(f'登録済みユーザー一覧（全 {len(users_dict)} アカウント）').classes('text-base font-bold text-zinc-200 mt-4')

            for u_name, u_info in sorted(users_dict.items()):
                with ui.card().classes('w-full p-4 bg-zinc-900 border border-zinc-800 rounded-xl gap-3'):
                    is_current_user = (u_name == self.username)
                    is_user_admin = bool(u_info.get('is_admin', False))

                    with ui.row().classes('w-full items-center justify-between'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('person', size='sm').classes('text-blue-400')
                            ui.label(u_name).classes('text-lg font-bold text-white')
                            if is_user_admin:
                                ui.badge('Admin', color='primary').props('outline')

                        if is_user_admin:
                            ui.badge('常時利用可能 (管理者)', color='positive').props('outline')
                        else:
                            is_allowed, status_msg = is_user_within_allowed_period(u_info)
                            if is_allowed:
                                ui.badge('利用可能', color='positive').props('outline')
                            else:
                                ui.badge(f'利用不可: {status_msg}', color='negative').props('outline')

                    with ui.row().classes('w-full items-center gap-3 pt-1'):
                        pass_input = ui.input(
                            '新パスワード',
                            placeholder='変更時のみ入力',
                            password=True,
                            password_toggle_button=True,
                        ).props('outlined dense').classes('w-44')

                        admin_check = ui.checkbox('管理者権限', value=is_user_admin).classes('text-zinc-300')
                        if is_current_user:
                            admin_check.disable()

                        from_input = ui.input(
                            '利用開始日時',
                            value='' if is_user_admin else str(u_info.get('valid_from', ''))
                        ).props('outlined dense placeholder="YYYY-MM-DD HH:MM"').classes('grow')
                        until_input = ui.input(
                            '利用終了日時',
                            value='' if is_user_admin else str(u_info.get('valid_until', ''))
                        ).props('outlined dense placeholder="YYYY-MM-DD HH:MM"').classes('grow')

                    # 既存ユーザー用のクイック加算ツールバー
                    with ui.row().classes('w-full items-center gap-2 p-1.5 bg-zinc-950/40 rounded border border-zinc-800 text-xs') as row_quick_bar:
                        ui.label('⏱ 終了日時のクイック加算:').classes('text-zinc-400 font-semibold')
                        r_btn_24h = ui.button('+24h', on_click=lambda target_in=until_input: target_in.set_value(calc_future_dt(hours=24))).props('dense outline size=xs')
                        r_btn_3d = ui.button('+3日', on_click=lambda target_in=until_input: target_in.set_value(calc_future_dt(days=3))).props('dense outline size=xs')
                        r_btn_7d = ui.button('+7日', on_click=lambda target_in=until_input: target_in.set_value(calc_future_dt(days=7))).props('dense outline size=xs')
                        r_btn_30d = ui.button('+30日', on_click=lambda target_in=until_input: target_in.set_value(calc_future_dt(days=30))).props('dense outline size=xs')

                        ui.label('│').classes('text-zinc-600')
                        row_days = ui.number('日', value=7, min=0, max=365).props('dense outlined size=xs').classes('w-14')
                        row_hours = ui.number('時間', value=0, min=0, max=23).props('dense outlined size=xs').classes('w-14')

                        def apply_row_custom(target_in=until_input, rd=row_days, rh=row_hours):
                            d = float(rd.value or 0)
                            h = float(rh.value or 0)
                            target_in.set_value(calc_future_dt(days=d, hours=h))

                        r_btn_apply = ui.button('加算セット', on_click=apply_row_custom).props('dense outline color=primary size=xs')
                        r_btn_clear = ui.button('制限解除 (空欄)', on_click=lambda target_in=until_input: target_in.set_value('')).props('dense outline color=grey size=xs')

                    # 管理者フラグに応じた入力制限（方針B）
                    def update_row_fields(is_adm: bool, fi=from_input, ui_=until_input, bar=row_quick_bar,
                                          widgets=(r_btn_24h, r_btn_3d, r_btn_7d, r_btn_30d, row_days, row_hours, r_btn_apply, r_btn_clear)):
                        if is_adm:
                            fi.disable()
                            ui_.disable()
                            fi.set_value('')
                            ui_.set_value('')
                            for w in widgets:
                                w.disable()
                            bar.classes(add='opacity-40 pointer-events-none')
                        else:
                            fi.enable()
                            ui_.enable()
                            for w in widgets:
                                w.enable()
                            bar.classes(remove='opacity-40 pointer-events-none')

                    # 初期状態の反映
                    update_row_fields(is_user_admin)
                    admin_check.on_value_change(lambda e, upd=update_row_fields: upd(bool(e.value)))

                    with ui.row().classes('w-full justify-end items-center gap-3 pt-1'):
                        async def handle_update(target=u_name, p=pass_input, a=admin_check, vf=from_input, vu=until_input):
                            new_admin_val = bool(a.value)

                            # 管理者権限剥奪の安全ガード
                            if users_dict[target].get('is_admin') and not new_admin_val:
                                if target == self.username:
                                    ui.notify('自分自身の管理者権限を外すことはできません．', type='negative')
                                    a.value = True
                                    return
                                admin_count = sum(1 for u in users_dict.values() if u.get('is_admin', False))
                                if admin_count <= 1:
                                    ui.notify('システム内に管理者がいなくなるため、最後の管理者権限を外すことはできません．', type='negative')
                                    a.value = True
                                    return

                            new_pw = (p.value or '').strip()
                            if new_pw:
                                users_dict[target]['password'] = hash_password(new_pw)

                            users_dict[target]['is_admin'] = new_admin_val
                            # 管理者の場合は日時設定を空文字にして保存
                            users_dict[target]['valid_from'] = '' if new_admin_val else (vf.value or '').strip()
                            users_dict[target]['valid_until'] = '' if new_admin_val else (vu.value or '').strip()
                            save_users(users_data)
                            ui.notify(f'ユーザー「{target}」の情報を更新しました．', type='positive')
                            self.refresh_user_management()

                        async def handle_delete(target=u_name):
                            if target == self.username:
                                ui.notify('現在ログイン中の自分自身を削除することはできません．', type='negative')
                                return
                            if len(users_dict) <= 1:
                                ui.notify('最後の1アカウントは削除できません．', type='negative')
                                return
                            del users_dict[target]
                            save_users(users_data)
                            ui.notify(f'ユーザー「{target}」を削除しました．', type='info')
                            self.refresh_user_management()

                        ui.button('変更を保存', on_click=handle_update).props('outline color=primary dense')
                        if not is_current_user:
                            ui.button('削除', on_click=handle_delete).props('outline color=negative dense')

    async def refresh_all(self) -> None:
        await self.refresh_views()
        await self.refresh_final_video()

    def build(self) -> None:
        ui.page_title('Slide Narrator')
        ui.dark_mode().enable()
        ui.colors(primary='#3b82f6')

        current_username = app.storage.user.get('username', '')
        is_guest = (current_username == 'guest')
        is_admin = app.storage.user.get('is_admin', False)

        def logout() -> None:
            app.storage.user.clear()
            ui.navigate.to('/login')

        with ui.header().classes('items-center w-full px-4 bg-slate-900 border-b border-slate-800'):
            ui.label('🎓 Slide Narrator').classes('text-h5 text-white')
            ui.space()
            if current_username:
                badge_role = ' (管理者)' if is_admin else ''
                ui.label(f'👤 {current_username}{badge_role}').classes('text-caption text-slate-400 mr-2')
            ui.button('ログアウト', on_click=logout).props('dense outline size=sm color=white').classes('mr-3')
            ui.label('TAKAGO_LAB. 2026').classes('text-subtitle2 font-mono tracking-wider text-slate-300 mr-2')

        with ui.left_drawer(value=True).props('width=320').classes('p-4'):
            ui.label('プロジェクト設定').classes('text-h5')
            ui.label(f'作業場所: webui_uploads/{self.username}/').classes('text-[11px] text-zinc-400 font-mono pb-1')

            self.uploader = (
                ui.upload(
                    label='プレゼンテーションPDFを選択',
                    auto_upload=True,
                    max_files=1,
                    on_upload=self.load_pdf,
                )
                .props('accept=.pdf')
                .classes('w-full')
            )
            self.uploader.on(
                'added',
                lambda: self.uploader.run_method(
                    'eval',
                    'if (this.files.length > 1) { this.removeFile(this.files[0]); }',
                ),
            )

            self.mode_select = ui.radio({'lecture': '🎓 講義', 'research': '🔬 研究発表'}, value=self.mode_code).props('inline')
            self.mode_select.on_value_change(lambda e: self._mode_changed(e.value))
            self.lang_select = ui.radio({'ja': '🇯🇵 日本語', 'en': '🇺🇸 英語'}, value=self.lang_code).props('inline')
            self.lang_select.on_value_change(lambda e: self._lang_changed(e.value))

            with ui.card().classes('w-full p-2.5 bg-zinc-900 border border-zinc-800 rounded-lg mt-2'):
                self.default_vlm_switch = ui.switch(
                    'VLMを活用してポインタ配置を決定する',
                    value=self.default_use_vlm,
                ).props('dense color=primary')
                self.default_vlm_switch.tooltip('ONにすると図形・数式・グラフの内部要素まで細かくポインティングします．OFFにするとPDFテキストのみを使用します．')
                self.default_vlm_switch.on_value_change(lambda e: self._default_vlm_changed(e.value))

            self.pages_input = (
                ui.input('ビデオ化対象', placeholder='例: 1-10,12', value=self.pages_spec)
                .classes('w-full mt-2')
                .props('outlined dense')
                .on_value_change(lambda e: self._range_changed())
            )
            self.active_count_label = ui.label('対象スライド: 0 / 0 スライド')
            ui.separator()
            ui.label('パイプライン実行').classes('text-h6')
            self.force_checkbox = ui.checkbox('キャッシュを破棄してやり直す', value=False)
            self.force_checkbox.tooltip('チェックを入れると、生成済みのナレーション・音声・動画ファイルをスキップせずにすべて作り直します．')
            self.force_checkbox.on_value_change(lambda e: setattr(self, 'force_run', bool(e.value)))
            self.pipeline_buttons = [
                ui.button('① ナレーション原稿の生成', on_click=lambda: self.pipeline('explain', '① ナレーション原稿を作成中…')).classes('w-full'),
                ui.button('② 翻訳，字幕生成，ポインタ配置', on_click=lambda: self.pipeline('align', '② 翻訳と字幕，ポインタ配置を決定中…')).classes('w-full'),
                ui.button('③ ナレーション音声の作成', on_click=lambda: self.pipeline('tts', f'③ ナレーション音声を作成中…')).classes('w-full'),
                ui.button('④ ナレーションビデオの作成', on_click=lambda: self.pipeline('video', '④ ナレーションビデオを作成中…')).classes('w-full'),
            ]
            ui.separator()
            ui.label('🎬 完成ビデオ').classes('text-subtitle1 font-bold text-zinc-200')
            self.final_video_container = ui.column().classes('w-full gap-2')
            ui.separator()
            ui.label('TAKAGO LAB., KIT, Japan.').classes('text-caption')
            ui.link('GitHub: takago/slide-narrator', 'https://github.com/takago/slide-narrator', new_tab=True)

        with ui.column().classes('w-full p-6'):
            with ui.tabs().classes('w-full') as tabs:
                self.tabs = tabs
                tab_gallery = ui.tab('🖼 スライドデッキ')
                tab_simple_edit = ui.tab('📋 ナレーション修正（簡易）')
                tab_edit = ui.tab('📝 ナレーション修正（詳細）')
                tab_slide_videos = ui.tab('🎞 ビデオデッキ')
                tab_logs = ui.tab('📜 ログ')
                if not is_guest:
                    tab_settings = ui.tab('⚙ 設定')
                if is_admin:
                    tab_user_manage = ui.tab('👥 ユーザー管理')

            with ui.tab_panels(tabs, value=tab_gallery).classes('w-full'):
                with ui.tab_panel(tab_gallery):
                    ui.label('🖼️ スライドデッキ').classes('text-h5')
                    self.gallery = ui.column().classes('w-full')
                with ui.tab_panel(tab_simple_edit):
                    ui.label('📋 ナレーション修正（簡易）').classes('text-h5')
                    self.simple_edit_container = ui.column().classes('w-full')
                with ui.tab_panel(tab_edit):
                    self.edit_container = ui.column().classes('w-full')
                with ui.tab_panel(tab_slide_videos):
                    ui.label('🎞 ビデオデッキ').classes('text-h5')
                    self.slide_video_gallery = ui.column().classes('w-full')
                with ui.tab_panel(tab_logs):
                    with ui.row().classes('w-full items-center justify-between pb-2'):
                        ui.label('📜 実行ログ履歴').classes('text-h5')
                        ui.button('ログをクリア', on_click=lambda: self.history_log_widget.clear() if self.history_log_widget else None).props('dense outline size=sm color=negative')
                    self.history_log_widget = ui.log(max_lines=2000).classes('w-full h-[650px] font-mono text-xs bg-zinc-900 border border-zinc-700 rounded-lg p-3 text-zinc-300')
                    if self.log:
                        for line in self.log.splitlines():
                            self.history_log_widget.push(line)
                if not is_guest:
                    with ui.tab_panel(tab_settings):
                        self.settings_container = ui.column().classes('w-full')
                if is_admin:
                    with ui.tab_panel(tab_user_manage):
                        self.user_manage_container = ui.column().classes('w-full')

        if not is_guest:
            self.refresh_settings()
        if is_admin:
            self.refresh_user_management()

    def _mode_changed(self, value: str) -> None:
        self.mode_code = value
        self.save_project_settings()

    def _lang_changed(self, value: str) -> None:
        self.lang_code = value
        self.save_project_settings()
        asyncio.create_task(self.refresh_simple_editor())
        asyncio.create_task(self.refresh_editor())

    def _default_vlm_changed(self, value: bool) -> None:
        self.default_use_vlm = bool(value)
        self.save_project_settings()

    def _range_changed(self) -> None:
        new_val = self.pages_input.value or ''
        if new_val == self.pages_spec:
            return
        try:
            self.apply_pages_spec(new_val, save_and_refresh=True)
        except Exception as exc:
            ui.notify(f'スライド範囲を解釈できません: {exc}', type='negative')


@ui.page('/')
def index_page():
    if not app.storage.user.get('authenticated', False):
        return RedirectResponse('/login')

    username = app.storage.user.get('username', '')
    users_data = load_users().get('users', {})
    u_info = users_data.get(username)
    if not u_info:
        app.storage.user.clear()
        return RedirectResponse('/login')

    allowed, err = is_user_within_allowed_period(u_info)
    if not allowed:
        app.storage.user.clear()
        return RedirectResponse('/login')

    app_instance = SlideNarratorApp(username=username)
    app_instance.build()


@ui.page('/login')
def login_page():
    if app.storage.user.get('authenticated', False):
        return RedirectResponse('/')

    ui.page_title('Slide Narrator - ログイン')
    ui.dark_mode().enable()
    ui.colors(primary='#3b82f6')

    def try_login() -> None:
        username = (username_input.value or '').strip()
        password = password_input.value or ''

        users_data = load_users()
        users_dict = users_data.get('users', {})
        user_info = users_dict.get(username)

        # パスワード検証（平文互換判定含む）
        if not user_info or not verify_password(password, str(user_info.get('password', ''))):
            ui.notify('ユーザー名またはパスワードが正しくありません．', type='negative')
            return

        # 既存の平文パスワードだった場合、初回認証成功時に自動でハッシュ化して保存更新
        stored_pw = str(user_info.get('password', ''))
        if not stored_pw.startswith('pbkdf2:sha256:'):
            user_info['password'] = hash_password(password)
            save_users(users_data)

        # 利用可能日時制限のチェック
        allowed, reason = is_user_within_allowed_period(user_info)
        if not allowed:
            ui.notify(f'ログイン拒否: {reason}', type='negative')
            return

        app.storage.user['authenticated'] = True
        app.storage.user['username'] = username
        app.storage.user['is_admin'] = bool(user_info.get('is_admin', False))
        ui.navigate.to('/')

    with ui.card().classes('absolute-center w-96 p-6 bg-zinc-900 border border-zinc-800 rounded-xl shadow-lg gap-4'):
        with ui.column().classes('w-full items-center gap-1'):
            ui.label('🎓 Slide Narrator').classes('text-h5 font-bold text-white')
            ui.label('サインインして続行してください').classes('text-xs text-zinc-400')

        username_input = ui.input('ユーザー名').props('outlined dense autofocus').classes('w-full')
        password_input = ui.input('パスワード', password=True, password_toggle_button=True).props('outlined dense').classes('w-full')
        password_input.on('keydown.enter', try_login)
        username_input.on('keydown.enter', try_login)

        ui.button('ログイン', on_click=try_login).props('color=primary').classes('w-full mt-2')


ui.run(title='Slide Narrator', reload=False, show=False, port=17171, host='0.0.0.0', storage_secret='slide-narrator-session-secret-key-change-in-prod')
