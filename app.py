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
import httpx
import json
import logging
import os
import re
from pathlib import Path
import secrets
import shutil
import signal
import sys
import time
from typing import Any

import pymupdf as fitz
import yaml
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from nicegui import app, run, ui, background_tasks
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
# Constants & UI Styling
# ----------------------------------------------------------------------

BADGE_COLORS: dict[str, tuple[str, str]] = {
    'image_subpart': ('bg-red-500/20 text-red-300 border-red-500/40', '注目要素'),
    'image': ('bg-amber-500/20 text-amber-300 border-amber-500/40', '画像・領域'),
    'text': ('bg-blue-500/20 text-blue-300 border-blue-500/40', 'テキスト'),
    'code_line': ('bg-cyan-500/20 text-cyan-300 border-cyan-500/40', 'コード/数式'),
}


# ----------------------------------------------------------------------
# Concurrency / Global Execution Lock (排他制御)
# ----------------------------------------------------------------------

class ExecutionLockManager:
    """複数ユーザーやセッションが同時に重い推論・レンダリング処理を行わないよう排他制御するマネージャ"""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.current_user: str | None = None
        self.task_name: str | None = None
        self.owner_app: Any | None = None

    def is_locked(self) -> bool:
        return self.lock.locked()

    async def acquire(self, username: str, task_name: str, owner_app: Any | None = None) -> bool:
        """ロックを即座に試行取得。他が実行中の場合はキューイングせず即座に False を返す"""
        if self.lock.locked():
            return False
        await self.lock.acquire()
        self.current_user = username
        self.task_name = task_name
        self.owner_app = owner_app
        return True

    def release(self) -> None:
        if self.lock.locked():
            self.lock.release()
        self.current_user = None
        self.task_name = None
        self.owner_app = None


GLOBAL_EXECUTION_LOCK = ExecutionLockManager()
LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# File IO Utilities (Atomic File Operations)
# ----------------------------------------------------------------------

def atomic_write_text(path: Path, content: str, encoding: str = 'utf-8') -> None:
    """同一ディレクトリに一時ファイルを書き込み、アトミックに置換する共通関数"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        temp_path.write_text(content, encoding=encoding)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


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

DEFAULT_USERS_DATA: dict[str, Any] = {
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
    content = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    atomic_write_text(USERS_FILE, content)


def is_user_within_allowed_period(user_info: dict[str, Any]) -> tuple[bool, str]:
    """利用可能日時制限の判定 (YYYY-MM-DD HH:MM または YYYY-MM-DD 形式)"""
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
# Configuration / Workspace / Job Tracking
# ----------------------------------------------------------------------

CONFIG_PATH = Path('config.yaml')
TTS_FILTER_PATH = Path('tts_filter.yaml')
BASE_UPLOAD_DIR = Path('webui_uploads')
BASE_UPLOAD_DIR.mkdir(exist_ok=True)


def get_user_workspace(username: str) -> Path:
    safe_name = "".join(c for c in username if c.isalnum() or c in ('_', '-')).strip() or 'unknown'
    p = BASE_UPLOAD_DIR / safe_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_last_project_file(username: str) -> Path:
    return get_user_workspace(username) / '.last_project.json'


def user_job_status_file(username: str) -> Path:
    return get_user_workspace(username) / '.job_status.json'


def user_job_log_file(username: str) -> Path:
    return get_user_workspace(username) / '.job_log.txt'


def append_user_job_log(username: str, line: str) -> None:
    path = user_job_log_file(username)
    stamp = datetime.now().isoformat(timespec='seconds')
    with path.open('a', encoding='utf-8') as f:
        f.write(f'[{stamp}] {line}\n')


def read_user_job_log(username: str, max_lines: int = 2000) -> str:
    try:
        lines = user_job_log_file(username).read_text(encoding='utf-8').splitlines()
    except OSError:
        return ''
    return '\n'.join(lines[-max_lines:])


def clear_user_job_log(username: str) -> None:
    user_job_log_file(username).write_text('', encoding='utf-8')


def read_user_job_status(username: str) -> dict[str, Any]:
    path = user_job_status_file(username)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def write_user_job_status(username: str, **updates: Any) -> None:
    path = user_job_status_file(username)
    status = read_user_job_status(username)
    status.update(updates)
    status['updated_at'] = datetime.now().isoformat(timespec='seconds')
    atomic_write_text(path, json.dumps(status, ensure_ascii=False, indent=2))


def safe_notify(*args: Any, **kwargs: Any) -> None:
    try:
        ui.notify(*args, **kwargs)
    except Exception:
        pass


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def save_config(path: Path, cfg: dict) -> None:
    content = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    atomic_write_text(path, content)


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
        'image_subpart': ('#ef4444', 3),
        'image': ('#f59e0b', 2),
        'text': ('#3b82f6', 2),
        'code_line': ('#06b6d4', 2),
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
    if not pages:
        return ''
    sorted_pages = sorted(set(pages))
    ranges: list[str] = []
    start = sorted_pages[0]
    prev = sorted_pages[0]

    for p in sorted_pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append(str(start) if start == prev else f'{start}-{prev}')
            start = p
            prev = p

    ranges.append(str(start) if start == prev else f'{start}-{prev}')
    return ','.join(ranges)


def file_url(path: Path) -> str:
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
# Application Main Controller
# ----------------------------------------------------------------------

STAGE_DEFINITIONS = [
    ('explain', '① ナレーション原稿の生成', '① ナレーション原稿を作成中…'),
    ('align', '② 翻訳，字幕生成，ポインタ配置', '② 翻訳と字幕，ポインタ配置を決定中…'),
    ('tts', '③ ナレーション音声の作成', '③ ナレーション音声を作成中…'),
    ('video', '④ ナレーションビデオの作成', '④ ナレーションビデオを作成中…'),
]

ORDERED_STAGES = [key for key, _, _ in STAGE_DEFINITIONS]


class SlideNarratorApp:
    def __init__(self, username: str) -> None:
        self.username = username
        self.cfg = load_config(CONFIG_PATH)
        user_info = load_users().get('users', {}).get(username, {})
        self.read_only = not is_user_within_allowed_period(user_info)[0]

        # ワークスペース
        self.user_dir = get_user_workspace(self.username)
        self.projects_dir = self.user_dir / 'projects'
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.test_audio_dir = self.user_dir / 'test_audio'
        self.test_audio_dir.mkdir(exist_ok=True)
        self.test_vlm_dir = self.user_dir / 'test_vlm'
        self.test_vlm_dir.mkdir(exist_ok=True)

        # プロジェクト状態
        self.pdf: Path | None = None
        self.paths: ProjectPaths | None = None
        self.images: list[Path] = []
        self.proj_cfg: dict[str, Any] = {}
        self.mode_code = 'lecture'
        self.lang_code = 'ja'
        self.default_use_vlm: bool = True
        self.slide_visual_modes: dict[str, str] = {}
        self.edit_page: int | None = None
        self.processing = False
        self._selected_pages: set[int] = set()

        # プロセス・非同期タスク管理
        self.current_process: asyncio.subprocess.Process | None = None
        self.current_task: asyncio.Task | None = None
        self.cancellation_requested = False
        self._loaded_log_text: str | None = None
        self._restore_started = False
        self._refresh_debounce_task: asyncio.Task | None = None

        # UI ウィジェット参照
        self.job_status_label = None
        self.history_log_widget = None
        self.tabs = None
        self.gallery = None
        self.simple_edit_container = None
        self.edit_container = None
        self.slide_video_gallery = None
        # 差分更新対象の画面部品．ブラウザ接続ごとに独立して保持する．
        self._simple_textareas: dict[int, Any] = {}
        self._simple_text_signatures: dict[int, tuple | None] = {}
        self._video_cards: dict[int, Any] = {}
        self._video_signatures: dict[int, tuple | None] = {}
        self._video_grid = None
        self._output_project_key: str | None = None
        self._detail_signatures: dict[str, tuple | None] = {}
        self._final_video_signature: tuple | None = None
        self.final_video_container = None
        self.settings_container = None
        self.user_manage_container = None
        self.uploader = None
        self.project_select = None
        self.project_name_input = None
        self.active_count_label = None
        self.mode_select = None
        self.lang_select = None
        self.default_vlm_switch = None
        self.pages_input = None
        self.pipeline_buttons: list[Any] = []
        self.stage_widgets: dict[str, dict[str, Any]] = {}
        self.pipeline_busy_row = None
        self.pipeline_busy_notice = None
        self.main_action_buttons: list[Any] = []

    # --- クライアントの寿命とジョブの寿命は別に管理する ---

    @staticmethod
    def _widget_available(widget: Any) -> bool:
        if widget is None or getattr(widget, 'is_deleted', False):
            return False
        try:
            return not widget.client.is_deleted
        except (AttributeError, RuntimeError):
            return False

    def _safe_view_update(self, callback, description: str) -> None:
        try:
            callback()
        except Exception:
            LOGGER.exception('UI更新に失敗しました（生成処理は継続）: %s', description)

    async def _safe_async_view_update(self, callback, description: str) -> None:
        try:
            await callback()
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception('UI更新に失敗しました（生成処理は継続）: %s', description)

    def _complete_background_job(self) -> None:
        # UIが破棄されていてもロック解放を最優先する．
        self.current_process = None
        self.processing = False
        self.current_task = None
        try:
            GLOBAL_EXECUTION_LOCK.release()
        finally:
            self._safe_view_update(lambda: self.set_processing(False), '処理終了時のボタン更新')

    def _start_background_job(self, worker_factory, *, busy_message: str) -> None:
        """重複起動の確認とジョブ登録を一箇所で管理する．"""
        if self.current_task and not self.current_task.done():
            safe_notify(busy_message, type='warning')
            return
        self.current_task = background_tasks.create(worker_factory(), name='slide-narrator-job')

    def _finalize_cancel_request(self, **extra_status: Any) -> None:
        """中断要求が残ったまま終了した場合にだけ状態を確定する．"""
        if not self.cancellation_requested:
            return
        saved = read_user_job_status(self.username)
        if saved.get('state') == 'running':
            self.update_job_status(
                'cancelled', cancellation_requested=False,
                message='処理を中断しました',
                finished_at=datetime.now().isoformat(timespec='seconds'),
                **extra_status,
            )

    def _invalidate_page_video_outputs(self, page: int) -> None:
        """ページの動画・完成動画・完成字幕を無効化する．音声と解析結果は保持する．"""
        assert self.paths is not None
        for path in (
            self.paths.video(page), self.paths.final_video,
            self.paths.final_ja_srt, self.paths.final_en_srt,
        ):
            path.unlink(missing_ok=True)

    # --- 認証・権限制御 ---

    def write_access_allowed(self, notify: bool = True) -> bool:
        user_info = load_users().get('users', {}).get(self.username, {})
        allowed, reason = is_user_within_allowed_period(user_info)
        self.read_only = not allowed
        if not allowed and notify:
            safe_notify(f'現在は閲覧モードです．編集・生成はできません．{reason}', type='warning')
        return allowed

    @staticmethod
    def set_button_enabled_visual(button: Any, enabled: bool) -> None:
        try:
            if enabled:
                button.enable()
                button.classes(remove='opacity-40 grayscale')
            else:
                button.disable()
                button.classes(add='opacity-40 grayscale')
        except Exception:
            pass

    def register_main_action_button(self, button: Any) -> Any:
        self.main_action_buttons.append(button)
        self.set_button_enabled_visual(button, not (GLOBAL_EXECUTION_LOCK.is_locked() or self.read_only))
        return button

    # --- ジョブ状態・ログ管理 ---

    def update_job_status(self, state: str | None = None, **updates: Any) -> dict[str, Any]:
        if state is not None:
            updates['state'] = state
        # 永続化をUI表示に依存させない．書き込み失敗はワーカーに通知する．
        write_user_job_status(self.username, **updates)
        status = read_user_job_status(self.username)
        if self._widget_available(self.job_status_label):
            self._safe_view_update(
                lambda: setattr(self.job_status_label, 'text', self.format_job_status(status)),
                'ジョブ状態表示',
            )
        return status

    def append_job_log(self, line: str) -> None:
        line = str(line)
        append_user_job_log(self.username, line)
        self._loaded_log_text = None
        if self._widget_available(self.history_log_widget):
            self._safe_view_update(
                lambda: self.history_log_widget.push(f'[{datetime.now():%H:%M:%S}] {line}'),
                'ログ表示',
            )

    def clear_job_log(self) -> None:
        clear_user_job_log(self.username)
        self._loaded_log_text = ''
        if self._widget_available(self.history_log_widget):
            self._safe_view_update(self.history_log_widget.clear, 'ログクリア')

    def refresh_history_log(self) -> None:
        if not self._widget_available(self.history_log_widget):
            return
        contents = read_user_job_log(self.username)
        if contents == self._loaded_log_text:
            return
        try:
            self.history_log_widget.clear()
            for line in contents.splitlines()[-2000:]:
                self.history_log_widget.push(line)
            self._loaded_log_text = contents
        except Exception:
            pass

    @staticmethod
    def format_job_status(status: dict[str, Any]) -> str:
        state = status.get('state', 'idle')
        labels = {
            'running': '生成処理中',
            'completed': '直近の生成処理は完了',
            'failed': '直近の生成処理は失敗',
            'cancelled': '直近の生成処理は中断',
            'idle': '生成処理の履歴はありません',
        }
        label = labels.get(state, str(state))
        detail = status.get('message') or status.get('stage') or ''
        updated = status.get('updated_at') or ''
        suffix = f' — {detail}' if detail else ''
        progress = status.get('progress')
        if state == 'running' and isinstance(progress, (int, float)):
            suffix += f'（進捗 {max(0, min(100, int(progress * 100)))}％）'
        page = status.get('current_page')
        if state == 'running' and page is not None:
            suffix += f'［スライド {page}］'
        return f'処理状態：{label}{suffix}' + (f'（{updated}）' if updated else '')

    @staticmethod
    def stage_button_key(stage: str | None) -> str | None:
        stage = str(stage or '')
        if any(word in stage for word in ('ナレーション原稿', 'ナレーション再生成', 'explain')):
            return 'explain'
        if any(word in stage for word in ('字幕', 'ポインタ', '翻訳', 'align')):
            return 'align'
        if any(word in stage for word in ('音声合成', '音声を作成', '音声', 'tts')):
            return 'tts'
        if any(word in stage for word in ('動画', 'ビデオ', 'video', 'concat')):
            return 'video'
        return None

    def get_active_stage_key(self, status: dict[str, Any]) -> str | None:
        stage_key = self.stage_button_key(status.get('stage'))
        if not stage_key and GLOBAL_EXECUTION_LOCK.is_locked():
            task_name = GLOBAL_EXECUTION_LOCK.task_name or ''
            if task_name.startswith('パイプライン (') and task_name.endswith(')'):
                stage_key = self.stage_button_key(task_name[len('パイプライン ('):-1])
            else:
                stage_key = self.stage_button_key(task_name)
        return stage_key

    @staticmethod
    def _file_signature(path: Path) -> tuple[int, int, int] | None:
        """更新内容の検知．存在しないファイルは None とする．"""
        try:
            stat = path.stat()
            return (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        except OSError:
            return None

    def _reset_output_watch(self) -> None:
        self._simple_textareas.clear()
        self._simple_text_signatures.clear()
        self._video_cards.clear()
        self._video_signatures.clear()
        self._video_grid = None
        self._detail_signatures.clear()
        self._final_video_signature = None

    def poll_generated_outputs(self) -> None:
        """生成物だけを差分反映する．既存の入力欄・再生中の動画には触れない．"""
        if not self.paths or not self.pdf:
            return
        project_key = str(self.pdf.resolve())
        if self._output_project_key != project_key:
            self._output_project_key = project_key
            self._reset_output_watch()
            # プロジェクト切替時の全描画は既存の refresh_all が担当する．
            return

        for page, textarea in list(self._simple_textareas.items()):
            if not self._widget_available(textarea):
                self._simple_textareas.pop(page, None)
                continue
            path = self.paths.explanation(page)
            signature = self._file_signature(path)
            if signature == self._simple_text_signatures.get(page):
                continue
            if signature is not None:
                try:
                    textarea.value = path.read_text(encoding='utf-8')
                except OSError:
                    LOGGER.exception('ナレーションフォームの同期に失敗: %s', path)
                    continue
            else:
                textarea.value = ''
            self._simple_text_signatures[page] = signature

        if self._video_grid is not None and self._widget_available(self._video_grid):
            for page in self.active_pages:
                signature = self._file_signature(self.paths.video(page))
                if signature != self._video_signatures.get(page) or page not in self._video_cards:
                    self._replace_video_card(page, signature)

        if self._widget_available(self.edit_container) and self.edit_page in self.active_pages:
            page = self.edit_page
            current = {
                'text': self._file_signature(self.paths.explanation(page)),
                'align': self._file_signature(self.paths.alignment(page)),
                'audio': self._file_signature(self.paths.audio(page)),
            }
            if self._detail_signatures and current != self._detail_signatures:
                # 詳細画面は現在表示している一枚だけを構築し直す．
                background_tasks.create(self._safe_async_view_update(
                    self.refresh_editor, '生成結果による詳細画面の更新'))
            self._detail_signatures = current

        final_signature = self._file_signature(self.paths.final_video)
        if final_signature != self._final_video_signature:
            self._final_video_signature = final_signature
            background_tasks.create(self._safe_async_view_update(
                self.refresh_final_video, '完成動画の差分更新'))

    def _replace_video_card(self, page: int, signature: tuple | None = None) -> None:
        """指定スライドのカードだけを置換．他の動画プレーヤーは保持する．"""
        if not self.paths or self._video_grid is None:
            return
        card = self._video_cards.get(page)
        if card is None or not self._widget_available(card):
            with self._video_grid:
                card = ui.card().classes('w-full p-3 gap-2 rounded-lg bg-zinc-800 border border-zinc-700 shadow-sm')
            self._video_cards[page] = card
        else:
            card.clear()
        with card:
            video = self.paths.video(page)
            img = self.paths.page_image(page)
            if video.exists():
                ui.video(file_url(video)).classes('w-full rounded shadow-sm')
            else:
                if img.exists():
                    ui.image(file_url(img)).classes('w-full opacity-60 rounded shadow-sm')
                ui.label('（単体動画 未生成）').classes('text-xs text-orange-400 font-medium')
            ui.label(f'スライド {page}').classes('text-sm font-bold text-white self-center pt-1')
        self._video_signatures[page] = signature if signature is not None else self._file_signature(self.paths.video(page))

    def refresh_job_status(self) -> None:
        """ポーリング時に画面上のボタン・進捗バー群を同期"""
        allowed_now = self.write_access_allowed(notify=False)
        if self.read_only != (not allowed_now):
            safe_notify('利用可能時間が変わったため，画面を更新します．', type='info')
            try:
                ui.navigate.reload()
            except Exception:
                pass
            return

        globally_busy = GLOBAL_EXECUTION_LOCK.is_locked()
        owner_name = GLOBAL_EXECUTION_LOCK.current_user or '' if globally_busy else ''

        if globally_busy:
            status = read_user_job_status(owner_name) if owner_name else {}
            status = {**status, 'stage': status.get('stage') or GLOBAL_EXECUTION_LOCK.task_name or '処理中'}
        else:
            status = read_user_job_status(self.username)
            if status.get('state') == 'running':
                status = {**status, 'state': 'failed', 'message': '実行中状態を確認できません（サーバ再起動等の可能性があります）'}

        # 再ログイン後の新しい画面にも，状態ファイルの内容を反映する．
        own_status = read_user_job_status(self.username)
        if self._widget_available(self.job_status_label):
            self._safe_view_update(
                lambda: setattr(self.job_status_label, 'text', self.format_job_status(
                    status if globally_busy and owner_name == self.username else own_status)),
                '再接続後の処理状態表示',
            )

        active_key = self.get_active_stage_key(status)
        owner_is_self = globally_busy and owner_name == self.username

        if self.pipeline_busy_row is not None and self.pipeline_busy_notice is not None:
            try:
                if globally_busy and not owner_is_self:
                    self.pipeline_busy_notice.text = f'他のユーザ（{owner_name or "xxxx"}）が処理中です．'
                    self.pipeline_busy_row.classes(remove='hidden')
                else:
                    self.pipeline_busy_row.classes(add='hidden')
            except Exception:
                pass

        if owner_name == self.username and globally_busy:
            self.processing = True  # 表示のためだけ．実行主体はowner_appのまま変更しない．
        elif not globally_busy:
            self.processing = False

        owner_app = GLOBAL_EXECUTION_LOCK.owner_app if globally_busy else None
        cancel_pending = bool(owner_app and owner_app.cancellation_requested)
        completed_stages = status.get('completed_stages', [])

        active_idx = ORDERED_STAGES.index(active_key) if (active_key in ORDERED_STAGES) else -1

        for key, w in self.stage_widgets.items():
            btn = w['button']
            box = w['box']
            p_label = w['label']
            p_spinner = w['spinner']
            p_bar = w['bar']
            p_indet = w['indeterminate']

            try:
                btn.classes(remove='animate-pulse rotate-infinite')
                btn.props(remove='icon')

                key_idx = ORDERED_STAGES.index(key) if key in ORDERED_STAGES else -1

                if owner_is_self and status.get('state') == 'running' and key == active_key:
                    btn.props('color=primary')
                    box.classes(remove='hidden')
                    p_spinner.classes(remove='hidden')

                    progress = status.get('progress')
                    slide_num = status.get('current_page')
                    cur_idx = status.get('current_index')
                    tot_cnt = status.get('total_count')

                    if slide_num is not None:
                        if cur_idx and tot_cnt:
                            slide_prefix = f'スライド {slide_num} ({cur_idx}/{tot_cnt})'
                        else:
                            slide_prefix = f'スライド {slide_num}'
                    else:
                        slide_prefix = '処理中…'

                    if isinstance(progress, (int, float)):
                        percent = max(0, min(100, int(progress * 100)))
                        p_label.text = f'{slide_prefix} : {percent}％'
                        p_bar.value = percent / 100.0
                        p_bar.classes(remove='hidden')
                        p_indet.classes(add='hidden')
                    else:
                        p_label.text = slide_prefix
                        p_bar.classes(add='hidden')
                        p_indet.classes(remove='hidden')

                    self.set_button_enabled_visual(btn, not cancel_pending)

                elif owner_is_self and (key in completed_stages or (active_idx >= 0 and key_idx < active_idx)):
                    box.classes(remove='hidden')
                    p_spinner.classes(add='hidden')
                    p_label.text = '完了 (100％)'
                    p_bar.value = 1.0
                    p_bar.classes(remove='hidden')
                    p_indet.classes(add='hidden')
                    self.set_button_enabled_visual(btn, False)

                else:
                    box.classes(add='hidden')
                    p_spinner.classes(add='hidden')
                    if globally_busy:
                        btn.props('color=grey-6')
                        self.set_button_enabled_visual(btn, False)
                    else:
                        btn.props('color=primary')
                        self.set_button_enabled_visual(btn, not self.read_only)
            except Exception:
                pass

        for button in self.main_action_buttons:
            self.set_button_enabled_visual(button, not (globally_busy or self.read_only))

        self.refresh_history_log()

    # --- プロジェクト / スライド選択管理 ---

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

        self._on_pages_updated(
            sync_input=(self.pages_input and self.pages_input.value != self.pages_spec),
            save=save_and_refresh,
            refresh=save_and_refresh,
        )

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
        background_tasks.create(self._safe_async_view_update(self.refresh_views, '全画面の再描画'), name=f'refresh-{self.username}')

    async def refresh_views(self) -> None:
        await self.refresh_gallery()
        await self.refresh_simple_editor()
        await self.refresh_editor()
        await self.refresh_slide_videos()

    def debounce_refresh_views(self, stages: tuple[str, ...]) -> None:
        """画面全体の再構築はしない．各接続のタイマーで差分を検出する．"""
        self._safe_view_update(self.poll_generated_outputs, '生成結果の差分反映')

    def refresh_output_views_on_tab_change(self) -> None:
        self._safe_view_update(self.poll_generated_outputs, 'タブ切替時の差分更新')
        self.refresh_history_log()

    # --- プロジェクト一覧・名前・削除 ---

    def list_user_projects(self) -> list[Path]:
        """IDごとのプロジェクトディレクトリ内にある元PDFを列挙する．"""
        projects: list[Path] = []
        if not self.projects_dir.exists():
            return projects
        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir() or not re.fullmatch(r"\d{8}\.[A-Za-z0-9_]{4,12}", project_dir.name):
                continue
            pdfs = sorted((p for p in project_dir.glob('*.pdf') if p.is_file()), key=lambda p: p.name.lower())
            if pdfs:
                projects.append(pdfs[0])
        return sorted(projects, key=lambda p: p.parent.stat().st_mtime, reverse=True)

    @staticmethod
    def project_display_name(pdf_path: Path) -> str:
        cfg = load_project_json(ProjectPaths(pdf_path).root)
        name = str(cfg.get('project_name') or '').strip()
        return name or pdf_path.stem

    def refresh_project_list(self) -> None:
        if self.project_select is None:
            return
        projects = self.list_user_projects()
        options = {pdf_path.parent.name: self.project_display_name(pdf_path) for pdf_path in projects}
        current_value = self.pdf.parent.name if self.pdf and self.pdf.exists() else None
        try:
            self.project_select.options = options
            self.project_select.value = current_value if current_value in options else None
            self.project_select.update()
        except Exception:
            pass

    def refresh_project_name_input(self) -> None:
        if self.project_name_input is not None:
            name = self.project_display_name(self.pdf) if self.pdf and self.pdf.exists() else ''
            self.project_name_input.value = name
            if self.pdf and self.pdf.exists() and not self.read_only:
                self.project_name_input.enable()
            else:
                self.project_name_input.disable()

    def request_rename_project(self) -> None:
        """現在のプロジェクト名をダイアログで変更する．"""
        if not self.write_access_allowed() or not self.pdf or not self.paths:
            return
        if GLOBAL_EXECUTION_LOCK.is_locked():
            ui.notify('処理中はプロジェクト名を変更できません．', type='warning')
            return
        current_name = self.project_display_name(self.pdf)
        with ui.dialog() as dialog, ui.card().classes('p-5 gap-4 w-full max-w-md'):
            ui.label('プロジェクト名の変更').classes('text-base font-bold')
            name_input = ui.input('プロジェクト名', value=current_name).props('outlined autofocus').classes('w-full')
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('キャンセル', on_click=dialog.close).props('outline color=grey')
                def save_name() -> None:
                    name = str(name_input.value or '').strip()
                    if not name:
                        ui.notify('プロジェクト名を入力してください．', type='warning')
                        return
                    if len(name) > 100:
                        ui.notify('プロジェクト名は100文字以内にしてください．', type='warning')
                        return
                    if not self.write_access_allowed() or not self.pdf or not self.paths:
                        return
                    if GLOBAL_EXECUTION_LOCK.is_locked():
                        ui.notify('処理中はプロジェクト名を変更できません．', type='warning')
                        return
                    self.proj_cfg['project_name'] = name
                    save_project_json(self.paths.root, self.proj_cfg)
                    self.refresh_project_list()
                    dialog.close()
                    ui.notify('プロジェクト名を変更しました．', type='positive')
                ui.button('保存', on_click=save_name, icon='save').props('color=primary')
        dialog.open()

    def save_project_name(self) -> None:
        if not self.write_access_allowed() or not self.pdf or not self.paths:
            return
        name = str(self.project_name_input.value or '').strip() if self.project_name_input else ''
        if not name:
            ui.notify('プロジェクト名を入力してください．', type='warning')
            return
        if len(name) > 100:
            ui.notify('プロジェクト名は100文字以内にしてください．', type='warning')
            return
        self.proj_cfg['project_name'] = name
        save_project_json(self.paths.root, self.proj_cfg)
        self.refresh_project_list()
        self.refresh_project_name_input()
        ui.notify('プロジェクト名を保存しました．', type='positive')

    async def select_existing_project(self, project_id: str | None) -> None:
        if not project_id:
            return
        if GLOBAL_EXECUTION_LOCK.is_locked() and GLOBAL_EXECUTION_LOCK.current_user != self.username:
            ui.notify('処理中はプロジェクトを切り替えられません．', type='warning')
            self.refresh_project_list()
            return
        if not re.fullmatch(r"\d{8}\.[A-Za-z0-9_]{4,12}", str(project_id)):
            ui.notify('プロジェクトIDが不正です．', type='negative')
            self.refresh_project_list()
            return
        if (GLOBAL_EXECUTION_LOCK.is_locked() and
                GLOBAL_EXECUTION_LOCK.current_user == self.username and
                GLOBAL_EXECUTION_LOCK.owner_app is not None and
                GLOBAL_EXECUTION_LOCK.owner_app.pdf is not None and
                project_id != GLOBAL_EXECUTION_LOCK.owner_app.pdf.parent.name):
            safe_notify('生成中は別プロジェクトに切り替えられません．', type='warning')
            return
        project_dir = (self.projects_dir / project_id).resolve()
        if project_dir.parent != self.projects_dir.resolve() or not project_dir.is_dir():
            ui.notify('選択したプロジェクトが見つかりません．', type='negative')
            self.refresh_project_list()
            return
        pdfs = [p for p in project_dir.glob('*.pdf') if p.is_file()]
        if len(pdfs) != 1:
            ui.notify('プロジェクト内の元PDFを一意に特定できません．', type='negative')
            self.refresh_project_list()
            return
        target_pdf = pdfs[0].resolve()
        if self.pdf and target_pdf == self.pdf.resolve():
            return
        try:
            self.pdf = target_pdf
            self.paths = ProjectPaths(target_pdf)
            self.proj_cfg = load_project_json(self.paths.root)
            self.mode_code = self.proj_cfg.get('mode') or self.cfg.get('mode', 'lecture')
            self.lang_code = self.proj_cfg.get('language') or self.cfg.get('language', 'ja')
            saved_vmode = self.proj_cfg.get('visual_mode') or self.cfg.get('visual_mode', 'vlm')
            self.default_use_vlm = saved_vmode in ('vlm', 'auto', True, 'true')
            self.slide_visual_modes = self.proj_cfg.get('slide_visual_modes', {})
            self.images = await run.io_bound(ensure_page_images, self.paths, int(self.cfg.get('pdf', {}).get('dpi', 120)))
            marker = user_last_project_file(self.username)
            atomic_write_text(marker, json.dumps({'project_id': project_id}, ensure_ascii=False))
            self.apply_pages_spec(self.proj_cfg.get('pages', ''), save_and_refresh=False)
            self.refresh_project_widgets()
            self.refresh_project_list()
            self.refresh_project_name_input()
            await self.refresh_all()
            ui.notify(f'プロジェクト「{self.project_display_name(target_pdf)}」を開きました．', type='positive')
        except Exception as exc:
            ui.notify(f'プロジェクトを開けませんでした: {exc}', type='negative')
            self.refresh_project_list()

    def request_delete_project(self) -> None:
        if not self.write_access_allowed() or not self.pdf or not self.paths:
            return
        if GLOBAL_EXECUTION_LOCK.is_locked():
            ui.notify('処理中のため，プロジェクトを削除できません．', type='warning')
            return
        pdf_path = self.pdf
        project_name = self.project_display_name(pdf_path)
        with ui.dialog() as dialog, ui.card().classes('p-5 gap-4 max-w-md'):
            dialog.props('persistent')
            with ui.row().classes('items-center gap-2 text-red-400'):
                ui.icon('delete_forever', size='md')
                ui.label('プロジェクトを完全に削除').classes('text-base font-bold')
            ui.label(
                f'「{project_name}」と，元PDF（{pdf_path.name}），生成済みの音声・動画・字幕・編集データを完全に削除します．この操作は取り消せません．'
            ).classes('text-sm text-zinc-300 leading-relaxed')
            with ui.row().classes('w-full justify-end gap-2 pt-2'):
                ui.button('キャンセル', on_click=dialog.close).props('outline color=grey')

                async def confirm_delete() -> None:
                    dialog.close()
                    await self.delete_project(pdf_path)

                ui.button('完全に削除', on_click=confirm_delete).props('color=negative')
        dialog.open()

    async def delete_project(self, pdf_path: Path) -> None:
        if not self.write_access_allowed():
            return
        if GLOBAL_EXECUTION_LOCK.is_locked():
            ui.notify('処理中のため，プロジェクトを削除できません．', type='warning')
            return
        project_dir = pdf_path.parent.resolve()
        if project_dir.parent != self.projects_dir.resolve() or not re.fullmatch(r"\d{8}\.[A-Za-z0-9_]{4,12}", project_dir.name):
            ui.notify('削除対象のプロジェクトを確認できませんでした．', type='negative')
            return
        pdfs = [p for p in project_dir.glob('*.pdf') if p.is_file()]
        if len(pdfs) != 1 or pdfs[0].resolve() != pdf_path.resolve():
            ui.notify('削除対象の元PDFを確認できませんでした．', type='negative')
            return
        deleted_project_name = self.project_display_name(pdf_path)
        deleted_id = project_dir.name
        was_active = bool(self.pdf and self.pdf.resolve() == pdf_path.resolve())
        try:
            shutil.rmtree(project_dir)
            remaining = self.list_user_projects()
            marker = user_last_project_file(self.username)
            if was_active:
                self.pdf = None
                self.paths = None
                self.images = []
                self.proj_cfg = {}
                self.slide_visual_modes = {}
                self._selected_pages.clear()
                self.edit_page = None
                self.mode_code = 'lecture'
                self.lang_code = 'ja'
                self.default_use_vlm = True
                if remaining:
                    await self.select_existing_project(remaining[0].parent.name)
                else:
                    marker.unlink(missing_ok=True)
                    self.refresh_project_widgets()
                    self.refresh_project_name_input()
                    await self.refresh_all()
            else:
                try:
                    last_id = json.loads(marker.read_text(encoding='utf-8')).get('project_id', '')
                except (OSError, ValueError, TypeError):
                    last_id = ''
                if last_id == deleted_id:
                    if remaining:
                        atomic_write_text(marker, json.dumps({'project_id': remaining[0].parent.name}, ensure_ascii=False))
                    else:
                        marker.unlink(missing_ok=True)
            self.refresh_project_list()
            ui.notify(f'プロジェクト「{deleted_project_name}」を完全に削除しました．', type='positive')
        except Exception as exc:
            ui.notify(f'プロジェクトの削除に失敗しました: {exc}', type='negative')
            self.refresh_project_list()

    # --- PDF 読み込み & 復元 ---

    async def load_pdf(self, e) -> None:
        """アップロードPDFごとに一意のIDを発行し，新規プロジェクトとして保存する．"""
        if not self.write_access_allowed():
            return
        if GLOBAL_EXECUTION_LOCK.is_locked():
            ui.notify('処理中は新しいプロジェクトを作成できません．', type='warning')
            return
        filename = Path(e.file.name).name
        if not filename or filename in ('.', '..') or Path(filename).suffix.lower() != '.pdf':
            ui.notify('PDFファイルを選択してください．', type='warning')
            return

        self.projects_dir.mkdir(parents=True, exist_ok=True)
        while True:
            project_id = datetime.now().strftime('%Y%m%d') + '.' + ''.join(secrets.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_') for _ in range(6))
            project_dir = self.projects_dir / project_id
            try:
                project_dir.mkdir()
                break
            except FileExistsError:
                continue

        target_pdf = project_dir / filename
        temp_pdf = project_dir / '.upload.tmp'
        try:
            await e.file.save(temp_pdf)
            if not self.write_access_allowed() or GLOBAL_EXECUTION_LOCK.is_locked():
                shutil.rmtree(project_dir, ignore_errors=True)
                ui.notify('権限または処理状態が変化したため，アップロードを中止しました．', type='warning')
                return
            temp_pdf.replace(target_pdf)
            self.pdf = target_pdf
            self.paths = ProjectPaths(target_pdf)
            self.proj_cfg = load_project_json(self.paths.root)
            self.proj_cfg.setdefault('project_name', target_pdf.stem)
            self.proj_cfg['project_id'] = project_id
            self.proj_cfg['source_pdf'] = filename
            save_project_json(self.paths.root, self.proj_cfg)

            marker = user_last_project_file(self.username)
            atomic_write_text(marker, json.dumps({'project_id': project_id}, ensure_ascii=False))

            self.mode_code = self.proj_cfg.get('mode') or self.cfg.get('mode', 'lecture')
            self.lang_code = self.proj_cfg.get('language') or self.cfg.get('language', 'ja')
            saved_vmode = self.proj_cfg.get('visual_mode') or self.cfg.get('visual_mode', 'vlm')
            self.default_use_vlm = saved_vmode in ('vlm', 'auto', True, 'true')
            self.slide_visual_modes = self.proj_cfg.get('slide_visual_modes', {})
            self.images = await run.io_bound(ensure_page_images, self.paths, int(self.cfg.get('pdf', {}).get('dpi', 120)))
            self.apply_pages_spec(self.proj_cfg.get('pages', ''), save_and_refresh=False)
            self.refresh_project_widgets()
            self.refresh_project_list()
            self.refresh_project_name_input()
            await self.refresh_all()
            ui.notify(f'新しいプロジェクト「{self.project_display_name(target_pdf)}」を作成しました．', type='positive')
        except Exception as exc:
            shutil.rmtree(project_dir, ignore_errors=True)
            ui.notify(f'プロジェクトの作成に失敗しました: {exc}', type='negative')
            self.refresh_project_list()

    async def restore_last_project(self) -> None:
        if self._restore_started:
            return
        self._restore_started = True
        try:
            marker = user_last_project_file(self.username)
            try:
                project_id = str(json.loads(marker.read_text(encoding='utf-8')).get('project_id', ''))
            except (OSError, ValueError, TypeError):
                project_id = ''
            projects = self.list_user_projects()
            if not project_id or not any(p.parent.name == project_id for p in projects):
                if not projects:
                    return
                project_id = projects[0].parent.name
                atomic_write_text(marker, json.dumps({'project_id': project_id}, ensure_ascii=False))
            await self.select_existing_project(project_id)
            if self.read_only:
                safe_notify('利用可能時間外のため，閲覧モードでプロジェクトを復元しました．', type='info')
        except Exception as exc:
            safe_notify(f'前回のプロジェクトを復元できませんでした: {exc}', type='warning')

    def refresh_project_widgets(self) -> None:
        if self.mode_select:
            self.mode_select.value = self.mode_code
        if self.lang_select:
            self.lang_select.value = self.lang_code
        if self.default_vlm_switch:
            self.default_vlm_switch.value = self.default_use_vlm
        if self.pages_input:
            self.pages_input.value = self.pages_spec if self.pdf else ''
        if self.active_count_label:
            if self.pdf:
                self.active_count_label.text = f'対象スライド: {len(self._selected_pages)} / {self.total_slides} スライド'
            else:
                self.active_count_label.text = '対象スライド: 0 / 0 スライド'

    def save_project_settings(self) -> None:
        if not self.write_access_allowed() or not self.paths:
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
        if not self.write_access_allowed():
            return
        if active:
            self._selected_pages.add(page_num)
        else:
            self._selected_pages.discard(page_num)
        self._on_pages_updated(sync_input=True, save=True, refresh=True)

    def select_all_slides(self) -> None:
        if not self.write_access_allowed():
            return
        self._selected_pages = set(range(1, self.total_slides + 1)) if (self.pdf and self.total_slides > 0) else set()
        self._on_pages_updated(sync_input=True, save=True, refresh=True)

    def clear_all_slides(self) -> None:
        if not self.write_access_allowed():
            return
        self._selected_pages = set()
        self._on_pages_updated(sync_input=True, save=True, refresh=True)

    # --- 生成データ初期化（PDF・設定・ページ画像は保存） ---

    def confirm_reset_generated_data(self) -> None:
        if not self.write_access_allowed() or not self.paths or not self.pdf:
            safe_notify('先にプロジェクトを選択してください．', type='warning')
            return
        if GLOBAL_EXECUTION_LOCK.is_locked():
            safe_notify('生成処理中は生成データを削除できません．', type='warning')
            return

        # 確認したプロジェクトと実際に削除するプロジェクトが一致するか再確認する．
        target_pdf = self.pdf.resolve()
        with ui.dialog() as dialog, ui.card().classes('min-w-[340px] max-w-lg p-5 gap-3'):
            ui.label('生成データを削除しますか？').classes('text-lg font-bold text-red-400')
            ui.label('このプロジェクトのナレーション原稿，翻訳・ポインタ情報，音声，単体動画，完成動画と字幕を削除します．')
            ui.label('元PDF，プロジェクト名・設定，スライド画像は残します．削除後に自動生成は開始しません．この操作は取り消せません．').classes('text-sm text-zinc-400')
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('キャンセル', on_click=dialog.close).props('flat')

                async def confirm() -> None:
                    dialog.close()
                    await self.reset_generated_data(target_pdf)

                ui.button('削除する', on_click=confirm).props('color=negative')
        dialog.open()

    async def reset_generated_data(self, target_pdf: Path) -> None:
        if not self.write_access_allowed():
            return
        # 他ユーザの生成や削除との競合を避けるため，削除操作自体も排他制御する．
        acquired = await GLOBAL_EXECUTION_LOCK.acquire(self.username, '生成データ削除', self)
        if not acquired:
            safe_notify('別の処理が実行中です．生成データを削除できません．', type='warning')
            return
        try:
            if not self.pdf or self.pdf.resolve() != target_pdf or not self.paths:
                safe_notify('プロジェクトが変更されたため，削除を取りやめました．', type='warning')
                return
            paths = self.paths
            expected_root = target_pdf.with_name(target_pdf.stem + '_lecture')
            if paths.root.resolve() != expected_root.resolve():
                raise RuntimeError('削除対象ディレクトリを検証できません')

            # プロジェクト設定(project.json)とpages/を含むディレクトリ全体は削除しない．
            for directory in (paths.explanations_dir, paths.audio_dir, paths.video_dir):
                if directory.exists():
                    shutil.rmtree(directory)

            # slide_lecture.py のforce処理同様，最終出力を初期化する．
            # project.json やPDF，ユーザ別ジョブ履歴には触れない．
            for output in (paths.final_video, paths.final_ja_srt, paths.final_en_srt):
                output.unlink(missing_ok=True)
            # 完成動画と同じstemで生成される一時・派生出力も対象にする．
            for output in paths.root.glob(f'{target_pdf.stem}*'):
                if output.is_file():
                    output.unlink()

            self.edit_page = self.active_pages[0] if self.active_pages else None
            self.update_job_status('idle', stage='', message='生成データを削除しました',
                                   completed_stages=[], progress=None, current_page=None,
                                   current_index=None, total_count=None,
                                   cancellation_requested=False)
            self.append_job_log('[RESET] 生成データを削除しました（元PDF・設定・ページ画像は保持）')
            await self.refresh_all()
            safe_notify('生成データを削除しました．生成ボタンから作り直せます．', type='positive')
        except Exception as exc:
            LOGGER.exception('生成データの削除に失敗しました')
            safe_notify(f'生成データの削除に失敗しました: {exc}', type='negative')
        finally:
            GLOBAL_EXECUTION_LOCK.release()
            self.refresh_job_status()

    # --- パイプライン実行・中断制御 ---

    def confirm_cancel(self) -> None:
        with ui.dialog() as dialog, ui.card().classes('min-w-[320px]'):
            ui.label('生成処理を中断しますか？').classes('text-lg font-bold')
            ui.label('実行中の処理によっては，現在の工程が戻るまで停止に時間がかかります．').classes('text-sm text-zinc-400')
            with ui.row().classes('w-full justify-end'):
                ui.button('戻る', on_click=dialog.close).props('flat')
                ui.button('中断を要求', color='negative', on_click=lambda: (dialog.close(), background_tasks.create(self.request_cancel())))
        dialog.open()

    async def request_cancel(self) -> None:
        if not GLOBAL_EXECUTION_LOCK.is_locked():
            safe_notify('中断できる処理はありません．', type='info')
            return
        if GLOBAL_EXECUTION_LOCK.current_user != self.username:
            safe_notify('他のユーザーの処理は中断できません．', type='warning')
            return

        owner = GLOBAL_EXECUTION_LOCK.owner_app
        if owner is None:
            safe_notify('処理の管理情報が見つからないため，中断要求を送れません．', type='negative')
            return
        if owner.cancellation_requested:
            safe_notify('すでに中断要求を受け付けています．', type='warning')
            return

        owner.cancellation_requested = True
        owner.update_job_status(
            'running', cancellation_requested=True,
            message='中断要求を受け付けました．実行中の処理を停止しています',
        )
        owner.append_job_log('[CANCEL] 中断要求を受け付けました')

        proc = owner.current_process
        if proc is not None and proc.returncode is None:
            try:
                if sys.platform != 'win32':
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
                safe_notify('中断要求を送信しました．停止を確認しています…', type='warning')
            except ProcessLookupError:
                safe_notify('処理はすでに停止しています．状態を確認しています…', type='info')
            except Exception as exc:
                owner.append_job_log(f'[CANCEL] プロセス停止要求に失敗: {exc}')
                safe_notify(f'プロセス停止要求に失敗しました: {exc}', type='negative')
        else:
            safe_notify('中断要求を受け付けました．実行中の処理が戻り次第，中断します．', type='warning')

    def set_processing(self, value: bool) -> None:
        self.processing = value
        if not any(self._widget_available(b) for b in self.pipeline_buttons):
            return
        status = read_user_job_status(self.username)
        if value:
            status = {**status, 'state': 'running'}
        elif not GLOBAL_EXECUTION_LOCK.is_locked():
            status = {**status, 'state': 'idle'}
        active_key = self.get_active_stage_key(status)
        for key, button in zip(('explain', 'align', 'tts', 'video'), self.pipeline_buttons):
            try:
                if value:
                    self.set_button_enabled_visual(button, key == active_key and not self.cancellation_requested)
                else:
                    self.set_button_enabled_visual(button, not self.read_only)
            except Exception:
                pass

    def base_args(self) -> list[str]:
        args = [
            '--mode', self.mode_code,
            '--lang', self.lang_code,
            '--visual-mode', 'vlm' if self.default_use_vlm else 'pdf',
        ]
        spec = self.pages_spec.strip()
        if spec and spec != 'none':
            args.extend(['--pages', spec])
        return args

    async def handle_pipeline_button(self, stage: str, initial_title: str) -> None:
        if GLOBAL_EXECUTION_LOCK.is_locked():
            if GLOBAL_EXECUTION_LOCK.current_user == self.username:
                owner = GLOBAL_EXECUTION_LOCK.owner_app
                status = read_user_job_status(self.username)
                active_key = self.get_active_stage_key(status)
                if active_key == stage and owner is not None:
                    self.confirm_cancel()
                else:
                    safe_notify('現在の処理が終わるまで，ほかの工程は開始できません．', type='info')
            else:
                safe_notify(f'{GLOBAL_EXECUTION_LOCK.current_user or "別のユーザー"} が処理中です．完了までお待ちください．', type='warning')
            return
        await self.pipeline(stage, initial_title)

    async def pipeline(self, stage: str, initial_title: str) -> None:
        if not self.write_access_allowed():
            return
        self._start_background_job(
            lambda: self._pipeline_worker(stage, initial_title),
            busy_message='このユーザーの生成処理はすでに実行中です．',
        )

    def _determine_starting_stage(self, target_stage: str) -> str:
        """必要な前工程の成果物を確認し，実行開始ステージを決定する．"""
        assert self.paths is not None
        stage_index = ORDERED_STAGES.index(target_stage)

        has_missing_explain = any(
            not self.paths.explanation(p).exists()
            or not self.paths.explanation(p).read_text(encoding='utf-8').strip()
            for p in self.active_pages
        )
        has_missing_align = any(
            not self.paths.alignment(p).exists()
            for p in self.active_pages
        )
        has_missing_tts = any(
            not self.paths.audio(p).exists()
            for p in self.active_pages
        )

        if stage_index >= ORDERED_STAGES.index('explain') and has_missing_explain:
            return 'explain'
        if stage_index >= ORDERED_STAGES.index('align') and has_missing_align:
            return 'align'
        if stage_index >= ORDERED_STAGES.index('tts') and has_missing_tts:
            return 'tts'
        return target_stage

    async def _pipeline_worker(self, stage: str, initial_title: str) -> None:
        if not self.pdf:
            safe_notify('先にプレゼンテーションPDFを選択してください．', type='warning')
            self.current_task = None
            return
        if not self.active_pages:
            safe_notify('処理対象となるスライドが1枚も選択されていません．', type='warning')
            self.current_task = None
            return
        if self.processing:
            safe_notify('別の処理が実行中です．処理が終わるまでお待ちください．', type='warning')
            self.current_task = None
            return

        from_stage = self._determine_starting_stage(stage)
        first_idx = ORDERED_STAGES.index(from_stage)
        target_idx = ORDERED_STAGES.index(stage)
        stages_to_run = ORDERED_STAGES[first_idx:target_idx + 1]

        lock_acquired = await GLOBAL_EXECUTION_LOCK.acquire(self.username, f"パイプライン ({stage})", self)
        if not lock_acquired:
            msg = f'他のユーザー（{GLOBAL_EXECUTION_LOCK.current_user or "誰か"}）が「{GLOBAL_EXECUTION_LOCK.task_name or "処理"}」を実行中です．完了するまでリクエストは受け付けられません．'
            safe_notify(msg, type='negative', duration=5)
            self.current_task = None
            return

        self.cancellation_requested = False
        self.current_task = asyncio.current_task()
        self.save_project_settings()
        total_active_slides = len(self.active_pages)
        completed_stages: list[str] = []
        all_logs: list[str] = []
        current_phase = from_stage
        succeeded = True
        cancelled = False

        self.update_job_status(
            'running',
            stage=from_stage,
            message='処理を開始します…',
            completed_stages=[],
            cancellation_requested=False,
            progress=0.0,
            current_page=self.active_pages[0],
            current_index=1,
            total_count=total_active_slides,
            started_at=datetime.now().isoformat(timespec='seconds'),
        )
        self.set_processing(True)

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        try:
            for run_index, run_stage in enumerate(stages_to_run):
                if self.cancellation_requested:
                    cancelled = True
                    succeeded = False
                    self.update_job_status(
                        'cancelled', stage=run_stage, completed_stages=list(completed_stages),
                        message='処理を中断しました', cancellation_requested=False,
                        finished_at=datetime.now().isoformat(timespec='seconds'),
                    )
                    break

                current_phase = run_stage
                stage_title = next((label for key, label, _ in STAGE_DEFINITIONS if key == run_stage), run_stage)
                stage_message = next((msg for key, _, msg in STAGE_DEFINITIONS if key == run_stage), '処理中…')
                self.update_job_status(
                    'running', stage=run_stage, completed_stages=list(completed_stages),
                    message=stage_message, progress=0.0,
                    current_page=self.active_pages[0], current_index=1,
                    total_count=total_active_slides,
                )
                self.append_job_log(f'=== {stage_title} 開始: --from {run_stage} ===')

                args = self.base_args()
                cmd = [
                    sys.executable, '-u', 'slide_lecture.py',
                    str(self.pdf), '--from', run_stage, *args,
                ]

                create_group_kwargs: dict[str, Any] = {}
                if sys.platform != 'win32':
                    create_group_kwargs['preexec_fn'] = os.setsid

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                    **create_group_kwargs,
                )
                self.current_process = proc

                while True:
                    line_bytes = await proc.stdout.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode('utf-8', errors='replace').rstrip()
                    all_logs.append(line)
                    self.append_job_log(line)

                    if line.startswith('[PROGRESS]'):
                        try:
                            p_data = json.loads(line[10:].strip())
                            phase = p_data.get('phase')
                            cur = p_data.get('current', 1)
                            tot = p_data.get('total', total_active_slides)
                            p_num = p_data.get('page')
                            msg = p_data.get('message', '')

                            effective_phase = 'video' if phase == 'concat' else phase
                            if effective_phase != run_stage:
                                continue

                            frac = 1.0 if phase == 'concat' else (cur / max(1, tot) if tot else 0.0)
                            self.update_job_status(
                                'running', stage=run_stage,
                                completed_stages=list(completed_stages),
                                message=msg or ('完成動画を結合・生成中…' if phase == 'concat' else stage_message),
                                progress=max(0.0, min(1.0, float(frac))),
                                current_page=p_num,
                                current_index=cur,
                                total_count=tot,
                            )
                            self.debounce_refresh_views((run_stage,))
                        except Exception:
                            pass

                    await asyncio.sleep(0.01)

                rc = await proc.wait()
                self.current_process = None

                if rc != 0:
                    succeeded = False
                    if self.cancellation_requested:
                        cancelled = True
                        self.update_job_status(
                            'cancelled', stage=run_stage,
                            completed_stages=list(completed_stages),
                            message='処理を中断しました', cancellation_requested=False,
                            finished_at=datetime.now().isoformat(timespec='seconds'),
                        )
                    else:
                        self.update_job_status(
                            'failed', stage=run_stage,
                            completed_stages=list(completed_stages),
                            message=f'工程「{stage_title}」が終了コード {rc} で失敗しました',
                            finished_at=datetime.now().isoformat(timespec='seconds'),
                        )
                        safe_notify(f'工程「{stage_title}」が終了コード {rc} で失敗したため，後続処理を中断しました．', type='negative')
                    break

                if self.cancellation_requested:
                    succeeded = False
                    cancelled = True
                    self.update_job_status(
                        'cancelled', stage=run_stage,
                        completed_stages=list(completed_stages),
                        message='処理を中断しました', cancellation_requested=False,
                        finished_at=datetime.now().isoformat(timespec='seconds'),
                    )
                    break

                if run_stage not in completed_stages:
                    completed_stages.append(run_stage)
                self.update_job_status(
                    'running', stage=run_stage,
                    completed_stages=list(completed_stages),
                    message=f'{stage_title} 完了', progress=1.0,
                    current_page=self.active_pages[-1],
                    current_index=total_active_slides,
                    total_count=total_active_slides,
                )
                self.debounce_refresh_views((run_stage,))

            if succeeded and not cancelled:
                self.update_job_status(
                    'completed', stage=stage,
                    completed_stages=list(completed_stages),
                    message='要求された処理がすべて完了しました',
                    progress=1.0,
                    cancellation_requested=False,
                    finished_at=datetime.now().isoformat(timespec='seconds'),
                )
                safe_notify('要求された処理がすべて完了しました．', type='positive')
            elif cancelled:
                saved = read_user_job_status(self.username)
                if saved.get('state') != 'cancelled':
                    self.update_job_status(
                        'cancelled', completed_stages=list(completed_stages),
                        message='処理を中断しました', cancellation_requested=False,
                        finished_at=datetime.now().isoformat(timespec='seconds'),
                    )

            await self._safe_async_view_update(self.refresh_all, '処理完了後の全画面更新')

        except asyncio.CancelledError:
            proc = self.current_process
            if proc is not None and proc.returncode is None:
                try:
                    if sys.platform != 'win32':
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    else:
                        proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    if proc.returncode is None:
                        if sys.platform != 'win32':
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        else:
                            proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                except Exception:
                    LOGGER.exception('子プロセスの終了処理に失敗しました')
            self.update_job_status(
                'cancelled', stage=current_phase, completed_stages=list(completed_stages),
                message='処理がキャンセルされました', cancellation_requested=False,
                finished_at=datetime.now().isoformat(timespec='seconds'),
            )
            safe_notify('処理がキャンセルされました．', type='warning')
        except Exception as exc:
            self.append_job_log(f'[ERROR] {exc}')
            self.update_job_status(
                'failed', stage=current_phase, completed_stages=list(completed_stages),
                message=str(exc)[:240], finished_at=datetime.now().isoformat(timespec='seconds'),
            )
            safe_notify(f'処理に失敗しました: {exc}', type='negative')
        finally:
            self._finalize_cancel_request(completed_stages=list(completed_stages))
            self._complete_background_job()

    # --- 個別スライド操作（原稿再生成・アライメント・TTS） ---

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

    async def regenerate_narration(self, page: int, text_area: Any) -> None:
        if not self.write_access_allowed():
            return
        self._start_background_job(
            lambda: self._regenerate_narration_worker(page, text_area),
            busy_message='このユーザーの処理はすでに実行中です．',
        )

    async def _regenerate_narration_worker(self, page: int, text_area: Any) -> None:
        if not self.write_access_allowed() or not self.pdf or not self.paths:
            return
        if self.processing:
            safe_notify('別の処理が実行中です．処理が終わるまでお待ちください．', type='warning')
            return

        lock_acquired = await GLOBAL_EXECUTION_LOCK.acquire(self.username, f"スライド {page} ナレーション再生成", self)
        if not lock_acquired:
            safe_notify(f'他のユーザーが処理を実行中です．完了するまでお待ちください．', type='negative', duration=5)
            return

        self.update_job_status(
            'running',
            stage='explain',
            message=f'スライド {page} の原稿を作成中…',
            progress=0.2,
            current_page=page,
            started_at=datetime.now().isoformat(timespec='seconds'),
        )
        self.append_job_log(f'[REGEN] スライド {page} ナレーション再生成開始')
        self.set_processing(True)
        self.current_task = asyncio.current_task()
        try:
            cfg = self.cfg
            client = await run.io_bound(make_client, cfg['llm'])

            self.update_job_status('running', stage='explain', message=f'スライド {page} をLLM推論中…', progress=0.6, current_page=page)

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
                try:
                    text_area.value = new_narration.strip()
                except Exception:
                    pass
                self.update_job_status('completed', message=f'スライド {page} のナレーション再生成完了', progress=1.0, finished_at=datetime.now().isoformat(timespec='seconds'))
                self.append_job_log(f'[REGEN] 完了: 新しいナレーションを保存しました')
                safe_notify('ナレーションを再生成しました（古い字幕・音声・動画を初期化しました）．', type='positive')
                self._safe_view_update(self.poll_generated_outputs, '個別生成結果の差分更新')
        except asyncio.CancelledError:
            self.update_job_status('cancelled', message='ナレーション再生成を中断しました', finished_at=datetime.now().isoformat(timespec='seconds'))
            safe_notify('ナレーション再生成を中断しました．', type='warning')
        except Exception as exc:
            self.update_job_status('failed', message=str(exc)[:240], finished_at=datetime.now().isoformat(timespec='seconds'))
            safe_notify(f'再生成に失敗しました: {exc}', type='negative')
        finally:
            self._finalize_cancel_request()
            self._complete_background_job()

    async def save_and_realign(self, page: int, text: str, page_use_vlm: bool | None = None) -> None:
        if not self.write_access_allowed():
            return
        self._start_background_job(
            lambda: self._save_and_realign_worker(page, text, page_use_vlm),
            busy_message='このユーザーの処理はすでに実行中です．',
        )

    async def _save_and_realign_worker(self, page: int, text: str, page_use_vlm: bool | None = None) -> None:
        if not self.write_access_allowed() or not self.pdf or not self.paths:
            return

        ep = self.paths.explanation(page)
        atomic_write_text(ep, text.rstrip() + '\n')

        mode_str = ('vlm' if page_use_vlm else 'pdf') if page_use_vlm is not None else self.slide_visual_modes.get(str(page), 'vlm' if self.default_use_vlm else 'pdf')
        self.slide_visual_modes[str(page)] = mode_str
        self.save_project_settings()

        if self.processing:
            safe_notify('別の処理が実行中です．処理が終わるまでお待ちください．', type='warning')
            return

        lock_acquired = await GLOBAL_EXECUTION_LOCK.acquire(self.username, f"スライド {page} 字幕・ポインタ再解析", self)
        if not lock_acquired:
            safe_notify(f'他のユーザーが処理を実行中です．完了するまでお待ちください．', type='negative', duration=5)
            return

        mode_label = "VLM併用" if mode_str == 'vlm' else "PDF基準"
        self.update_job_status('running', stage='align', message=f'スライド {page} 解析中（{mode_label}）', progress=0.3, current_page=page, started_at=datetime.now().isoformat(timespec='seconds'))
        self.append_job_log(f'[ALIGN] スライド {page} の要素抽出 ({mode_label}) と視線誘導を再計算中...')
        self.set_processing(True)
        self.current_task = asyncio.current_task()
        try:
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
                atomic_write_text(alp, json.dumps(data, ensure_ascii=False, indent=2))

                self.paths.cleanup_downstream(page, include_alignment=False)
                self.update_job_status('completed', message=f'スライド {page} の解析完了', progress=1.0, finished_at=datetime.now().isoformat(timespec='seconds'))
                self.append_job_log(f'[ALIGN] 完了: 字幕・ポインタアライメントを保存しました')
                safe_notify(f'スライド {page} の字幕とポインタを再生成しました（{mode_label}）．', type='positive')
                self._safe_view_update(self.poll_generated_outputs, '個別生成結果の差分更新')
        except asyncio.CancelledError:
            self.update_job_status('cancelled', message='字幕・ポインタ再解析を中断しました', finished_at=datetime.now().isoformat(timespec='seconds'))
            safe_notify('解析を中断しました．', type='warning')
        except Exception as exc:
            self.update_job_status('failed', message=str(exc)[:240], finished_at=datetime.now().isoformat(timespec='seconds'))
            safe_notify(f'再解析に失敗しました: {exc}', type='negative')
        finally:
            self._finalize_cancel_request()
            self._complete_background_job()

    async def save_alignment(self, page: int, rows: list[dict], align_data: dict) -> None:
        if not self.write_access_allowed() or not self.paths:
            return
        alp = self.paths.alignment(page)
        align_data['alignments'] = rows
        atomic_write_text(alp, json.dumps(align_data, ensure_ascii=False, indent=2))

        self._invalidate_page_video_outputs(page)
        ui.notify('字幕とポインタの修正を保存しました．', type='positive')
        await self.refresh_editor()

    async def generate_slide_tts(self, page: int) -> None:
        if not self.write_access_allowed():
            return
        self._start_background_job(
            lambda: self._generate_slide_tts_worker(page),
            busy_message='このユーザーの処理はすでに実行中です．',
        )

    async def _generate_slide_tts_worker(self, page: int) -> None:
        if not self.write_access_allowed() or not self.pdf or not self.paths:
            return
        txt_path = self.paths.explanation(page)
        if not txt_path.exists() or not txt_path.read_text(encoding='utf-8').strip():
            safe_notify(f'スライド {page} のナレーション原稿がありません．先に作成してください．', type='warning')
            return
        if self.processing:
            safe_notify('別の処理が実行中です．処理が終わるまでお待ちください．', type='warning')
            return

        lock_acquired = await GLOBAL_EXECUTION_LOCK.acquire(self.username, f"スライド {page} 音声合成", self)
        if not lock_acquired:
            safe_notify(f'他のユーザーが処理を実行中です．完了するまでお待ちください．', type='negative', duration=5)
            return

        self.update_job_status('running', stage='tts', message=f'スライド {page} 音声合成中…', progress=0.4, current_page=page, started_at=datetime.now().isoformat(timespec='seconds'))
        self.append_job_log(f'[TTS] スライド {page} の音声合成を開始 (lang={self.lang_code})')
        self.set_processing(True)
        self.current_task = asyncio.current_task()
        try:
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
                self._invalidate_page_video_outputs(page)

                self.update_job_status('completed', message=f'スライド {page} の音声合成完了', progress=1.0, finished_at=datetime.now().isoformat(timespec='seconds'))
                self.append_job_log(f'[TTS] スライド {page} の音声を保存しました: {out_mp3.name}')
                safe_notify(f'スライド {page} の音声を合成しました．', type='positive')
                self._safe_view_update(self.poll_generated_outputs, '音声生成結果の差分更新')
        except asyncio.CancelledError:
            self.update_job_status('cancelled', message='音声合成を中断しました', finished_at=datetime.now().isoformat(timespec='seconds'))
            safe_notify('音声合成を中断しました．', type='warning')
        except Exception as exc:
            self.update_job_status('failed', message=str(exc)[:240], finished_at=datetime.now().isoformat(timespec='seconds'))
            safe_notify(f'音声合成に失敗しました: {exc}', type='negative')
        finally:
            self._finalize_cancel_request()
            self._complete_background_job()

    # --- UI レンダリング: スライドデッキ / エディタ / ギャラリー ---

    async def refresh_gallery(self) -> None:
        if not self.gallery:
            return
        self.gallery.clear()
        if not self.images:
            self.gallery.add(ui.label('プレゼンテーションPDFを選択してください．'))
            return

        with self.gallery:
            with ui.row().classes('w-full items-center justify-between pb-2'):
                desc = f'全 {self.total_slides} スライド' + ('（閲覧モード）' if self.read_only else '（カードをクリックまたはチェックボックスで対象を切り替えられます）')
                ui.label(desc).classes('text-sm text-gray-500 dark:text-gray-400')
                if not self.read_only:
                    with ui.row().classes('gap-2'):
                        self.register_main_action_button(ui.button('全スライドを選択', on_click=self.select_all_slides).props('dense outline size=sm'))
                        self.register_main_action_button(ui.button('すべて解除', on_click=self.clear_all_slides).props('dense outline size=sm color=negative'))

            with ui.grid(columns=5).classes('w-full gap-4'):
                for n, img in enumerate(self.images, 1):
                    is_active = n in self._selected_pages
                    card_cls = 'w-full cursor-pointer transition-all border p-3 rounded-lg gap-2 bg-zinc-800 border-zinc-700 shadow-sm '
                    img_cls = 'w-full rounded shadow-sm'
                    if not is_active:
                        card_cls += 'opacity-40 hover:border-zinc-600'
                        img_cls += ' grayscale'
                    if self.read_only:
                        card_cls = card_cls.replace('cursor-pointer', 'cursor-default')

                    card = ui.card().classes(card_cls)
                    with card:
                        if not self.read_only:
                            card.on('click', lambda _, page=n, cur=is_active: self.toggle_slide_active(page, not cur))
                        ui.image(file_url(img)).classes(img_cls)

                        with ui.row().classes('w-full items-center justify-between pt-1'):
                            cb = ui.checkbox(
                                f'スライド {n}',
                                value=is_active,
                                on_change=lambda e, page=n: self.toggle_slide_active(page, e.value),
                            ).props('dense dark color=blue').classes('text-white font-medium text-sm')
                            if self.read_only:
                                cb.disable()

    async def refresh_simple_editor(self) -> None:
        if not self.simple_edit_container:
            return
        self.simple_edit_container.clear()
        self._simple_textareas.clear()
        self._simple_text_signatures.clear()
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
                                ).props('outlined').classes('w-full h-[180px]').style('height: 180px; min-height: 180px;')
                                ta.props('input-style="height: 140px; resize: vertical;"')
                                self._simple_textareas[p] = ta
                                self._simple_text_signatures[p] = self._file_signature(txt_path)

                                with ui.row().classes('w-full items-center justify-end gap-3 pt-1'):
                                    self.register_main_action_button(ui.button(
                                        '✨ ナレーションを再生成',
                                        on_click=lambda _, page=p, area=ta: self.regenerate_narration(page, area),
                                    ).props('outline dense'))

                                    self.register_main_action_button(ui.button(
                                        '🔄 保存して字幕・ポインタを再解析',
                                        on_click=lambda _, page=p, area=ta: self.save_and_realign(page, area.value),
                                    ).props('dense color=primary'))

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
        self._detail_signatures = {
            'text': self._file_signature(current_text_path),
            'align': self._file_signature(self.paths.alignment(page)),
            'audio': self._file_signature(self.paths.audio(page)),
        }

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
                    sel.on_value_change(lambda e: background_tasks.create(self.select_edit_page(e.value)))

                    ui.button('◀ 前', on_click=self.prev_edit).props(f'disable={idx == 0} outlined dense')
                    ui.button('次 ▶', on_click=self.next_edit).props(f'disable={idx == len(self.active_pages)-1} outlined dense')

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
                            self.register_main_action_button(ui.button(
                                '🔄 このスライドの音声を再生成',
                                on_click=lambda _, p=page: self.generate_slide_tts(p),
                            ).props('dense outline size=sm').classes('w-full mt-1'))
                        else:
                            ui.label('（音声未生成）').classes('text-xs text-zinc-500 py-1')
                            self.register_main_action_button(ui.button(
                                '🔊 このスライドの音声を生成',
                                on_click=lambda _, p=page: self.generate_slide_tts(p),
                            ).props('dense color=primary size=sm').classes('w-full'))

                with ui.column().classes('flex-[3] min-w-0 gap-3'):
                    ui.label('🎯 検出された要素一覧（バッジ対応）').classes('text-xs font-semibold text-zinc-400')
                    if not blocks:
                        ui.label('（検出された要素はありません）').classes('text-xs text-zinc-500 italic')
                    else:
                        with ui.column().classes('w-full gap-1.5 max-h-[450px] overflow-y-auto pr-1'):
                            for b in blocks:
                                bid = b.get('block_id')
                                label = b.get('label') or '名称なし'
                                btype = b.get('type', 'image_subpart')
                                style_cls, type_name = BADGE_COLORS.get(btype, BADGE_COLORS['image_subpart'])

                                with ui.row().classes(
                                    'w-full items-center justify-between p-2 rounded bg-zinc-900/90 '
                                    'border border-zinc-800 hover:border-zinc-700 transition-colors'
                                ):
                                    with ui.row().classes('items-center gap-2 grow'):
                                        ui.label(f'#{bid}').classes(f'text-xs font-bold font-mono px-2 py-0.5 rounded border {style_cls}')
                                        ui.label(label).classes('text-xs text-zinc-200 font-medium break-all')
                                    ui.badge(type_name).props('outline').classes('text-[10px] text-zinc-400 shrink-0')

            with ui.column().classes('w-full gap-3 pt-2'):
                main_lang = '日本語' if self.lang_code == 'ja' else '英語'
                text_area = ui.textarea(f'主言語ナレーション原稿（{main_lang}）', value=current_text).props('outlined').classes('w-full').style('min-height: 180px')
                with ui.row().classes('w-full gap-2'):
                    self.register_main_action_button(ui.button('✨ ナレーションを再生成', on_click=lambda: self.regenerate_narration(page, text_area)).classes('grow').props('outline'))
                    self.register_main_action_button(ui.button('🔄 保存して字幕・ポインタを再解析', on_click=lambda: self.save_and_realign(page, text_area.value, page_vlm_switch.value)).classes('grow').props('color=primary'))

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
                        for r in rows:
                            bid = r['_select'].value
                            tr = r['_trans'].value or ''
                            if self.lang_code == 'ja':
                                out.append({'sentence': r['sentence'], 'block_id': bid, 'ja_sentence': r['sentence'], 'en_sentence': tr})
                            else:
                                out.append({'sentence': r['sentence'], 'block_id': bid, 'en_sentence': r['sentence'], 'ja_sentence': tr})
                        await self.save_alignment(page, out, align_data)

                    self.register_main_action_button(ui.button('💾 字幕・ポインタ修正を保存', on_click=save_rows).classes('w-full mt-2'))

    async def refresh_slide_videos(self) -> None:
        if not self.slide_video_gallery:
            return
        self.slide_video_gallery.clear()
        self._video_cards.clear()
        self._video_signatures.clear()
        self._video_grid = None
        if not self.paths:
            with self.slide_video_gallery:
                ui.label('プレゼンテーションPDFを選択してください．').classes('text-blue-500 dark:text-blue-400')
            return

        with self.slide_video_gallery:
            self._video_grid = ui.grid(columns=4).classes('w-full gap-4')
        for page in self.active_pages:
            self._replace_video_card(page)

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

    async def refresh_all(self) -> None:
        self._output_project_key = str(self.pdf.resolve()) if self.pdf else None
        await self.refresh_views()
        await self.refresh_final_video()

    # --- 設定タブ (LLM / TTS / 辞書) ---

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
                llm_model_select = ui.select(
                    options=[current_model] if current_model else [],
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

            with ui.card().classes('w-full p-3 bg-zinc-950 border border-zinc-800 rounded-lg gap-2'):
                ui.label('🧪 LLM 接続テスト（テキスト応答）').classes('text-xs font-bold text-zinc-300')
                with ui.row().classes('w-full items-center gap-2'):
                    llm_test_input = ui.input('テストプロンプト', value='こんにちは！自己紹介を1文でしてください．').props('dense outlined').classes('grow')
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
                        test_cfg = {'base_url': llm_base.value.strip(), 'api_key': (llm_key.value or '').strip() or 'dummy'}
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

            with ui.card().classes('w-full p-3 bg-zinc-950 border border-zinc-800 rounded-lg gap-3 mt-2'):
                with ui.row().classes('items-center justify-between w-full'):
                    ui.label('👁️ VLM 画像認識 & アライメントテスト').classes('text-xs font-bold text-zinc-300')
                    ui.label('任意の画像（写真・イラスト・図表など何でも可）を放り込んで認識とバウンディングボックス抽出をテスト').classes('text-xs text-zinc-500')

                uploaded_vlm_img = {'path': None}

                with ui.row().classes('w-full items-start gap-4'):
                    with ui.column().classes('w-80 shrink-0 gap-2'):
                        vlm_uploader = ui.upload(label='画像をアップロード（何でも可）', auto_upload=True, max_files=1).props('accept="image/*" dense').classes('w-full')
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
                            vlm_description_box.text = ''
                            vlm_status_label.text = '画像を受信しました．自動解析を開始します…'
                            with vlm_preview_container:
                                ui.image(file_url(saved_path)).classes('w-full rounded border border-zinc-700 shadow-sm')
                            await run_vlm_test()

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
                        test_cfg = {'base_url': llm_base.value.strip(), 'api_key': (llm_key.value or '').strip() or 'dummy'}
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

                        with vlm_preview_container:
                            ui.label('🎯 アライメントプレビュー & 検出要素一覧').classes('text-xs font-bold text-zinc-300 mt-2')
                            with ui.row().classes('w-full items-start gap-4'):
                                ui.image(f'{file_url(preview_path)}&ts={ts}').classes('w-full max-w-xl rounded-lg border border-zinc-700 shadow-md')
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
                                                style_cls, type_name = BADGE_COLORS.get(btype, BADGE_COLORS['image_subpart'])

                                                with ui.row().classes('w-full items-center justify-between p-2 rounded bg-zinc-900/90 border border-zinc-800 hover:border-zinc-700 transition-colors'):
                                                    with ui.row().classes('items-center gap-2 grow'):
                                                        ui.label(f'#{bid}').classes(f'text-xs font-bold font-mono px-2 py-0.5 rounded border {style_cls}')
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

        def _build_single_tts_card(lang_tag: str, lang_name: str, cfg_data: dict[str, Any], default_text: str, test_filename: str):
            with ui.card().classes('w-full p-5 bg-zinc-900 border border-zinc-800 rounded-xl gap-4 shadow-sm mt-2'):
                with ui.row().classes('w-full items-center justify-between border-b border-zinc-800 pb-2'):
                    with ui.row().classes('items-center gap-2'):
                        ui.label('🇯🇵' if lang_tag == 'ja' else '🇺🇸').classes('text-xl')
                        ui.label(f'{lang_name} TTS 設定 (`config.yaml: tts.{lang_tag}`)').classes('text-lg font-bold text-zinc-100')
                    ui.label('※互換API未実装のTTSサーバでは一覧取得できません（直接入力可）').classes('text-xs text-zinc-400')

                with ui.row().classes('w-full items-center gap-4'):
                    base_in = ui.input(f'{lang_name} Base URL', value=cfg_data.get('base_url', '')).classes('grow')
                    key_in = ui.input(f'{lang_name} API Key', value=cfg_data.get('api_key', 'dummy'), password=True, password_toggle_button=True).classes('w-72')

                cur_model = cfg_data.get('model', '')
                cur_voice = cfg_data.get('voice', '')

                with ui.row().classes('w-full items-center gap-4'):
                    model_sel = ui.select(options=[cur_model] if cur_model else [], value=cur_model, label=f'{lang_name} Model').props('use-input new-value-mode="add-unique" outlined dense').classes('grow')
                    m_btn = ui.button('🔄 モデル取得').props('dense outline')
                    voice_sel = ui.select(options=[cur_voice] if cur_voice else [], value=cur_voice, label=f'{lang_name} Voice / 話者').props('use-input new-value-mode="add-unique" outlined dense').classes('grow')
                    v_btn = ui.button('🗣 ボイス取得').props('dense outline')

                async def fetch_models():
                    b = (base_in.value or '').strip()
                    k = (key_in.value or '').strip() or 'dummy'
                    if not b:
                        ui.notify(f'先に {lang_name} TTS Base URL を入力してください．', type='warning')
                        return
                    m_btn.disable()
                    try:
                        client = await run.io_bound(make_client, {'base_url': b, 'api_key': k})
                        resp = await run.io_bound(client.models.list)
                        m_ids = sorted([m.id for m in resp.data])
                        if not m_ids:
                            ui.notify('モデルが見つかりませんでした．', type='warning')
                            return
                        model_sel.options = m_ids
                        if model_sel.value not in m_ids:
                            model_sel.value = m_ids[0]
                        model_sel.update()
                        ui.notify(f'{len(m_ids)} 個のモデルを取得しました．', type='positive')
                    except Exception as err:
                        ui.notify(f'モデル一覧取得に失敗しました: {err}', type='negative')
                    finally:
                        m_btn.enable()

                async def fetch_voices():
                    b = (base_in.value or '').strip()
                    k = (key_in.value or '').strip() or 'dummy'
                    if not b:
                        ui.notify(f'先に {lang_name} TTS Base URL を入力してください．', type='warning')
                        return
                    v_btn.disable()
                    try:
                        voices = await run.io_bound(self._fetch_voices_from_server, b, k)
                        if not voices:
                            ui.notify('ボイス一覧を取得できませんでした（一覧API未対応の可能性）．', type='warning')
                            return
                        voice_sel.options = voices
                        if voice_sel.value not in voices:
                            voice_sel.value = voices[0]
                        voice_sel.update()
                        ui.notify(f'{len(voices)} 件のボイスを取得しました．', type='positive')
                    except Exception as err:
                        ui.notify(f'ボイス一覧取得に失敗しました: {err}', type='negative')
                    finally:
                        v_btn.enable()

                m_btn.on_click(fetch_models)
                v_btn.on_click(fetch_voices)

                with ui.card().classes('w-full p-3 bg-zinc-950 border border-zinc-800 rounded-lg gap-2'):
                    ui.label(f'🧪 {lang_name} TTS 接続・音声再生テスト').classes('text-xs font-bold text-zinc-300')
                    with ui.row().classes('w-full items-center gap-2'):
                        test_text_in = ui.input('読み上げテキスト', value=default_text).props('dense outlined').classes('grow')
                        test_btn = ui.button('🔊 音声を生成・再生').props('dense outline')

                    audio_container = ui.column().classes('w-full')

                    async def run_test():
                        txt = (test_text_in.value or '').strip()
                        if not txt:
                            ui.notify('読み上げテキストを入力してください．', type='warning')
                            return
                        m = (str(model_sel.value) if model_sel.value is not None else '').strip()
                        v = (str(voice_sel.value) if voice_sel.value is not None else '').strip()
                        if not base_in.value or not m or not v:
                            ui.notify(f'{lang_name} TTS の Base URL, Model, Voice を指定してください．', type='warning')
                            return

                        test_btn.disable()
                        audio_container.clear()
                        with audio_container:
                            ui.label('音声を合成中…').classes('text-xs text-zinc-400')
                        try:
                            t_cfg = {
                                'base_url': base_in.value.strip(),
                                'api_key': (key_in.value or '').strip() or 'dummy',
                                'model': m,
                                'voice': v,
                                'response_format': 'mp3',
                            }
                            out_p = self.test_audio_dir / test_filename

                            def _call():
                                cl = make_client(t_cfg)
                                resp = cl.audio.speech.create(model=m, voice=v, input=txt, response_format='mp3')
                                resp.write_to_file(out_p)

                            await run.io_bound(_call)
                            audio_container.clear()
                            ts = int(time.time() * 1000)
                            with audio_container:
                                ui.audio(f"{file_url(out_p)}?t={ts}").props('autoplay').classes('w-full max-w-lg mt-1')
                                ui.label(f'Model: {m} / Voice: {v} で生成完了').classes('text-xs text-zinc-400')
                            ui.notify(f'{lang_name} 音声を合成しました．', type='positive')
                        except Exception as err:
                            audio_container.clear()
                            with audio_container:
                                ui.label(f'【TTS合成失敗】: {err}').classes('text-xs text-red-400')
                            ui.notify(f'{lang_name} TTS テストに失敗しました: {err}', type='negative')
                        finally:
                            test_btn.enable()

                    test_btn.on_click(run_test)

            return base_in, key_in, model_sel, voice_sel

        ja_widgets = _build_single_tts_card('ja', '日本語', ja, 'こんにちは。日本語の音声合成テストです。正常に聞こえますか？', 'test_ja.mp3')
        en_widgets = _build_single_tts_card('en', '英語', en, 'Hello! This is a test for English text-to-speech synthesis.', 'test_en.mp3')
        return ja_widgets, en_widgets

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

            cur_filter_model = server_cfg.get('model', '')
            with ui.row().classes('w-full items-center gap-4'):
                filter_base = ui.input('フィルタ用 LLM Base URL', value=server_cfg.get('base_url', '')).classes('grow')
                filter_key = ui.input('フィルタ用 API Key', value=server_cfg.get('api_key', 'dummy'), password=True, password_toggle_button=True).classes('w-72')

            with ui.row().classes('w-full items-center gap-4'):
                filter_model_sel = ui.select(options=[cur_filter_model] if cur_filter_model else [], value=cur_filter_model, label='フィルタ用 LLM Model (選択または直接入力)').props('use-input new-value-mode="add-unique" outlined dense').classes('grow')
                filter_fetch_btn = ui.button('🔄 モデル一覧を取得').props('dense outline')
                filter_temp = ui.number('Temperature', value=float(gen_cfg.get('temperature', 0)), min=0, max=2, step=0.1).classes('w-36')

            async def fetch_filter_models() -> None:
                b = (filter_base.value or '').strip()
                k = (filter_key.value or '').strip() or 'dummy'
                if not b:
                    ui.notify('先にフィルタ用 Base URL を入力してください．', type='warning')
                    return
                filter_fetch_btn.disable()
                try:
                    client = await run.io_bound(make_tts_filter_client, {'server': {'base_url': b, 'api_key': k}})
                    resp = await run.io_bound(client.models.list)
                    m_ids = sorted([m.id for m in resp.data])
                    if not m_ids:
                        ui.notify('モデルが見つかりませんでした．', type='warning')
                        return
                    filter_model_sel.options = m_ids
                    if filter_model_sel.value not in m_ids and m_ids:
                        filter_model_sel.value = m_ids[0]
                    filter_model_sel.update()
                    ui.notify(f'{len(m_ids)} 個のモデルを取得しました．', type='positive')
                except Exception as err:
                    ui.notify(f'モデル一覧取得に失敗しました: {err}', type='negative')
                finally:
                    filter_fetch_btn.enable()

            filter_fetch_btn.on_click(fetch_filter_models)

            with ui.card().classes('w-full p-3 bg-zinc-950 border border-zinc-800 rounded-lg gap-2'):
                ui.label('🧪 ヨミ変換フィルタ リアルタイムテスト').classes('text-xs font-bold text-zinc-300')
                with ui.row().classes('w-full items-center gap-2'):
                    filter_test_in = ui.input('テスト入力文（技術文書・プログラムなど）', value='argc と argv を確認し、/usr/bin/python で実行します。cnt++ でカウンタを増やします。').props('dense outlined').classes('grow')
                    filter_test_btn = ui.button('🔄 ヨミ変換テスト実行').props('dense outline')

                filter_test_res = ui.label('').classes('text-xs text-zinc-300 font-mono p-2 bg-zinc-900 border border-zinc-800 rounded min-h-[36px] w-full whitespace-pre-wrap')

                async def run_filter_test() -> None:
                    src_txt = (filter_test_in.value or '').strip()
                    if not src_txt:
                        ui.notify('テスト対象の文章を入力してください．', type='warning')
                        return
                    m = (str(filter_model_sel.value) if filter_model_sel.value is not None else '').strip()
                    b = (filter_base.value or '').strip()
                    k = (filter_key.value or '').strip() or 'dummy'
                    if not b or not m:
                        ui.notify('フィルタ用の Base URL と Model を設定してください．', type='warning')
                        return

                    filter_test_btn.disable()
                    filter_test_res.text = 'ヨミ変換中…'
                    try:
                        t_cfg = {
                            'server': {'base_url': b, 'api_key': k, 'model': m},
                            'generation': {'temperature': float(filter_temp.value or 0)},
                            'dictionary': filter_dict,
                            'prompt': filter_cfg.get('prompt'),
                        }
                        client = await run.io_bound(make_tts_filter_client, t_cfg)
                        s_prompt = build_tts_filter_prompt(t_cfg)
                        res = await run.io_bound(tts_filter_transform, client=client, model=m, system_prompt=s_prompt, text=src_txt, generation=t_cfg['generation'])
                        filter_test_res.text = res
                        ui.notify('ヨミ変換フィルタを適用しました．', type='positive')
                    except Exception as err:
                        filter_test_res.text = f'【エラー】\n{err}'
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

            grid = ui.aggrid({
                'columnDefs': [
                    {'headerName': '単語 / 識別子', 'field': '単語 / 識別子', 'editable': True},
                    {'headerName': '読みの目安', 'field': '読みの目安', 'editable': True},
                ],
                'rowData': [{'単語 / 識別子': k, '読みの目安': v} for k, v in filter_dict.items()],
                ':getRowId': '(params) => params.data[\"単語 / 識別子\"]',
                'defaultColDef': {'flex': 1, 'resizable': True},
                'animateRows': False,
                'stopEditingWhenCellsLoseFocus': True,
            }).classes('w-full h-80')

            async def save_filter_configuration() -> None:
                await grid.load_client_data()
                data = grid.options.get('rowData', [])
                new_dict = {}
                for row in data or []:
                    w = str(row.get('単語 / 識別子', '')).strip()
                    r = str(row.get('読みの目安', '')).strip()
                    if w and r and w != 'nan' and r != 'nan':
                        new_dict[w] = r

                filter_cfg['server'] = {
                    'base_url': (filter_base.value or '').strip(),
                    'api_key': (filter_key.value or '').strip() or 'dummy',
                    'model': (str(filter_model_sel.value) if filter_model_sel.value is not None else '').strip(),
                }
                filter_cfg['generation'] = {'temperature': float(filter_temp.value or 0)}
                filter_cfg['dictionary'] = new_dict
                save_config(TTS_FILTER_PATH, filter_cfg)
                ui.notify(f'tts_filter.yaml を更新しました（辞書全 {len(new_dict)} 件）．', type='positive')

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

    # --- ユーザー管理タブ ---

    def refresh_user_management(self) -> None:
        if not self.user_manage_container:
            return
        self.user_manage_container.clear()

        users_data = load_users()
        users_dict = users_data.get('users', {})

        def calc_future_dt(days: float = 0, hours: float = 0) -> str:
            return (datetime.now() + timedelta(days=days, hours=hours)).strftime('%Y-%m-%d %H:%M')

        with self.user_manage_container:
            with ui.row().classes('w-full items-center justify-between pb-2'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('manage_accounts', size='md').classes('text-blue-400')
                    ui.label('ユーザーアカウント管理 (`users.yaml`)').classes('text-h4 font-bold')
                ui.label('パスワードはソルト付きハッシュで安全に保護されます．一般ユーザーの利用可能日時を制限できます（管理者は常時無制限）').classes('text-xs text-zinc-400')

            # 新規ユーザー追加
            with ui.card().classes('w-full p-5 bg-zinc-900 border border-zinc-800 rounded-xl gap-3'):
                ui.label('➕ 新規ユーザーの追加').classes('text-lg font-bold text-zinc-100')
                with ui.row().classes('w-full items-center gap-3'):
                    add_name = ui.input('ユーザー名 (英数字)').props('outlined dense').classes('w-40')
                    add_pass = ui.input('パスワード', password=True, password_toggle_button=True).props('outlined dense').classes('w-40')
                    add_admin = ui.checkbox('管理者権限').classes('text-zinc-300')
                    add_from = ui.input('利用開始日時', placeholder='例: 2026-10-09 13:00').props('outlined dense').classes('grow')
                    add_until = ui.input('利用終了日時', placeholder='例: 2026-10-16 13:00').props('outlined dense').classes('grow')

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

            ui.label(f'登録済みユーザー一覧（全 {len(users_dict)} アカウント）').classes('text-base font-bold text-zinc-200 mt-4')

            # 既存ユーザー一覧カード
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
                            ui.badge('利用可能' if is_allowed else f'利用不可: {status_msg}', color='positive' if is_allowed else 'negative').props('outline')

                    with ui.row().classes('w-full items-center gap-3 pt-1'):
                        pass_input = ui.input('新パスワード', placeholder='変更時のみ入力', password=True, password_toggle_button=True).props('outlined dense').classes('w-44')
                        admin_check = ui.checkbox('管理者権限', value=is_user_admin).classes('text-zinc-300')
                        if is_current_user:
                            admin_check.disable()

                        from_input = ui.input('利用開始日時', value='' if is_user_admin else str(u_info.get('valid_from', ''))).props('outlined dense placeholder="YYYY-MM-DD HH:MM"').classes('grow')
                        until_input = ui.input('利用終了日時', value='' if is_user_admin else str(u_info.get('valid_until', ''))).props('outlined dense placeholder="YYYY-MM-DD HH:MM"').classes('grow')

                    with ui.row().classes('w-full items-center gap-2 p-1.5 bg-zinc-950/40 rounded border border-zinc-800 text-xs') as row_quick_bar:
                        ui.label('⏱ 終了日時のクイック加算:').classes('text-zinc-400 font-semibold')
                        r_btn_24h = ui.button('+24h', on_click=lambda target_in=until_input: target_in.set_value(calc_future_dt(hours=24))).props('dense outline size=xs')
                        r_btn_3d = ui.button('+3日', on_click=lambda target_in=until_input: target_in.set_value(calc_future_dt(days=3))).props('dense outline size=xs')
                        r_btn_7d = ui.button('+7日 (1週間)', on_click=lambda target_in=until_input: target_in.set_value(calc_future_dt(days=7))).props('dense outline size=xs')
                        r_btn_30d = ui.button('+30日 (1ヶ月)', on_click=lambda target_in=until_input: target_in.set_value(calc_future_dt(days=30))).props('dense outline size=xs')

                        ui.label('│').classes('text-zinc-600')
                        row_days = ui.number('日', value=7, min=0, max=365).props('dense outlined size=xs').classes('w-14')
                        row_hours = ui.number('時間', value=0, min=0, max=23).props('dense outlined size=xs').classes('w-14')

                        def apply_row_custom(target_in=until_input, rd=row_days, rh=row_hours):
                            d = float(rd.value or 0)
                            h = float(rh.value or 0)
                            target_in.set_value(calc_future_dt(days=d, hours=h))

                        r_btn_apply = ui.button('加算セット', on_click=apply_row_custom).props('dense outline color=primary size=xs')
                        r_btn_clear = ui.button('制限解除 (空欄)', on_click=lambda target_in=until_input: target_in.set_value('')).props('dense outline color=grey size=xs')

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

                    update_row_fields(is_user_admin)
                    admin_check.on_value_change(lambda e, upd=update_row_fields: upd(bool(e.value)))

                    with ui.row().classes('w-full justify-end items-center gap-3 pt-1'):
                        async def handle_update(target=u_name, p=pass_input, a=admin_check, vf=from_input, vu=until_input):
                            new_admin_val = bool(a.value)
                            if users_dict[target].get('is_admin') and not new_admin_val:
                                if target == self.username:
                                    ui.notify('自分自身の管理者権限を外すことはできません．', type='negative')
                                    a.value = True
                                    return
                                if sum(1 for u in users_dict.values() if u.get('is_admin', False)) <= 1:
                                    ui.notify('システム内に管理者がいなくなるため、最後の管理者権限を外すことはできません．', type='negative')
                                    a.value = True
                                    return

                            new_pw = (p.value or '').strip()
                            if new_pw:
                                users_dict[target]['password'] = hash_password(new_pw)

                            users_dict[target]['is_admin'] = new_admin_val
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

    # --- 画面レイアウト構築 ---

    def build(self) -> None:
        ui.page_title('Slide Narrator')
        ui.dark_mode().enable()
        ui.colors(primary='#3b82f6')

        current_username = app.storage.user.get('username', '')
        is_admin = app.storage.user.get('is_admin', False)

        def logout() -> None:
            app.storage.user.clear()
            ui.navigate.to('/login')

        # ヘッダー
        with ui.header().classes('items-center w-full px-4 bg-slate-900 border-b border-slate-800'):
            ui.label('🎓 Slide Narrator').classes('text-h5 text-white')
            ui.space()
            if current_username:
                badge_role = ' (管理者)' if is_admin else ''
                ui.label(f'👤 {current_username}{badge_role}').classes('text-caption text-slate-400 mr-2')
            ui.button('ログアウト', on_click=logout).props('dense outline size=sm color=white').classes('mr-3')
            ui.label('TAKAGO_LAB. 2026').classes('text-subtitle2 font-mono tracking-wider text-slate-300 mr-2')

        # 左サイドドロワー
        with ui.left_drawer(value=True).props('width=320').classes('p-4'):
            if self.read_only:
                ui.label('閲覧モード：利用可能時間外のため編集・生成はできません．').classes('text-sm text-amber-400')
            ui.timer(2.0, self.refresh_job_status)
            ui.timer(2.0, lambda: self._safe_view_update(
                self.poll_generated_outputs, '生成ファイルの差分同期'))

            self.uploader = (
                ui.upload(
                    auto_upload=True,
                    max_files=1,
                    on_upload=self.load_pdf,
                )
                .props('accept=.pdf')
                .classes('hidden')
            )
            if self.read_only:
                self.uploader.disable()

            self.uploader.on('added', lambda: self.uploader.run_method('eval', 'if (this.files.length > 1) { this.removeFile(this.files[0]); }'))

            with ui.row().classes('w-full items-center justify-between mt-2'):
                ui.label('プロジェクト').classes('text-h6')
                with ui.row().classes('items-center gap-1'):
                    add_project_button = ui.button(icon='add', on_click=lambda: self.uploader.run_method('pickFiles')).props('flat dense round size=sm')
                    add_project_button.tooltip('PDFを追加して新しいプロジェクトを作成')
                    with ui.button(icon='more_vert').props('flat dense round size=sm'):
                        with ui.menu():
                            ui.menu_item('名前を変更', on_click=self.request_rename_project).props('dense')
                            ui.menu_item('完全削除', on_click=self.request_delete_project).props('dense')
            self.project_select = ui.select(
                options={}, value=None, label='プロジェクトを選択',
                on_change=lambda e: background_tasks.create(self.select_existing_project(e.value)),
            ).props('outlined dense options-dense').classes('w-full')
            if self.read_only:
                add_project_button.disable()
            self.refresh_project_list()
            self.refresh_project_name_input()

            self.mode_select = ui.radio({'lecture': '🎓 講義', 'research': '🔬 研究発表'}, value=self.mode_code).props('inline')
            self.mode_select.on_value_change(lambda e: self._mode_changed(e.value))
            self.lang_select = ui.radio({'ja': '🇯🇵 日本語', 'en': '🇺🇸 英語'}, value=self.lang_code).props('inline')
            self.lang_select.on_value_change(lambda e: self._lang_changed(e.value))

            with ui.card().classes('w-full p-2.5 bg-zinc-900 border border-zinc-800 rounded-lg mt-2'):
                self.default_vlm_switch = ui.switch('VLMを活用してポインタ配置を決定する', value=self.default_use_vlm).props('dense color=primary')
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

            with ui.row().classes('hidden w-full items-center gap-2 p-2 bg-amber-500/10 border border-amber-500/30 rounded-lg mb-2') as self.pipeline_busy_row:
                ui.spinner(size='xs', color='warning')
                self.pipeline_busy_notice = ui.label('').classes('text-xs font-bold text-amber-400')

            self.stage_widgets.clear()
            self.pipeline_buttons.clear()

            for key, label, initial_msg in STAGE_DEFINITIONS:
                with ui.column().classes('w-full gap-1 mb-2'):
                    btn = ui.button(label, on_click=lambda k=key, m=initial_msg: self.handle_pipeline_button(k, m)).props('color=primary').classes('w-full')
                    self.pipeline_buttons.append(btn)

                    with ui.column().classes('w-full px-1 hidden gap-1') as progress_box:
                        with ui.row().classes('w-full items-center justify-between text-[11px]'):
                            with ui.row().classes('items-center gap-1.5'):
                                p_spinner = ui.spinner(size='xs', color='primary').classes('hidden')
                                p_label = ui.label('準備中…').classes('font-bold text-zinc-300')

                        p_bar = ui.linear_progress(value=0.0, show_value=False).props('rounded size=6px color=positive').classes('w-full hidden')
                        p_indet = ui.linear_progress(value=0.0, show_value=False).props('indeterminate rounded size=6px color=positive').classes('w-full')

                    self.stage_widgets[key] = {
                        'button': btn,
                        'box': progress_box,
                        'spinner': p_spinner,
                        'label': p_label,
                        'bar': p_bar,
                        'indeterminate': p_indet,
                    }

            self.reset_generated_button = self.register_main_action_button(
                ui.button('生成データを削除', icon='delete_outline', on_click=self.confirm_reset_generated_data)
                .props('outline color=negative').classes('w-full mt-2')
            )
            self.reset_generated_button.tooltip('元PDF，プロジェクト設定，ページ画像を残し，生成済みの原稿・字幕・音声・動画を削除します．')

            if self.read_only:
                for button in self.pipeline_buttons:
                    button.disable()
                self.mode_select.disable()
                self.lang_select.disable()
                self.default_vlm_switch.disable()
                self.pages_input.disable()

            ui.separator()
            ui.label('🎬 完成ビデオ').classes('text-subtitle1 font-bold text-zinc-200')
            self.final_video_container = ui.column().classes('w-full gap-2')
            ui.separator()
            ui.label('TAKAGO LAB., KIT, Japan.').classes('text-caption')
            ui.link('GitHub: takago/slide-narrator', 'https://github.com/takago/slide-narrator', new_tab=True)

        # メインタブエリア
        with ui.column().classes('w-full p-6'):
            with ui.tabs().classes('w-full') as tabs:
                self.tabs = tabs
                tab_gallery = ui.tab('🖼 スライドデッキ')
                if not self.read_only:
                    tab_simple_edit = ui.tab('📋 ナレーション修正（簡易）')
                    tab_edit = ui.tab('📝 ナレーション修正（詳細）')
                tab_slide_videos = ui.tab('🎞 ビデオデッキ')
                tab_logs = ui.tab('📜 ログ')
                if is_admin and not self.read_only:
                    tab_settings = ui.tab('⚙ 設定')
                if is_admin:
                    tab_user_manage = ui.tab('👥 ユーザー管理')

            with ui.tab_panels(tabs, value=tab_gallery).classes('w-full'):
                with ui.tab_panel(tab_gallery):
                    ui.label('🖼️ スライドデッキ').classes('text-h5')
                    self.gallery = ui.column().classes('w-full')
                if not self.read_only:
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
                        ui.button('ログをクリア', on_click=self.clear_job_log).props('dense outline size=sm color=negative')
                    self.history_log_widget = ui.log(max_lines=2000).classes('w-full h-[650px] font-mono text-xs bg-zinc-900 border border-zinc-700 rounded-lg p-3')
                    self.refresh_history_log()

                if is_admin and not self.read_only:
                    with ui.tab_panel(tab_settings):
                        self.settings_container = ui.column().classes('w-full')
                if is_admin:
                    with ui.tab_panel(tab_user_manage):
                        self.user_manage_container = ui.column().classes('w-full')

            tabs.on_value_change(lambda e: self.refresh_output_views_on_tab_change())

        if is_admin and not self.read_only:
            self.refresh_settings()
        if is_admin:
            self.refresh_user_management()

        background_tasks.create(self.restore_last_project())

    # --- ユーザー入力イベント ---

    def _mode_changed(self, value: str) -> None:
        if not self.write_access_allowed():
            return
        self.mode_code = value
        self.save_project_settings()

    def _lang_changed(self, value: str) -> None:
        if not self.write_access_allowed():
            return
        self.lang_code = value
        self.save_project_settings()
        background_tasks.create(self.refresh_simple_editor())
        background_tasks.create(self.refresh_editor())

    def _default_vlm_changed(self, value: bool) -> None:
        if not self.write_access_allowed():
            return
        self.default_use_vlm = bool(value)
        self.save_project_settings()

    def _range_changed(self) -> None:
        if not self.write_access_allowed():
            return
        new_val = self.pages_input.value or ''
        if new_val == self.pages_spec:
            return
        try:
            self.apply_pages_spec(new_val, save_and_refresh=True)
        except Exception as exc:
            ui.notify(f'スライド範囲を解釈できません: {exc}', type='negative')


# ----------------------------------------------------------------------
# Application Routing & Authentication Pages
# ----------------------------------------------------------------------

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

    app_instance = SlideNarratorApp(username=username)
    app_instance.build()


@ui.page('/login')
def login_page():
    if app.storage.user.get('authenticated', False):
        return RedirectResponse('/')

    ui.page_title('Slide Narrator - ログイン')
    ui.dark_mode().enable()
    ui.colors(primary='#3b82f6')

    # ページを開いた瞬間にロック中か判定（タイマーを使わないためエラーが出ません）
    is_busy = GLOBAL_EXECUTION_LOCK.is_locked()
    busy_user = GLOBAL_EXECUTION_LOCK.current_user or '誰か'
    task_label = GLOBAL_EXECUTION_LOCK.task_name or '処理'

    def try_login() -> None:
        username = (username_input.value or '').strip()
        password = password_input.value or ''

        users_data = load_users()
        users_dict = users_data.get('users', {})
        user_info = users_dict.get(username)

        if not user_info or not verify_password(password, str(user_info.get('password', ''))):
            ui.notify('ユーザー名またはパスワードが正しくありません．', type='negative')
            return

        stored_pw = str(user_info.get('password', ''))
        if not stored_pw.startswith('pbkdf2:sha256:'):
            user_info['password'] = hash_password(password)
            save_users(users_data)

        app.storage.user['authenticated'] = True
        app.storage.user['username'] = username
        app.storage.user['is_admin'] = bool(user_info.get('is_admin', False))
        ui.navigate.to('/')

    with ui.column().classes('w-full min-h-screen items-center justify-center p-4 bg-gradient-to-br from-slate-950 via-zinc-900 to-slate-950'):
        # カード幅を max-w-sm から max-w-md に変更
        with ui.card().classes('w-full max-w-md p-6 bg-zinc-900/90 backdrop-blur border border-zinc-800 rounded-2xl shadow-2xl gap-4'):

            # ヘッダー部
            with ui.column().classes('w-full items-center gap-1'):
                with ui.row().classes('items-center justify-center gap-4 mb-1'):
                    ui.icon('picture_as_pdf', size='48px').classes('text-red-400')
                    ui.icon('arrow_forward', size='28px').classes('text-zinc-400')
                    ui.icon('movie', size='48px').classes('text-blue-400')
                ui.label('Slide Narrator').classes('text-xl font-bold tracking-tight text-white')
                # whitespace-nowrap を追加（1行維持）
                ui.label('Turn PDF slides into narrated videos automatically with AI.').classes('text-xs text-zinc-400 whitespace-nowrap')
            # サーバーが実行中だった場合のみ静的にスピナーを表示
            if is_busy:
                with ui.row().classes('w-full items-center gap-2.5 p-2.5 bg-amber-500/10 border border-amber-500/30 rounded-lg'):
                    ui.spinner(size='sm', color='warning').classes('shrink-0')
                    with ui.column().classes('grow gap-0 leading-tight'):
                        ui.label(f'ユーザー{busy_user}が生成処理中です').classes('text-[11px] font-bold text-amber-400')
                        ui.label('しばらく待ってからお試しください')

            # 入力フィールド
            with ui.column().classes('w-full gap-3 pt-1'):
                username_input = ui.input('ユーザー名').props('outlined dense autofocus').classes('w-full')
                password_input = ui.input('パスワード', password=True, password_toggle_button=True).props('outlined dense').classes('w-full')
                username_input.on('keydown.enter', try_login)
                password_input.on('keydown.enter', try_login)

            ui.button('ログイン', on_click=try_login).props('color=primary unelevated').classes('w-full font-semibold mt-1')

            # フッター部
            with ui.column().classes('w-full items-center gap-0.5 pt-3 border-t border-zinc-800/80'):
                ui.label('TAKAGO_LAB. 2026').classes('text-[12px] font-mono tracking-widest text-zinc-400')
                ui.label('Kanazawa Institute of Technology').classes('text-[12px] text-zinc-600')

ui.run(
    title='Slide Narrator',
    reload=False,
    show=False,
    port=17171,
    host='0.0.0.0',
    storage_secret='slide-narrator-session-secret-key-change-in-prod',
)
