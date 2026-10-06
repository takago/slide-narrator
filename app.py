#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import pymupdf as fitz
import yaml
from fastapi.responses import FileResponse, PlainTextResponse
from nicegui import app, run, ui
from PIL import Image, ImageDraw

from slide_lecture import (
    build_course_overview,
    extract_page_text,
    generate_single_alignment,
    load_project_json,
    make_client,
    make_explanation,
    parse_page_ranges,
    save_project_json,
)
from tts_filter import load_config as load_tts_filter_config


# ----------------------------------------------------------------------
# Configuration / filesystem helpers
# ----------------------------------------------------------------------

CONFIG_PATH = Path('config.yaml')
TTS_FILTER_PATH = Path('tts_filter.yaml')
UPLOAD_DIR = Path('webui_uploads')
UPLOAD_DIR.mkdir(exist_ok=True)
TEST_AUDIO_DIR = UPLOAD_DIR / 'test_audio'
TEST_AUDIO_DIR.mkdir(exist_ok=True)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def save_config(path: Path, cfg: dict) -> None:
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )


def run_command(args: list[str]) -> tuple[int, str]:
    p = subprocess.run(args, capture_output=True, text=True)
    return p.returncode, p.stdout + '\n' + p.stderr


def project_root(pdf: Path) -> Path:
    return pdf.with_name(pdf.stem + '_lecture')


def explanation_path(pdf: Path, page: int) -> Path:
    return project_root(pdf) / 'explanations' / f'{page:03d}.txt'


def alignment_path(pdf: Path, page: int) -> Path:
    return project_root(pdf) / 'explanations' / f'{page:03d}_align.json'


def page_image_path(pdf: Path, page: int) -> Path:
    return project_root(pdf) / 'pages' / f'{page:03d}.png'


def audio_path(pdf: Path, page: int) -> Path:
    return project_root(pdf) / 'audio' / f'{page:03d}.mp3'


def video_path(pdf: Path, page: int) -> Path:
    return project_root(pdf) / 'video' / f'{page:03d}.mp4'


def final_video_path(pdf: Path) -> Path:
    return project_root(pdf) / f'{pdf.stem}.mp4'


def final_ja_srt_path(pdf: Path) -> Path:
    return project_root(pdf) / f'{pdf.stem}_ja.srt'


def final_en_srt_path(pdf: Path) -> Path:
    return project_root(pdf) / f'{pdf.stem}_en.srt'


def cleanup_downstream_media(pdf: Path, page: int | None = None, include_alignment: bool = False) -> None:
    for f in [final_video_path(pdf), final_ja_srt_path(pdf), final_en_srt_path(pdf)]:
        if f.exists():
            f.unlink()

    if page is not None:
        if include_alignment:
            f = alignment_path(pdf, page)
            if f.exists():
                f.unlink()
        for f in [audio_path(pdf, page), video_path(pdf, page)]:
            if f.exists():
                f.unlink()
    else:
        root = project_root(pdf)
        if include_alignment:
            exp_d = root / 'explanations'
            if exp_d.exists():
                for f in exp_d.glob('*_align.json'):
                    f.unlink()
        for d in [root / 'audio', root / 'video']:
            if d.exists():
                for f in d.glob('*'):
                    if f.is_file():
                        f.unlink()


def ensure_page_images(pdf: Path, dpi: int = 120) -> list[Path]:
    pages_dir = project_root(pdf) / 'pages'
    pages_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf)
    result: list[Path] = []
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    for i, page in enumerate(doc, 1):
        out = pages_dir / f'{i:03d}.png'
        result.append(out)
        if not out.exists():
            page.get_pixmap(matrix=matrix, alpha=False).save(out)
    return result


def draw_block_preview(img_path: Path, blocks: list[dict]) -> Image.Image:
    im = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(im)
    for b in blocks:
        box = b.get('bbox')
        bid = b.get('block_id')
        if box:
            draw.rectangle(box, outline='blue', width=2)
            draw.rectangle([box[0], box[1] - 20, box[0] + 35, box[1]], fill='blue')
            draw.text((box[0] + 4, box[1] - 18), f'#{bid}', fill='white')
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
    """Return a safe URL handled by the FastAPI file endpoint below."""
    return '/files/' + path.resolve().relative_to(Path.cwd().resolve()).as_posix()


@app.get('/files/{path:path}')
async def serve_file(path: str):
    root = Path.cwd().resolve()
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        return PlainTextResponse('Forbidden', status_code=403)
    if not target.is_file():
        return PlainTextResponse('Not found', status_code=404)
    return FileResponse(target)


@app.get('/download/{path:path}')
async def download_file(path: str):
    root = Path.cwd().resolve()
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        return PlainTextResponse('Forbidden', status_code=403)
    if not target.is_file():
        return PlainTextResponse('Not found', status_code=404)
    return FileResponse(target, filename=target.name)


# ----------------------------------------------------------------------
# Application state
# ----------------------------------------------------------------------

class SlideNarratorApp:
    def __init__(self) -> None:
        self.cfg = load_config(CONFIG_PATH)
        self.pdf: Path | None = None
        self.images: list[Path] = []
        self.proj_cfg: dict[str, Any] = {}
        self.mode_code = 'lecture'
        self.lang_code = 'ja'
        self.pages_spec = ''
        self.force_run = False
        self.active_pages: list[int] = []
        self.edit_page: int | None = None
        self.log = ''
        self.processing = False

        # プロセスおよび非同期キャンセルの追跡
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
        self.mode_select = None
        self.lang_select = None
        self.pages_input = None
        self.force_checkbox = None
        self.pipeline_buttons = []

    # ---- project ------------------------------------------------------

    async def load_pdf(self, e) -> None:
        filename = Path(e.file.name).name
        pdf = UPLOAD_DIR / filename
        if not pdf.exists() or pdf.stat().st_size != e.file.size():
            await e.file.save(pdf)
        self.pdf = pdf
        self.proj_cfg = load_project_json(project_root(pdf))
        self.mode_code = self.proj_cfg.get('mode') or self.cfg.get('mode', 'lecture')
        self.lang_code = self.proj_cfg.get('language') or self.cfg.get('language', 'ja')
        self.pages_spec = self.proj_cfg.get('pages', '')
        self.images = await run.io_bound(
            ensure_page_images, pdf, int(self.cfg.get('pdf', {}).get('dpi', 120))
        )
        self.refresh_project_widgets()
        await self.refresh_all()
        ui.notify(f'プレゼンテーションを読み込みました: {pdf.name}', type='positive')

    def refresh_project_widgets(self) -> None:
        if not self.pdf:
            return
        if self.mode_select:
            self.mode_select.value = self.mode_code
        if self.lang_select:
            self.lang_select.value = self.lang_code
        if self.pages_input:
            self.pages_input.value = self.pages_spec

    def save_project_settings(self) -> None:
        if not self.pdf:
            return
        root = project_root(self.pdf)
        self.proj_cfg.update({
            'mode': self.mode_code,
            'language': self.lang_code,
            'pages': self.pages_spec,
            'skip_pages': '',
        })
        save_project_json(root, self.proj_cfg)

    def recalculate_pages(self) -> None:
        if not self.pdf:
            self.active_pages = []
            return
        total = len(self.images)
        active = set(range(1, total + 1))
        if self.pages_spec.strip() and self.pages_spec.strip() != 'none':
            active &= parse_page_ranges(self.pages_spec.strip())
        elif self.pages_spec.strip() == 'none':
            active = set()
        self.active_pages = sorted(active)
        if self.edit_page not in self.active_pages:
            self.edit_page = self.active_pages[0] if self.active_pages else None
        if self.active_count_label:
            self.active_count_label.text = f'対象スライド: {len(self.active_pages)} / {total} スライド'

    def toggle_slide_active(self, page_num: int, active: bool) -> None:
        cur = set(self.active_pages)
        if active:
            cur.add(page_num)
        else:
            cur.discard(page_num)

        total = len(self.images)
        if len(cur) == total:
            self.pages_spec = ''
        elif not cur:
            self.pages_spec = 'none'
        else:
            self.pages_spec = format_page_ranges(sorted(cur))

        if self.pages_input:
            self.pages_input.value = self.pages_spec

        self.recalculate_pages()
        self.save_project_settings()
        asyncio.create_task(self.refresh_gallery())
        asyncio.create_task(self.refresh_simple_editor())
        asyncio.create_task(self.refresh_editor())
        asyncio.create_task(self.refresh_slide_videos())

    def select_all_slides(self) -> None:
        self.pages_spec = ''
        if self.pages_input:
            self.pages_input.value = ''
        self.recalculate_pages()
        self.save_project_settings()
        asyncio.create_task(self.refresh_gallery())
        asyncio.create_task(self.refresh_simple_editor())
        asyncio.create_task(self.refresh_editor())
        asyncio.create_task(self.refresh_slide_videos())

    def clear_all_slides(self) -> None:
        self.pages_spec = 'none'
        if self.pages_input:
            self.pages_input.value = 'none'
        self.active_pages = []
        self.edit_page = None
        if self.active_count_label:
            self.active_count_label.text = f'対象スライド: 0 / {len(self.images)} スライド'
        self.save_project_settings()
        asyncio.create_task(self.refresh_gallery())
        asyncio.create_task(self.refresh_simple_editor())
        asyncio.create_task(self.refresh_editor())
        asyncio.create_task(self.refresh_slide_videos())

    # ---- pipeline / process control -----------------------------------

    def open_processing_dialog(self, initial_title: str) -> tuple[ui.dialog, Callable[[str, float | None, str | None, int | None], None], Callable[[str], None]]:
        dialog = ui.dialog()
        dialog.props('persistent')
        self.cancellation_requested = False

        with dialog, ui.card().classes('items-center p-6 gap-3 min-w-[620px] max-w-[760px]'):
            ui.spinner(size='lg')
            title_label = ui.label(initial_title).classes('text-base font-bold text-center text-zinc-100')
            status_label = ui.label('準備中…').classes('text-sm text-zinc-400 text-center')

            with ui.row().classes('w-full items-center gap-2'):
                progress_bar = ui.linear_progress(value=0.0, show_value=False).props('rounded size=14px').classes('grow')
                percent_label = ui.label('0%').classes('text-xs font-mono font-bold w-12 text-right text-zinc-300')

            # --- スライド3枚プレビュー（前・現在・次） ---
            slide_preview_row = ui.row().classes('w-full items-center justify-center gap-3 py-2 bg-zinc-900/60 rounded-lg border border-zinc-800')
            with slide_preview_row:
                # 前スライド (縮小・半透明)
                with ui.column().classes('items-center w-28 opacity-45'):
                    prev_label = ui.label('前スライド').classes('text-[11px] text-zinc-400 mb-0.5')
                    prev_img_box = ui.column().classes('w-28 h-18 bg-zinc-800 rounded border border-zinc-700 items-center justify-center overflow-hidden')
                    with prev_img_box:
                        prev_image = ui.image('').classes('w-full rounded').style('display: none')
                        prev_placeholder = ui.label('-').classes('text-xs text-zinc-500')

                # 現在処理中のスライド (中央・拡大・中立なグレー枠)
                with ui.column().classes('items-center w-48 shadow-lg scale-105 transition-all'):
                    curr_label = ui.label('現在処理中').classes('text-xs font-bold text-zinc-300 mb-0.5')
                    curr_img_box = ui.column().classes('w-48 h-30 bg-zinc-800 rounded-lg border border-zinc-700 shadow-md items-center justify-center overflow-hidden')
                    with curr_img_box:
                        curr_image = ui.image('').classes('w-full rounded').style('display: none')
                        curr_placeholder = ui.label('スライド待機中').classes('text-xs text-zinc-400')

                # 次スライド (縮小・半透明)
                with ui.column().classes('items-center w-28 opacity-45'):
                    next_label = ui.label('次スライド').classes('text-[11px] text-zinc-400 mb-0.5')
                    next_img_box = ui.column().classes('w-28 h-18 bg-zinc-800 rounded border border-zinc-700 items-center justify-center overflow-hidden')
                    with next_img_box:
                        next_image = ui.image('').classes('w-full rounded').style('display: none')
                        next_placeholder = ui.label('-').classes('text-xs text-zinc-500')

            # 処理中のダイアログ内リアルタイム・スクロール可能ログ
            with ui.expansion('詳細ログを表示', icon='terminal').classes('w-full border border-zinc-700 rounded-lg text-xs mt-1'):
                dialog_log = ui.log(max_lines=300).classes('w-full h-40 font-mono text-xs bg-zinc-900 text-zinc-300 p-2')

            with ui.row().classes('w-full justify-center pt-2'):
                ui.button('🛑 処理を中断', on_click=self.request_cancel, color='negative').props('outline')

        dialog.open()

        def update_slide_preview(current_page: int | None) -> None:
            if not self.pdf or current_page is None:
                return
            pages_list = self.active_pages if self.active_pages else list(range(1, len(self.images) + 1))
            if current_page not in pages_list:
                return

            idx = pages_list.index(current_page)
            prev_p = pages_list[idx - 1] if idx > 0 else None
            next_p = pages_list[idx + 1] if idx + 1 < len(pages_list) else None

            # 前スライド
            if prev_p is not None:
                p_path = page_image_path(self.pdf, prev_p)
                if p_path.exists():
                    prev_image.set_source(file_url(p_path))
                    prev_image.style('display: block')
                    prev_placeholder.style('display: none')
                    prev_label.text = f'スライド {prev_p}'
            else:
                prev_image.style('display: none')
                prev_placeholder.style('display: block')
                prev_label.text = ''

            # 現在スライド
            c_path = page_image_path(self.pdf, current_page)
            if c_path.exists():
                curr_image.set_source(file_url(c_path))
                curr_image.style('display: block')
                curr_placeholder.style('display: none')
                curr_label.text = f'スライド {current_page}'

            # 次スライド
            if next_p is not None:
                n_path = page_image_path(self.pdf, next_p)
                if n_path.exists():
                    next_image.set_source(file_url(n_path))
                    next_image.style('display: block')
                    next_placeholder.style('display: none')
                    next_label.text = f'スライド {next_p}'
            else:
                next_image.style('display: none')
                next_placeholder.style('display: block')
                next_label.text = ''

        def update_status(text: str, frac: float | None = None, title: str | None = None, current_page: int | None = None) -> None:
            if title is not None:
                title_label.text = title
            status_label.text = text
            if frac is None:
                progress_bar.value = 0.0
                percent_label.text = '--'
            else:
                clamped = max(0.0, min(1.0, float(frac)))
                progress_bar.value = clamped
                percent = int(round(clamped * 100))
                percent_label.text = f'{percent}%'

            if current_page is not None:
                update_slide_preview(current_page)

        return dialog, update_status, dialog_log.push

    async def request_cancel(self) -> None:
        """実行中のプロセスおよびタスクを中断します．"""
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
        args = ['--mode', self.mode_code, '--lang', self.lang_code]
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

        self.save_project_settings()
        dialog, update_status, push_log = self.open_processing_dialog(initial_title)
        self.set_processing(True)

        total_slides = len(self.active_pages) or 1
        cmd = [sys.executable, '-u', 'slide_lecture.py', str(self.pdf), '--from', stage, *self.base_args()]

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

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

                # ① ナレーション生成フェーズ
                if '[LLM] ナレーション生成:' in line:
                    m_p = re.search(r'page (\d+)/', line)
                    m_frac = re.search(r'\((\d+)/(\d+)\)', line)
                    if m_p and m_frac:
                        p_num = int(m_p.group(1))
                        cur, tot = int(m_frac.group(1)), int(m_frac.group(2))
                        update_status(f'スライド {cur} / {tot}', cur / tot, title='① ナレーション原稿を作成中…', current_page=p_num)

                elif '[LLM] reuse narration' in line:
                    m = re.search(r'(\d{3})\.txt', line)
                    if m:
                        p_num = int(m.group(1))
                        idx = self.active_pages.index(p_num) + 1 if p_num in self.active_pages else 1
                        update_status(f'既存原稿を再利用: スライド {p_num} ({idx}/{total_slides})', idx / total_slides, title='① ナレーション原稿を確認中…', current_page=p_num)

                # ② 字幕・視線誘導アライメントフェーズ
                elif '[ALIGN] 字幕＆ポインタ解析:' in line:
                    m = re.search(r'page (\d+)/', line)
                    if m:
                        p_num = int(m.group(1))
                        idx = self.active_pages.index(p_num) + 1 if p_num in self.active_pages else 1
                        update_status(f'スライド {p_num} ({idx}/{total_slides})', idx / total_slides, title='② 字幕・ポインタを解析中…', current_page=p_num)

                elif '[ALIGN] reuse' in line:
                    m = re.search(r'(\d{3})_align\.json', line)
                    if m:
                        p_num = int(m.group(1))
                        idx = self.active_pages.index(p_num) + 1 if p_num in self.active_pages else 1
                        update_status(f'既存データを再利用: スライド {p_num} ({idx}/{total_slides})', idx / total_slides, title='② 字幕・ポインタを確認中…', current_page=p_num)

                # ③ TTS 音声合成フェーズ
                elif '[TTS] 音声合成中:' in line or '[TTS' in line:
                    m_page = re.search(r'page (\d+)', line) or re.search(r'(\d{3})\.mp3', line)
                    m_prog = re.search(r'\((\d+)/(\d+)\)', line)
                    p_num = int(m_page.group(1)) if m_page else None
                    if m_prog:
                        cur, tot = int(m_prog.group(1)), int(m_prog.group(2))
                        update_status(f'スライド ({cur}/{tot})', cur / tot, title=f'③ 音声を合成中 ({self.lang_code})…', current_page=p_num)
                    elif p_num:
                        idx = self.active_pages.index(p_num) + 1 if p_num in self.active_pages else 1
                        action = '既存音声を再利用' if 'reuse' in line else '音声合成完了'
                        update_status(f'{action}: スライド {p_num} ({idx}/{total_slides})', idx / total_slides, title=f'③ 音声を合成中 ({self.lang_code})…', current_page=p_num)

                # ④ 動画生成フェーズ
                elif '[VIDEO] スライド動画生成中:' in line or '[VIDEO]' in line:
                    weight = 0.85 if stage == 'video' else 1.0
                    m_page = re.search(r'page (\d+)', line) or re.search(r'(\d{3})\.mp4', line)
                    m_prog = re.search(r'\((\d+)/(\d+)\)', line)
                    p_num = int(m_page.group(1)) if m_page else None

                    if m_prog:
                        cur, tot = int(m_prog.group(1)), int(m_prog.group(2))
                        frac = (cur / tot) * weight
                        update_status(f'スライド ({cur}/{tot})', frac, title='④ スライド動画をレンダリング中…', current_page=p_num)
                    elif p_num:
                        idx = self.active_pages.index(p_num) + 1 if p_num in self.active_pages else 1
                        frac = (idx / total_slides) * weight
                        action = '既存動画を再利用' if 'reuse' in line else 'レンダリング完了'
                        update_status(f'{action}: スライド {p_num} ({idx}/{total_slides})', frac, title='④ スライド動画をレンダリング中…', current_page=p_num)

                elif '[CONCAT]' in line or 'concat' in line.lower():
                    update_status('全スライド動画の結合＆字幕トラック埋め込み中…', 0.92, title='④ 完成動画を結合・生成中…')

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
            dialog.close()

    # ---- edit tab -----------------------------------------------------

    def alignment_data(self, page: int) -> tuple[dict, list[dict], list[dict]]:
        if not self.pdf:
            return {}, [], []
        p = alignment_path(self.pdf, page)
        if not p.exists():
            return {}, [], []
        data = json.loads(p.read_text(encoding='utf-8'))
        return data, data.get('blocks', []), data.get('alignments', [])

    async def select_edit_page(self, page: int) -> None:
        self.edit_page = int(page)
        await self.refresh_editor()

    async def prev_edit(self) -> None:
        if self.edit_page in self.active_pages:
            i = self.active_pages.index(self.edit_page)
            if i > 0:
                await self.select_edit_page(self.active_pages[i - 1])

    async def next_edit(self) -> None:
        if self.edit_page in self.active_pages:
            i = self.active_pages.index(self.edit_page)
            if i + 1 < len(self.active_pages):
                await self.select_edit_page(self.active_pages[i + 1])

    async def regenerate_narration(self, page: int, text_area) -> None:
        if not self.pdf:
            return
        if self.processing:
            ui.notify('別の処理が実行中です．処理が終わるまでお待ちください．', type='warning')
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
            page_texts = await run.io_bound(extract_page_text, self.pdf)
            exp_dir = project_root(self.pdf) / 'explanations'
            overview_path = exp_dir / '_course_overview.txt'
            if overview_path.exists():
                overview = overview_path.read_text(encoding='utf-8')
            else:
                update_status('プレゼンテーション全体の概要を構築中…', 0.4, current_page=page)
                await asyncio.sleep(0.01)
                overview = await run.io_bound(
                    build_course_overview, client, cfg['llm'], page_texts,
                    mode=self.mode_code, lang=self.lang_code,
                )
                exp_dir.mkdir(parents=True, exist_ok=True)
                overview_path.write_text(overview + '\n', encoding='utf-8')

            cur_idx = self.active_pages.index(page)
            prev_idx = self.active_pages[cur_idx - 1] - 1 if cur_idx > 0 else None
            next_idx = self.active_pages[cur_idx + 1] - 1 if cur_idx + 1 < len(self.active_pages) else None
            prev_text = page_texts[prev_idx] if prev_idx is not None else None
            next_text = page_texts[next_idx] if next_idx is not None else None
            prev_exp_path = explanation_path(self.pdf, self.active_pages[cur_idx - 1]) if cur_idx > 0 else None
            prev_explanation = prev_exp_path.read_text(encoding='utf-8').strip() if prev_exp_path and prev_exp_path.exists() else ''

            update_status(f'LLMでスライド {page} の解説文を推論中…', 0.7, current_page=page)
            push_log(f'[REGEN] LLM推論中...')
            await asyncio.sleep(0.01)
            new_narration = await run.io_bound(
                make_explanation,
                client=client,
                cfg=cfg['llm'],
                image=page_image_path(self.pdf, page),
                current_page=page,
                current_text=page_texts[page - 1],
                prev_page=prev_idx + 1 if prev_idx is not None else None,
                prev_text=prev_text,
                next_page=next_idx + 1 if next_idx is not None else None,
                next_text=next_text,
                previous_explanation=prev_explanation,
                course_overview=overview,
                total_active=len(self.active_pages),
                active_idx=cur_idx,
                mode=self.mode_code,
                lang=self.lang_code,
            )

            if not self.cancellation_requested:
                ep = explanation_path(self.pdf, page)
                ep.parent.mkdir(parents=True, exist_ok=True)
                temp_ep = ep.with_name(f".{ep.name}.tmp")
                temp_ep.write_text(new_narration.strip() + '\n', encoding='utf-8')
                temp_ep.replace(ep)

                cleanup_downstream_media(self.pdf, page, include_alignment=True)
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
            dialog.close()

    async def save_and_realign(self, page: int, text: str) -> None:
        if not self.pdf:
            return
        ep = explanation_path(self.pdf, page)
        ep.parent.mkdir(parents=True, exist_ok=True)
        temp_ep = ep.with_name(f".{ep.name}.tmp")
        temp_ep.write_text(text.rstrip() + '\n', encoding='utf-8')
        temp_ep.replace(ep)

        if self.processing:
            ui.notify('別の処理が実行中です．処理が終わるまでお待ちください．', type='warning')
            return
        dialog, update_status, push_log = self.open_processing_dialog('文分割・字幕翻訳・ポインタを再解析中…')
        self.set_processing(True)
        self.current_task = asyncio.current_task()
        try:
            update_status(f'スライド {page} の要素抽出と対訳・視線誘導を再計算中…', 0.5, current_page=page)
            push_log(f'[ALIGN] スライド {page} の要素抽出と視線誘導を再計算中...')
            await asyncio.sleep(0.01)
            client = await run.io_bound(make_client, self.cfg['llm'])
            dpi = int(self.cfg.get('pdf', {}).get('dpi', 150))
            data = await run.io_bound(
                generate_single_alignment,
                client, self.cfg['llm'], self.pdf, page,
                page_image_path(self.pdf, page), text.strip(), dpi, self.lang_code,
            )

            if not self.cancellation_requested:
                alp = alignment_path(self.pdf, page)
                alp.parent.mkdir(parents=True, exist_ok=True)
                temp_alp = alp.with_name(f".{alp.name}.tmp")
                temp_alp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
                temp_alp.replace(alp)

                cleanup_downstream_media(self.pdf, page, include_alignment=False)
                update_status('完了しました！', 1.0, current_page=page)
                push_log(f'[ALIGN] 完了: 字幕・ポインタアライメントを保存しました')
                await asyncio.sleep(0.3)
                ui.notify('字幕とポインタを再生成しました（古い音声・動画をリセットしました）．', type='positive')
                await self.refresh_simple_editor()
                await self.refresh_editor()
        except asyncio.CancelledError:
            ui.notify('解析を中断しました．', type='warning')
        except Exception as exc:
            ui.notify(f'再解析に失敗しました: {exc}', type='negative')
        finally:
            self.current_task = None
            self.set_processing(False)
            dialog.close()

    async def save_alignment(self, page: int, rows: list[dict], align_data: dict) -> None:
        if not self.pdf:
            return
        alp = alignment_path(self.pdf, page)
        align_data['alignments'] = rows
        temp_alp = alp.with_name(f".{alp.name}.tmp")
        temp_alp.write_text(json.dumps(align_data, ensure_ascii=False, indent=2), encoding='utf-8')
        temp_alp.replace(alp)

        for f in [video_path(self.pdf, page), final_video_path(self.pdf), final_ja_srt_path(self.pdf), final_en_srt_path(self.pdf)]:
            if f.exists():
                f.unlink()
        ui.notify('字幕とポインタの修正を保存しました．', type='positive')
        await self.refresh_editor()

    # ---- rendering ----------------------------------------------------

    async def refresh_gallery(self) -> None:
        if not self.gallery:
            return
        self.gallery.clear()
        if not self.images:
            self.gallery.add(ui.label('プレゼンテーションPDFを選択してください．'))
            return
        with self.gallery:
            with ui.row().classes('w-full items-center justify-between pb-2'):
                ui.label(f'全 {len(self.images)} スライド（カードをクリックまたはチェックボックスで対象を切り替えられます）').classes('text-sm text-gray-500 dark:text-gray-400')
                with ui.row().classes('gap-2'):
                    ui.button('全スライドを選択', on_click=self.select_all_slides).props('dense outline size=sm')
                    ui.button('すべて解除', on_click=self.clear_all_slides).props('dense outline size=sm color=negative')

            with ui.grid(columns=5).classes('w-full gap-4'):
                for n, img in enumerate(self.images, 1):
                    is_active = n in self.active_pages
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
        if not self.active_pages or not self.pdf:
            with self.simple_edit_container:
                ui.label('⚠ 処理対象となるスライドがありません．「スライドデッキ」タブで対象スライドを選択してください．').classes('text-orange-500 dark:text-orange-400')
            return

        with self.simple_edit_container:
            ui.label(f'対象スライド一覧（全 {len(self.active_pages)} スライド）').classes('text-sm text-gray-400 pb-2')

            with ui.column().classes('w-full gap-6'):
                for p in self.active_pages:
                    img_path = page_image_path(self.pdf, p)
                    txt_path = explanation_path(self.pdf, p)
                    curr_txt = txt_path.read_text(encoding='utf-8') if txt_path.exists() else ''

                    with ui.card().classes('w-full p-4 bg-zinc-900 border border-zinc-800 rounded-lg shadow-sm'):
                        with ui.row().classes('w-full items-start gap-4 flex-nowrap'):
                            # 左側: スライド画像プレビューおよび下部にスライド番号ラベル
                            with ui.column().classes('w-72 shrink-0 items-center gap-1.5'):
                                if img_path.exists():
                                    ui.image(file_url(img_path)).classes('w-full rounded border border-zinc-700 shadow-sm')
                                else:
                                    ui.label('（スライド画像なし）').classes('text-xs text-zinc-500')
                                ui.label(f'スライド {p}').classes('text-sm font-bold text-zinc-300 self-center')

                            # 右側: ナレーション原稿テキストフォーム＋下部ボタン列
                            with ui.column().classes('grow gap-2'):
                                ta = ui.textarea(
                                    label=f'ナレーション原稿（スライド {p}）',
                                    value=curr_txt,
                                ).props('outlined').classes('w-full h-[180px]').style(
                                    'height: 180px; min-height: 180px;'
                                )
                                ta.props('input-style="height: 140px; resize: vertical;"')

                                # テキスト下部のボタン列
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
        if not self.active_pages or not self.pdf:
            with self.edit_container:
                ui.label('⚠ 処理対象となるスライドがありません．「スライドデッキ」タブで対象スライドを選択してください．').classes('text-orange-500 dark:text-orange-400')
            return
        page = self.edit_page or self.active_pages[0]
        self.edit_page = page
        idx = self.active_pages.index(page)
        align_data, blocks, alignments = self.alignment_data(page)
        img = page_image_path(self.pdf, page)
        current_text_path = explanation_path(self.pdf, page)
        current_text = current_text_path.read_text(encoding='utf-8') if current_text_path.exists() else ''

        with self.edit_container:
            with ui.row().classes('w-full items-center gap-3 pb-2'):
                sel = ui.select(
                    {p: f'スライド {p} ({self.active_pages.index(p)+1} / {len(self.active_pages)} スライド)' for p in self.active_pages},
                    value=page,
                    label='編集スライド選択',
                ).classes('min-w-[280px] max-w-[340px]').props('outlined dense')
                sel.on_value_change(lambda e: asyncio.create_task(self.select_edit_page(e.value)))

                ui.button('◀ 前のスライド', on_click=self.prev_edit).props(f'disable={idx == 0} outlined')
                ui.button('次のスライド ▶', on_click=self.next_edit).props(f'disable={idx == len(self.active_pages)-1} outlined')

            with ui.row().classes('w-full items-start gap-6'):
                with ui.column().classes('w-5/12'):
                    if img.exists():
                        preview = draw_block_preview(img, blocks) if blocks else None
                        if preview is not None:
                            preview_path = project_root(self.pdf) / 'pages' / f'{page:03d}_preview.png'
                            preview.save(preview_path)
                            ui.image(file_url(preview_path)).classes('w-full')
                        else:
                            ui.image(file_url(img)).classes('w-full')
                        ui.label(f'スライド {page}').classes('text-sm text-gray-500 dark:text-gray-400')
                    ap = audio_path(self.pdf, page)
                    if ap.exists():
                        ui.label('🔊 生成済み音声プレビュー').classes('text-h6')
                        ui.audio(file_url(ap)).classes('w-full')

                with ui.column().classes('w-7/12'):
                    main_lang = '日本語' if self.lang_code == 'ja' else '英語'
                    text_area = ui.textarea(
                        f'主言語ナレーション原稿（{main_lang}）',
                        value=current_text,
                    ).props('outlined').classes('w-full').style('min-height: 180px')
                    with ui.row().classes('w-full'):
                        ui.button('✨ 現在のスライドのナレーションを再生成',
                                  on_click=lambda: self.regenerate_narration(page, text_area)).classes('grow')
                        ui.button('🔄 保存して字幕・ポインタを再解析',
                                  on_click=lambda: self.save_and_realign(page, text_area.value)).classes('grow')

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
        if not self.pdf:
            with self.slide_video_gallery:
                ui.label('プレゼンテーションPDFを選択してください．').classes('text-blue-500 dark:text-blue-400')
            return

        with self.slide_video_gallery:
            with ui.grid(columns=4).classes('w-full gap-4'):
                for p in self.active_pages:
                    vp = video_path(self.pdf, p)
                    img = page_image_path(self.pdf, p)
                    with ui.card().classes('w-full p-3 gap-2 rounded-lg bg-zinc-800 border border-zinc-700 shadow-sm'):
                        if vp.exists():
                            ui.video(file_url(vp)).classes('w-full rounded shadow-sm')
                        else:
                            if img.exists():
                                ui.image(file_url(img)).classes('w-full opacity-60 rounded shadow-sm')
                            ui.label('（単体動画 未生成）').classes('text-xs text-orange-400 font-medium')
                        # スライド番号ラベルをビデオ／画像の下側に配置
                        ui.label(f'スライド {p}').classes('text-sm font-bold text-white self-center pt-1')

    async def refresh_final_video(self) -> None:
        if not self.final_video_container:
            return
        self.final_video_container.clear()
        if not self.pdf:
            with self.final_video_container:
                ui.label('PDFを選択してください．').classes('text-xs text-zinc-500')
            return

        final = final_video_path(self.pdf)

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

    # ---- settings -----------------------------------------------------

    def refresh_settings(self) -> None:
        if not self.settings_container:
            return
        self.settings_container.clear()
        filter_cfg = load_tts_filter_config(TTS_FILTER_PATH) if TTS_FILTER_PATH.exists() else {}
        filter_dict = filter_cfg.setdefault('dictionary', {})
        with self.settings_container:
            ui.label('システム設定').classes('text-h4')

            # --- LLM 設定 ---
            ui.label('LLM 設定 (`config.yaml`)').classes('text-h5')
            llm = self.cfg.setdefault('llm', {})
            with ui.row().classes('w-full'):
                llm_base = ui.input('LLM Base URL', value=llm.get('base_url', '')).classes('grow')
                llm_model = ui.input('LLM Model', value=llm.get('model', '')).classes('grow')
                llm_temp = ui.number('Temperature', value=float(llm.get('temperature', 0.3)), min=0, max=2, step=0.1).classes('w-40')

            # LLM 接続テストエリア
            with ui.card().classes('w-full p-4 bg-zinc-900 border border-zinc-700 rounded-lg gap-2 mt-1'):
                ui.label('🧪 LLM 接続テスト（シングルターン会話）').classes('text-sm font-bold text-zinc-200')
                with ui.row().classes('w-full items-center gap-2'):
                    llm_test_input = ui.input(
                        'テストプロンプト',
                        value='こんにちは！自己紹介を1文でしてください．',
                    ).props('dense outlined').classes('grow')
                    llm_test_btn = ui.button('💬 LLM 接続テスト送信').props('dense outline')

                llm_test_result = ui.label('').classes('text-xs text-zinc-300 font-mono p-2 bg-zinc-800 rounded min-h-[36px] w-full whitespace-pre-wrap')

                async def run_llm_test() -> None:
                    prompt = (llm_test_input.value or '').strip()
                    if not prompt:
                        ui.notify('プロンプトを入力してください．', type='warning')
                        return
                    if not llm_base.value or not llm_model.value:
                        ui.notify('Base URL と Model を入力してください．', type='warning')
                        return

                    llm_test_btn.disable()
                    llm_test_result.text = 'LLMにリクエスト中…'
                    try:
                        test_cfg = {
                            'base_url': llm_base.value.strip(),
                            'api_key': llm.get('api_key', 'dummy'),
                        }
                        client = await run.io_bound(make_client, test_cfg)

                        def call_llm() -> str:
                            res = client.chat.completions.create(
                                model=llm_model.value.strip(),
                                temperature=float(llm_temp.value or 0.3),
                                max_tokens=1000,
                                messages=[{'role': 'user', 'content': prompt}],
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

            # --- 日本語 TTS 設定 ---
            ui.separator().classes('my-4')
            ui.label('🇯🇵 日本語 TTS 設定 (`config.yaml: tts.ja`)').classes('text-h5')
            tts = self.cfg.setdefault('tts', {})
            ja = tts.setdefault('ja', {})
            with ui.row().classes('w-full'):
                ja_base = ui.input('日本語 Base URL', value=ja.get('base_url', '')).classes('grow')
                ja_model = ui.input('日本語 Model', value=ja.get('model', '')).classes('grow')
                ja_voice = ui.input('日本語 Voice', value=ja.get('voice', '')).classes('grow')

            # 日本語 TTS 接続テストエリア
            with ui.card().classes('w-full p-4 bg-zinc-900 border border-zinc-700 rounded-lg gap-2 mt-1'):
                ui.label('🧪 日本語 TTS 接続・音声再生テスト').classes('text-sm font-bold text-zinc-200')
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
                    if not ja_base.value or not ja_model.value or not ja_voice.value:
                        ui.notify('日本語 TTS の Base URL, Model, Voice を指定してください．', type='warning')
                        return

                    ja_test_btn.disable()
                    ja_audio_container.clear()
                    with ja_audio_container:
                        ui.label('音声を合成中…').classes('text-xs text-zinc-400')
                    try:
                        tts_cfg = {
                            'base_url': ja_base.value.strip(),
                            'api_key': 'dummy',
                            'model': ja_model.value.strip(),
                            'voice': ja_voice.value.strip(),
                            'response_format': 'mp3',
                        }
                        out_path = TEST_AUDIO_DIR / 'test_ja.mp3'

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
                        with ja_audio_container:
                            ui.audio(file_url(out_path)).props('autoplay').classes('w-full max-w-lg mt-1')
                        ui.notify('日本語音声を合成しました．', type='positive')
                    except Exception as err:
                        ja_audio_container.clear()
                        with ja_audio_container:
                            ui.label(f'【TTS合成失敗】: {err}').classes('text-xs text-red-400')
                        ui.notify(f'日本語 TTS テストに失敗しました: {err}', type='negative')
                    finally:
                        ja_test_btn.enable()

                ja_test_btn.on_click(run_ja_tts_test)

            # --- 英語 TTS 設定 ---
            ui.separator().classes('my-4')
            ui.label('🇺🇸 英語 TTS 設定 (`config.yaml: tts.en`)').classes('text-h5')
            en = tts.setdefault('en', {})
            with ui.row().classes('w-full'):
                en_base = ui.input('英語 Base URL', value=en.get('base_url', '')).classes('grow')
                en_model = ui.input('英語 Model', value=en.get('model', '')).classes('grow')
                en_voice = ui.input('英語 Voice', value=en.get('voice', '')).classes('grow')

            # 英語 TTS 接続テストエリア
            with ui.card().classes('w-full p-4 bg-zinc-900 border border-zinc-700 rounded-lg gap-2 mt-1'):
                ui.label('🧪 英語 TTS 接続・音声再生テスト').classes('text-sm font-bold text-zinc-200')
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
                    if not en_base.value or not en_model.value or not en_voice.value:
                        ui.notify('英語 TTS の Base URL, Model, Voice を指定してください．', type='warning')
                        return

                    en_test_btn.disable()
                    en_audio_container.clear()
                    with en_audio_container:
                        ui.label('Synthesizing speech...').classes('text-xs text-zinc-400')
                    try:
                        tts_cfg = {
                            'base_url': en_base.value.strip(),
                            'api_key': 'dummy',
                            'model': en_model.value.strip(),
                            'voice': en_voice.value.strip(),
                            'response_format': 'mp3',
                        }
                        out_path = TEST_AUDIO_DIR / 'test_en.mp3'

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
                        ui.notify('英語音声を合成しました．', type='positive')
                    except Exception as err:
                        en_audio_container.clear()
                        with en_audio_container:
                            ui.label(f'【TTS合成失敗】: {err}').classes('text-xs text-red-400')
                        ui.notify(f'英語 TTS テストに失敗しました: {err}', type='negative')
                    finally:
                        en_test_btn.enable()

                en_test_btn.on_click(run_en_tts_test)

            # --- config.yaml 保存ボタン ---
            def save_main_config() -> None:
                self.cfg['llm']['base_url'] = llm_base.value
                self.cfg['llm']['model'] = llm_model.value
                self.cfg['llm']['temperature'] = llm_temp.value
                self.cfg['tts']['ja'] = {'base_url': ja_base.value, 'api_key': 'dummy', 'model': ja_model.value, 'voice': ja_voice.value, 'response_format': 'mp3'}
                self.cfg['tts']['en'] = {'base_url': en_base.value, 'api_key': 'dummy', 'model': en_model.value, 'voice': en_voice.value, 'response_format': 'mp3'}
                save_config(CONFIG_PATH, self.cfg)
                ui.notify('config.yaml を保存しました．', type='positive')

            ui.button('💾 config.yaml を保存', on_click=save_main_config).classes('w-full mt-4')

            # --- 日本語TTS用ヨミ変換フィルタ ---
            ui.separator().classes('my-4')
            ui.label('🤖 日本語TTS用ヨミ変換フィルタ (`tts_filter.yaml`)').classes('text-h5')
            ui.label('技術用語・識別子・コマンド等の読み仮名辞書を編集できます．')

            with ui.row().classes('w-full items-end'):
                new_word = ui.input('単語・識別子（例: argc）').classes('grow')
                new_reading = ui.input('読みの目安（例: アーギューシー）').classes('grow')
                def add_word() -> None:
                    if new_word.value and new_reading.value:
                        filter_dict[new_word.value.strip()] = new_reading.value.strip()
                        save_config(TTS_FILTER_PATH, filter_cfg)
                        ui.notify(f'「{new_word.value.strip()} → {new_reading.value.strip()}」を追加しました．', type='positive')
                        self.refresh_settings()
                ui.button('辞書に追加', on_click=add_word)

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
            }).classes('w-full h-96')

            async def save_dictionary() -> None:
                await grid.load_client_data()
                data = grid.options.get('rowData', [])
                new_dictionary = {}
                for row in data or []:
                    w = str(row.get('単語 / 識別子', '')).strip()
                    r = str(row.get('読みの目安', '')).strip()
                    if w and r and w != 'nan' and r != 'nan':
                        new_dictionary[w] = r
                filter_cfg['dictionary'] = new_dictionary
                save_config(TTS_FILTER_PATH, filter_cfg)
                ui.notify(f'tts_filter.yaml を更新しました（全 {len(new_dictionary)} 件）．', type='positive')

            ui.button('💾 ヨミ変換辞書 (tts_filter.yaml) を保存', on_click=save_dictionary).classes('w-full')

    # ---- global refresh ----------------------------------------------

    async def refresh_all(self) -> None:
        self.recalculate_pages()
        await self.refresh_gallery()
        await self.refresh_simple_editor()
        await self.refresh_editor()
        await self.refresh_slide_videos()
        await self.refresh_final_video()

    # ---- UI -----------------------------------------------------------

    def build(self) -> None:
        ui.page_title('Slide Narrator')
        # ダークテーマに固定
        ui.dark_mode().enable()
        ui.colors(primary='#3b82f6')

        with ui.header().classes('items-center w-full px-4 bg-slate-900 border-b border-slate-800'):
            ui.label('🎓 Slide Narrator').classes('text-h5 text-white')
            ui.space()
            ui.label('TAKAGO_LAB. 2026').classes('text-subtitle2 font-mono tracking-wider text-slate-300 mr-2')

        with ui.left_drawer(value=True).props('width=320').classes('p-4'):
            ui.label('プロジェクト設定').classes('text-h5')
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

            self.pages_input = (
                ui.input('ビデオ化対象', placeholder='例: 1-10,12', value=self.pages_spec)
                .classes('w-full')
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
                ui.button('① ナレーション原稿を一括生成', on_click=lambda: self.pipeline('explain', '① ナレーション原稿を作成中…')).classes('w-full'),
                ui.button('② 字幕翻訳＆ポインタを一括解析', on_click=lambda: self.pipeline('align', '② 字幕・ポインタを解析中…')).classes('w-full'),
                ui.button('③ TTS音声を生成', on_click=lambda: self.pipeline('tts', f'③ 音声を合成中 ({self.lang_code})…')).classes('w-full'),
                ui.button('④ 動画を生成', on_click=lambda: self.pipeline('video', '④ 動画を生成中…')).classes('w-full'),
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
                tab_settings = ui.tab('⚙ 設定')
                tab_logs = ui.tab('📜 ログ')

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
                with ui.tab_panel(tab_settings):
                    self.settings_container = ui.column().classes('w-full')
                with ui.tab_panel(tab_logs):
                    with ui.row().classes('w-full items-center justify-between pb-2'):
                        ui.label('📜 実行ログ履歴').classes('text-h5')
                        ui.button('ログをクリア', on_click=lambda: self.history_log_widget.clear() if self.history_log_widget else None).props('dense outline size=sm color=negative')
                    self.history_log_widget = ui.log(max_lines=2000).classes('w-full h-[650px] font-mono text-xs bg-zinc-900 border border-zinc-700 rounded-lg p-3 text-zinc-300')
                    if self.log:
                        for line in self.log.splitlines():
                            self.history_log_widget.push(line)

        self.refresh_settings()
        self.recalculate_pages()

    def _mode_changed(self, value: str) -> None:
        self.mode_code = value
        self.save_project_settings()

    def _lang_changed(self, value: str) -> None:
        self.lang_code = value
        self.save_project_settings()
        asyncio.create_task(self.refresh_simple_editor())
        asyncio.create_task(self.refresh_editor())

    def _range_changed(self) -> None:
        self.pages_spec = self.pages_input.value or ''
        try:
            self.recalculate_pages()
            self.save_project_settings()
            asyncio.create_task(self.refresh_gallery())
            asyncio.create_task(self.refresh_simple_editor())
            asyncio.create_task(self.refresh_editor())
            asyncio.create_task(self.refresh_slide_videos())
        except Exception as exc:
            ui.notify(f'スライド範囲を解釈できません: {exc}', type='negative')


application = SlideNarratorApp()
application.build()

ui.run(title='Slide Narrator', reload=False)