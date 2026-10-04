#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

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
        self.skip_pages_spec = ''
        self.force_run = False
        self.active_pages: list[int] = []
        self.edit_page: int | None = None
        self.video_page: int | None = None
        self.log = ''
        self.processing = False

        self.page_count_label = None
        self.active_count_label = None
        self.pdf_label = None
        self.log_area = None
        self.gallery = None
        self.edit_container = None
        self.video_container = None
        self.settings_container = None
        self.mode_select = None
        self.lang_select = None
        self.pages_input = None
        self.skip_input = None
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
        self.skip_pages_spec = self.proj_cfg.get('skip_pages', '')
        self.images = await run.io_bound(
            ensure_page_images, pdf, int(self.cfg.get('pdf', {}).get('dpi', 120))
        )
        self.refresh_project_widgets()
        await self.refresh_all()
        ui.notify(f'PDFを読み込みました: {pdf.name}', type='positive')

    def refresh_project_widgets(self) -> None:
        if not self.pdf:
            return
        if self.mode_select:
            self.mode_select.value = self.mode_code
        if self.lang_select:
            self.lang_select.value = self.lang_code
        if self.pages_input:
            self.pages_input.value = self.pages_spec
        if self.skip_input:
            self.skip_input.value = self.skip_pages_spec
        if self.pdf_label:
            self.pdf_label.text = f'PDF: {self.pdf.name} ({len(self.images)} ページ)'

    def save_project_settings(self) -> None:
        if not self.pdf:
            return
        root = project_root(self.pdf)
        self.proj_cfg.update({
            'mode': self.mode_code,
            'language': self.lang_code,
            'pages': self.pages_spec,
            'skip_pages': self.skip_pages_spec,
        })
        save_project_json(root, self.proj_cfg)

    def recalculate_pages(self) -> None:
        if not self.pdf:
            self.active_pages = []
            return
        total = len(self.images)
        active = set(range(1, total + 1))
        if self.pages_spec.strip():
            active &= parse_page_ranges(self.pages_spec.strip())
        if self.skip_pages_spec.strip():
            active -= parse_page_ranges(self.skip_pages_spec.strip())
        self.active_pages = sorted(active)
        if self.edit_page not in self.active_pages:
            self.edit_page = self.active_pages[0] if self.active_pages else None
        if self.video_page not in self.active_pages:
            self.video_page = self.active_pages[0] if self.active_pages else None
        if self.active_count_label:
            self.active_count_label.text = f'対象スライド: {len(self.active_pages)} / {total} 枚'

    # ---- pipeline -----------------------------------------------------

    def open_processing_dialog(self, label: str):
        dialog = ui.dialog()
        dialog.props('persistent')
        with dialog, ui.card().classes('items-center'):
            ui.spinner(size='lg')
            ui.label(label)
        dialog.open()
        return dialog

    def set_processing(self, value: bool) -> None:
        self.processing = value
        for button in self.pipeline_buttons:
            button.disable() if value else button.enable()

    def base_args(self) -> list[str]:
        args = ['--mode', self.mode_code, '--lang', self.lang_code]
        if self.force_run:
            args.append('--force')
        if self.pages_spec.strip():
            args.extend(['--pages', self.pages_spec.strip()])
        if self.skip_pages_spec.strip():
            args.extend(['--skip-pages', self.skip_pages_spec.strip()])
        return args

    async def pipeline(self, stage: str, label: str) -> None:
        if not self.pdf:
            ui.notify('先にPDFを選択してください．', type='warning')
            return
        if self.processing:
            ui.notify('別の処理が実行中です．処理が終わるまでお待ちください．', type='warning')
            return
        self.save_project_settings()
        dialog = self.open_processing_dialog(label)
        self.set_processing(True)
        try:
            rc, out = await run.io_bound(
                run_command,
                ['python', 'slide_lecture.py', str(self.pdf), '--from', stage, *self.base_args()],
            )
            self.log = out
            self.log_area.value = out
            if rc == 0:
                ui.notify(f'{label} 完了', type='positive')
            else:
                ui.notify(f'処理が終了コード {rc} で終了しました．ログを確認してください．', type='negative')
            await self.refresh_all()
        except Exception as exc:
            self.log = str(exc)
            self.log_area.value = self.log
            ui.notify(f'処理に失敗しました: {exc}', type='negative')
        finally:
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
        dialog = self.open_processing_dialog(f'スライド P.{page} のナレーションをLLMで再作成中…')
        self.set_processing(True)
        try:
            cfg = self.cfg
            client = await run.io_bound(make_client, cfg['llm'])
            page_texts = await run.io_bound(extract_page_text, self.pdf)
            exp_dir = project_root(self.pdf) / 'explanations'
            overview_path = exp_dir / '_course_overview.txt'
            if overview_path.exists():
                overview = overview_path.read_text(encoding='utf-8')
            else:
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
            ep = explanation_path(self.pdf, page)
            ep.parent.mkdir(parents=True, exist_ok=True)
            ep.write_text(new_narration.strip() + '\n', encoding='utf-8')
            cleanup_downstream_media(self.pdf, page, include_alignment=True)
            text_area.value = new_narration.strip()
            ui.notify('ナレーションを再生成しました（古い字幕・音声・動画を初期化しました）．', type='positive')
            await self.refresh_editor()
        except Exception as exc:
            ui.notify(f'再生成に失敗しました: {exc}', type='negative')
        finally:
            self.set_processing(False)
            dialog.close()

    async def save_and_realign(self, page: int, text: str) -> None:
        if not self.pdf:
            return
        ep = explanation_path(self.pdf, page)
        ep.parent.mkdir(parents=True, exist_ok=True)
        ep.write_text(text.rstrip() + '\n', encoding='utf-8')
        if self.processing:
            ui.notify('別の処理が実行中です．処理が終わるまでお待ちください．', type='warning')
            return
        dialog = self.open_processing_dialog('文分割・字幕翻訳・ポインタを再解析中…')
        self.set_processing(True)
        try:
            client = await run.io_bound(make_client, self.cfg['llm'])
            dpi = int(self.cfg.get('pdf', {}).get('dpi', 150))
            data = await run.io_bound(
                generate_single_alignment,
                client, self.cfg['llm'], self.pdf, page,
                page_image_path(self.pdf, page), text.strip(), dpi, self.lang_code,
            )
            alp = alignment_path(self.pdf, page)
            alp.parent.mkdir(parents=True, exist_ok=True)
            alp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            cleanup_downstream_media(self.pdf, page, include_alignment=False)
            ui.notify('字幕とポインタを再生成しました（古い音声・動画をリセットしました）．', type='positive')
            await self.refresh_editor()
        except Exception as exc:
            ui.notify(f'再解析に失敗しました: {exc}', type='negative')
        finally:
            self.set_processing(False)
            dialog.close()

    async def save_alignment(self, page: int, rows: list[dict], align_data: dict) -> None:
        if not self.pdf:
            return
        alp = alignment_path(self.pdf, page)
        align_data['alignments'] = rows
        alp.write_text(json.dumps(align_data, ensure_ascii=False, indent=2), encoding='utf-8')
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
            self.gallery.add(ui.label('PDFを選択してください．'))
            return
        with self.gallery:
            with ui.grid(columns=5).classes('w-full gap-4'):
                for n, img in enumerate(self.images, 1):
                    with ui.card().classes('w-full'):
                        ui.image(file_url(img)).classes('w-full')
                        ui.label(('✅' if n in self.active_pages else '❌') + f' P.{n} ' + ('(対象)' if n in self.active_pages else '(除外)'))

    async def refresh_editor(self) -> None:
        if not self.edit_container:
            return
        self.edit_container.clear()
        if not self.active_pages or not self.pdf:
            with self.edit_container:
                ui.label('⚠ 処理対象となるスライドがありません．左側の範囲指定を確認してください．').classes('text-orange-700')
            return
        page = self.edit_page or self.active_pages[0]
        self.edit_page = page
        idx = self.active_pages.index(page)
        align_data, blocks, alignments = self.alignment_data(page)
        img = page_image_path(self.pdf, page)
        current_text_path = explanation_path(self.pdf, page)
        current_text = current_text_path.read_text(encoding='utf-8') if current_text_path.exists() else ''

        with self.edit_container:
            with ui.row().classes('w-full items-center'):
                ui.button('◀ 前のスライド', on_click=self.prev_edit).props(f'disable={idx == 0}')
                sel = ui.select(
                    {p: f'スライド P.{p} ({self.active_pages.index(p)+1} / {len(self.active_pages)})' for p in self.active_pages},
                    value=page,
                    label='編集スライド選択',
                ).classes('grow')
                sel.on_value_change(lambda e: asyncio.create_task(self.select_edit_page(e.value)))
                ui.button('次のスライド ▶', on_click=self.next_edit).props(f'disable={idx == len(self.active_pages)-1}')

            with ui.row().classes('w-full items-start gap-6'):
                with ui.column().classes('w-5/12'):
                    if img.exists():
                        preview = draw_block_preview(img, blocks) if blocks else None
                        if preview is not None:
                            # Use the original file for stable browser loading when no temp path is needed;
                            # the block overlay is saved to a project-local preview file.
                            preview_path = project_root(self.pdf) / 'pages' / f'{page:03d}_preview.png'
                            preview.save(preview_path)
                            ui.image(file_url(preview_path)).classes('w-full')
                        else:
                            ui.image(file_url(img)).classes('w-full')
                        ui.label(f'スライド P.{page}').classes('text-sm text-gray-600')
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
                            with ui.row().classes('w-full items-center'):
                                ui.label(f'文 {i}').classes('w-12')
                                ui.label(row['sentence']).classes('grow')
                                select = ui.select(block_options, value=row['block_id'], label='ポインタ先').classes('w-48')
                                trans = ui.input('対訳字幕', value=row['translation']).classes('grow')
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

                        ui.button('💾 字幕・ポインタ修正を保存', on_click=save_rows).classes('w-full')

    async def refresh_video(self) -> None:
        if not self.video_container:
            return
        self.video_container.clear()
        if not self.pdf:
            with self.video_container:
                ui.label('先にPDFを選択してください．').classes('text-blue-700')
            return
        final = final_video_path(self.pdf)
        ja_srt = final_ja_srt_path(self.pdf)
        en_srt = final_en_srt_path(self.pdf)
        with self.video_container:
            if final.exists():
                ui.video(file_url(final)).classes('w-full')
                ui.label('📥 成果物のダウンロード').classes('text-h6')
                with ui.row().classes('w-full'):
                    ui.link('動画 (MP4: 日英字幕・チャプター内蔵)', '/download/' + final.resolve().relative_to(Path.cwd().resolve()).as_posix(), new_tab=True)
                    if ja_srt.exists():
                        ui.link('日本語字幕 (.srt)', '/download/' + ja_srt.resolve().relative_to(Path.cwd().resolve()).as_posix(), new_tab=True)
                    if en_srt.exists():
                        ui.link('英語字幕 (.srt)', '/download/' + en_srt.resolve().relative_to(Path.cwd().resolve()).as_posix(), new_tab=True)
            else:
                ui.label('「④ 動画を生成」を実行するとここに全編結合動画と字幕ファイルが表示されます．').classes('text-blue-700')
            ui.separator()
            ui.label('🔍 ページごとの単体動画プレビュー').classes('text-h6')
            if self.active_pages:
                page = self.video_page or self.active_pages[0]
                idx = self.active_pages.index(page)
                with ui.row().classes('w-full items-center'):
                    ui.button('◀ 前のスライド', on_click=self.prev_video).props(f'disable={idx == 0}')
                    sel = ui.select({p: f'スライド P.{p} ({self.active_pages.index(p)+1} / {len(self.active_pages)})' for p in self.active_pages}, value=page).classes('grow')
                    sel.on_value_change(lambda e: asyncio.create_task(self.select_video_page(e.value)))
                    ui.button('次のスライド ▶', on_click=self.next_video).props(f'disable={idx == len(self.active_pages)-1}')
                vp = video_path(self.pdf, page)
                if vp.exists():
                    ui.video(file_url(vp)).classes('w-full')
                else:
                    ui.label(f'スライド P.{page} の個別動画はまだ生成されていません．')

    async def select_video_page(self, page: int) -> None:
        self.video_page = int(page)
        await self.refresh_video()

    async def prev_video(self) -> None:
        if self.video_page in self.active_pages:
            i = self.active_pages.index(self.video_page)
            if i > 0:
                await self.select_video_page(self.active_pages[i - 1])

    async def next_video(self) -> None:
        if self.video_page in self.active_pages:
            i = self.active_pages.index(self.video_page)
            if i + 1 < len(self.active_pages):
                await self.select_video_page(self.active_pages[i + 1])

    # ---- settings -----------------------------------------------------

    def refresh_settings(self) -> None:
        if not self.settings_container:
            return
        self.settings_container.clear()
        filter_cfg = load_tts_filter_config(TTS_FILTER_PATH) if TTS_FILTER_PATH.exists() else {}
        filter_dict = filter_cfg.setdefault('dictionary', {})
        with self.settings_container:
            ui.label('システム設定').classes('text-h4')
            ui.label('LLM 設定 (`config.yaml`)').classes('text-h5')
            llm = self.cfg.setdefault('llm', {})
            with ui.row().classes('w-full'):
                llm_base = ui.input('LLM Base URL', value=llm.get('base_url', '')).classes('grow')
                llm_model = ui.input('LLM Model', value=llm.get('model', '')).classes('grow')
                llm_temp = ui.number('Temperature', value=float(llm.get('temperature', 0.3)), min=0, max=2, step=0.1).classes('w-40')

            tts = self.cfg.setdefault('tts', {})
            ui.label('🇯🇵 日本語 TTS 設定 (`config.yaml: tts.ja`)').classes('text-h5')
            ja = tts.setdefault('ja', {})
            with ui.row().classes('w-full'):
                ja_base = ui.input('日本語 Base URL', value=ja.get('base_url', '')).classes('grow')
                ja_model = ui.input('日本語 Model', value=ja.get('model', '')).classes('grow')
                ja_voice = ui.input('日本語 Voice', value=ja.get('voice', '')).classes('grow')

            ui.label('🇺🇸 英語 TTS 設定 (`config.yaml: tts.en`)').classes('text-h5')
            en = tts.setdefault('en', {})
            with ui.row().classes('w-full'):
                en_base = ui.input('英語 Base URL', value=en.get('base_url', '')).classes('grow')
                en_model = ui.input('英語 Model', value=en.get('model', '')).classes('grow')
                en_voice = ui.input('英語 Voice', value=en.get('voice', '')).classes('grow')

            def save_main_config() -> None:
                self.cfg['llm']['base_url'] = llm_base.value
                self.cfg['llm']['model'] = llm_model.value
                self.cfg['llm']['temperature'] = llm_temp.value
                self.cfg['tts']['ja'] = {'base_url': ja_base.value, 'api_key': 'dummy', 'model': ja_model.value, 'voice': ja_voice.value, 'response_format': 'mp3'}
                self.cfg['tts']['en'] = {'base_url': en_base.value, 'api_key': 'dummy', 'model': en_model.value, 'voice': en_voice.value, 'response_format': 'mp3'}
                save_config(CONFIG_PATH, self.cfg)
                ui.notify('config.yaml を保存しました．', type='positive')

            ui.button('config.yaml を保存', on_click=save_main_config).classes('w-full')
            ui.separator()
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
        await self.refresh_editor()
        await self.refresh_video()

    # ---- UI -----------------------------------------------------------

    def build(self) -> None:
        ui.page_title('Slide Narrator')
        ui.colors(primary='#1976d2')

        with ui.header().classes('items-center'):
            ui.label('🎓 Slide Narrator').classes('text-h5')
            ui.label('講義＆研究発表モード対応・日英ナレーション＆字幕連動スライド動画生成システム').classes('text-sm')

        with ui.left_drawer(value=True).classes('p-4'):
            ui.label('プロジェクト設定').classes('text-h5')
            ui.upload(label='PDFを選択', auto_upload=True, on_upload=self.load_pdf).props('accept=.pdf').classes('w-full')
            self.pdf_label = ui.label('まずPDFを選択してください．')
            self.mode_select = ui.radio({'lecture': '🎓 講義 (Lecture)', 'research': '🔬 研究発表 (Research)'}, value=self.mode_code).props('inline')
            self.mode_select.on_value_change(lambda e: self._mode_changed(e.value))
            self.lang_select = ui.radio({'ja': '🇯🇵 日本語 (ja)', 'en': '🇺🇸 英語 (en)'}, value=self.lang_code).props('inline')
            self.lang_select.on_value_change(lambda e: self._lang_changed(e.value))
            self.pages_input = ui.input('含めるスライド (例: 1-10,12)', value=self.pages_spec).on_value_change(lambda e: self._range_changed())
            self.skip_input = ui.input('除外するスライド (例: 5,11-13)', value=self.skip_pages_spec).on_value_change(lambda e: self._range_changed())
            self.active_count_label = ui.label('対象スライド: 0 / 0 枚')
            ui.separator()
            ui.label('パイプライン実行').classes('text-h6')
            self.force_checkbox = ui.checkbox('既存ファイルを強制上書き (--force)', value=False)
            self.force_checkbox.on_value_change(lambda e: setattr(self, 'force_run', bool(e.value)))
            self.pipeline_buttons = [
                ui.button('① ナレーション原稿を一括生成', on_click=lambda: self.pipeline('explain', 'スライド解説ドラフトを作成中…')).classes('w-full'),
                ui.button('② 字幕翻訳＆ポインタを一括解析', on_click=lambda: self.pipeline('align', '文分割・対訳字幕生成・視線誘導を設計中…')).classes('w-full'),
                ui.button('③ TTS音声を生成', on_click=lambda: self.pipeline('tts', f'音声を生成中 ({self.lang_code})…')).classes('w-full'),
                ui.button('④ 動画を生成', on_click=lambda: self.pipeline('video', '動画および日英字幕トラックを生成中…')).classes('w-full'),
            ]
            ui.separator()
            self.log_area = ui.textarea('処理ログ').props('readonly').classes('w-full').style('min-height: 180px')
            ui.label('TAKAGO LAB., KIT, Japan.').classes('text-caption')
            ui.link('GitHub: takago/slide-narrator', 'https://github.com/takago/slide-narrator', new_tab=True)

        with ui.column().classes('w-full p-6'):
            with ui.tabs().classes('w-full') as tabs:
                tab_gallery = ui.tab('🖼️ スライド一覧')
                tab_edit = ui.tab('📝 処理対象スライドの詳細編集')
                tab_video = ui.tab('🎬 スライドショービデオの確認')
                tab_settings = ui.tab('⚙ 設定')
            with ui.tab_panels(tabs, value=tab_gallery).classes('w-full'):
                with ui.tab_panel(tab_gallery):
                    ui.label('🖼️ 全スライド一覧').classes('text-h5')
                    self.gallery = ui.column().classes('w-full')
                with ui.tab_panel(tab_edit):
                    self.edit_container = ui.column().classes('w-full')
                with ui.tab_panel(tab_video):
                    self.video_container = ui.column().classes('w-full')
                with ui.tab_panel(tab_settings):
                    self.settings_container = ui.column().classes('w-full')

        self.refresh_settings()
        self.recalculate_pages()

    def _mode_changed(self, value: str) -> None:
        self.mode_code = value
        self.save_project_settings()

    def _lang_changed(self, value: str) -> None:
        self.lang_code = value
        self.save_project_settings()
        asyncio.create_task(self.refresh_editor())

    def _range_changed(self) -> None:
        self.pages_spec = self.pages_input.value or ''
        self.skip_pages_spec = self.skip_input.value or ''
        try:
            self.recalculate_pages()
            self.save_project_settings()
            asyncio.create_task(self.refresh_gallery())
            asyncio.create_task(self.refresh_editor())
        except Exception as exc:
            ui.notify(f'スライド範囲を解釈できません: {exc}', type='negative')


application = SlideNarratorApp()
application.build()

ui.run(title='Slide Narrator', reload=False)
