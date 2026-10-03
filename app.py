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

import json
import subprocess
from pathlib import Path

import pandas as pd
import pymupdf as fitz
import streamlit as st
import yaml
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
from tts_filter import (
    load_config as load_tts_filter_config,
    build_system_prompt as build_tts_filter_prompt,
    make_client as make_tts_filter_client,
    transform_text as tts_filter_transform,
)


st.set_page_config(
    page_title="Slide Presentation Studio",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Slide Presentation Studio")
st.caption("講義＆研究発表モード対応・日英ナレーション＆字幕連動スライド動画生成システム")


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_config(path: Path, cfg: dict) -> None:
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def run_command(args: list[str]) -> tuple[int, str]:
    p = subprocess.run(
        args,
        capture_output=True,
        text=True,
    )
    return p.returncode, p.stdout + "\n" + p.stderr


def project_root(pdf: Path) -> Path:
    return pdf.with_name(pdf.stem + "_lecture")


def explanation_path(pdf: Path, page: int) -> Path:
    return project_root(pdf) / "explanations" / f"{page:03d}.txt"


def alignment_path(pdf: Path, page: int) -> Path:
    return project_root(pdf) / "explanations" / f"{page:03d}_align.json"


def page_image_path(pdf: Path, page: int) -> Path:
    return project_root(pdf) / "pages" / f"{page:03d}.png"


def audio_path(pdf: Path, page: int) -> Path:
    return project_root(pdf) / "audio" / f"{page:03d}.mp3"


def video_path(pdf: Path, page: int) -> Path:
    return project_root(pdf) / "video" / f"{page:03d}.mp4"


def final_video_path(pdf: Path) -> Path:
    return project_root(pdf) / f"{pdf.stem}.mp4"


def final_ja_srt_path(pdf: Path) -> Path:
    return project_root(pdf) / f"{pdf.stem}_ja.srt"


def final_en_srt_path(pdf: Path) -> Path:
    return project_root(pdf) / f"{pdf.stem}_en.srt"


def cleanup_downstream_media(pdf: Path, page: int | None = None, include_alignment: bool = False) -> None:
    """上流更新時、依存する古い音声・個別動画・結合成果物を確実に削除する"""
    for f in [final_video_path(pdf), final_ja_srt_path(pdf), final_en_srt_path(pdf)]:
        if f.exists():
            f.unlink()

    if page is not None:
        if include_alignment:
            alp = alignment_path(pdf, page)
            if alp.exists():
                alp.unlink()
        ap = audio_path(pdf, page)
        vp = video_path(pdf, page)
        if ap.exists():
            ap.unlink()
        if vp.exists():
            vp.unlink()
    else:
        root = project_root(pdf)
        if include_alignment:
            exp_d = root / "explanations"
            if exp_d.exists():
                for item in exp_d.glob("*_align.json"):
                    item.unlink()
        for d in [root / "audio", root / "video"]:
            if d.exists():
                for item in d.glob("*"):
                    if item.is_file():
                        item.unlink()


def ensure_page_images(pdf: Path, dpi: int = 120) -> list[Path]:
    pages_dir = project_root(pdf) / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf)
    result = []
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    for i, page in enumerate(doc, 1):
        out = pages_dir / f"{i:03d}.png"
        result.append(out)
        if not out.exists():
            page.get_pixmap(matrix=matrix, alpha=False).save(out)
    return result


def draw_block_preview(img_path: Path, blocks: list[dict]) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    for b in blocks:
        box = b.get("bbox")
        bid = b.get("block_id")
        if box:
            draw.rectangle(box, outline="blue", width=2)
            draw.rectangle([box[0], box[1] - 20, box[0] + 35, box[1]], fill="blue")
            draw.text((box[0] + 4, box[1] - 18), f"#{bid}", fill="white")
    return im


config_path = Path("config.yaml")
cfg = load_config(config_path)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("プロジェクト設定")

    uploaded = st.file_uploader("PDFを選択", type=["pdf"])

    if uploaded is not None:
        upload_dir = Path("webui_uploads")
        upload_dir.mkdir(exist_ok=True)
        pdf_path = upload_dir / uploaded.name
        if not pdf_path.exists() or pdf_path.stat().st_size != uploaded.size:
            pdf_path.write_bytes(uploaded.getvalue())
        st.session_state["pdf_path"] = str(pdf_path)

    if "pdf_path" not in st.session_state:
        st.info("まずPDFを選択してください．")
        st.stop()

    pdf = Path(st.session_state["pdf_path"])
    root = project_root(pdf)

    # project.json から設定をロード
    proj_cfg = load_project_json(root)

    current_mode = proj_cfg.get("mode") or cfg.get("mode", "lecture")
    selected_mode = st.radio(
        "発表の種別 (Mode)",
        ["🎓 講義 (Lecture)", "🔬 研究発表 (Research)"],
        index=0 if current_mode == "lecture" else 1,
    )
    mode_code = "lecture" if "講義" in selected_mode else "research"

    current_lang = proj_cfg.get("language") or cfg.get("language", "ja")
    selected_lang = st.radio(
        "スライド・ナレーションの主言語",
        ["🇯🇵 日本語 (ja)", "🇺🇸 英語 (en)"],
        index=0 if current_lang == "ja" else 1,
    )
    lang_code = "ja" if "日本語" in selected_lang else "en"

    images = ensure_page_images(pdf, dpi=int(cfg.get("pdf", {}).get("dpi", 120)))

    st.write(f"**PDF:** `{pdf.name}` ({len(images)} ページ)")

    st.divider()
    st.header("🎯 スライド範囲指定")

    init_pages = proj_cfg.get("pages", "")
    init_skip_pages = proj_cfg.get("skip_pages", "")

    pages_spec = st.text_input("含めるスライド (例: 1-10,12)", value=init_pages, help="未指定時は全スライドが対象")
    skip_pages_spec = st.text_input("除外するスライド (例: 5,11-13)", value=init_skip_pages, help="除外したいスライド番号")

    # 変更を検知して project.json に自動保存
    if (
        mode_code != proj_cfg.get("mode")
        or lang_code != proj_cfg.get("language")
        or pages_spec != proj_cfg.get("pages")
        or skip_pages_spec != proj_cfg.get("skip_pages")
    ):
        proj_cfg.update({
            "mode": mode_code,
            "language": lang_code,
            "pages": pages_spec,
            "skip_pages": skip_pages_spec,
        })
        save_project_json(root, proj_cfg)

    # 処理対象スライド集合の算出
    total_pages = len(images)
    active_pages_set = set(range(1, total_pages + 1))
    if pages_spec.strip():
        active_pages_set &= parse_page_ranges(pages_spec.strip())
    if skip_pages_spec.strip():
        active_pages_set -= parse_page_ranges(skip_pages_spec.strip())
    active_pages_list = sorted(list(active_pages_set))

    st.caption(f"対象スライド: **{len(active_pages_list)}** / {total_pages} 枚")

    st.divider()
    st.header("パイプライン実行")

    force_run = st.checkbox("既存ファイルを強制上書き (--force)", value=False)
    base_args = ["--mode", mode_code, "--lang", lang_code]
    if force_run:
        base_args.append("--force")
    if pages_spec.strip():
        base_args.extend(["--pages", pages_spec.strip()])
    if skip_pages_spec.strip():
        base_args.extend(["--skip-pages", skip_pages_spec.strip()])

    if st.button("① ナレーション原稿を一括生成", use_container_width=True):
        with st.spinner("スライド解説ドラフトを作成中…"):
            rc, out = run_command(["python", "slide_lecture.py", str(pdf), "--from", "explain", *base_args])
        st.session_state["log"] = out
        st.rerun()

    if st.button("② 字幕翻訳＆ポインタを一括解析", use_container_width=True):
        with st.spinner("文分割・対訳字幕生成・視線誘導を設計中…"):
            rc, out = run_command(["python", "slide_lecture.py", str(pdf), "--from", "align", *base_args])
        st.session_state["log"] = out
        st.rerun()

    if st.button("③ TTS音声を生成", use_container_width=True):
        with st.spinner(f"音声を生成中 ({lang_code})…"):
            rc, out = run_command(["python", "slide_lecture.py", str(pdf), "--from", "tts", *base_args])
        st.session_state["log"] = out
        st.rerun()

    if st.button("④ 動画を生成", use_container_width=True):
        with st.spinner("動画および日英字幕トラックを生成中…"):
            rc, out = run_command(["python", "slide_lecture.py", str(pdf), "--from", "video", *base_args])
        st.session_state["log"] = out
        st.rerun()

    if "log" in st.session_state:
        with st.expander("処理ログ"):
            st.code(st.session_state["log"])


# ----------------------------------------------------------------------
# Main Tabs (4タブ構成)
# ----------------------------------------------------------------------

tabs = st.tabs([
    "🖼️ スライド一覧",
    "📝 処理対象スライドの詳細編集",
    "🎬 スライドショービデオの確認",
    "⚙ 設定",
])


# ----------------------------------------------------------------------
# Tab 1: スライド一覧 (閲覧用ギャラリー)
# ----------------------------------------------------------------------

with tabs[0]:
    st.subheader(f"🖼️ 全スライド一覧（全 {len(images)} ページ）")
    st.caption("※ 除外したいスライドの番号は、左側ペインの「除外するスライド」に入力してください。設定は project.json に自動保存されます。")

    cols_per_row = 5
    rows = [images[i : i + cols_per_row] for i in range(0, len(images), cols_per_row)]
    for r_idx, row in enumerate(rows):
        cols = st.columns(cols_per_row)
        for c_idx, img_p in enumerate(row):
            p_num = r_idx * cols_per_row + c_idx + 1
            is_active = p_num in active_pages_set
            with cols[c_idx]:
                st.image(str(img_p), use_container_width=True)
                if is_active:
                    st.caption(f"✅ **P.{p_num}** (対象)")
                else:
                    st.caption(f"❌ **P.{p_num}** (除外)")


# ----------------------------------------------------------------------
# Tab 2: 処理対象スライドの詳細編集
# ----------------------------------------------------------------------

with tabs[1]:
    st.subheader("📝 処理対象スライドの詳細編集")

    if not active_pages_list:
        st.warning("処理対象となるスライドが1枚もありません．左ペインの範囲指定を確認してください．")
    else:
        if "edit_page" not in st.session_state or st.session_state["edit_page"] not in active_pages_list:
            st.session_state["edit_page"] = active_pages_list[0]

        cur_page = st.session_state["edit_page"]
        cur_idx = active_pages_list.index(cur_page)

        c_nav_prev, c_nav_sel, c_nav_next = st.columns([1, 3, 1])

        with c_nav_prev:
            if st.button("◀ 前のスライド", use_container_width=True, disabled=(cur_idx <= 0), key="btn_prev_edit"):
                st.session_state["edit_page"] = active_pages_list[cur_idx - 1]
                st.rerun()

        with c_nav_sel:
            sel_target = st.selectbox(
                "編集スライド選択",
                options=active_pages_list,
                index=cur_idx,
                format_func=lambda x: f"スライド P.{x} ({active_pages_list.index(x) + 1} / {len(active_pages_list)})",
                label_visibility="collapsed",
            )
            if sel_target != st.session_state["edit_page"]:
                st.session_state["edit_page"] = sel_target
                st.rerun()

        with c_nav_next:
            if st.button("次のスライド ▶", use_container_width=True, disabled=(cur_idx >= len(active_pages_list) - 1), key="btn_next_edit"):
                st.session_state["edit_page"] = active_pages_list[cur_idx + 1]
                st.rerun()

        page = st.session_state["edit_page"]

        col_left, col_right = st.columns([1.1, 1.2])

        alp = alignment_path(pdf, page)
        align_data = {}
        blocks = []
        alignments = []
        if alp.exists():
            align_data = json.loads(alp.read_text(encoding="utf-8"))
            blocks = align_data.get("blocks", [])
            alignments = align_data.get("alignments", [])

        # 左側：スライド画像とポインタ枠プレビュー
        with col_left:
            img = page_image_path(pdf, page)
            if img.exists():
                if blocks:
                    preview_img = draw_block_preview(img, blocks)
                    st.image(preview_img, caption=f"スライド P.{page}（青枠: 抽出要素）", use_container_width=True)
                else:
                    st.image(str(img), caption=f"スライド P.{page}", use_container_width=True)

            ap = audio_path(pdf, page)
            if ap.exists():
                st.markdown("##### 🔊 生成済み音声プレビュー")
                st.audio(str(ap), format="audio/mp3")

        # 右側：テキスト・ポインタ・字幕編集
        with col_right:
            ep = explanation_path(pdf, page)
            current_text = ep.read_text(encoding="utf-8") if ep.exists() else ""

            # テキストエリアの世代管理キー（再生成時にインクリメントすることでUIを即時更新）
            ver_key = f"explanation_ver_{page}"
            if ver_key not in st.session_state:
                st.session_state[ver_key] = 0

            text = st.text_area(
                f"主言語ナレーション原稿 ({'日本語' if lang_code == 'ja' else '英語'})",
                value=current_text,
                height=160,
                key=f"explanation_text_{page}_{st.session_state[ver_key]}",
            )

            c_act1, c_act2 = st.columns([1.2, 1.4])
            with c_act1:
                if st.button("✨ 現在のスライドのナレーションを再生成", key=f"regen_narration_{page}", use_container_width=True):
                    with st.spinner(f"スライド P.{page} のナレーションをLLMで再作成中…"):
                        client = make_client(cfg["llm"])
                        page_texts = extract_page_text(pdf)
                        exp_dir = project_root(pdf) / "explanations"
                        overview_path = exp_dir / "_course_overview.txt"
                        if overview_path.exists():
                            overview = overview_path.read_text(encoding="utf-8")
                        else:
                            overview = build_course_overview(client, cfg["llm"], page_texts, mode=mode_code, lang=lang_code)
                            exp_dir.mkdir(parents=True, exist_ok=True)
                            overview_path.write_text(overview + "\n", encoding="utf-8")

                        prev_idx = active_pages_list[cur_idx - 1] - 1 if cur_idx > 0 else None
                        next_idx = active_pages_list[cur_idx + 1] - 1 if cur_idx < len(active_pages_list) - 1 else None

                        prev_text = page_texts[prev_idx] if prev_idx is not None else None
                        next_text = page_texts[next_idx] if next_idx is not None else None

                        prev_exp_path = explanation_path(pdf, active_pages_list[cur_idx - 1]) if cur_idx > 0 else None
                        prev_explanation = prev_exp_path.read_text(encoding="utf-8").strip() if (prev_exp_path and prev_exp_path.exists()) else ""

                        new_narration = make_explanation(
                            client=client,
                            cfg=cfg["llm"],
                            image=page_image_path(pdf, page),
                            current_page=page,
                            current_text=page_texts[page - 1],
                            prev_page=prev_idx + 1 if prev_idx is not None else None,
                            prev_text=prev_text,
                            next_page=next_idx + 1 if next_idx is not None else None,
                            next_text=next_text,
                            previous_explanation=prev_explanation,
                            course_overview=overview,
                            total_active=len(active_pages_list),
                            active_idx=cur_idx,
                            mode=mode_code,
                            lang=lang_code,
                        )

                        ep.parent.mkdir(parents=True, exist_ok=True)
                        ep.write_text(new_narration.strip() + "\n", encoding="utf-8")

                        # ウィジェットの世代を進めて再描画時に新しいテキストを確実に表示
                        st.session_state[ver_key] += 1

                        # 原稿更新に伴い下流の古い align.json・音声・動画を初期化
                        cleanup_downstream_media(pdf, page, include_alignment=True)

                    st.success("ナレーションを再生成しました（古い字幕・音声・動画を初期化しました）．")
                    st.rerun()

            with c_act2:
                if st.button("🔄 保存して字幕・ポインタを再解析", key=f"re_align_{page}", use_container_width=True):
                    ep.parent.mkdir(parents=True, exist_ok=True)
                    ep.write_text(text.rstrip() + "\n", encoding="utf-8")
                    with st.spinner("文分割・字幕翻訳・ポインタを再解析中…"):
                        client = make_client(cfg["llm"])
                        dpi = int(cfg.get("pdf", {}).get("dpi", 150))
                        new_align_data = generate_single_alignment(
                            client, cfg["llm"], pdf, page, page_image_path(pdf, page), text.strip(), dpi, lang_code
                        )
                        alp.write_text(json.dumps(new_align_data, ensure_ascii=False, indent=2), encoding="utf-8")
                    cleanup_downstream_media(pdf, page, include_alignment=False)
                    st.success("字幕とポインタを再生成しました（古い音声・動画をリセットしました）．")
                    st.rerun()

            if alignments:
                st.markdown("---")
                st.markdown("##### 🎯 文ごとのポインタ先 & 対訳字幕の微調整")
                block_options = [None] + [b["block_id"] for b in blocks]
                block_labels = {
                    None: "（ポインタなし）",
                    **{b["block_id"]: f"#{b['block_id']}: {b['text'][:25]}..." for b in blocks}
                }

                updated_alignments = []
                for s_idx, item in enumerate(alignments):
                    s_text = item["sentence"]
                    cur_bid = item.get("block_id")
                    if cur_bid not in block_options:
                        cur_bid = None

                    c_sel, c_sub = st.columns([1, 1.3])
                    with c_sel:
                        selected_bid = st.selectbox(
                            f"文 {s_idx + 1} のポインタ先",
                            options=block_options,
                            index=block_options.index(cur_bid),
                            format_func=lambda x: block_labels[x],
                            key=f"align_{page}_{s_idx}",
                            help=s_text,
                        )
                    with c_sub:
                        if lang_code == "ja":
                            target_trans = st.text_input(f"文 {s_idx + 1} の英語字幕", value=item.get("en_sentence", ""), key=f"en_{page}_{s_idx}")
                            item_to_add = {"sentence": s_text, "block_id": selected_bid, "ja_sentence": s_text, "en_sentence": target_trans}
                        else:
                            target_trans = st.text_input(f"文 {s_idx + 1} の日本語字幕", value=item.get("ja_sentence", ""), key=f"ja_{page}_{s_idx}")
                            item_to_add = {"sentence": s_text, "block_id": selected_bid, "en_sentence": s_text, "ja_sentence": target_trans}

                    updated_alignments.append(item_to_add)

                if st.button("字幕・ポインタ修正を保存", key=f"save_alignment_only_{page}", use_container_width=True):
                    align_data["alignments"] = updated_alignments
                    alp.write_text(json.dumps(align_data, ensure_ascii=False, indent=2), encoding="utf-8")
                    vp = video_path(pdf, page)
                    if vp.exists():
                        vp.unlink()
                    for f in [final_video_path(pdf), final_ja_srt_path(pdf), final_en_srt_path(pdf)]:
                        if f.exists():
                            f.unlink()
                    st.success("字幕とポインタの修正を保存しました（古い動画成果物をリセットしました）．")
                    st.rerun()


# ----------------------------------------------------------------------
# Tab 3: スライドショービデオの確認
# ----------------------------------------------------------------------

with tabs[2]:
    st.subheader("🎬 スライドショービデオの確認")

    final = final_video_path(pdf)
    ja_srt = final_ja_srt_path(pdf)
    en_srt = final_en_srt_path(pdf)

    if final.exists():
        st.video(str(final))

        st.markdown("##### 📥 成果物のダウンロード")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            st.download_button(
                "動画 (MP4: 日英字幕・チャプター内蔵)",
                data=final.read_bytes(),
                file_name=final.name,
                mime="video/mp4",
                use_container_width=True,
            )
        with col_dl2:
            if ja_srt.exists():
                st.download_button(
                    "日本語字幕 (.srt)",
                    data=ja_srt.read_bytes(),
                    file_name=ja_srt.name,
                    mime="text/plain",
                    use_container_width=True,
                )
        with col_dl3:
            if en_srt.exists():
                st.download_button(
                    "英語字幕 (.srt)",
                    data=en_srt.read_bytes(),
                    file_name=en_srt.name,
                    mime="text/plain",
                    use_container_width=True,
                )
    else:
        st.info("「④ 動画を生成」を実行するとここに全編結合動画と字幕ファイルが表示されます．")

    st.divider()

    with st.expander("🔍 ページごとの単体動画プレビュー（動作検証用）"):
        if active_pages_list:
            if "video_page" not in st.session_state or st.session_state["video_page"] not in active_pages_list:
                st.session_state["video_page"] = active_pages_list[0]

            v_page = st.session_state["video_page"]
            v_idx = active_pages_list.index(v_page)

            v_prev, v_sel, v_next = st.columns([1, 3, 1])

            with v_prev:
                if st.button("◀ 前のスライド", use_container_width=True, disabled=(v_idx <= 0), key="btn_prev_video"):
                    st.session_state["video_page"] = active_pages_list[v_idx - 1]
                    st.rerun()

            with v_sel:
                sel_v_target = st.selectbox(
                    "確認スライド選択",
                    options=active_pages_list,
                    index=v_idx,
                    format_func=lambda x: f"スライド P.{x} ({active_pages_list.index(x) + 1} / {len(active_pages_list)})",
                    label_visibility="collapsed",
                )
                if sel_v_target != st.session_state["video_page"]:
                    st.session_state["video_page"] = sel_v_target
                    st.rerun()

            with v_next:
                if st.button("次のスライド ▶", use_container_width=True, disabled=(v_idx >= len(active_pages_list) - 1), key="btn_next_video"):
                    st.session_state["video_page"] = active_pages_list[v_idx + 1]
                    st.rerun()

            prev_page = st.session_state["video_page"]
            vp = video_path(pdf, prev_page)
            if vp.exists():
                st.video(str(vp))
            else:
                st.caption(f"スライド P.{prev_page} の個別動画はまだ生成されていません．")


# ----------------------------------------------------------------------
# Tab 4: 設定
# ----------------------------------------------------------------------

with tabs[3]:
    st.header("システム設定")

    filter_config_path = Path("tts_filter.yaml")
    filter_cfg = load_tts_filter_config(filter_config_path) if filter_config_path.exists() else {}

    st.subheader("LLM 設定 (`config.yaml`)")
    c_l1, c_l2, c_l3 = st.columns([2, 2, 1])
    llm_base = c_l1.text_input("LLM Base URL", cfg["llm"]["base_url"])
    llm_model = c_l2.text_input("LLM Model", cfg["llm"]["model"])
    llm_temp = c_l3.number_input("Temperature", min_value=0.0, max_value=2.0, value=float(cfg["llm"].get("temperature", 0.3)), step=0.1)

    st.divider()
    st.subheader("🇯🇵 日本語 TTS 設定 (`config.yaml: tts.ja`)")
    tts_ja = cfg.get("tts", {}).get("ja", {})
    c_j1, c_j2, c_j3 = st.columns([2, 1.5, 1.5])
    ja_base = c_j1.text_input("日本語 Base URL", tts_ja.get("base_url", ""))
    ja_model = c_j2.text_input("日本語 Model", tts_ja.get("model", ""))
    ja_voice = c_j3.text_input("日本語 Voice", tts_ja.get("voice", ""))

    st.subheader("🇺🇸 英語 TTS 設定 (`config.yaml: tts.en`)")
    tts_en = cfg.get("tts", {}).get("en", {})
    c_e1, c_e2, c_e3 = st.columns([2, 1.5, 1.5])
    en_base = c_e1.text_input("英語 Base URL", tts_en.get("base_url", ""))
    en_model = c_e2.text_input("英語 Model", tts_en.get("model", ""))
    en_voice = c_e3.text_input("英語 Voice", tts_en.get("voice", ""))

    if st.button("config.yaml を保存", use_container_width=True):
        cfg["llm"]["base_url"] = llm_base
        cfg["llm"]["model"] = llm_model
        cfg["llm"]["temperature"] = llm_temp

        cfg.setdefault("tts", {})
        cfg["tts"]["ja"] = {"base_url": ja_base, "api_key": "dummy", "model": ja_model, "voice": ja_voice, "response_format": "mp3"}
        cfg["tts"]["en"] = {"base_url": en_base, "api_key": "dummy", "model": en_model, "voice": en_voice, "response_format": "mp3"}

        save_config(config_path, cfg)
        st.success("config.yaml を保存しました．")

    st.divider()
    st.subheader("🤖 日本語TTS用ヨミ変換フィルタ (`tts_filter.yaml`)")
    st.caption("スライド解説文中の技術用語・識別子・コマンド等の読み仮名辞書を直接編集・更新できます．")

    filter_dict = filter_cfg.setdefault("dictionary", {})

    with st.expander("➕ 新しい単語・読みの個別追加", expanded=True):
        with st.form("add_filter_word"):
            c_word, c_reading, c_btn = st.columns([2, 2, 1])
            new_word = c_word.text_input("単語・識別子（例: argc）")
            new_reading = c_reading.text_input("読みの目安（例: アーギューシー）")
            if c_btn.form_submit_button("辞書に追加") and new_word and new_reading:
                filter_dict[new_word.strip()] = new_reading.strip()
                save_config(filter_config_path, filter_cfg)
                st.success(f"「{new_word.strip()} → {new_reading.strip()}」を追加しました．")
                st.rerun()

    dict_rows = [{"単語 / 識別子": k, "読みの目安": v} for k, v in filter_dict.items()]
    df_dict = pd.DataFrame(dict_rows)

    st.markdown(f"**登録済み辞書一覧（{len(dict_rows)} 件）**")
    st.caption("セルをダブルクリックして直接編集できます．最下行で新規追加、行選択＋Deleteキーで行削除も可能です．")

    edited_df = st.data_editor(
        df_dict,
        num_rows="dynamic",
        use_container_width=True,
        key="tts_filter_editor",
    )

    if st.button("💾 ヨミ変換辞書 (tts_filter.yaml) を保存", type="primary", use_container_width=True):
        new_dictionary = {}
        for _, row in edited_df.iterrows():
            w = str(row.get("単語 / 識別子", "")).strip()
            r = str(row.get("読みの目安", "")).strip()
            if w and r and w != "nan" and r != "nan":
                new_dictionary[w] = r

        filter_cfg["dictionary"] = new_dictionary
        save_config(filter_config_path, filter_cfg)
        st.success(f"tts_filter.yaml を更新しました（全 {len(new_dictionary)} 件）．")
        st.rerun()
