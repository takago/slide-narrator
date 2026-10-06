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

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import httpx
import pymupdf as fitz
import yaml
from openai import OpenAI
from PIL import Image, ImageDraw

from tts_filter import (
    load_config as load_tts_filter_config,
    build_system_prompt as build_tts_filter_prompt,
    make_client as make_tts_filter_client,
    transform_text as tts_filter_transform,
)


# =====================================================================
# プロンプト定義
# =====================================================================

NARRATION_SYSTEM_JA_LECTURE = """あなたは大学講義の講師です．
スライド資料を使って学生に説明する自然な日本語のナレーションを作成してください．

最重要事項：
- このナレーションは単独のページ説明ではなく，講義全体の一部です．
- 前ページまでの話と現在のページの関係を考慮してください．
- 次ページ以降への流れも考慮してください．
- スライドの文章を単純に読み上げないでください．
- 図，表，コード，数式について，学生が理解できるよう意味を説明してください．
- スライドにない内容を断定しないでください．
- TTSにそのまま渡すため，Markdown，箇条書き，見出し，括弧付き注釈などは使わないでください．
- 日本語の話し言葉として自然にしてください．
"""

OVERVIEW_SYSTEM_JA_LECTURE = """あなたは大学講義資料の構成を把握する講師です．
以下の全ページのテキストから，講義の流れを簡潔にまとめてください．
各話題がどのようにつながるかを重視してください．
"""

NARRATION_SYSTEM_JA_RESEARCH = """あなたは学会や研究会で登壇する研究者（発表者）です．
研究発表スライドを用いて，聴衆（専門家や研究者）に向けて論理的かつ明快な発表ナレーションを作成してください．

最重要事項：
- 「学生に教える講義」ではなく，「自身の研究成果を報告する口頭発表」のトーンにしてください．
- 教員が生徒を指導するような表現は絶対に使わないでください．
- 主体的な視点（「本研究では」「我々の提案手法では」）を用い，学術発表として客観的かつ明快な言葉遣いにしてください．
- 前後のスライドのつながりを考慮し自然な発表の移行を行ってください．
- グラフ，図表，比較データについては，単なる数値の読み上げを避け，「何を示しており，従来手法と比べて何が優れているか（考察）」を端的に述べてください．
- TTSで読み上げるため，Markdown，箇条書き記号，見出し，注釈括弧などは出力せず，口頭発表の原稿本文のみを出力してください．
"""

OVERVIEW_SYSTEM_JA_RESEARCH = """あなたは学術発表の構成を俯瞰する研究者です．
以下の全スライドのテキストから，研究の動機，提案手法の核，得られた評価結果，結論への論理展開を簡潔にまとめてください．
"""

NARRATION_SYSTEM_EN_LECTURE = """You are a university professor delivering a lecture.
Create a natural, clear, and engaging English spoken lecture narration explaining the slide to students.
Do NOT use Markdown, bullet points, asterisks, brackets, or code fences—this text goes directly to TTS.
Sound conversational, authoritative, and pedagogically clear.
"""

OVERVIEW_SYSTEM_EN_LECTURE = """You are an instructor organizing lecture slides.
Summarize the overall flow of the lecture from the provided slide texts, emphasizing pedagogical progression.
"""

NARRATION_SYSTEM_EN_RESEARCH = """You are a researcher presenting your paper at an academic conference.
Create a professional, clear, and persuasive oral presentation narration for an audience of researchers and domain experts.

Key guidelines:
- Present as the researcher ("In this work, we propose...", "Our approach focuses on...").
- Do NOT sound like an instructor teaching students.
- Emphasize research motivation, methodology, novel contributions, and empirical results.
- For charts, graphs, and benchmark tables, highlight key trends and scientific insights rather than merely reading numbers.
- Do NOT use Markdown, asterisks, brackets, or bullet formatting. Output plain spoken presentation script only.
"""

OVERVIEW_SYSTEM_EN_RESEARCH = """You are a researcher outlining a conference presentation.
Summarize the core narrative: problem definition, key technical innovation, experimental findings, and overall impact.
"""

ALIGN_SYSTEM_JA = """あなたはプレゼンテーション動画の視線誘導（レーザーポインタ位置）を設計するアシスタントです．
スライド画像と，スライド上の要素一覧（通常テキスト，プログラムの各行「Line N: ...」，図・画像領域），および発表ナレーション文一覧が与えられます．

スライド画像とテキスト内容を確認し，ナレーションの各文が「スライド内のどの要素・どの行を見せようとしているか」を判定してください．

判定方針：
1. プログラム・ソースコードへの言及：該当する code_line（Line N）の block_id をピンポイントで選んでください．
2. 図・画像領域への言及：該当する図・画像領域の block_id を選んでください．
3. 通常テキスト・箇条書きへの言及：該当するテキストブロックの block_id を選んでください．
4. 冒頭の挨拶やつなぎの言葉など，特定の要素を指すのが明らかに不自然な文のみ null にしてください．それ以外は文脈上もっとも近い block_id を割り当ててください．

必ず以下の形式のJSON配列のみを返してください：
[
  {"sentence_index": 0, "block_id": 1},
  {"sentence_index": 1, "block_id": 2}
]
"""

ALIGN_SYSTEM_EN = """You are an assistant designing laser-pointer focus for a slide presentation video.
Given slide elements and narration sentences, assign each sentence to the most relevant block_id (code_line, image, or text).
Assign null only for broad opening or transitional phrases that have no visual anchor.

Return ONLY a JSON array:
[
  {"sentence_index": 0, "block_id": 1},
  {"sentence_index": 1, "block_id": 2}
]
"""

TRANSLATE_TO_EN_SYSTEM = """You are a professional technical translator for academic lectures and conference presentations.
Translate the following presentation narration sentences into natural, clear, and concise English subtitles.
Keep technical terms, commands, and code identifiers accurate.

Return ONLY a JSON array of translated strings matching the order and length of the input array.
"""

TRANSLATE_TO_JA_SYSTEM = """あなたは学術講義および研究発表の専門翻訳者です．
以下の英語ナレーション文を，自然で簡潔な日本語の字幕テキストに翻訳してください．
専門用語やコード識別子の正確さを保ってください．

Return ONLY a JSON array of translated strings matching the order and length of the input array.
"""


# =====================================================================
# ファイルシステム／アトミック書き出し用ヘルパー
# =====================================================================

def emit_progress(phase: str, current: int, total: int, page: int | None = None, message: str = "", reused: bool = False) -> None:
    """GUIおよび外部プロセス向けの構造化進捗イベントを出力します．"""
    payload = {
        "phase": phase,
        "current": current,
        "total": total,
        "page": page,
        "message": message,
        "reused": reused,
    }
    print(f"[PROGRESS] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def atomic_write_text(target: Path, text: str, encoding: str = "utf-8") -> None:
    """一時ファイルを経由してアトミックにテキストを書き出します．"""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_name(f".{target.stem}.tmp{target.suffix}")
    try:
        temp_target.write_text(text, encoding=encoding)
        temp_target.replace(target)
    finally:
        if temp_target.exists():
            temp_target.unlink()


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg が見つかりません．PATHを確認してください．")


@lru_cache(maxsize=1)
def detect_best_video_encoder() -> tuple[str, list[str]]:
    """実行環境をチェックし、利用可能な最適な動画エンコーダとオプションを返します．"""
    try:
        res = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
        encoders = res.stdout
    except Exception:
        return "libx264", ["-preset", "ultrafast"]

    if "h264_nvenc" in encoders:
        test_cmd = [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "nullsrc=s=128x128:d=0.1",
            "-c:v", "h264_nvenc", "-f", "null", "-"
        ]
        if subprocess.run(test_cmd).returncode == 0:
            print("[ENCODER] ハードウェアアクセラレーション: NVIDIA NVENC を使用します．")
            return "h264_nvenc", ["-preset", "p1"]

    if "h264_videotoolbox" in encoders:
        test_cmd = [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "nullsrc=s=128x128:d=0.1",
            "-c:v", "h264_videotoolbox", "-f", "null", "-"
        ]
        if subprocess.run(test_cmd).returncode == 0:
            print("[ENCODER] ハードウェアアクセラレーション: Apple VideoToolbox を使用します．")
            return "h264_videotoolbox", ["-realtime", "1"]

    if "h264_qsv" in encoders:
        test_cmd = [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "nullsrc=s=128x128:d=0.1",
            "-c:v", "h264_qsv", "-f", "null", "-"
        ]
        if subprocess.run(test_cmd).returncode == 0:
            print("[ENCODER] ハードウェアアクセラレーション: Intel Quick Sync Video (QSV) を使用します．")
            return "h264_qsv", ["-preset", "veryfast"]

    print("[ENCODER] ソフトウェアエンコード: libx264 (ultrafast) を使用します．")
    return "libx264", ["-preset", "ultrafast"]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_project_json(project_dir: Path) -> dict:
    p = project_dir / "project.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_project_json(project_dir: Path, data: dict) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    p = project_dir / "project.json"
    atomic_write_text(p, json.dumps(data, ensure_ascii=False, indent=2))


def make_client(cfg: dict) -> OpenAI:
    http_client = httpx.Client(
        trust_env=False,
        verify=False,
    )
    return OpenAI(
        base_url=cfg["base_url"],
        api_key=cfg.get("api_key", "dummy"),
        http_client=http_client,
    )


def apply_tts_filter(text: str, filter_config_path: Path = Path("tts_filter.yaml")) -> str:
    if not filter_config_path.exists():
        return text

    try:
        filter_cfg = load_tts_filter_config(filter_config_path)
        client = make_tts_filter_client(filter_cfg)
        model = filter_cfg.get("server", {}).get("model")
        if not model:
            return text
        system_prompt = build_tts_filter_prompt(filter_cfg)
        generation = filter_cfg.get("generation", {})

        return tts_filter_transform(
            client=client,
            model=model,
            system_prompt=system_prompt,
            text=text,
            generation=generation,
        )
    except Exception as e:
        print(f"[TTS_FILTER WARN] フィルタ処理中にエラーが発生しました: {e}")
        return text


def parse_page_ranges(spec: str) -> set[int]:
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start_s, end_s = part.split("-", 1)
                pages.update(range(int(start_s), int(end_s) + 1))
            except ValueError:
                continue
        else:
            try:
                pages.add(int(part))
            except ValueError:
                continue
    return pages


def pdf_to_images(pdf: Path, out_dir: Path, dpi: int, force: bool) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf)
    result = []
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)

    for i, page in enumerate(doc, 1):
        out = out_dir / f"{i:03d}.png"
        result.append(out)
        if out.exists() and not force:
            continue
        temp_out = out.with_name(f".{out.stem}.tmp.png")
        try:
            page.get_pixmap(matrix=matrix, alpha=False).save(temp_out)
            temp_out.replace(out)
            print(f"[PDF] {out}")
        finally:
            if temp_out.exists():
                temp_out.unlink()

    return result


def extract_page_text(pdf: Path) -> list[str]:
    doc = fitz.open(pdf)
    return [page.get_text("text").strip() for page in doc]


def extract_page_blocks(pdf: Path, page_index: int, dpi: int) -> list[dict]:
    doc = fitz.open(pdf)
    page = doc[page_index]
    scale = dpi / 72.0
    targets = []

    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        b_type = block.get("type", 0)
        bbox = [int(v * scale) for v in block.get("bbox", [0, 0, 0, 0])]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        if w < 20 or h < 10:
            continue

        if b_type == 0:
            lines = block.get("lines", [])
            full_text = "".join(
                "".join(span.get("text", "") for span in line.get("spans", []))
                for line in lines
            ).strip()

            if not full_text:
                continue

            is_code_like = len(lines) >= 3 and any(
                c in full_text for c in [";", "{", "}", "#include", "def ", "return", "=", "(", ")", "->"]
            )

            if is_code_like:
                for line_idx, line in enumerate(lines, 1):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                    if not line_text:
                        continue
                    l_bbox = [int(v * scale) for v in line.get("bbox", [0, 0, 0, 0])]
                    px = max(10, l_bbox[0] - 16)
                    py = l_bbox[1] + (l_bbox[3] - l_bbox[1]) // 2

                    targets.append({
                        "block_id": len(targets),
                        "type": "code_line",
                        "text": f"Line {line_idx}: {line_text}",
                        "x": px,
                        "y": py,
                        "bbox": l_bbox,
                    })
            else:
                px = max(10, bbox[0] - 16)
                py = bbox[1] + int(10 * scale)
                targets.append({
                    "block_id": len(targets),
                    "type": "text",
                    "text": full_text,
                    "x": px,
                    "y": py,
                    "bbox": bbox,
                })

        elif b_type == 1:
            px = bbox[0] + w // 2
            py = bbox[1] + h // 2
            targets.append({
                "block_id": len(targets),
                "type": "image",
                "text": "[図・画像領域]",
                "x": px,
                "y": py,
                "bbox": bbox,
            })

    existing_boxes = [t["bbox"] for t in targets]
    for img_info in page.get_images():
        xref = img_info[0]
        rects = page.get_image_rects(xref)
        for r in rects:
            bbox = [int(r.x0 * scale), int(r.y0 * scale), int(r.x1 * scale), int(r.y1 * scale)]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < 40 or h < 40:
                continue

            is_dup = any(
                abs(bbox[0] - eb[0]) < 20 and abs(bbox[1] - eb[1]) < 20
                for eb in existing_boxes
            )
            if not is_dup:
                px = bbox[0] + w // 2
                py = bbox[1] + h // 2
                targets.append({
                    "block_id": len(targets),
                    "type": "image",
                    "text": "[挿入画像・図表]",
                    "x": px,
                    "y": py,
                    "bbox": bbox,
                })
                existing_boxes.append(bbox)

    return targets


def split_sentences(text: str, lang: str = "ja") -> list[str]:
    if lang == "ja":
        raw_sentences = re.split(r'(?<=[。！？\n])', text)
    else:
        raw_sentences = re.split(r'(?<=[.!?\n])\s+', text)
    return [s.strip() for s in raw_sentences if s.strip()]


def get_audio_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(p.stdout.strip())
    except Exception:
        return 5.0


def format_srt_time(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def split_ja_subtitle_chunks(text: str, max_chars_per_line: int = 24, max_lines_per_chunk: int = 2) -> list[str]:
    text = text.strip()
    if not text:
        return []

    lines = []
    if "、" in text:
        parts = text.split("、")
        current_line = ""
        for i, part in enumerate(parts):
            suffix = "、" if i < len(parts) - 1 else ""
            chunk = part + suffix

            while len(chunk) > max_chars_per_line:
                avail = max_chars_per_line - len(current_line)
                if avail > 0:
                    current_line += chunk[:avail]
                    lines.append(current_line)
                    chunk = chunk[avail:]
                    current_line = ""
                else:
                    lines.append(current_line)
                    current_line = ""

            if not current_line:
                current_line = chunk
            elif len(current_line) + len(chunk) <= max_chars_per_line:
                current_line += chunk
            else:
                lines.append(current_line)
                current_line = chunk
        if current_line:
            lines.append(current_line)
    else:
        while len(text) > max_chars_per_line:
            lines.append(text[:max_chars_per_line])
            text = text[max_chars_per_line:]
        if text:
            lines.append(text)

    chunks = []
    for i in range(0, len(lines), max_lines_per_chunk):
        chunks.append("\n".join(lines[i : i + max_lines_per_chunk]))
    return chunks


def split_en_subtitle_chunks(text: str, max_chars_per_line: int = 48, max_lines_per_chunk: int = 2) -> list[str]:
    text = text.strip()
    if not text:
        return []

    words = text.split()
    lines = []
    current_line = ""

    for w in words:
        if not current_line:
            current_line = w
        elif len(current_line) + 1 + len(w) <= max_chars_per_line:
            current_line += " " + w
        else:
            lines.append(current_line)
            current_line = w

    if current_line:
        lines.append(current_line)

    chunks = []
    for i in range(0, len(lines), max_lines_per_chunk):
        chunks.append("\n".join(lines[i : i + max_lines_per_chunk]))
    return chunks


def translate_sentences(client: OpenAI, cfg: dict, sentences: list[str], target_lang: str) -> list[str]:
    if not sentences:
        return []

    sys_prompt = TRANSLATE_TO_JA_SYSTEM if target_lang == "ja" else TRANSLATE_TO_EN_SYSTEM
    prompt = json.dumps(sentences, ensure_ascii=False)
    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            temperature=0.1,
            max_tokens=cfg.get("max_tokens", 8000),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            extra_body={"reasoning_effort": "none"} 
        )
        translations = extract_json(response.choices[0].message.content)
        if isinstance(translations, list) and len(translations) == len(sentences):
            return [str(s).strip() for s in translations]
    except Exception as e:
        print(f"[WARN] 字幕翻訳 ({target_lang}) 中にエラーが発生しました: {e}")

    return sentences


def create_full_srt(
    all_page_alignments: list[tuple[float, list[dict]]],
    out_srt_file: Path,
    lang: str,
) -> None:
    srt_lines = []
    sub_idx = 1

    for offset, alignments in all_page_alignments:
        for item in alignments:
            t_start = offset + item["start"]
            t_end = offset + item["end"]
            duration = t_end - t_start

            if duration <= 0:
                continue

            if lang == "ja":
                text = item.get("ja_sentence", item.get("sentence", "")).strip()
                chunks = split_ja_subtitle_chunks(text, max_chars_per_line=24, max_lines_per_chunk=2)
            else:
                text = item.get("en_sentence", item.get("sentence", "")).strip()
                chunks = split_en_subtitle_chunks(text, max_chars_per_line=48, max_lines_per_chunk=2)

            if not chunks or not text:
                continue

            total_chars = sum(len(c.replace("\n", " ")) for c in chunks)
            chunk_start = t_start

            for c in chunks:
                c_len = max(1, len(c.replace("\n", " ")))
                c_duration = duration * (c_len / total_chars)
                chunk_end = min(t_end, chunk_start + c_duration)

                srt_lines.append(str(sub_idx))
                srt_lines.append(f"{format_srt_time(chunk_start)} --> {format_srt_time(chunk_end)}")
                srt_lines.append(c)
                srt_lines.append("")

                sub_idx += 1
                chunk_start = chunk_end

    atomic_write_text(out_srt_file, "\n".join(srt_lines) + "\n")


def create_laser_dot_image(out_path: Path, radius: int = 14) -> Path:
    if out_path.exists():
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(f".{out_path.stem}.tmp.png")
    size = radius * 2
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for r in range(radius, 0, -1):
        ratio = r / float(radius)
        alpha = int(240 * (1.0 - ratio ** 1.8))
        if r <= max(2, radius // 3):
            color = (255, 140, 140, 250)
        else:
            color = (255, 30, 30, alpha)
        draw.ellipse([radius - r, radius - r, radius + r, radius + r], fill=color)

    try:
        img.save(temp_path, format="PNG")
        temp_path.replace(out_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return out_path


def image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def extract_json(text: str) -> dict | list:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start_brace = text.find("{")
    start_bracket = text.find("[")

    if start_brace == -1 and start_bracket == -1:
        raise ValueError("LLMからJSONを取得できませんでした．")

    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        end = text.rfind("}")
        return json.loads(text[start_brace:end + 1])
    else:
        end = text.rfind("]")
        return json.loads(text[start_bracket:end + 1])


def align_narration_with_blocks(
    client: OpenAI,
    cfg: dict,
    image_path: Path,
    blocks: list[dict],
    sentences: list[str],
    lang: str = "ja",
) -> list[dict]:
    if not blocks or not sentences:
        return [{"sentence": s, "block_id": None} for s in sentences]

    block_data = []
    for b in blocks:
        item = {
            "block_id": b["block_id"],
            "type": b["type"],
            "bbox": b["bbox"],
        }
        if b["type"] in ("text", "code_line"):
            item["text"] = b["text"][:150]
        else:
            item["description"] = "スライド内の画像・図表領域" if lang == "ja" else "Slide image/figure region"
        block_data.append(item)

    sentence_data = [
        {"sentence_index": idx, "sentence": s}
        for idx, s in enumerate(sentences)
    ]

    header_block = "スライド内の要素・行一覧:" if lang == "ja" else "Slide elements & code lines:"
    header_sent = "ナレーションの文一覧:" if lang == "ja" else "Narration sentences:"

    prompt_text = f"""{header_block}
{json.dumps(block_data, ensure_ascii=False, indent=2)}

{header_sent}
{json.dumps(sentence_data, ensure_ascii=False, indent=2)}
"""

    content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
    ]

    sys_prompt = ALIGN_SYSTEM_JA if lang == "ja" else ALIGN_SYSTEM_EN

    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            temperature=0.0,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": content},
            ],
            extra_body={"reasoning_effort": "none"} 
        )

        data = extract_json(response.choices[0].message.content)
        if isinstance(data, dict) and "alignments" in data:
            data = data["alignments"]

        mapping = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    s_idx = item.get("sentence_index")
                    b_id = item.get("block_id")
                    if s_idx is not None:
                        mapping[s_idx] = b_id if isinstance(b_id, int) and 0 <= b_id < len(blocks) else None

        result = []
        last_block_id = None

        for i, s in enumerate(sentences):
            bid = mapping.get(i, None)
            if bid is not None:
                last_block_id = bid
            elif last_block_id is not None and len(s) > 10:
                bid = last_block_id

            result.append({
                "sentence": s,
                "block_id": bid,
            })
        return result
    except Exception as e:
        print(f"[ALIGN WARN] アライメント解析でエラーが発生しました: {e}")
        return [{"sentence": s, "block_id": None} for s in sentences]


def build_pointer_schedule(
    blocks: list[dict],
    alignments: list[dict],
    duration: float,
) -> tuple[list[dict], list[dict]]:
    if not alignments or duration <= 0:
        return [], []

    block_map = {b["block_id"]: b for b in blocks}
    total_weight = sum(
        max(1, len(item.get("tts_sentence", item["sentence"])))
        for item in alignments
    )
    cur_t = 0.0
    schedule = []
    subtitle_timing = []

    for item in alignments:
        s_weight = max(1, len(item.get("tts_sentence", item["sentence"])))
        b_id = item.get("block_id")
        span = (s_weight / total_weight) * duration
        t_start = cur_t
        t_end = min(duration, cur_t + span)
        cur_t = t_end

        timing_entry = {
            "start": t_start,
            "end": t_end,
            "sentence": item["sentence"],
        }
        if "ja_sentence" in item:
            timing_entry["ja_sentence"] = item["ja_sentence"]
        if "en_sentence" in item:
            timing_entry["en_sentence"] = item["en_sentence"]

        subtitle_timing.append(timing_entry)

        if b_id is not None and b_id in block_map:
            target = block_map[b_id]
            schedule.append({
                "start": round(t_start, 2),
                "end": round(t_end, 2),
                "x": target["x"],
                "y": target["y"],
            })

    return schedule, subtitle_timing


def get_narration_prompt(mode: str, lang: str) -> str:
    if lang == "ja":
        return NARRATION_SYSTEM_JA_RESEARCH if mode == "research" else NARRATION_SYSTEM_JA_LECTURE
    else:
        return NARRATION_SYSTEM_EN_RESEARCH if mode == "research" else NARRATION_SYSTEM_EN_LECTURE


def get_overview_prompt(mode: str, lang: str) -> str:
    if lang == "ja":
        return OVERVIEW_SYSTEM_JA_RESEARCH if mode == "research" else OVERVIEW_SYSTEM_JA_LECTURE
    else:
        return OVERVIEW_SYSTEM_EN_RESEARCH if mode == "research" else OVERVIEW_SYSTEM_EN_LECTURE


def build_course_overview(
    client: OpenAI,
    cfg: dict,
    page_texts: list[str],
    mode: str = "lecture",
    lang: str = "ja",
) -> str:
    material = "\n\n".join(
        f"--- Slide {i} ---\n{text[:5000]}"
        for i, text in enumerate(page_texts, 1)
    )

    sys_prompt = get_overview_prompt(mode, lang)

    response = client.chat.completions.create(
        model=cfg["model"],
        temperature=cfg.get("temperature", 0.2),
        max_tokens=cfg.get("max_tokens", 8000),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": material},
        ],
    )
    return response.choices[0].message.content.strip()


def make_explanation(
    client: OpenAI,
    cfg: dict,
    image: Path,
    current_page: int,
    current_text: str,
    prev_page: int | None,
    prev_text: str | None,
    next_page: int | None,
    next_text: str | None,
    previous_explanation: str,
    course_overview: str,
    total_active: int,
    active_idx: int,
    mode: str = "lecture",
    lang: str = "ja",
) -> str:
    if lang == "ja":
        title_overview = "発表全体の概要：" if mode == "research" else "講義全体の概要："
        instruction_line = "このスライドの研究発表ナレーションを作成してください．" if mode == "research" else "このスライドの講義ナレーションを作成してください．"

        prev_section = f"前のスライド (P.{prev_page}) のテキスト:\n{prev_text[:2000]}\n\n前スライドのナレーション:\n{previous_explanation[-2000:]}\n" if prev_page else "（ここが最初のスライドです）\n"
        next_section = f"次のスライド (P.{next_page}) のテキスト（予告や橋渡しに活用可）:\n{next_text[:2000]}\n" if next_page else "（ここが最後のスライドです）\n"

        prompt = f"""{title_overview}
{course_overview}

前後スライドの流れ：
{prev_section}
{next_section}
現在のスライド (P.{current_page}) のテキスト：
{current_text[:8000]}

現在は対象{total_active}枚中、{active_idx + 1}枚目のスライドです．

{instruction_line}
出力はナレーション本文だけにしてください．
"""
    else:
        title_overview = "Presentation Overview:" if mode == "research" else "Lecture Overview:"
        instruction_line = "Create a professional spoken English conference presentation narration for this slide." if mode == "research" else "Create a natural spoken English lecture narration for this slide."

        prev_section = f"Previous Slide (P.{prev_page}) Text:\n{prev_text[:2000]}\n\nPrevious Narration:\n{previous_explanation[-2000:]}\n" if prev_page else "(This is the first slide)\n"
        next_section = f"Next Slide (P.{next_page}) Text:\n{next_text[:2000]}\n" if next_page else "(This is the last slide)\n"

        prompt = f"""{title_overview}
{course_overview}

Context & Slide Flow:
{prev_section}
{next_section}
Current Slide (P.{current_page}) Text:
{current_text[:8000]}

Currently presenting slide {active_idx + 1} of {total_active}.

{instruction_line}
Return ONLY the raw narration text.
"""

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_data_url(image)}},
    ]

    sys_prompt = get_narration_prompt(mode, lang)

    response = client.chat.completions.create(
        model=cfg["model"],
        temperature=cfg.get("temperature", 0.3),
        max_tokens=cfg.get("max_tokens", 5000),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content},
        ],
    )
    return response.choices[0].message.content.strip()


def generate_single_explanation(
    client: OpenAI,
    cfg: dict,
    pdf: Path,
    images: list[Path],
    out_dir: Path,
    page: int,
    active_pages: list[int] | None = None,
    mode: str = "lecture",
    lang: str = "ja",
) -> str:
    """単一スライドのナレーション原稿を前後の文脈を考慮して生成・保存します．"""
    out_dir.mkdir(parents=True, exist_ok=True)
    page_texts = extract_page_text(pdf)

    overview_path = out_dir / "_course_overview.txt"
    if overview_path.exists():
        overview = overview_path.read_text(encoding="utf-8")
    else:
        overview = build_course_overview(client, cfg, page_texts, mode=mode, lang=lang)
        atomic_write_text(overview_path, overview + "\n")

    pages_list = active_pages if active_pages is not None else list(range(1, len(images) + 1))
    cur_idx = pages_list.index(page) if page in pages_list else 0

    prev_p = pages_list[cur_idx - 1] if cur_idx > 0 else None
    next_p = pages_list[cur_idx + 1] if cur_idx + 1 < len(pages_list) else None

    prev_text = page_texts[prev_p - 1] if prev_p is not None else None
    next_text = page_texts[next_p - 1] if next_p is not None else None

    prev_exp_file = out_dir / f"{prev_p:03d}.txt" if prev_p else None
    prev_explanation = (
        prev_exp_file.read_text(encoding="utf-8").strip()
        if prev_exp_file and prev_exp_file.exists()
        else ""
    )

    image_path = images[page - 1]
    new_text = make_explanation(
        client=client,
        cfg=cfg,
        image=image_path,
        current_page=page,
        current_text=page_texts[page - 1],
        prev_page=prev_p,
        prev_text=prev_text,
        next_page=next_p,
        next_text=next_text,
        previous_explanation=prev_explanation,
        course_overview=overview,
        total_active=len(pages_list),
        active_idx=cur_idx,
        mode=mode,
        lang=lang,
    )

    out_file = out_dir / f"{page:03d}.txt"
    atomic_write_text(out_file, new_text + "\n")
    return new_text


def generate_single_alignment(
    client: OpenAI,
    cfg: dict,
    pdf: Path,
    page_num: int,
    image_path: Path,
    text: str,
    dpi: int,
    lang: str,
) -> dict:
    blocks = extract_page_blocks(pdf, page_num - 1, dpi)
    sentences = split_sentences(text, lang=lang)
    alignments = align_narration_with_blocks(client, cfg, image_path, blocks, sentences, lang=lang)

    target_lang = "en" if lang == "ja" else "ja"
    translations = translate_sentences(client, cfg, sentences, target_lang=target_lang)

    for s_idx, a_item in enumerate(alignments):
        trans_text = translations[s_idx] if s_idx < len(translations) else a_item["sentence"]
        if lang == "ja":
            a_item["ja_sentence"] = a_item["sentence"]
            a_item["en_sentence"] = trans_text
        else:
            a_item["en_sentence"] = a_item["sentence"]
            a_item["ja_sentence"] = trans_text

    return {
        "page": page_num,
        "blocks": blocks,
        "alignments": alignments,
    }


def generate_single_tts(
    text_path: Path,
    out_path: Path,
    tts_cfg: dict,
    filter_config_path: Path,
    force: bool,
    lang: str = "ja",
) -> None:
    """単一スライドの音声をTTSエンジンで合成・保存します．"""
    generate_tts(
        text_path=text_path,
        out_path=out_path,
        tts_cfg=tts_cfg,
        filter_config_path=filter_config_path,
        force=force,
        lang=lang,
    )


def generate_single_page_video(
    image: Path,
    audio: Path,
    out: Path,
    schedule: list[dict],
    laser_img: Path,
    video_cfg: dict,
    force: bool,
) -> None:
    """単一スライドの動画をレンダリングします．"""
    generate_page_video(
        image=image,
        audio=audio,
        out=out,
        schedule=schedule,
        laser_img=laser_img,
        cfg=video_cfg,
        force=force,
    )


def generate_explanations(
    pdf: Path,
    images: list[Path],
    out_dir: Path,
    cfg: dict,
    force: bool,
    mode: str = "lecture",
    lang: str = "ja",
    active_pages: set[int] | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(cfg)
    page_texts = extract_page_text(pdf)

    overview_path = out_dir / "_course_overview.txt"
    if overview_path.exists() and not force:
        overview = overview_path.read_text(encoding="utf-8")
    else:
        overview = build_course_overview(client, cfg, page_texts, mode=mode, lang=lang)
        atomic_write_text(overview_path, overview + "\n")

    result = []
    previous_explanation = ""

    active_indices = [
        idx for idx in range(len(images))
        if active_pages is None or (idx + 1) in active_pages
    ]

    for k, i_idx in enumerate(active_indices):
        i = i_idx + 1
        image = images[i_idx]
        out = out_dir / f"{i:03d}.txt"
        result.append(out)

        prev_idx = active_indices[k - 1] if k > 0 else None
        next_idx = active_indices[k + 1] if k < len(active_indices) - 1 else None

        if out.exists() and not force:
            previous_explanation = out.read_text(encoding="utf-8").strip()
            emit_progress("explain", k + 1, len(active_indices), page=i, message=f"スライド {i}（{k + 1}/{len(active_indices)}）", reused=True)
            print(f"[LLM] reuse narration {out}")
            continue

        emit_progress("explain", k + 1, len(active_indices), page=i, message=f"スライド {i}（{k + 1}/{len(active_indices)}）")
        print(f"[LLM] ナレーション生成: page {i}/{len(images)} ({k + 1}/{len(active_indices)}) (mode={mode}, lang={lang})")

        text = make_explanation(
            client,
            cfg,
            image,
            current_page=i,
            current_text=page_texts[i_idx],
            prev_page=prev_idx + 1 if prev_idx is not None else None,
            prev_text=page_texts[prev_idx] if prev_idx is not None else None,
            next_page=next_idx + 1 if next_idx is not None else None,
            next_text=page_texts[next_idx] if next_idx is not None else None,
            previous_explanation=previous_explanation,
            course_overview=overview,
            total_active=len(active_indices),
            active_idx=k,
            mode=mode,
            lang=lang,
        )
        atomic_write_text(out, text + "\n")
        previous_explanation = text

    return result


def generate_alignments(
    pdf: Path,
    images: list[Path],
    out_dir: Path,
    dpi: int,
    cfg: dict,
    force: bool,
    lang: str = "ja",
    active_pages: set[int] | None = None,
) -> None:
    client = make_client(cfg)
    active_indices = [
        idx for idx in range(len(images))
        if active_pages is None or (idx + 1) in active_pages
    ]

    for k, i_idx in enumerate(active_indices, 1):
        i = i_idx + 1
        image = images[i_idx]
        text_file = out_dir / f"{i:03d}.txt"
        align_out = out_dir / f"{i:03d}_align.json"

        if not text_file.exists():
            continue

        if align_out.exists() and not force:
            emit_progress("align", k, len(active_indices), page=i, message=f"スライド {i}（{k}/{len(active_indices)}）", reused=True)
            print(f"[ALIGN] reuse {align_out}")
            continue

        emit_progress("align", k, len(active_indices), page=i, message=f"スライド {i}（{k}/{len(active_indices)}）")
        print(f"[ALIGN] 字幕＆ポインタ解析: page {i}/{len(images)}")
        raw_text = text_file.read_text(encoding="utf-8").strip()
        align_data = generate_single_alignment(client, cfg, pdf, i, image, raw_text, dpi, lang)
        atomic_write_text(align_out, json.dumps(align_data, ensure_ascii=False, indent=2))


def generate_tts(
    text_path: Path,
    out_path: Path,
    tts_cfg: dict,
    filter_config_path: Path,
    force: bool,
    lang: str = "ja",
) -> None:
    if out_path.exists() and not force:
        print(f"[TTS] reuse {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_out = out_path.with_name(f".{out_path.stem}.tmp.mp3")

    try:
        client = make_client(tts_cfg)
        raw_text = text_path.read_text(encoding="utf-8").strip()

        tts_text = apply_tts_filter(raw_text, filter_config_path) if lang == "ja" else raw_text

        response = client.audio.speech.create(
            model=tts_cfg["model"],
            voice=tts_cfg["voice"],
            input=tts_text,
            response_format=tts_cfg.get("response_format", "mp3"),
        )
        response.write_to_file(temp_out)
        temp_out.replace(out_path)
        print(f"[TTS:{lang}] {out_path}")
    finally:
        if temp_out.exists():
            temp_out.unlink()


def generate_page_video(
    image: Path,
    audio: Path,
    out: Path,
    schedule: list[dict],
    laser_img: Path,
    cfg: dict,
    force: bool,
) -> None:
    if out.exists() and not force:
        print(f"[VIDEO] reuse {out}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    temp_out = out.with_name(f".{out.stem}.tmp.mp4")
    filter_script = out.parent / f".{out.stem}_filter.txt"

    vcodec, encoder_opts = detect_best_video_encoder()

    crf = str(cfg.get("crf", 20))
    fps = str(cfg.get("fps", 15))
    audio_codec = str(cfg.get("audio_codec", "aac"))
    audio_bitrate = str(cfg.get("audio_bitrate", "192k"))

    quality_opts = {
        "libx264": ["-crf", crf],
        "h264_nvenc": ["-cq", crf],
        "h264_qsv": ["-global_quality", crf],
    }.get(vcodec, [])

    try:
        if not schedule:
            cmd = [
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-loop", "1",
                "-i", str(image),
                "-i", str(audio),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", vcodec,
                *encoder_opts,
                *quality_opts,
                "-r", fps,
                "-pix_fmt", "yuv420p",
                "-c:a", audio_codec,
                "-b:a", audio_bitrate,
                "-shortest",
                "-movflags", "+faststart",
                "-f", "mp4",
                str(temp_out),
            ]
            run(cmd)
        else:
            expr_x_parts = [
                f"between(t,{s['start']},{s['end']})*({s['x']}+2*sin(4*PI*t))"
                for s in schedule
            ]
            expr_y_parts = [
                f"between(t,{s['start']},{s['end']})*({s['y']}+2*cos(4*PI*t))"
                for s in schedule
            ]
            expr_x = "+".join(expr_x_parts) if expr_x_parts else "-100"
            expr_y = "+".join(expr_y_parts) if expr_y_parts else "-100"

            filter_complex = (
                f"[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2[bg];\n"
                f"[bg][1:v]overlay=x='{expr_x}':y='{expr_y}':eval=frame[v]\n"
            )

            filter_script.write_text(filter_complex, encoding="utf-8")

            cmd = [
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-loop", "1", "-i", str(image),
                "-loop", "1", "-i", str(laser_img),
                "-i", str(audio),
                "-filter_complex_script", str(filter_script),
                "-map", "[v]",
                "-map", "2:a:0",
                "-c:v", vcodec,
                *encoder_opts,
                *quality_opts,
                "-r", fps,
                "-pix_fmt", "yuv420p",
                "-c:a", audio_codec,
                "-b:a", audio_bitrate,
                "-shortest",
                "-movflags", "+faststart",
                "-f", "mp4",
                str(temp_out),
            ]
            run(cmd)

        temp_out.replace(out)
        print(f"[VIDEO] {out}")
    finally:
        if filter_script.exists():
            filter_script.unlink()
        if temp_out.exists():
            temp_out.unlink()


def create_chapter_metadata(videos: list[Path], slide_indices: list[int], out_metadata_file: Path) -> None:
    lines = [";FFMETADATA1"]
    current_time_ms = 0

    for s_idx, video in zip(slide_indices, videos):
        duration_sec = get_audio_duration(video)
        duration_ms = int(duration_sec * 1000)
        end_time_ms = current_time_ms + duration_ms

        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={current_time_ms}")
        lines.append(f"END={end_time_ms}")
        lines.append(f"title=Slide {s_idx}")

        current_time_ms = end_time_ms

    atomic_write_text(out_metadata_file, "\n".join(lines) + "\n")


def concat_videos(
    videos: list[Path],
    slide_indices: list[int],
    ja_srt: Path | None,
    en_srt: Path | None,
    out: Path,
    force: bool,
    primary_lang: str = "ja",
) -> None:
    if out.exists() and not force:
        print(f"[CONCAT] reuse {out}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    temp_out = out.with_name(f".{out.stem}.tmp.mp4")

    list_file = out.parent / "concat.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for video in videos:
            f.write(f"file '{video.resolve().as_posix()}'\n")

    meta_file = out.parent / "metadata.txt"
    create_chapter_metadata(videos, slide_indices, meta_file)

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
    ]

    input_count = 1
    tracks_to_add = [("ja", ja_srt), ("en", en_srt)] if primary_lang == "ja" else [("en", en_srt), ("ja", ja_srt)]
    track_map = []

    for l_code, srt_p in tracks_to_add:
        if srt_p and srt_p.exists():
            cmd.extend(["-i", str(srt_p)])
            track_map.append((l_code, input_count))
            input_count += 1

    cmd.extend(["-i", str(meta_file)])
    meta_idx = input_count

    cmd.extend(["-map_metadata", str(meta_idx)])
    cmd.extend(["-map", "0:v", "-map", "0:a"])

    for s_idx, (l_code, file_in_idx) in enumerate(track_map):
        cmd.extend(["-map", f"{file_in_idx}:0"])
        lang_code = "jpn" if l_code == "ja" else "eng"
        title_text = "Japanese" if l_code == "ja" else "English"
        cmd.extend([
            f"-metadata:s:s:{s_idx}", f"language={lang_code}",
            f"-metadata:s:s:{s_idx}", f"title={title_text}",
        ])

    cmd.extend(["-c:v", "copy", "-c:a", "copy"])
    if track_map:
        cmd.extend(["-c:s", "mov_text"])

    cmd.extend([
        "-movflags", "+faststart",
        "-f", "mp4",
        str(temp_out),
    ])

    try:
        run(cmd)
        temp_out.replace(out)
    finally:
        if temp_out.exists():
            temp_out.unlink()


STAGES = ("pdf", "explain", "align", "tts", "video")


def _force_cleanup(root: Path, pdf: Path, start: str) -> None:
    """開始ステージに応じて，そのステージ以降の生成物を初期化します．"""
    print(f"[FORCE CLEANUP] 開始ステージ '{start}' に応じて下流ファイルを削除・初期化します．")

    for path in root.glob(f"{pdf.stem}*"):
        if path.is_file():
            path.unlink()

    start_idx = STAGES.index(start)

    directories = {
        "pdf": root / "pages",
        "explain": root / "explanations",
        "tts": root / "audio",
        "video": root / "video",
    }

    for stage, directory in directories.items():
        if start_idx <= STAGES.index(stage) and directory.exists():
            shutil.rmtree(directory)

    if start == "align":
        explanations = root / "explanations"
        if explanations.exists():
            for path in explanations.glob("*_align.json"):
                path.unlink()


def _resolve_active_pages(total_pages: int, pages_spec: str, skip_pages_spec: str) -> list[int]:
    """対象スライドを昇順で返します．"""
    active_pages = set(range(1, total_pages + 1))

    if pages_spec:
        active_pages &= parse_page_ranges(pages_spec)
    if skip_pages_spec:
        active_pages -= parse_page_ranges(skip_pages_spec)

    return sorted(active_pages)


def _run_tts_stage(
    sorted_active_pages: list[int],
    explanations: Path,
    audio: Path,
    tts_cfg: dict,
    filter_config_path: Path,
    force: bool,
    lang: str,
) -> None:
    total = len(sorted_active_pages)

    for index, page_num in enumerate(sorted_active_pages, 1):
        text_file = explanations / f"{page_num:03d}.txt"
        mp3 = audio / f"{page_num:03d}.mp3"
        if not text_file.exists():
            continue

        emit_progress(
            "tts",
            index,
            total,
            page=page_num,
            message=f"スライド {page_num}（{index}/{total}）",
        )
        print(f"[TTS] 音声合成中: page {page_num} ({index}/{total})")
        generate_tts(
            text_file,
            mp3,
            tts_cfg,
            filter_config_path,
            force,
            lang=lang,
        )


def _run_video_stage(
    pdf: Path,
    root: Path,
    pages: Path,
    explanations: Path,
    audio: Path,
    video: Path,
    sorted_active_pages: list[int],
    laser_img: Path,
    video_cfg: dict,
    force: bool,
    lang: str,
) -> None:
    videos: list[Path] = []
    video_slide_indices: list[int] = []
    all_page_subtitles: list[tuple[float, list[dict]]] = []
    accumulated_offset = 0.0
    total = len(sorted_active_pages)

    for index, page_num in enumerate(sorted_active_pages, 1):
        image = pages / f"{page_num:03d}.png"
        mp3_file = audio / f"{page_num:03d}.mp3"
        text_file = explanations / f"{page_num:03d}.txt"
        out = video / f"{page_num:03d}.mp4"
        align_file = explanations / f"{page_num:03d}_align.json"

        if not mp3_file.exists() or not image.exists():
            continue

        emit_progress(
            "video",
            index,
            total,
            page=page_num,
            message=f"スライド {page_num}（{index}/{total}）",
        )
        print(f"[VIDEO] スライド動画生成中: page {page_num} ({index}/{total})")

        duration = get_audio_duration(mp3_file)

        if align_file.exists():
            align_data = json.loads(align_file.read_text(encoding="utf-8"))
            blocks = align_data.get("blocks", [])
            alignments = align_data.get("alignments", [])

            # build_pointer_schedule は tts_sentence があればそれを時間配分に使用する．
            for item in alignments:
                item["tts_sentence"] = item["sentence"]

            schedule, page_subtitles = build_pointer_schedule(
                blocks, alignments, duration
            )
        else:
            text = text_file.read_text(encoding="utf-8").strip() if text_file.exists() else ""
            page_subtitles = [{
                "start": 0.0,
                "end": duration,
                "sentence": text,
                "ja_sentence": text if lang == "ja" else "",
                "en_sentence": text if lang == "en" else "",
            }]
            schedule = []

        all_page_subtitles.append((accumulated_offset, page_subtitles))
        accumulated_offset += duration

        generate_page_video(
            image,
            mp3_file,
            out,
            schedule,
            laser_img,
            video_cfg,
            force,
        )
        videos.append(out)
        video_slide_indices.append(page_num)

    emit_progress("concat", len(videos), len(videos), message="完成動画を結合・生成中…")

    final_ja_srt = root / f"{pdf.stem}_ja.srt"
    final_en_srt = root / f"{pdf.stem}_en.srt"
    create_full_srt(all_page_subtitles, final_ja_srt, lang="ja")
    create_full_srt(all_page_subtitles, final_en_srt, lang="en")

    final = root / f"{pdf.stem}.mp4"
    concat_videos(
        videos,
        video_slide_indices,
        final_ja_srt,
        final_en_srt,
        final,
        force,
        primary_lang=lang,
    )



def main() -> int:
    parser = argparse.ArgumentParser(
        description="PDFから講義・研究発表動画を自動生成します．"
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--from",
        dest="start",
        choices=list(STAGES),
        default="pdf",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=["lecture", "research"], default=None, help="発表種別")
    parser.add_argument("--lang", choices=["ja", "en"], default=None, help="主言語")
    parser.add_argument("--pages", type=str, default=None, help="対象スライド番号または範囲（例: '1-10,12'）")
    parser.add_argument("--skip-pages", type=str, default=None, help="除外スライド番号または範囲（例: '5,11-13'）")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"PDFがありません: {args.pdf}", file=sys.stderr)
        return 1

    check_ffmpeg()
    cfg = load_config(args.config)
    root = args.output or args.pdf.with_name(args.pdf.stem + "_lecture")
    filter_config_path = Path("tts_filter.yaml")

    proj_cfg = load_project_json(root)
    mode = args.mode or proj_cfg.get("mode") or cfg.get("mode", "lecture")
    lang = args.lang or proj_cfg.get("language") or cfg.get("language", "ja")
    pages_spec = args.pages if args.pages is not None else proj_cfg.get("pages", "")
    skip_pages_spec = (
        args.skip_pages
        if args.skip_pages is not None
        else proj_cfg.get("skip_pages", "")
    )

    proj_cfg.update({
        "mode": mode,
        "language": lang,
        "pages": pages_spec,
        "skip_pages": skip_pages_spec,
    })
    save_project_json(root, proj_cfg)

    pages = root / "pages"
    explanations = root / "explanations"
    audio = root / "audio"
    video = root / "video"

    if args.force:
        _force_cleanup(root, args.pdf, args.start)

    for directory in (pages, explanations, audio, video):
        directory.mkdir(parents=True, exist_ok=True)

    dpi = int(cfg["pdf"].get("dpi", 150))
    images = pdf_to_images(args.pdf, pages, dpi, args.force)

    if args.start == "pdf":
        print("PDF変換完了．")
        print("次: python slide_lecture.py lecture.pdf --from explain")
        return 0

    sorted_active_pages = _resolve_active_pages(
        len(images),
        pages_spec,
        skip_pages_spec,
    )
    print(
        f"対象スライド数: {len(sorted_active_pages)} / "
        f"{len(images)} ページ: {sorted_active_pages}"
    )

    if not sorted_active_pages:
        print(
            "対象となるスライドが1枚もありません．指定を確認してください．",
            file=sys.stderr,
        )
        return 1

    active_pages = set(sorted_active_pages)

    generate_explanations(
        args.pdf,
        images,
        explanations,
        cfg["llm"],
        args.force,
        mode=mode,
        lang=lang,
        active_pages=active_pages,
    )

    if args.start == "explain":
        print(f"ナレーション原稿作成完了 (mode={mode}, lang={lang})．")
        print("次: ユーザによる原稿編集 ➔ python slide_lecture.py lecture.pdf --from align")
        return 0

    generate_alignments(
        args.pdf,
        images,
        explanations,
        dpi,
        cfg["llm"],
        args.force,
        lang=lang,
        active_pages=active_pages,
    )

    if args.start == "align":
        print(f"字幕翻訳＆視線誘導アライメント完了 (lang={lang})．")
        print("次: ユーザによる修正 ➔ python slide_lecture.py lecture.pdf --from tts")
        return 0

    tts_all = cfg.get("tts", {})
    tts_cfg = tts_all.get(lang, tts_all)
    _run_tts_stage(
        sorted_active_pages,
        explanations,
        audio,
        tts_cfg,
        filter_config_path,
        args.force,
        lang,
    )

    if args.start == "tts":
        print(f"TTS生成完了 (lang={lang})．")
        print("次: python slide_lecture.py lecture.pdf --from video")
        return 0

    laser_img = root / "laser_dot.png"
    create_laser_dot_image(laser_img, radius=14)

    video_count = _run_video_stage(
        args.pdf,
        root,
        pages,
        explanations,
        audio,
        video,
        sorted_active_pages,
        laser_img,
        cfg["video"],
        args.force,
        lang,
    )

    final = root / f"{args.pdf.stem}.mp4"
    print()
    print("========================================")
    print(
        f"完成しました (種別: {mode}, 主言語: {lang}, "
        f"対象スライド: {video_count}枚)"
    )
    print(final)
    print("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
