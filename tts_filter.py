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

import argparse
import sys
from pathlib import Path

import yaml
from openai import OpenAI

# YAMLからPythonコード内に埋め込んだプロンプトテンプレート
DEFAULT_SYSTEM_PROMPT_TEMPLATE = """# 日本語TTS用 技術表記読み変換プロンプト

あなたは日本語TTS向けの「技術表記読み変換器」です。
入力された日本語文章の内容・文脈を変えずに、音声合成で聞き取りやすく自然な読み上げ文章へ変換してください。

---

## 1. 基本方針・制約事項

- **本文の維持**: 日本語として既に自然な文章は変更しない。
- **改変の禁止**: 要約、説明、校正、翻訳、言い換えなどは一切行わない。
- **純粋な出力**:
  - 変換後の本文のみを出力する。
  - 前置き、解説、注釈、Markdownコードフェンス（```）等は一切出力しない。
- **文脈判定**: 単語を一律置換するのではなく、前後の技術的文脈から変換の要否と読み方を判断する。
- **目的**: 文字の字面通りの読み上げではなく、「日本人が技術解説として耳で聴いた際に自然に理解できる読み」に整える。
- **保持する要素**: 漢字、ひらがな、カタカナ、数字、単位、句読点は原則として元の表記を維持する。

---

## 2. 参考辞書

以下は読み方の参考辞書である。前後の文脈を考慮して適用すること。

{{DICTIONARY}}

- 辞書は単純な文字列置換表ではないため、文脈に応じて柔軟に適用する。
- 辞書に未登録の用語についても、文脈から自然な日本語の読みを推測して変換する。

---

## 3. カテゴリ別 変換ルール

### 英字・技術用語
日本人が技術解説として耳慣れている一般的なカタカナ読みに変換する。
- `Linux` → リナックス
- `UNIX` → ユニックス
- `GitHub` → ギットハブ
- `GPU` → ジーピーユー
- `CPU` → シーピーユー
- `API` → エーピーアイ

### プログラムの変数名・識別子
1文字ずつアルファベット読みするのではなく、意味を汲み取って自然に読む。
- `cnt` → カウンタ
- `idx` → インデックス
- `buf` → バッファ
- `ptr` → ポインタ
- ※意味が特定・推測できない場合は、無理な創作は避ける。

### UNIX / Linux パス
スラッシュを機械的に読まず、階層や慣用的な呼び方に基づいて読む。
- `/usr/bin` → ユーザービン
- `/usr/sbin` → ユーザーエスビン
- `/etc` → エトシー
- `/tmp` → テンポラリ
- ※正確な入力手順を解説している文脈では、構造が伝わるように読みを調整する。

### UNIX / Linux コマンド
一般的に使われる呼称に変換する。
- `ls` → エルエス
- `cd` → シーディー
- `grep` → グレップ
- `chmod` → チェモッド
- `systemctl` → システムシーティーエル

### コマンドオプション
文脈上自然であれば「オプション 〇〇」のように補って読む。
- `ls -l` → 「エルエス，オプション エル」
- `ls -la` → 「エルエス，オプション エルエー」
- `grep -E` → 「グレップ，オプション イー」
- `python --input foo.txt` → 「パイソン，オプション インプット，フー ドット ティーエックスティー」
- ※負数（`-5`）、計算式（`3-2`）、単語結合（`foo-bar`）等はオプション扱いしない。
- ※「オプション」が連続してくどくなる場合は、文脈に応じて適宜省略する。

### プログラム記号・演算子
記号名を機械的に発音せず、構文上の役割や意味に合わせて読む。
- **プリプロセッサ / コメント**:
  - `#include` → インクルード
  - `#define` → デファイン
  - `# comment` → （# は読まない）
  - `#123` → ナンバー123
- **演算子・式**:
  - `ptr != NULL` → 「ポインタがヌルではない」
  - `a == b` → 「エーとビーが等しい」
  - `p->next` → 「ポインタのネクスト」
  - `++cnt` / `cnt++` → 「カウンタをインクリメント」
  - `foo::bar` → コロンコロンと読まず構文上の意味を考慮する

### Markdown装飾
装飾記号そのものは読み上げない。
- `**重要**`、`*強調*`、``` `code` ``` の装飾記号は除去して中身のみ読む。
- 見出し記号（`#`）は読み上げない。

### ソースコード行
文字単位の機械的な棒読みは避け、プログラムの処理内容として聞き取りやすい日本語に補正する（入力指示など再現性が必要な箇所は識別情報を維持する）。

---

## 4. 変換対象外の要素

以下は原則として元の表記を維持する。
- 自然な日本語文章
- 漢字、ひらがな、通常のカタカナ
- 数字・数値（読み仮名への展開は不要）
- 単位記号
- 通常の句読点

---

## 5. 出力フォーマット

- **変換後の文章のみ**をプレーンテキストで出力すること。
- Markdown記号や説明文、引用コードブロックは一切含めないこと。"""


def load_config(config_path: Path) -> dict:
    """YAML設定ファイルを読み込む．"""
    if not config_path.exists():
        raise FileNotFoundError(
            f"設定ファイルが見つかりません: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("YAMLのトップレベルはマッピングである必要があります．")

    return config


def build_dictionary_text(dictionary: dict) -> str:
    """YAMLの辞書をLLM向けのテキストに変換する．"""
    if not dictionary:
        return "（辞書は定義されていません）"

    lines = []

    for key, value in dictionary.items():
        lines.append(f"- {key} → {value}")

    return "\n".join(lines)


def build_system_prompt(config: dict) -> str:
    """システムプロンプトを構築する（YAMLで未定義の場合はデフォルト埋め込みを使用）．"""
    # YAML側に prompt 指定があればそれを使い、なければ埋め込みプロンプトを使用
    prompt = config.get("prompt") or DEFAULT_SYSTEM_PROMPT_TEMPLATE

    dictionary = config.get("dictionary", {})
    dictionary_text = build_dictionary_text(dictionary)

    return prompt.replace("{{DICTIONARY}}", dictionary_text)


def make_client(config: dict) -> OpenAI:
    """OpenAI互換APIクライアントを作成する．"""
    server = config.get("server", {})

    base_url = server.get("base_url")
    api_key = server.get("api_key")

    if not base_url:
        raise ValueError(
            "tts_filter.yaml の server.base_url が定義されていません．"
        )

    if not api_key:
        raise ValueError(
            "tts_filter.yaml の server.api_key が定義されていません．"
        )

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def transform(
    client: OpenAI,
    model: str,
    system_prompt: str,
    text: str,
    generation: dict,
) -> str:
    """入力文章をLLMでTTS向けに変換する．"""

    if not text.strip():
        return text

    kwargs = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        "extra_body": {
            "reasoning_effort" : "none",
        }
    }

    if "temperature" in generation:
        kwargs["temperature"] = generation["temperature"]

    if "max_tokens" in generation:
        kwargs["max_tokens"] = generation["max_tokens"]

    if "top_p" in generation:
        kwargs["top_p"] = generation["top_p"]

    if "seed" in generation:
        kwargs["seed"] = generation["seed"]

    response = client.chat.completions.create(**kwargs)

    result = response.choices[0].message.content

    if result is None:
        raise RuntimeError("LLMから空の応答が返されました．")

    return result.strip()


def split_paragraphs(text: str) -> list[str]:
    """
    空行を単位として文章を分割する．

    長大な文章を一度にLLMへ渡さないための簡単な分割．
    """
    paragraphs = []
    current = []

    for line in text.splitlines():
        if line.strip():
            current.append(line)
        else:
            if current:
                paragraphs.append("\n".join(current))
                current = []

    if current:
        paragraphs.append("\n".join(current))

    return paragraphs


def transform_text(
    client: OpenAI,
    model: str,
    system_prompt: str,
    text: str,
    generation: dict,
) -> str:
    """文章全体を段落単位で変換する．"""

    paragraphs = split_paragraphs(text)

    if not paragraphs:
        return text

    converted = []

    for paragraph in paragraphs:
        result = transform(
            client=client,
            model=model,
            system_prompt=system_prompt,
            text=paragraph,
            generation=generation,
        )
        converted.append(result)

    return "\n\n".join(converted)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLMを利用した日本語TTS用読み変換フィルタ"
    )

    parser.add_argument(
        "-c",
        "--config",
        default="tts_filter.yaml",
        help="設定ファイル（デフォルト: tts_filter.yaml）",
    )

    parser.add_argument(
        "-i",
        "--input",
        help="入力ファイル．指定しない場合は標準入力から読む",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="出力ファイル．指定しない場合は標準出力へ出力",
    )

    args = parser.parse_args()

    try:
        config_path = Path(args.config).resolve()
        config = load_config(config_path)

        server = config.get("server", {})
        model = server.get("model")

        if not model:
            raise ValueError(
                "tts_filter.yaml の server.model が定義されていません．"
            )

        generation = config.get("generation", {})
        system_prompt = build_system_prompt(config)
        client = make_client(config)

        if args.input:
            input_path = Path(args.input)

            with input_path.open("r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = sys.stdin.read()

        result = transform_text(
            client=client,
            model=model,
            system_prompt=system_prompt,
            text=text,
            generation=generation,
        )

        if args.output:
            output_path = Path(args.output)

            with output_path.open("w", encoding="utf-8") as f:
                f.write(result)
                f.write("\n")
        else:
            print(result)

        return 0

    except KeyboardInterrupt:
        return 130

    except Exception as e:
        print(
            f"tts_filter: エラー: {e}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
