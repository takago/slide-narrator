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
    """YAMLからシステムプロンプトを構築する．"""
    prompt = config.get("prompt", "")

    if not prompt:
        raise ValueError(
            "tts_filter.yaml に prompt が定義されていません．"
        )

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
