# `tts_filter.py` 内部仕様・解説ドキュメント

## 1. 概要

`tts_filter.py` は、技術文書やコマンドライン操作ログなどのテキストを、音声合成（TTS: Text-to-Speech）エンジンが日本語として自然に発音できるよう前処理を行うためのフィルタリング・スクリプトです。

OpenAI API 互換のエンドポイント（Ollama、vLLM、OpenWebUI、または OpenAI 公式など）を利用し、LLM に対して専門的な技術表記の読み替えプロンプトと辞書情報を与えることで、文脈に沿ったカタカナ化や読み下しを実現します。

---

## 2. システム構成と動作の流れ

```
[ 入力テキスト (標準入力 / ファイル) ]
                 │
                 ▼
     [ split_paragraphs() ]  ──── 空行区切りで段落ごとに分割
                 │
                 ▼
     [ build_system_prompt() ]  ── デフォルト埋め込みプロンプト
                 │                 + YAMLの辞書 ({{DICTIONARY}})
                 ▼
        [ transform() ]  ──────── OpenAI互換 API (Chat Completion)
                 │
                 ▼
[ 出力テキスト (標準出力 / ファイル) ]
```

---

## 3. 主要コンポーネント・関数の解説

### 3.1. 定数: `DEFAULT_SYSTEM_PROMPT_TEMPLATE`
LLM に渡す基底システムプロンプトです。元々 YAML 側に定義されていた内容をスクリプト内に内包しており、以下の重要な制約・ルールが定義されています。

1. **基本方針・制約事項**: 原文の意味を変えない、解説や Markdown 装飾を出力せず本文のみ返す等の出力純度の担保。
2. **参考辞書プレースホルダー (`{{DICTIONARY}}`)**: ユーザー定義辞書を展開するスロット。
3. **カテゴリ別変換ルール**:
   - 英字・略語（`GPU` → ジーピーユー、`Linux` → リナックス）
   - 変数名・識別子（`cnt` → カウンタ、`ptr` → ポインタ）
   - パス・コマンド（`/usr/bin` → ユーザービン、`ls -la` → エルエス，オプション エルエー）
   - 演算子・記号（`a == b` → エーとビーが等しい、`p->next` → ポインタのネクスト）
4. **対象外・フォーマット規約**: 漢字・ひらがな・数字・単位は元の表記を維持。

---

### 3.2. 設定の読み込みとプロンプト構築

#### `load_config(config_path: Path) -> dict`
- YAML 設定ファイルを `yaml.safe_load` でパースします。
- ファイルの存在確認およびトップレベルが辞書構造（マッピング）であるかを検証します。

#### `build_dictionary_text(dictionary: dict) -> str`
- YAML 内の `dictionary:` に定義されたキーバリューペアを展開し、以下のような LLM 用の箇条書きテキストに整形します。
  ```text
  - Linux → リナックス
  - cnt → カウンタ
  ```

#### `build_system_prompt(config: dict) -> str`
- YAML 側に `prompt:` の上書き設定があればそれを採用し、指定がなければ埋め込みの `DEFAULT_SYSTEM_PROMPT_TEMPLATE` を採用します。
- プロンプト内の `{{DICTIONARY}}` 部分を `build_dictionary_text()` の生成結果で置換して最終的なシステムプロンプトを完成させます。

---

### 3.3. API クライアント生成と変換処理

#### `make_client(config: dict) -> OpenAI`
- `openai.OpenAI` クライアントを初期化します。
- YAML の `server.base_url` および `server.api_key` を読み取ります。
  ※ ローカル LLM（Ollama や vLLM など）を利用する場合、`api_key` にダミー文字列（例: `"dummy"`）を指定して動作させることが可能です。

#### `split_paragraphs(text: str) -> list[str]`
- 入力テキストを「空行」単位で段落ごとに分割します。
- **目的**: 巨大な文章を一度に API に投入すると、コンテキスト長制限の圧迫やハルシネーション（要約・省略・フォーマット崩れ）が発生しやすくなるため、適切な単位に切り分けます。

#### `transform(client, model, system_prompt, text, generation) -> str`
- 単一の段落テキストを LLM に渡し、変換結果を取得します。
- `system` ロールに変換ルールと辞書、`user` ロールに対象段落をセットして Chat Completion API を呼び出します。
- `generation` 辞書から `temperature`, `max_tokens`, `top_p`, `seed` などの推論パラメータを抽出して適用します（`temperature=0` に設定することで再現性とルールの厳格遵守を高めています）。

#### `transform_text(client, model, system_prompt, text, generation) -> str`
- 分割された全段落を順次 `transform()` で変換し、最終的に `\n\n` で再結合して 1 つのテキストにまとめます。

---

### 3.4. エントリポイント (`main`)

- コマンドライン引数をパースします（`argparse`）：
  - `-c`, `--config`: 設定 YAML ファイルのパス（デフォルト: `tts_filter.yaml`）
  - `-i`, `--input`: 変換対象の入力テキストファイル（未指定時は `sys.stdin` からパイプ入力）
  - `-o`, `--output`: 変換結果の出力先ファイル（未指定時は `sys.stdout` へ出力）
- 標準入出力に対応しているため、UNIX パイプライン（`cat text.txt | ./tts_filter.py | espeak-ng` 等）に容易に組み込める設計になっています。

---

## 4. 設定ファイル (`tts_filter.yaml`) の役割

プロンプトが Python スクリプト側に内包されたため、YAML 側は接続先と個別辞書のみをシンプルに管理できるようになっています。

```yaml
server:
  base_url: https://your-llm-host:port/v1
  api_key: dummy
  model: huihui_ai/Qwen3.8-abliterated:latest

generation:
  temperature: 0

dictionary:
  Daisuke Takago: たかごう だいすけ
  cnt: カウンタ
  ls: エルエス
  # 固有表現や誤読しやすい用語を自由に追加可能
```

---

## 5. 主な使用例

### ① ファイルからファイルへ変換
```bash
python3 tts_filter.py -i input.txt -o output.txt
```

### ② パイプライン連携（標準入出力）
```bash
echo "ls -la を実行して cnt を確認してください。" | python3 tts_filter.py
# 出力例: エルエス，オプション エルエー を実行して カウンタ を確認してください。
```

### ③ 設定ファイルを切り替えて実行
```bash
python3 tts_filter.py -c custom_config.yaml -i document.md
```