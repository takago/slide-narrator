# slide_lecture.py CLI 利用ガイド・コマンド実行例

`slide_lecture.py` は、PDFスライドから講義・研究発表用動画（TTS音声合成、レーザーポインタ視線誘導、日英字幕多重化、チャプター付きMP4）を自動生成・再構築するコマンドラインツールです。

---

## 1. 段階的に進める標準実行フロー

各段階で中間出力を確認・微修正しながら進める標準的な手順です。

### (1) PDFからスライド画像を抽出 (`pages/` に PNG を展開)
```bash
python slide_lecture.py lecture.pdf --from pdf
```

### (2) 全体構成を把握し、各ページの解説原稿ドラフトを一括作成
```bash
python slide_lecture.py lecture.pdf --from explain
```
> **Note**: 必要に応じて `explanations/001.txt` などのテキストを手動修正できます。

### (3) 文分割・対訳字幕の翻訳・レーザーポインタ位置の自動解析
```bash
python slide_lecture.py lecture.pdf --from align
```
> **Note**: 必要に応じて、翻訳文を手動で修正してください。

### (4) ナレーション音声をTTSで一括生成 (`audio/` に MP3 を出力)
```bash
python slide_lecture.py lecture.pdf --from tts
```

### (5) スライド別動画の描画、日英SRT字幕作成、チャプター付き結合動画を出力
```bash
python slide_lecture.py lecture.pdf --from video
```

---

## 2. スライド範囲の指定実行

タイトルスライドや末尾の質疑応答スライドを除外したり、特定範囲のみをテスト・動画化する場合に指定します。

* **スライド 2〜15 ページのみを対象に動画を生成:**
  ```bash
  python slide_lecture.py lecture.pdf --from video --pages 2-15
  ```

* **スライド 1, 5, 12 ページを除外して原稿を生成:**
  ```bash
  python slide_lecture.py lecture.pdf --from explain --skip-pages 1,5,12
  ```

* **飛び飛びのページを指定（2〜5ページ、8ページ、10〜12ページ）:**
  ```bash
  python slide_lecture.py lecture.pdf --from video --pages 2-5,8,10-12
  ```

---

## 3. モード・主言語の切り替え

口頭発表のトーン（講義 or 研究発表）や、主言語（日 or 英）を指定します。

* **講義モード（学生向けの丁寧な解説トーン / 日本語）:**
  ```bash
  python slide_lecture.py lecture.pdf --from explain --mode lecture --lang ja
  ```

* **研究発表モード（学会・口頭発表向けの論理的トーン / 日本語）:**
  ```bash
  python slide_lecture.py paper_slides.pdf --from explain --mode research --lang ja
  ```

* **英語スライド向け（英語ナレーション＆日本語字幕構成）:**
  ```bash
  python slide_lecture.py english_talk.pdf --from explain --mode research --lang en
  ```

---

## 4. 下流キャッシュの段階的強制削除 (`--force`)

指定した開始ステージ (`--from`) に応じて、依存する下流ディレクトリのみが自動的に初期化・削除されます（手前の工程の成果物は安全に保持されます）。

* **【原稿から全再作成】**  
  画像 (`pages/`) を残し、`explanations/`, `audio/`, `video/` を初期化して再実行:
  ```bash
  python slide_lecture.py lecture.pdf --from explain --force
  ```

* **【字幕・ポインタ再解析】**  
  原稿 (`*.txt`) を残し、`*_align.json`, `audio/`, `video/` を初期化して再実行:
  ```bash
  python slide_lecture.py lecture.pdf --from align --force
  ```

* **【音声から再合成】**  
  原稿・字幕アライメントを残し、`audio/`, `video/` を初期化して再実行:
  ```bash
  python slide_lecture.py lecture.pdf --from tts --force
  ```

* **【動画のみ再エンコード】**  
  音声・原稿を残し、`video/` のみを初期化して再実行:
  ```bash
  python slide_lecture.py lecture.pdf --from video --force
  ```

---

## 5. 出力先ディレクトリの指定

デフォルトでは `{PDFファイル名}_lecture/` に作業フォルダが作成されますが、`--output` オプションで自由に変更可能です。

```bash
python slide_lecture.py lecture.pdf --from video --output ./output_project
```

---

## 6. コマンドライン引数一覧

| 引数 | 必須 / デフォルト | 説明 |
| :--- | :--- | :--- |
| `pdf` | **必須** | 処理対象のPDFファイルパス |
| `--from` | `pdf` | 開始ステージ (`pdf`, `explain`, `align`, `tts`, `video`) |
| `--mode` | - | 発表トーン種別 (`lecture`: 講義, `research`: 研究発表) |
| `--lang` | - | 主ナレーション言語 (`ja`: 日本語, `en`: 英語) |
| `--pages` | - | 対象とするスライドページ範囲 (例: `'1-5,8,10'`) |
| `--skip-pages` | - | 除外するスライドページ (例: `'1,6-7'`) |
| `--force` | - | 指定ステージ以降の下流ディレクトリ・生成物を削除して再構築 |
| `--output` | `{PDF名}_lecture` | 出力先ディレクトリパス |
| `--config` | `config.yaml` | 設定YAMLファイルパス |