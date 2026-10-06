# `slide_lecture.py` の使い方

`slide_lecture.py` は，PDF形式の講義・研究発表スライドから，ナレーション，字幕，レーザーポインタ位置，音声，スライド動画を順に生成するCLIプログラムです．

## 基本的な使い方

```bash
python slide_lecture.py presentation.pdf
```

PDFを画像に変換します．

以降の処理は，生成済みのファイルを利用しながら段階的に実行できます．

```text
PDF
 │
 ▼
pdf
 │  PDF → PNG
 ▼
explain
 │  LLM → ナレーション原稿
 │  ※ここで原稿を確認・編集
 ▼
align
 │  LLM → 字幕翻訳＋レーザーポインタ位置
 │  ※ここで必要に応じて修正
 ▼
tts
 │  TTS → 音声
 ▼
video
 │  スライド＋音声＋レーザーポインタ
 ▼
完成動画（MP4）
```

---

## ステージ指定 `--from`

処理を開始するステージを指定できます．

```bash
--from pdf
--from explain
--from align
--from tts
--from video
```

### `--from pdf`

PDFをスライド画像に変換します．

```bash
python slide_lecture.py presentation.pdf --from pdf
```

生成されるディレクトリ：

```text
presentation_lecture/
└── pages/
    ├── 001.png
    ├── 002.png
    └── ...
```

この段階ではLLM，TTS，動画生成は行いません．

---

### `--from explain`

ナレーション原稿を生成します．

```bash
python slide_lecture.py presentation.pdf --from explain
```

LLMがスライド全体の流れを考慮して，各ページのナレーションを生成します．

生成例：

```text
presentation_lecture/
└── explanations/
    ├── _course_overview.txt
    ├── 001.txt
    ├── 002.txt
    └── ...
```

**この段階で原稿を確認・編集することを推奨します．**

原稿を編集した後は，

```bash
python slide_lecture.py presentation.pdf --from align
```

として次の段階へ進めます．

---

### `--from align`

ナレーションとスライド上の要素を対応付けます．

```bash
python slide_lecture.py presentation.pdf --from align
```

以下の処理を行います．

* ナレーションを文単位に分割
* スライド上のテキスト・コード・図などを抽出
* ナレーション各文とスライド要素をLLMで対応付け
* 日本語／英語字幕を生成するための翻訳
* レーザーポインタの位置を決定するための情報を生成

生成されるファイル：

```text
presentation_lecture/
└── explanations/
    ├── 001.txt
    ├── 001_align.json
    ├── 002.txt
    ├── 002_align.json
    └── ...
```

必要に応じて `*_align.json` を編集した後，

```bash
python slide_lecture.py presentation.pdf --from tts
```

へ進みます．

---

### `--from tts`

ナレーションから音声を生成します．

```bash
python slide_lecture.py presentation.pdf --from tts
```

生成される音声：

```text
presentation_lecture/
└── audio/
    ├── 001.mp3
    ├── 002.mp3
    └── ...
```

日本語の場合，`tts_filter.yaml` が存在すれば，TTS前に読み上げ用テキストフィルタも適用されます．

---

### `--from video`

スライド，音声，アライメント情報から動画を生成し，最後に全ページを結合します．

```bash
python slide_lecture.py presentation.pdf --from video
```

生成される主なファイル：

```text
presentation_lecture/
├── video/
│   ├── 001.mp4
│   ├── 002.mp4
│   └── ...
├── presentation_lecture_ja.srt
├── presentation_lecture_en.srt
└── presentation.mp4
```

完成した動画は，

```text
presentation_lecture/presentation.mp4
```

です．

---

# 推奨する実行方法

通常は，次のように段階的に実行します．

## 1．PDFを読み込む

```bash
python slide_lecture.py presentation.pdf --from pdf
```

## 2．ナレーションを生成する

```bash
python slide_lecture.py presentation.pdf --from explain
```

ここで，

```text
presentation_lecture/explanations/*.txt
```

を確認・編集します．

## 3．字幕・レーザーポインタ位置を生成する

```bash
python slide_lecture.py presentation.pdf --from align
```

必要なら，

```text
presentation_lecture/explanations/*_align.json
```

を確認・修正します．

## 4．音声を生成する

```bash
python slide_lecture.py presentation.pdf --from tts
```

## 5．動画を生成する

```bash
python slide_lecture.py presentation.pdf --from video
```

---

# 発表種別

`--mode` でナレーションのスタイルを指定できます．

### 講義

```bash
python slide_lecture.py presentation.pdf --mode lecture
```

大学講義として学生に説明するナレーションを生成します．

### 研究発表

```bash
python slide_lecture.py presentation.pdf --mode research
```

学会・研究会での研究発表を想定したナレーションを生成します．

---

# 主言語

`--lang` で主言語を指定できます．

### 日本語

```bash
python slide_lecture.py presentation.pdf --lang ja
```

### 英語

```bash
python slide_lecture.py presentation.pdf --lang en
```

主言語に応じてナレーション，TTS，字幕の扱いが切り替わります．

日本語の場合は，日本語ナレーションから英語字幕を生成します．

英語の場合は，英語ナレーションから日本語字幕を生成します．

---

# 処理対象ページの指定

`--pages` を使用すると，処理するスライドを限定できます．

## 1ページだけ

```bash
python slide_lecture.py presentation.pdf --from explain --pages 5
```

## 複数ページ

```bash
python slide_lecture.py presentation.pdf --from explain --pages 1,3,5
```

## 範囲

```bash
python slide_lecture.py presentation.pdf --from explain --pages 1-10
```

## 範囲と個別ページの組み合わせ

```bash
python slide_lecture.py presentation.pdf --from explain --pages 1-5,8,10-12
```

---

# ページを除外する

`--skip-pages` で特定のページを処理対象から除外できます．

例えば，

```bash
python slide_lecture.py presentation.pdf --from explain --skip-pages 5
```

5ページを除外します．

複数ページの場合：

```bash
python slide_lecture.py presentation.pdf --from explain --skip-pages 5,8-10
```

`--pages` と `--skip-pages` を組み合わせることもできます．

```bash
python slide_lecture.py presentation.pdf \
    --from explain \
    --pages 1-20 \
    --skip-pages 5,10-12
```

この場合，1～20ページのうち，5，10～12ページを除外します．

---

# `--force`

通常，既に生成済みのファイルがある場合は，そのファイルを再利用します．

強制的に再生成する場合は `--force` を指定します．

```bash
python slide_lecture.py presentation.pdf --from explain --force
```

`--force` は，指定した開始ステージより後の生成物を削除してから再生成します．

例えば，

```bash
python slide_lecture.py presentation.pdf --from tts --force
```

では，TTS以降の生成物が削除され，TTSから再生成されます．

また，

```bash
python slide_lecture.py presentation.pdf --from align --force
```

では，アライメント情報と，それ以降の音声・動画が再生成されます．

---

# 出力ディレクトリの変更

デフォルトでは，PDFと同じディレクトリに

```text
PDFファイル名_lecture/
```

というディレクトリが作られます．

例えば，

```text
lecture.pdf
```

なら，

```text
lecture_lecture/
```

になります．

`--output` で変更できます．

```bash
python slide_lecture.py lecture.pdf \
    --output ./output
```

この場合，

```text
output/
├── pages/
├── explanations/
├── audio/
└── video/
```

のように生成されます．

---

# 設定ファイル

設定ファイルはデフォルトで

```text
config.yaml
```

を使用します．

別の設定ファイルを使用する場合：

```bash
python slide_lecture.py presentation.pdf \
    --config my_config.yaml
```

---

# よく使うコマンド

## 講義・日本語

```bash
python slide_lecture.py lecture.pdf \
    --mode lecture \
    --lang ja
```

## 研究発表・日本語

```bash
python slide_lecture.py presentation.pdf \
    --mode research \
    --lang ja
```

## 研究発表・英語

```bash
python slide_lecture.py presentation.pdf \
    --mode research \
    --lang en
```

## 特定ページだけ処理

```bash
python slide_lecture.py lecture.pdf \
    --from explain \
    --pages 1-10
```

## 特定ページを除外

```bash
python slide_lecture.py lecture.pdf \
    --from explain \
    --skip-pages 1,5,10-12
```

## 原稿を最初から作り直す

```bash
python slide_lecture.py lecture.pdf \
    --from explain \
    --force
```

## 音声から作り直す

```bash
python slide_lecture.py lecture.pdf \
    --from tts \
    --force
```

## 動画だけ作り直す

```bash
python slide_lecture.py lecture.pdf \
    --from video \
    --force
```

---

# オプション一覧

| オプション          | 内容                       | デフォルト            |
| -------------- | ------------------------ | ---------------- |
| `pdf`          | 入力PDF                    | 必須               |
| `--config`     | 設定ファイル                   | `config.yaml`    |
| `--from`       | 開始ステージ                   | `pdf`            |
| `--force`      | 既存ファイルを削除して再生成           | 無効               |
| `--output`     | 出力ディレクトリ                 | `<PDF名>_lecture` |
| `--mode`       | `lecture` または `research` | 設定ファイル等から決定      |
| `--lang`       | `ja` または `en`            | 設定ファイル等から決定      |
| `--pages`      | 処理対象ページ                  | 全ページ             |
| `--skip-pages` | 処理対象から除外するページ            | なし               |

---

# 処理途中から再開する場合

このプログラムでは，各ステージの生成物を保存しているため，最初からやり直す必要はありません．

例えば，ナレーションを修正した場合：

```bash
python slide_lecture.py lecture.pdf --from align
```

音声だけを作り直したい場合：

```bash
python slide_lecture.py lecture.pdf --from tts --force
```

動画だけを作り直したい場合：

```bash
python slide_lecture.py lecture.pdf --from video --force
```

というように，必要なステージから再開できます．

---

# 典型的なワークフロー

```text
                     ┌──────────────┐
                     │ presentation │
                     │     .pdf     │
                     └──────┬───────┘
                            │
                            ▼
                      --from pdf
                            │
                            ▼
                    ┌───────────────┐
                    │   pages/*.png │
                    └───────┬───────┘
                            │
                            ▼
                   --from explain
                            │
                            ▼
                  ┌──────────────────┐
                  │ explanations/*.txt│
                  └────────┬─────────┘
                           │
                     原稿を確認・編集
                           │
                           ▼
                    --from align
                           │
                           ▼
                 ┌────────────────────┐
                 │ *_align.json       │
                 │ 字幕・視線誘導情報 │
                 └─────────┬──────────┘
                           │
                     必要なら修正
                           │
                           ▼
                     --from tts
                           │
                           ▼
                     ┌──────────┐
                     │ audio/*.mp3 │
                     └─────┬────┘
                           │
                           ▼
                    --from video
                           │
                           ▼
                  ┌─────────────────┐
                  │ 完成動画 .mp4   │
                  │ 日本語/英語字幕 │
                  └─────────────────┘
```

このように，**LLM生成 → 人間による原稿確認 → アライメント → TTS → 動画生成**という段階的なワークフローを想定しています．

