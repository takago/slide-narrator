# `app.py` プログラム解説

## 1．このプログラムの役割

`app.py` は，`slide_lecture.py` をバックエンドとして利用する **NiceGUIベースのWeb UI** です．

役割を大きく分けると，次の4つです．

1. PDFを受け取り，スライドをWeb UIに表示する
2. スライドやナレーションを編集・確認する
3. `slide_lecture.py` を外部プロセスとして実行する
4. 生成された音声・動画・字幕などをWeb UIに表示する

つまり，実際の生成処理そのものをすべて `app.py` が行っているわけではありません．

```text
                    app.py
              NiceGUI Web UI
                    │
        ┌───────────┼───────────┐
        │           │           │
      PDF管理    編集・確認    設定管理
        │           │           │
        └───────────┼───────────┘
                    │
                    ▼
          slide_lecture.py
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      LLM         TTS         FFmpeg
        │           │           │
        └───────────┼───────────┘
                    ▼
                 動画
```

したがって，`app.py` を理解するときには，

> **「UIから操作を受け取り，プロジェクトの状態を管理し，`slide_lecture.py` に処理を依頼し，結果を再びUIへ反映するプログラム」**

と考えると分かりやすいです．

---

# 2．プログラム全体の構造

ソースコードは大きく次の順序で構成されています．

```text
app.py
│
├─ 1. import
│
├─ 2. 設定・ファイルシステム関連
│    ├─ CONFIG_PATH
│    ├─ TTS_FILTER_PATH
│    ├─ UPLOAD_DIR
│    ├─ TEST_AUDIO_DIR
│    ├─ load_config()
│    └─ save_config()
│
├─ 3. ProjectPaths
│    └─ プロジェクト内のファイルパスを管理
│
├─ 4. PDF・画像・ページ範囲関連のヘルパー
│    ├─ ensure_page_images()
│    ├─ draw_block_preview()
│    └─ format_page_ranges()
│
├─ 5. FastAPIのファイル配信
│    ├─ /files/...
│    └─ /download/...
│
└─ 6. SlideNarratorApp
     │
     ├─ 状態管理
     ├─ スライド選択
     ├─ PDF読み込み
     ├─ パイプライン実行
     ├─ 処理キャンセル
     ├─ ナレーション編集
     ├─ アライメント編集
     ├─ スライド動画表示
     ├─ 最終動画表示
     ├─ 設定画面
     ├─ TTS辞書編集
     └─ UI構築
```

最後に，

```python
application = SlideNarratorApp()
application.build()

ui.run(title='Slide Narrator', reload=False)
```

によってアプリケーションを起動します．

---

# 3．外部モジュールとの関係

`app.py` は単独で全処理を実装していません．

特に重要なのが `slide_lecture.py` です．

```python
from slide_lecture import (
    build_course_overview,
    extract_page_text,
    generate_single_alignment,
    generate_single_explanation,
    generate_single_tts,
    generate_single_page_video,
    generate_tts,
    load_project_json,
    make_client,
    make_explanation,
    parse_page_ranges,
    save_project_json,
)
```

ここで，

* LLMによる解説生成
* アライメント
* TTS
* 動画生成
* プロジェクト設定
* ページ範囲の解析

などを利用しています．

一方，Web UI側では `NiceGUI` を中心に，

```python
from nicegui import app, run, ui
```

を使います．

`run.io_bound()` によって，CPU処理や同期的な外部処理をイベントループから切り離している点も重要です．

---

# 4．設定ファイル

冒頭で主要なファイルを定義しています．

```python
CONFIG_PATH = Path('config.yaml')
TTS_FILTER_PATH = Path('tts_filter.yaml')
UPLOAD_DIR = Path('webui_uploads')
```

役割は次の通りです．

| 定数                | 用途                   |
| ----------------- | -------------------- |
| `CONFIG_PATH`     | LLM，TTS，PDF，動画などの設定  |
| `TTS_FILTER_PATH` | 日本語TTS用読み変換辞書        |
| `UPLOAD_DIR`      | Web UIからアップロードされたPDF |
| `TEST_AUDIO_DIR`  | TTS接続テストで生成する音声      |

設定ファイルは，

```python
load_config()
save_config()
```

で読み書きします．

---

# 5．ProjectPaths

## 5.1 目的

`ProjectPaths` は，プロジェクト内のファイル名を一元管理するクラスです．

例えば，

```text
lecture.pdf
```

を入力すると，プロジェクトディレクトリは，

```text
lecture_lecture/
```

になります．

その下に，

```text
lecture_lecture/
├── pages/
├── explanations/
├── audio/
├── video/
├── lecture.mp4
├── lecture_ja.srt
└── lecture_en.srt
```

という構造を作ります．

---

## 5.2 ファイルパス生成

例えば，

```python
self.paths.page_image(3)
```

は，

```text
lecture_lecture/pages/003.png
```

を返します．

同様に，

```python
self.paths.explanation(3)
```

なら，

```text
lecture_lecture/explanations/003.txt
```

です．

その他，

```text
alignment(3)
→ explanations/003_align.json

audio(3)
→ audio/003.mp3

video(3)
→ video/003.mp4
```

となります．

このクラスの重要な点は，

> **ファイル名やディレクトリ構造をプログラム全体に直接書かない**

ことです．

後からディレクトリ構造を変更する場合も，ここを変更すれば済みます．

---

# 6．cleanup_downstream()

`ProjectPaths.cleanup_downstream()` は，あるページの原稿などが変更されたときに，その後の生成物を削除するための処理です．

例えば，

```text
003.txt
```

を変更した場合，

```text
003_align.json
003.mp3
003.mp4
```

は古い原稿に基づいている可能性があります．

そこで，下流生成物を削除します．

また，最終的な，

```text
lecture.mp4
lecture_ja.srt
lecture_en.srt
```

も削除対象になります．

これは，

```text
原稿
 ↓
アライメント
 ↓
音声
 ↓
動画
 ↓
完成動画
```

という依存関係を維持するための処理です．

---

# 7．ensure_page_images()

```python
ensure_page_images(paths, dpi)
```

はPDFをPNG画像へ変換します．

重要なのは，

```python
if not out.exists():
```

という条件があることです．

つまり，すでに

```text
pages/001.png
pages/002.png
...
```

が存在していれば，毎回PDFを再変換しません．

このため，Web UIを再読み込みしても既存のページ画像を再利用できます．

---

# 8．draw_block_preview()

```python
draw_block_preview(img_path, blocks)
```

は，スライド画像にLLMアライメント用のブロック情報を重ねて表示するための関数です．

各ブロックについて，

```text
bbox
block_id
```

を利用し，

```text
┌──────────────┐
│ #1           │
│ スライド本文 │
│              │
└──────────────┘
```

のように矩形と番号を描きます．

主に編集UIで，

> 「LLMがスライドのどの領域を認識しているか」

を確認するために使います．

---

# 9．ページ範囲の管理

`format_page_ranges()` は，

```python
[1, 2, 3, 4, 7, 9, 10]
```

を，

```text
1-4,7,9-10
```

のような文字列に変換します．

この文字列はプロジェクト設定や `slide_lecture.py` のCLIに渡されます．

---

# 10．FastAPIのファイル配信

NiceGUIのUIから画像や動画を表示するため，FastAPIのエンドポイントを用意しています．

## `/files/...`

通常のブラウザ表示用です．

```text
/files/...
```

でファイルを返します．

## `/download/...`

ダウンロード用です．

```text
/download/...
```

では `filename=` を指定して返します．

どちらも，

```python
target = (root / path).resolve()
```

として実体パスを求めた後，

```python
if root not in target.parents and target != root:
```

で，カレントディレクトリの外へ脱出しないか確認しています．

これはパストラバーサル対策です．

---

# 11．SlideNarratorApp

このプログラムの中心です．

```python
class SlideNarratorApp:
```

の中に，Web UIの状態とほぼすべての操作が集約されています．

大きく見ると，

```text
SlideNarratorApp
│
├─ プロジェクト状態
│
├─ スライド選択状態
│
├─ UI部品への参照
│
├─ パイプライン実行
│
├─ 編集機能
│
├─ 表示更新
│
└─ 設定画面
```

という構造です．

---

# 12．アプリケーション状態

`__init__()` では，アプリケーション全体の状態を保持します．

主要なものは次の通りです．

```python
self.cfg
```

システム設定．

```python
self.pdf
```

現在読み込んでいるPDF．

```python
self.paths
```

現在のプロジェクトのファイルパス管理．

```python
self.images
```

スライド画像一覧．

```python
self.proj_cfg
```

プロジェクト固有の設定．

```python
self.mode_code
self.lang_code
```

講義／研究発表，日本語／英語の状態．

```python
self.processing
```

現在パイプライン処理中かどうか．

---

# 13．スライド選択状態

特に重要なのが，

```python
self._selected_pages: set[int]
```

です．

これは現在処理対象になっているページ番号を保持します．

例えば，

```python
{1, 2, 3, 7, 8}
```

なら，

```text
1～3ページ
7～8ページ
```

が処理対象です．

この状態を「Single Source of Truth」として扱っています．

つまり，

```text
チェックボックス
ページ範囲入力欄
対象スライド数
編集対象
パイプライン
```

などが，それぞれ別々のページ情報を持つのではなく，基本的に `_selected_pages` を基準にします．

これはこのプログラムを理解するうえで非常に重要です．

---

# 14．active_pages と pages_spec

## active_pages

```python
@property
def active_pages(self)
```

は，

```python
_selected_pages
```

をソートしてリストとして返します．

例えば，

```python
_selected_pages = {5, 2, 3}
```

なら，

```python
active_pages
→ [2, 3, 5]
```

です．

## pages_spec

```python
@property
def pages_spec(self)
```

は，これをCLIで利用できる形式に変換します．

例えば，

```text
[1,2,3,4,7,9,10]
```

なら，

```text
1-4,7,9-10
```

です．

全ページが選択されている場合は空文字列になります．

1ページも選択されていない場合は，

```text
none
```

になります．

---

# 15．apply_pages_spec()

ページ範囲入力欄などから，

```text
1-5,8,10-12
```

のような指定を受け取ります．

そして，

```python
parse_page_ranges()
```

を使って整数の集合に変換します．

さらに，

```python
1 <= page <= total_slides
```

となるように現在のPDFのページ数で制限します．

その後，

```python
_on_pages_updated()
```

を呼び出して，

* UI
* 設定ファイル
* ギャラリー
* 編集画面

などを更新します．

---

# 16．_on_pages_updated()

スライド選択状態が変わったときの共通処理です．

ここでは，

1. 編集対象ページを調整
2. ページ範囲入力欄を更新
3. 「対象スライド数」を更新
4. プロジェクト設定を保存
5. 各種ビューを再構築

を行います．

ページ選択処理を複数の場所から直接書かず，この関数に集約しているのがポイントです．

---

# 17．PDF読み込み

中心となる処理が，

```python
async def load_pdf(self, e)
```

です．

処理の流れは，

```text
PDFアップロード
    ↓
webui_uploads/ に保存
    ↓
ProjectPaths生成
    ↓
プロジェクト設定読み込み
    ↓
mode / language 読み込み
    ↓
PDF → PNG
    ↓
保存済みページ選択を復元
    ↓
UI更新
```

です．

PDF画像変換は，

```python
await run.io_bound(
    ensure_page_images,
    ...
)
```

として実行します．

これは，PDF処理によってNiceGUIのイベントループを長時間止めないためです．

---

# 18．プロジェクト設定

プロジェクト設定は，

```python
load_project_json()
save_project_json()
```

を利用します．

主な項目は，

```text
mode
language
pages
skip_pages
```

です．

例えば，

```json
{
    "mode": "lecture",
    "language": "ja",
    "pages": "1-20",
    "skip_pages": ""
}
```

のような情報を保存します．

Web UIを閉じても，PDFを再度読み込めば設定を復元できます．

---

# 19．UIの更新

主要な更新処理は，

```python
refresh_views()
```

に集約されています．

順番は，

```text
refresh_gallery()
      ↓
refresh_simple_editor()
      ↓
refresh_editor()
      ↓
refresh_slide_videos()
```

です．

さらに，

```python
refresh_all()
```

では，

```text
refresh_views()
      ↓
refresh_final_video()
```

まで実行します．

つまり，

> **プロジェクトの状態が変わったら，関連する画面を再構築する**

という設計です．

---

# 20．スライドギャラリー

`refresh_gallery()` は，PDFの各ページをカード状に並べます．

各カードには，

* スライド画像
* スライド番号
* 選択状態

があります．

選択中でないページは，

```text
opacity
grayscale
```

などによって薄く表示されます．

カードまたはチェックボックスをクリックすると，

```python
toggle_slide_active()
```

が呼ばれます．

---

# 21．簡易ナレーション編集

`refresh_simple_editor()` は，選択された全スライドのナレーションを一覧表示します．

各ページについて，

```text
┌─────────────┐
│ スライド画像 │
└─────────────┘
┌──────────────────────┐
│ ナレーション原稿      │
│                      │
└──────────────────────┘
```

というUIを作ります．

さらに，

```text
ナレーションを再生成
```

と，

```text
保存して字幕・ポインタを再解析
```

のボタンがあります．

---

# 22．詳細編集画面

`refresh_editor()` は，より詳細な1ページ単位の編集画面を作ります．

ここでは，

* 編集対象ページの選択
* 前ページ／次ページへの移動
* スライド画像
* ナレーション
* アライメント情報
* スライド上のブロック
* 字幕
* ポインタ位置

などを扱います．

ページを変更すると，

```python
select_edit_page()
```

を通じて再描画します．

---

# 23．alignment_data()

```python
alignment_data(page)
```

は，

```text
explanations/003_align.json
```

などのアライメントファイルを読み込みます．

戻り値は，

```python
data
blocks
alignments
```

の3つです．

つまり，

```text
JSON全体
  ├─ blocks
  └─ alignments
```

というデータ構造をUI側から扱いやすくしています．

---

# 24．ナレーション再生成

```python
regenerate_narration()
```

は，特定スライドだけLLMでナレーションを再生成する処理です．

大まかな流れは，

```text
対象ページ
   ↓
全体概要・前後ページなどの文脈を取得
   ↓
LLM
   ↓
ナレーション生成
   ↓
001.txtなどを更新
   ↓
下流生成物を無効化
   ↓
UI更新
```

です．

これは全ページを再生成するCLI処理とは別に，

> **1ページだけ修正・再生成する**

ためのWeb UI用機能です．

---

# 25．save_and_realign()

ナレーション原稿を人間が修正した後，

```text
保存して字幕・ポインタを再解析
```

を押すと，この処理が使われます．

重要なのは，

```text
原稿変更
 ↓
アライメント再生成
 ↓
音声・動画など下流を無効化
```

という依存関係を維持することです．

---

# 26．処理パイプライン

このプログラムで最も重要な部分の一つです．

```python
async def pipeline(self, stage, initial_title)
```

が，UIから `slide_lecture.py` を起動する共通処理になっています．

例えば，

```text
ナレーション生成
字幕・ポインタ解析
TTS
動画生成
```

の各ボタンが，基本的にこの共通処理を利用します．

---

# 27．pipeline() の実行順序

概念的には，

```text
UIボタン
   ↓
pipeline(stage)
   ↓
入力チェック
   ↓
プロジェクト設定保存
   ↓
処理ダイアログ表示
   ↓
ボタン無効化
   ↓
slide_lecture.py 起動
   ↓
標準出力を読み取る
   ↓
[PROGRESS] を解析
   ↓
UI進捗更新
   ↓
終了待ち
   ↓
結果判定
   ↓
画面再読み込み
   ↓
ダイアログ終了
```

となります．

---

# 28．slide_lecture.py の起動

実際には，

```python
cmd = [
    sys.executable,
    '-u',
    'slide_lecture.py',
    str(self.pdf),
    '--from',
    stage,
    *self.base_args(),
]
```

というコマンドを構築します．

例えば，

```text
python slide_lecture.py lecture.pdf \
    --from explain \
    --mode lecture \
    --lang ja \
    --pages 1-20
```

のようなコマンドになります．

つまり，Web UIはCLIを別の方法で再実装しているのではなく，

> **Web UIからCLIプログラムを操作している**

と考えると分かりやすいです．

---

# 29．base_args()

```python
base_args()
```

は，Web UIで設定されている共通オプションをCLI引数に変換します．

現在は，

```text
--mode
--lang
--force
--pages
```

を構築します．

そのため，Web UIの設定とCLIの設定が食い違わないようになっています．

---

# 30．非同期プロセス実行

`slide_lecture.py` は，

```python
asyncio.create_subprocess_exec()
```

で起動します．

標準出力を，

```python
stdout=asyncio.subprocess.PIPE
```

で受け取ります．

そして1行ずつ，

```python
line = ...
```

として読み取ります．

この設計により，

```text
slide_lecture.py
       │
       │ stdout
       ▼
     app.py
       │
       ├─ ログ表示
       └─ 進捗表示
```

というリアルタイム連携ができます．

---

# 31．[PROGRESS] の処理

`slide_lecture.py` が出力する構造化進捗イベント，

```text
[PROGRESS] {...}
```

だけを特別に解釈します．

JSONから，

```text
phase
current
total
page
message
```

などを取得します．

これを利用して，

```text
現在の処理
進捗率
現在のスライド
```

をUIに反映します．

通常のログ文字列を無理に解析するのではなく，

> **構造化された `[PROGRESS]` イベントだけをUI進捗情報として利用する**

のがポイントです．

---

# 32．処理ダイアログ

`open_processing_dialog()` は，処理中に表示するモーダルダイアログを構築します．

表示されるものは，

```text
処理タイトル
処理状態
進捗バー
パーセント
現在のスライド
ログ
キャンセルボタン
```

などです．

また，スライド画像について，

```text
前のスライド
現在のスライド
次のスライド
```

を表示できるようになっています．

---

# 33．キャンセル処理

```python
request_cancel()
```

は処理中断を担当します．

Linuxなどでは，

```python
os.killpg(...)
```

によってプロセスグループ全体へ `SIGTERM` を送ります．

これは重要です．

`slide_lecture.py` がさらに

```text
ffmpeg
```

などを起動しているため，親プロセスだけを停止するよりも，プロセスグループをまとめて終了させる方が適切だからです．

Windowsでは，

```python
process.terminate()
```

を利用します．

---

# 34．processing状態

```python
self.processing
```

は，

> 現在パイプライン処理中か

を表します．

処理中は，

```python
set_processing(True)
```

によってパイプライン関連ボタンを無効化します．

これにより，

```text
動画生成中
   ↓
ユーザがもう一度動画生成
```

のような二重実行を防ぎます．

---

# 35．処理終了後

プロセス終了後は，

```text
return code == 0
```

なら成功，

```text
return code != 0
```

ならエラーとして扱います．

その後，

```python
await self.refresh_all()
```

を実行します．

つまり，生成されたファイルをUIへ反映します．

最後に，

```python
self.current_process = None
self.set_processing(False)
dialog.close()
```

として後始末を行います．

---

# 36．TTS接続テスト

設定画面には，TTSの接続テストがあります．

日本語・英語それぞれについて，

```text
Base URL
Model
Voice
```

を入力できます．

テストボタンを押すと，

```text
入力テキスト
    ↓
OpenAI互換TTS API
    ↓
MP3
    ↓
Web UIで再生
```

という処理を行います．

テスト音声は，

```text
webui_uploads/test_audio/
```

に保存されます．

---

# 37．LLM接続テスト

設定画面にはLLM接続テストもあります．

ユーザが任意のプロンプトを入力すると，

```python
client.chat.completions.create(...)
```

でLLMへ送信します．

ここでは本番のナレーション生成とは別に，

> **LLMのBase URLとモデルが正しく設定されているか**

を確認できます．

---

# 38．設定画面

`refresh_settings()` が設定タブ全体を構築します．

大きく，

```text
システム設定
│
├─ LLM設定
│
├─ 日本語TTS設定
│
├─ 英語TTS設定
│
└─ TTS読み変換辞書
```

という構造です．

設定を変更して，

```text
config.yaml を保存
```

すると設定ファイルへ書き戻されます．

---

# 39．LLM設定

`_build_llm_settings()` では，

```text
Base URL
Model
Temperature
```

などを設定できます．

さらに，入力したプロンプトを使ってLLMへ直接問い合わせるテスト機能があります．

ここは，

> **実際のナレーション生成ではなく接続確認用**

です．

---

# 40．TTS設定

`_build_tts_settings()` では，日本語と英語を別々に設定します．

それぞれ，

```text
Base URL
Model
Voice
```

を持ちます．

設定は，

```text
config.yaml
└── tts
    ├── ja
    └── en
```

に対応しています．

---

# 41．TTS読み変換辞書

`_build_dict_editor()` は，

```text
tts_filter.yaml
```

をWeb UIから編集するための画面です．

辞書は，

```text
単語 / 識別子
        ↓
読みの目安
```

という対応になっています．

例えば，

```text
argc → アーギューシー
```

のような技術用語の読みを登録できます．

AG Gridを使って表形式で編集できます．

---

# 42．設定保存の流れ

設定画面で変更して保存すると，

```text
Web UI
  ↓
self.cfg
  ↓
save_config()
  ↓
config.yaml
```

となります．

TTS辞書の場合は，

```text
AG Grid
  ↓
grid.load_client_data()
  ↓
dictionary
  ↓
save_config()
  ↓
tts_filter.yaml
```

です．

---

# 43．build()

`build()` はWeb UIそのものを構築する巨大な関数です．

ここでは，

```text
ヘッダ
左サイドバー
メインタブ
```

を作ります．

---

# 44．左サイドバー

左側には主に，

```text
PDFアップロード
発表種別
言語
スライド範囲
処理ボタン
```

などがあります．

PDFは，

```python
ui.upload(...)
```

で受け取ります．

---

# 45．メイン画面のタブ

現在のUIは概ね次のタブ構成です．

```text
┌──────────────────────────────────┐
│ Slide Narrator                   │
├──────────────────────────────────┤
│                                  │
│  スライドデッキ                  │
│  ナレーション修正（簡易）        │
│  詳細編集                        │
│  ビデオデッキ                    │
│  設定                            │
│  実行ログ                        │
│                                  │
└──────────────────────────────────┘
```

それぞれのコンテナを，

```python
self.gallery
self.simple_edit_container
self.edit_container
self.slide_video_gallery
self.settings_container
self.history_log_widget
```

として保持しています．

---

# 46．UIコンテナを保持する理由

例えば，

```python
self.gallery
```

を保持しておけば，後から，

```python
self.gallery.clear()
```

として中身を再構築できます．

つまりUIを，

```text
最初に一度作って終わり
```

ではなく，

```text
状態変更
 ↓
container.clear()
 ↓
現在の状態に合わせて再構築
```

という方式で管理しています．

---

# 47．言語変更

```python
_lang_changed()
```

では，

```text
lang_code変更
 ↓
プロジェクト設定保存
 ↓
簡易編集画面更新
 ↓
詳細編集画面更新
```

を行います．

日本語／英語によって字幕やTTSの扱いが変わるため，関連UIも更新します．

---

# 48．モード変更

```python
_mode_changed()
```

では，

```text
lecture
research
```

を変更して，プロジェクト設定へ保存します．

この設定は，後で `slide_lecture.py` の

```text
--mode
```

として渡されます．

---

# 49．スライド範囲変更

```python
_range_changed()
```

はページ範囲入力欄の変更を処理します．

例えば，

```text
1-10,15,20-25
```

を入力すると，

```text
parse_page_ranges()
```

によって内部のページ集合に変換されます．

不正な入力の場合は，

```text
ui.notify(...)
```

でユーザに知らせます．

---

# 50．動画表示

`refresh_slide_videos()` は各ページのMP4を表示します．

また，`refresh_final_video()` は完成した，

```text
<PDF名>.mp4
```

を表示します．

そのため，処理終了後に

```python
refresh_all()
```

を呼ぶことで，生成された動画をすぐUIへ反映できます．

---

# 51．このプログラムのデータフロー

最も重要な部分だけをまとめると，次のようになります．

```text
             PDF
              │
              ▼
        ┌─────────────┐
        │ load_pdf()  │
        └──────┬──────┘
               │
               ▼
          pages/*.png
               │
               ▼
        ┌─────────────┐
        │ スライド選択 │
        └──────┬──────┘
               │
               ▼
       _selected_pages
               │
               ▼
        ┌─────────────┐
        │  pipeline() │
        └──────┬──────┘
               │
               ▼
       slide_lecture.py
               │
      ┌────────┼────────┐
      ▼        ▼        ▼
   explain   align     tts
      │        │        │
      └────────┼────────┘
               ▼
             video
               │
               ▼
           完成動画
```

---

# 52．「状態」と「処理」を分けて読む

`app.py` を後から読むときは，まず次の2つを分けて考えると理解しやすくなります．

## 状態

```text
self.pdf
self.paths
self.images
self.proj_cfg
self.mode_code
self.lang_code
self._selected_pages
self.edit_page
self.processing
```

これは，

> 「今，アプリがどんな状態なのか」

を表します．

## 処理

```text
load_pdf()
apply_pages_spec()
toggle_slide_active()
pipeline()
regenerate_narration()
save_and_realign()
refresh_*
```

こちらは，

> 「状態をどう変化させるのか」

を表します．

この2つを分けて読むと，かなり追いやすくなります．

---

# 53．処理の中心は3本

実際にこのプログラムを追いかける場合，最初から全関数を読む必要はありません．

まず見るべきなのは次の3本です．

## ① PDF読み込み

```text
load_pdf()
```

ここからプロジェクトが始まります．

## ② 全体処理

```text
pipeline()
```

ここが `slide_lecture.py` との接続点です．

## ③ 画面更新

```text
refresh_views()
refresh_all()
```

ここが生成結果をUIへ反映する部分です．

この3本を理解した後で，各編集機能や設定機能を見ると構造が分かりやすくなります．

---

# 54．処理ステージと対応関数

| 処理        | 主な担当                                           |
| --------- | ---------------------------------------------- |
| PDFアップロード | `load_pdf()`                                   |
| PDF→PNG   | `ensure_page_images()`                         |
| ページ選択     | `apply_pages_spec()` / `toggle_slide_active()` |
| 原稿生成      | `pipeline('explain', ...)`                     |
| アライメント    | `pipeline('align', ...)`                       |
| TTS       | `pipeline('tts', ...)`                         |
| 動画生成      | `pipeline('video', ...)`                       |
| 1ページ再生成   | `regenerate_narration()`                       |
| 再アライメント   | `save_and_realign()`                           |
| スライド一覧    | `refresh_gallery()`                            |
| 原稿一覧      | `refresh_simple_editor()`                      |
| 詳細編集      | `refresh_editor()`                             |
| ページ動画     | `refresh_slide_videos()`                       |
| 完成動画      | `refresh_final_video()`                        |
| 設定        | `refresh_settings()`                           |
| TTS辞書     | `_build_dict_editor()`                         |

---

# 55．`app.py` と `slide_lecture.py` の責務

この2ファイルは，役割を分けて考えることが重要です．

```text
app.py
│
│  Web UI
│  状態管理
│  ユーザ操作
│  プロセス管理
│  結果表示
│
└──────────────► slide_lecture.py
                    │
                    │ 実際の生成処理
                    │
                    ├─ LLM
                    ├─ アライメント
                    ├─ TTS
                    ├─ FFmpeg
                    └─ 字幕
```

つまり，

> `app.py` は「操作盤」

で，

> `slide_lecture.py` は「処理エンジン」

という関係です．

この境界を意識しておくと，今後の修正箇所を判断しやすくなります．

---

# 56．今後コードを修正するときの目安

## UIの見た目を変更したい

主に，

```text
build()
refresh_gallery()
refresh_simple_editor()
refresh_editor()
refresh_slide_videos()
refresh_settings()
```

を見る．

## 処理ボタンの動作を変更したい

```text
pipeline()
base_args()
```

を見る．

## PDF読み込みを変更したい

```text
load_pdf()
ensure_page_images()
ProjectPaths
```

を見る．

## スライド選択を変更したい

```text
_selected_pages
active_pages
pages_spec
apply_pages_spec()
_on_pages_updated()
toggle_slide_active()
select_all_slides()
clear_all_slides()
```

を見る．

## ナレーション編集を変更したい

```text
refresh_simple_editor()
refresh_editor()
regenerate_narration()
save_and_realign()
```

を見る．

## LLM設定を変更したい

```text
_build_llm_settings()
refresh_settings()
```

を見る．

## TTS設定を変更したい

```text
_build_tts_settings()
_build_dict_editor()
```

を見る．

## キャンセル処理を変更したい

```text
open_processing_dialog()
request_cancel()
pipeline()
set_processing()
```

を見る．

---

# 57．プログラムを読む順番

初めてこのファイルを読み返す場合は，次の順番がおすすめです．

```text
① ProjectPaths
       ↓
② SlideNarratorApp.__init__()
       ↓
③ active_pages / pages_spec
       ↓
④ load_pdf()
       ↓
⑤ pipeline()
       ↓
⑥ refresh_views()
       ↓
⑦ refresh_gallery()
       ↓
⑧ refresh_simple_editor()
       ↓
⑨ refresh_editor()
       ↓
⑩ refresh_settings()
       ↓
⑪ build()
```

`build()` はコード量が多いですが，**最初に読む必要はありません**．

`build()` は基本的に，

> 「どんなUI部品を配置して，どの関数をイベントに接続しているか」

を定義している場所だからです．

---

# 58．最終的な理解

このプログラムを一言で表すと，

> **PDFから講義動画を生成する処理エンジン `slide_lecture.py` を，NiceGUIから操作するための状態管理付きWebフロントエンド**

です．

中心となる考え方は，

```text
PDF
 ↓
ProjectPaths
 ↓
_selected_pages
 ↓
pipeline()
 ↓
slide_lecture.py
 ↓
生成ファイル
 ↓
refresh_views()
 ↓
Web UI
```

です．

特に重要なのは，

```text
ProjectPaths
_selected_pages
pipeline()
refresh_views()
```

の4つです．

この4つを押さえれば，`app.py` の大部分のコードが「何のために存在するのか」を追えるようになります．

