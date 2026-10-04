# `app.py` 徹底解説ドキュメント

`app.py` は、プレゼンテーションスライド（PDF）からナレーション原稿作成、字幕翻訳、レーザーポインタ視線誘導アライメント、TTS（音声合成）、動画合成までを自動化するツール **「Slide Narrator」** の WebUI 実装ファイルです。

PythonベースのWebUIフレームワークである **[NiceGUI](https://nicegui.io/)**（FastAPIおよびVue/Quasar上に構築）を採用し、対話的かつ直感的な操作パネルとプレビュー画面を提供します。

---

## 1. 全体像とアーキテクチャ

`app.py` は、CLIツールである `slide_lecture.py` およびTTS読み仮名正規化モジュール `tts_filter.py` をUI層からオーケストレーションするハブの役割を果たします。

```
                       [ ユーザーブラウザ ]
                               │ (WebSocket / HTTP)
                               ▼
                        [ NiceGUI / FastAPI ]
                            (app.py)
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    【状態管理・UI描画】    【FastAPI静的配信】    【非同期ワーカ (run.io_bound)】
    - SlideNarratorApp     - /files/* (プレビュー) - PyMuPDF (PDFレンダリング)
    - 4つのタブ画面        - /download/* (DL)     - subprocess (slide_lecture.py)
    - 設定/辞書編集                               - LLM API / TTS API 呼び出し
                                                  │
                                                  ▼
                                      [ ディレクトリ / 出力物 ]
                                      - webui_uploads/
                                      - {pdf_stem}_lecture/
                                          ├─ pages/*.png
                                          ├─ explanations/*.txt, *.json
                                          ├─ audio/*.mp3
                                          ├─ video/*.mp4
                                          └─ 最終結合 .mp4 / .srt
```

---

## 2. モジュール構成と主要な依存関係

| パッケージ / モジュール | 主な役割 |
| :--- | :--- |
| **`nicegui` (`app`, `ui`, `run`)** | Webアプリケーション本体、リアクティブUI構築、非同期タスク管理 |
| **`fastapi.responses`** | 生成メディア（画像・音声・動画・SRT）のインライン配信およびダウンロード制御 |
| **`pymupdf` (`fitz`)** | PDF各ページの画像抽出・ラスタライズ（解像度DPI制御） |
| **`PIL` (`Image`, `ImageDraw`)** | アライメント検出されたテキスト/画像バウンディングボックスのオーバーレイ描画 |
| **`slide_lecture`** | コアロジック（LLMクライアント生成、ナレーション/アライメント生成、設定入出力） |
| **`tts_filter`** | 日本語TTS専用の技術用語・記号読み仮名変換フィルタ |

---

## 3. ファイルシステムヘルパーとパス設計

各プロジェクトは、アップロードされたPDFのファイル名をベースにディレクトリ階層（`{stem}_lecture`）を構成して作業成果物を永続化します。

```python
def project_root(pdf: Path) -> Path:
    return pdf.with_name(pdf.stem + '_lecture')
```

### パス解決用ヘルパー群
* `explanation_path(...)`: `explanations/{page:03d}.txt`（ナレーション原稿）
* `alignment_path(...)`: `explanations/{page:03d}_align.json`（ブロックバウンディングボックス＋文対応＋対訳）
* `page_image_path(...)`: `pages/{page:03d}.png`（スライド画像）
* `audio_path(...)` / `video_path(...)`: ページごとの音声（`.mp3`）および動画（`.mp4`）
* `final_video_path(...)`, `final_ja_srt_path(...)`, `final_en_srt_path(...)`: 全ページ結合後の最終成果物

### `cleanup_downstream_media`（カスケード削除機構）
ナレーション原稿やアライメント設定がユーザーによって再編集・再生成された際、**不整合が生じる古い音声（MP3）や動画（MP4）、字幕ファイルを即座に破棄（unlink）** します。これにより、編集後の再ビルド時に古いメディアが使い回される事故を防ぎます。

---

## 4. FastAPI エンドポイント（安全なファイル配信）

NiceGUI の背後で稼働している FastAPI インスタンスを活用し、セキュアなメディア配信を行っています。

* `@app.get('/files/{path:path}')`: ブラウザ上での画像、音声、動画のインラインプレビュー用。
* `@app.get('/download/{path:path}')`: 完成した結合動画や字幕SRTファイルの直接ダウンロード用。

**セキュリティ対策（パストラバーサル防止）:**
```python
root = Path.cwd().resolve()
target = (root / path).resolve()
if root not in target.parents and target != root:
    return PlainTextResponse('Forbidden', status_code=403)
```
カレントワーキングディレクトリ外へのディレクトリ走査（`../` を使った任意ファイル読み取り）を厳格に遮断します。

---

## 5. アプリケーションクラス: `SlideNarratorApp`

UIと状態ロジックは単一のクラス `SlideNarratorApp` にカプセル化されています。

### 5.1 状態管理（State Management）
* **PDF / プロジェクト設定**: `pdf`, `proj_cfg`, `images`, `active_pages`
* **動作モード**: `mode_code` (`lecture`: 講義 / `research`: 研究発表), `lang_code` (`ja` / `en`)
* **ページフィルタリング**: `pages_spec`（対象範囲 例: `1-5,8`）, `skip_pages_spec`（除外範囲）
* **UI状態**: `edit_page`（現在編集中のページ）, `video_page`, `processing`（実行中ロックフラグ）

---

### 5.2 コア機能の詳細解説

#### ① パイプライン実行 (`pipeline` メソッド)
「①ナレーション原稿を一括生成」「②字幕翻訳＆ポインタ解析」「③TTS生成」「④動画生成」の4ステップを管理します。

```python
rc, out = await run.io_bound(
    run_command,
    ['python', 'slide_lecture.py', str(self.pdf), '--from', stage, *self.base_args()],
)
```
* **ノンブロッキング処理**: `run.io_bound()` を用いることで、バックグラウンドで `subprocess` を実行しても NiceGUI のイベントループやブラウザUI描画がフリーズしません。
* **スピナーダイアログ**: 実行中はモーダルダイアログ（`open_processing_dialog`）が表示され、多重実行を防止します。
* **ログ出力**: 標準出力および標準エラー出力をリアルタイムにキャプチャし、UI左下の `処理ログ` テキストエリアに反映します。

#### ② スライド個別再生成 (`regenerate_narration`)
特定の1ページだけナレーションを書き直したい場合に使用されます。
* 前後ページの文脈（`prev_text`, `next_text`）および前スライドのナレーション（`prev_explanation`）を引き継いでLLMを呼び出し、連続性を壊さずに指定スライドの原稿のみをアップデートします。

#### ③ 視線誘導プレビューとアライメント微調整 (`draw_block_preview` & `save_alignment`)
* スライド画像内のテキストブロックや図表領域を検出し、青色の枠線と `#ID` を描画した `_preview.png` を生成・表示します。
* ユーザーは「どの文でどのブロック（ポインタ先）を指すか」「対訳字幕のテキスト」をWeb画面上で1行ずつセレクトボックス・インプットから直接微調整・上書き保存できます。

---

### 5.3 UIレイアウト構造 (`build` メソッド)

画面は **左側ドロワー（サイドバー）** と **メインコンテンツエリア（タブ構成）** に分かれています。

```
┌─────────────────┬─────────────────────────────────────────────────────────┐
│ [Slide Narrator]│  [🖼️ スライド一覧] [📝 詳細編集] [🎬 ビデオ確認] [⚙ 設定] │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ ▼ プロジェクト設定│                                                         │
│ ・PDFアップロード│ (タブパネルの内容)                                      │
│ ・講義/研究ラジオ│                                                         │
│ ・言語/範囲指定 │                                                         │
│                 │ - ギャラリー表示                                        │
│ ▼ パイプライン実行│ - バウンディングボックス付きプレビュー＆字幕微調整UI     │
│ [① 原稿一括生成] │ - 全編動画 / ページ別個別動画プレビュー                 │
│ [② アライメント] │ - LLM/TTSパラメータ設定 & Ag-Gridによる読み仮名辞書編集  │
│ [③ TTS音声生成] │                                                         │
│ [④ 動画生成]    │                                                         │
│                 │                                                         │
│ ▼ 処理ログ出力  │                                                         │
└─────────────────┴─────────────────────────────────────────────────────────┘
```

#### 各タブの役割
1. **🖼️ 全スライド一覧 (`tab_gallery`)**:
   抽出された全ページを5カラムグリッドで一覧表示。除外・対象スライドがアイコン（✅/❌）で視覚的に把握可能。
2. **📝 処理対象スライドの詳細編集 (`tab_edit`)**:
   「スライド画像＋レーザーポインタ位置のプレビュー」「生成済み音声プレイヤー」「原稿テキストエリア」「文ごとのポインタ対象＆対訳字幕の編集テーブル」。
3. **🎬 スライドショービデオの確認 (`tab_video`)**:
   完成したチャプター・字幕内蔵MP4の再生、および `.srt` ファイルのダウンロードリンク。個別スライドごとの動画プレビューも可能。
4. **⚙ 設定 (`tab_settings`)**:
   * `config.yaml`（LLMのAPIエンドポイント、Model、Temperature、各言語のTTSモデル設定）の編集・保存。
   * `tts_filter.yaml`（専門用語・識別子の読み仮名辞書）を **Ag-Grid** コンポーネントを使って表形式で直接編集・追加・保存。

---

## 6. 特筆すべき実装テクニック

1. **`run.io_bound` によるイベントループの分離**
   PDFのレンダリング（PyMuPDF）や画像描画（Pillow）、ローカルファイルIO、コマンド実行（`subprocess`）など、CPU/IOバウンドな処理はすべて `await run.io_bound(...)` で別スレッドにオフロードされ、UIの快適なレスポンスが維持されています。
2. **設定の自動同期とカスケード保存**
   ページ範囲やモード（講義/研究）を変更すると、即座にプロジェクトルートの `project.json` へ書き出され、次回起動時やCLI実行時にも矛盾なく状態が引き継がれます。
3. **Ag-Grid による辞書エディタ**
   `tts_filter.yaml` の読み辞書をNiceGUI組み込みの `ui.aggrid` でバインドし、Webブラウザ上でスプレッドシート感覚で単語と読みをインライン編集可能にしています。