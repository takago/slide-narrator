# slide-narrator
Automated AI presenter that turns PDF slides into chaptered videos with voice, subtitles, and laser focus.

講義や研究発表用のPDF形式スライドから音声付きスライドショー動画（日本語・英語字幕対応）を自動的に生成するシステムです。
 - 日本語スライド → 日本語音声＋和英字幕
 - 英語スライド → 英語音声＋和英字幕
 - 音声サンプルがあれば，その声で解説音声にできます．
 - レーザポインタ風マーカーでどこを話しているかをポイントしますので，ある程度は視聴者の視線を誘導できます．プログラムコードやベクター系の図であれば部分的にポイントはできます（期待は厳禁）．
 - 全て処理をローカル環境で済ませることが可能です（OpenAI互換APIをもったLLM/TTSサーバをローカルで稼働させてください）．以下は私が使っている環境です．
   - LLMサーバ: ollama ( https://ollama.com/ )
     - LLMは"Qwen3.8-27b ( https://huggingface.co/Qwen/Qwen3.8-27B )
   - 英語TTSサーバ: remskyさんの https://github.com/remsky/Kokoro-FastAPI 
   - 日本語TTSサーバ: Aratakoさんの https://github.com/Aratako/Irodori-TTS-Server 
     -　Irodori-TTS用の参照音声としてはhadouさんの　https://huggingface.co/datasets/hadou1225/Hadou-Voice-Dataset

## セットアップ(Linuxの場合)
```bash
sudo apt install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -p 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

vi config.yaml        # OpenAI互換エンドポイントを持ったLLM，TTSサーバを指定してください．
vi tts_filter.yaml    # OpenAI互換エンドポイントを持ったLLMを指定してください．
```

## 起動
```bash

streamlit run app.py
```

## ライセンス (License)
本プロジェクトは GNU General Public License v3.0 (GPLv3) の下で公開します。
詳細は LICENSE ファイルをご確認ください。
