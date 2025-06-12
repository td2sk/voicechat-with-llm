# LLM VoiceChat

## 概要

このプロジェクトは、音声認識・音声合成・大規模言語モデル(LLM)を組み合わせた日本語音声対話システムです。リアルタイムで音声入力を認識し、LLM で応答を生成し、音声で返答します。

## 主な機能

- 音声認識 (Whisper)
- 音声合成 (VOICEVOX)
- LLM による対話 (Ollama)
- VAD(Voice Activity Detection)による自動音声検出
- マルチスレッドによるリアルタイム処理

## 必要環境

- Python 3.8 以降
- Windows (他 OS は未検証)
- Ollama
- VOICEVOX
- 必要な Python パッケージは`pyproject.toml`参照

## セットアップ

1. 必要なパッケージを[uv](https://github.com/astral-sh/uv)でインストール:

```sh
uv sync
```

2. キャラクター設定(システムプロンプト)をテキストファイルで用意してください。

-　利用可能なサンプルとして `examples/system-prompt.txt` があります

3. VOICEVOX エンジンが必要です。ローカルまたは API で利用できるようにしてください。
4. Ollama サーバーを起動し、利用したいモデルをダウンロードしてください。

## 実行方法

```sh
# 入力オーディオデバイスIDの確認
uv run ./audio/device.py

# 実行
 uv run main.py \
   --system-prompt-path ./examples/system_prompt.txt \
   --ollama-host 127.0.0.1:11434 \
   --whisper-model turbo \
   --whisper-device cuda \
   --whisper-type int8 \
   --whisper-beam-size 1 \
   --whisper-language ja \
   --ollama-model qwen3:32b \
   --vad-mode 3 \
   --voicevox-character ずんだもん \
   --audio-device-id 0 # 確認したオーディオデバイスID
```

- `--system-prompt-path` : LLM に指示するシステムプロンプトファイルのパス
- `--ollama-host` : Ollama サーバーの URL (デフォルト値: 127.0.0.1:11434)
- `--ollama-model` : 使用する LLM モデル名
- `--voicevox-endpoint` : VOICEVOX の接続先 (デフォルト値: http://127.0.0.1:50021)
- `--voicevox-character` : VOICEVOX で利用するキャラクター (デフォルト値: 四国めたん)
- `--vad-mode` : 無音検出の感度。0~ 3 で指定。大きいほど無音判定されやすい (デフォルト値: 3)
- `--audio-device-id` : マイクデバイス ID (デフォルト値: 0)
- `--whisper-mode` : Whisper の動作環境。local か remote で指定 (デフォルト値: local)
  - local の場合
    - `--whisper-model` : Whisper モデル名 (デフォルト値: turbo)
    - `--whisper-device` : Whisper の動作環境。cpu or cuda
    - `--whisper-type` : Whisper の精度 (デフォルト値: int8)
  - remote の場合
    - `--whisper-endpoint` : Whisper の接続先 (デフォルト値: http://127.0.0.1:8000)
  - 共通オプション
    - `--whisper-beam-size` : Whisper の検索サイズ (デフォルト値: 1)
    - `--whisper-language` : Whisper の聞き取る言語 (デフォルト値: None (自動推定))

## ファイル構成

- `main.py` : メインスクリプト
- `audio/` : 音声入出力・VAD・認識関連
- `llm/` : LLM クライアント
- `voice/` : VOICEVOX 連携
- `utils/` : 補助ユーティリティ
- `examples/` : サンプルのシステムプロンプト

## 注意事項

- VOICEVOX エンジンや Ollama サーバーは別途起動が必要です

## システムプロンプト

キャラクター設定をシステムプロンプトとしてファイルに記述して --system-prompt-path で指定してください。
`examples/system_prompt.txt` に記述例があります。

設定内では以下の変数が利用できます。

- `{%DEFAULT_OUTPUT_FORMAT}`: LLM に対する出力フォーマット指定 (後述)
- `{%VOICEVOX_CHARACTER}`: --voicevox-character で指定したキャラクター名
- `{%VOICEVOX_TONES}`: --voicevox-character で指定したキャラクター名で使える声のスタイルのリスト

### 出力フォーマット指定

本システムでは、LLM からの応答が特定のフォーマットになることを前提としています。

LLM からの応答例

```jsonc
{
  // 返答内容
  "content": "うーん、今日はカレーライスにするかな～。辛さは少なめで。",
  // VOICEVOX のスタイル(喋り方)
  "tone": "普通"
}
```

```json
{
  "content": "う、少しだけ辛いのもいいけど、今日は胃がちょっと弱いんだよ…。",
  "tone": "ツンツン"
}
```

そのため、フォーマット指定をかならずプロンプト内に記載してください。

フォーマットについてカスタマイズが不要な場合は変数 {%DEFAULT_OUTPUT_FORMAT} をプロンプト内に含めてください。以下のように置換されます。

```markdown
- json 形式で出力してください
- content 属性には、会話の内容を指定してください
- ユーザー側のセリフやナレーションは書かないでください
- tone 属性では、あなたの話のトーン、感情を指定します。返事の内容に沿って以下の中から選んでください
{%VOICEVOX_TONES}
```

※ Structured Output 機能により出力をコントロールしてはいますが、明示的に指示したほうが回答が安定します。

## ライセンス

MIT License

## 参考

[「WebRTC VAD」を試す - Zenn](https://zenn.dev/kun432/scraps/ec4666f467832c)
