# Cookie Mold Maker

手書き画像からクッキー型の3D STLファイルを生成するツール。

## セットアップ

```bash
# Docker環境で実行
docker compose build
```

### AI前処理を使う場合

1. [Google AI Studio](https://aistudio.google.com/app/apikey)でAPIキーを取得
2. `.env`ファイルを作成:
```bash
cp .env.example .env
# GOOGLE_API_KEYを設定
```

## 使い方

### 基本的な使い方（綺麗な画像）

```bash
docker compose run --rm app python main.py <画像パス>
```

例:
```bash
docker compose run --rm app python main.py test/bear.jpg
```

### 方眼紙の画像（AI前処理あり）

```bash
docker compose run --rm app python main.py <画像パス> --ai
```

例:
```bash
docker compose run --rm app python main.py test/nuts.jpeg --ai
```

## オプション

| フラグ | 説明 | デフォルト |
|--------|------|-----------|
| `--ai` | AI前処理を有効化（方眼紙除去） | 無効 |
| `-o`, `--output` | 出力ディレクトリ | `output` |
| `--height` | STLの高さ(mm) | 10.0 |
| `--wall` | 壁の厚さ(mm) | 1.0 |
| `--size` | 目標サイズ(mm) | 50.0 |
| `--no-stl` | STL生成をスキップ | - |
| `--no-smooth` | スムージングを無効化 | - |
| `--smoothing` | スムージングの強さ | 0.2 |
| `--all-contours` | すべての輪郭を出力 | - |

## 出力ファイル

- `output/<name>.svg` - ベクター画像（ベジェ曲線）
- `output/<name>.stl` - 3Dプリント用クッキー型
- `output/<name>_cleaned.png` - AI処理後の画像（`--ai`使用時）

## パイプライン

```
入力画像
    ↓
[AI前処理] ← --ai オプション（方眼紙除去、文字除去）
    ↓
グレースケール変換
    ↓
二値化
    ↓
輪郭検出
    ↓
SVG出力（ベジェ曲線スムージング）
    ↓
STL生成（クッキー型）
```
