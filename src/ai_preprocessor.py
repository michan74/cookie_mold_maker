"""AI画像前処理モジュール - Google GenAI SDKを使った画像クリーンアップ"""

import os
from pathlib import Path

from google import genai
from google.genai import types


def init_genai(api_key: str = None):
    """Google GenAI SDKを初期化

    Args:
        api_key: Google AI Studio APIキー（Noneの場合は環境変数から取得）

    Returns:
        初期化されたクライアント
    """
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "Google API Keyが必要です。"
            "環境変数 GOOGLE_API_KEY を設定するか、引数で指定してください。"
        )

    client = genai.Client(api_key=api_key)
    return client


def _get_mime_type(path: Path) -> str:
    """画像のMIMEタイプを取得"""
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(ext, "image/jpeg")


def _generate_image(
    client,
    contents: list,
    model_name: str = "gemini-2.5-flash-image"
) -> bytes:
    """AIで画像を生成"""
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            temperature=0.1,
        ),
    )

    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.inline_data:
                    return part.inline_data.data
        # コンテンツがない場合の詳細情報
        finish_reason = getattr(candidate, 'finish_reason', None)
        safety_ratings = getattr(candidate, 'safety_ratings', None)
        raise ValueError(
            f"AIからの画像レスポンスを取得できませんでした。"
            f"finish_reason: {finish_reason}, safety_ratings: {safety_ratings}"
        )

    raise ValueError("AIからの画像レスポンスを取得できませんでした（candidatesが空）")


def _save_image(image_data: bytes, output_path: str) -> str:
    """画像を保存"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(image_data)
    print(f"保存: {output_file}")
    return str(output_file)


def remove_background(
    image_path: str,
    output_path: str,
    model_name: str = "gemini-2.5-flash-image"
) -> str:
    """背景削除

    - 方眼用紙などの背景を削除
    - 背景は白一色にする

    Args:
        image_path: 入力画像のパス
        output_path: 出力画像のパス
        model_name: 使用するモデル名

    Returns:
        出力画像のパス
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    client = init_genai()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime_type = _get_mime_type(path)

    prompt = """画像下処理: 背景削除

要件:
- 方眼用紙などの背景のみを削除する
- 背景は純粋な白(#FFFFFF)にする
- それ以外は何も変更しない

画像のみを出力してください。"""

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        prompt,
    ]

    image_data = _generate_image(client, contents, model_name)
    return _save_image(image_data, output_path)


def normalize_lines(
    image_path: str,
    output_path: str,
    model_name: str = "gemini-2.5-flash-image"
) -> str:
    """線の正規化

    - 全ての線を黒に統一
    - 薄い線も含めて全て保持
    - ノイズ除去

    Args:
        image_path: 入力画像のパス（背景削除済みの画像）
        output_path: 出力画像のパス
        model_name: 使用するモデル名

    Returns:
        出力画像のパス
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    client = init_genai()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime_type = _get_mime_type(path)

    prompt = """画像下処理: 線の正規化

要件:
- 全ての線を黒(#000000)に統一する
- 背景は純粋な白(#FFFFFF)のまま維持する
- 線のノイズ（端の細いひげ、意図しない飛び出し）は除去する
- 元の画像にない線を追加しないこと

画像のみを出力してください。"""

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        prompt,
    ]

    image_data = _generate_image(client, contents, model_name)
    return _save_image(image_data, output_path)


def extract_contour(
    image_path: str,
    output_path: str,
    model_name: str = "gemini-2.5-flash-image"
) -> str:
    """輪郭抽出（クッキー型用）

    - 一番外側の輪郭のみを抽出
    - 一筆書きで描ける閉じた形状にする
    - 内部の線、文字、模様は全て削除

    Args:
        image_path: 入力画像のパス（線の正規化済みの画像）
        output_path: 出力画像のパス
        model_name: 使用するモデル名

    Returns:
        出力画像のパス
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    client = init_genai()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime_type = _get_mime_type(path)

    prompt = """クッキー型の輪郭抽出

この画像の「シルエット」の輪郭線だけを描いてください。

やること:
1. 図形全体を黒く塗りつぶしたと想像する
2. その塗りつぶした形の外周線だけを描く

出力する線:
- 図形の一番外側の境界線のみ（1本の閉じた線）

削除するもの:
- 内部の線（縦線、横線、斜め線など全て）
- 文字
- 模様
- 図形の中にあるもの全て

出力形式:
- 背景: 白(#FFFFFF)
- 輪郭線: 黒(#000000)
- 線は滑らかで均等な太さ

画像のみを出力してください。"""

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        prompt,
    ]

    image_data = _generate_image(client, contents, model_name)
    return _save_image(image_data, output_path)


def extract_stamp(
    image_path: str,
    contour_image_path: str,
    output_path: str,
    model_name: str = None  # 未使用（後方互換のため残す）
) -> str:
    """スタンプ抽出（外枠を除去）- OpenCV版

    - 元画像から輪郭画像の線を差し引く
    - 内部の要素（文字、模様など）のみを残す

    Args:
        image_path: 入力画像のパス（線の正規化済みの画像）
        contour_image_path: クッキー型用輪郭画像のパス（extract_contourで生成した画像）
        output_path: 出力画像のパス
        model_name: 未使用（後方互換のため）

    Returns:
        出力画像のパス
    """
    import cv2
    import numpy as np

    path = Path(image_path)
    contour_path = Path(contour_image_path)
    if not path.exists():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
    if not contour_path.exists():
        raise FileNotFoundError(f"輪郭画像ファイルが見つかりません: {contour_image_path}")

    # 画像を読み込み（グレースケール）
    original = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    contour = cv2.imread(str(contour_path), cv2.IMREAD_GRAYSCALE)

    # サイズを合わせる（必要な場合）
    if original.shape != contour.shape:
        contour = cv2.resize(contour, (original.shape[1], original.shape[0]))

    # 二値化（白背景=255、黒線=0）
    _, original_bin = cv2.threshold(original, 127, 255, cv2.THRESH_BINARY)
    _, contour_bin = cv2.threshold(contour, 127, 255, cv2.THRESH_BINARY)

    # 輪郭線を膨張させて確実に消す（線の太さの違いを吸収）
    kernel = np.ones((5, 5), np.uint8)
    contour_dilated = cv2.dilate(255 - contour_bin, kernel, iterations=2)

    # 差分計算: 元画像の線から輪郭の線を除去
    # 元画像の線（黒=0）で、輪郭にない部分を残す
    result = cv2.bitwise_or(original_bin, contour_dilated)

    # 出力
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), result)
    print(f"保存: {output_file}")

    return str(output_file)
