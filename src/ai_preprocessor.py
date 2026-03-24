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


def separate_outline(
    image_path: str,
    output_path: str,
    model_name: str = "gemini-2.5-flash-image"
) -> str:
    """外枠と内側の線を分離

    - 外枠と内側の線がつながっている部分を切り離す
    - 外枠は閉じた状態を維持
    - 内側の線は外枠から離れた状態にする

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

    prompt = """Image editing: Disconnect internal lines from outline

Task:
- Find where internal lines touch the outer boundary
- Shorten the internal lines slightly to create a small gap
- Do NOT modify the outer boundary shape at all

Rules:
- Do NOT add any new lines
- Do NOT make lines thicker or thinner
- Do NOT duplicate any lines
- Keep all existing lines in their original positions

Output:
- Background: white (#FFFFFF)
- Lines: black (#000000)

Output image only."""

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

    prompt = """画像編集: 外枠だけを残す

この画像から、図形の一番外側の枠線だけを残してください。

消すもの:
- 図形の中央を通る線（縦線、斜め線など）
- 文字（「nuts」など）
- 模様
- 外枠以外の全ての線

残すもの:
- 図形の外周を囲む線だけ（クッキー型の形になる部分）

重要:
- 外枠の線は元の位置・太さをそのまま維持する
- 外枠は閉じた1本の線になるようにする

出力形式:
- 背景: 白(#FFFFFF)
- 外枠線: 黒(#000000)

画像のみを出力してください。"""

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        prompt,
    ]

    image_data = _generate_image(client, contents, model_name)
    return _save_image(image_data, output_path)


def extract_contour_cv(
    image_path: str,
    output_path: str,
) -> str:
    """輪郭抽出（クッキー型用）- OpenCV版

    内側エッジからのflood fillで外枠を抽出:
    1. 外側からflood fillで外側領域をマーク
    2. 外枠の内側エッジを検出
    3. 内側領域を白で塗りつぶし → 外枠だけ残る

    Args:
        image_path: 入力画像のパス（線の正規化済みの画像）
        output_path: 出力画像のパス

    Returns:
        出力画像のパス
    """
    import cv2
    import numpy as np

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    # 画像を読み込み
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = image.shape

    # 二値化（白背景=255、黒線=0）
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Step 1: 外側からflood fillで外側領域をマーク
    # 作業用にコピー（flood fillは画像を変更する）
    flood_mask = binary.copy()

    # 4隅からflood fill（外側の白領域を灰色=128でマーク）
    corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
    for x, y in corners:
        if flood_mask[y, x] == 255:  # 白（背景）なら
            cv2.floodFill(flood_mask, None, (x, y), 128)

    # 端の全周からもflood fill（隅だけでは届かない場合）
    for x in range(w):
        if flood_mask[0, x] == 255:
            cv2.floodFill(flood_mask, None, (x, 0), 128)
        if flood_mask[h-1, x] == 255:
            cv2.floodFill(flood_mask, None, (x, h-1), 128)
    for y in range(h):
        if flood_mask[y, 0] == 255:
            cv2.floodFill(flood_mask, None, (0, y), 128)
        if flood_mask[y, w-1] == 255:
            cv2.floodFill(flood_mask, None, (w-1, y), 128)

    # Step 2: 内側領域を特定
    # 128=外側、0=線、255=内側領域
    inside_mask = (flood_mask == 255).astype(np.uint8) * 255

    # Step 3: 内側領域の輪郭を検出し、その内部を全て白で塗りつぶす
    contours, _ = cv2.findContours(inside_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = binary.copy()
    if contours:
        # 一番大きい内側領域（外枠の内側エッジ）で塗りつぶす
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(result, [largest], -1, 255, -1)  # 内側を全て白で塗りつぶす

    # 出力
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), result)
    print(f"保存: {output_file}")

    return str(output_file)


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
    kernel = np.ones((7, 7), np.uint8)
    contour_dilated = cv2.dilate(255 - contour_bin, kernel, iterations=3)

    # 差分計算: 元画像の線から輪郭の線を除去
    # 元画像の線（黒=0）で、輪郭にない部分を残す
    result = cv2.bitwise_or(original_bin, contour_dilated)

    # 出力
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), result)
    print(f"保存: {output_file}")

    return str(output_file)
