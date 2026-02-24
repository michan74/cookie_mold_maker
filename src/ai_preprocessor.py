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
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return part.inline_data.data

    raise ValueError("AIからの画像レスポンスを取得できませんでした")


def _save_image(image_data: bytes, output_path: str) -> str:
    """画像を保存"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(image_data)
    print(f"保存: {output_file}")
    return str(output_file)


def preprocess_step1_clean(
    image_path: str,
    output_path: str,
    model_name: str = "gemini-2.5-flash-image"
) -> str:
    """画像下処理1: 背景削除

    - 方眼用紙などの背景を削除
    - 背景は白一色にする
    - 線はなめらかな曲線にする
    - 枠の部分は黒色で均等な太さにする
    - 線の端の跳ねやはみ出しを処理して綺麗にする

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
- 方眼用紙などの背景を削除する
- 背景は純粋な白(#FFFFFF)にする
- クッキー型にするために線は、なめらかな曲線にする
- 枠の部分は黒色(#000000)で均等な太さにする
- 線の端の跳ねやはみ出しを処理して、線の端を綺麗に処理する

画像のみを出力してください。"""

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        prompt,
    ]

    image_data = _generate_image(client, contents, model_name)
    return _save_image(image_data, output_path)


def preprocess_step2_contour(
    image_path: str,
    output_path: str,
    model_name: str = "gemini-2.5-flash-image"
) -> str:
    """画像下処理2: クッキー型用枠

    - 一番外側の輪郭のみを抽出
    - 一筆書きで描ける閉じた形状にする
    - 内部の線、文字、模様は全て削除

    Args:
        image_path: 入力画像のパス（step1で処理済みの画像）
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

    prompt = """画像下処理: クッキー型用枠の抽出

要件:
- 渡された画像から一番外側の輪郭(クッキー型にする部分)となる部分のみを抜き出す
- 一筆書きで描ける閉じた形状にする（完全に閉じた1本の輪郭線のみ）
- 内部を横切る線は全て削除する
- それ以外の部分（文字、模様、内側の線など）は全て削除する
- 背景は純粋な白(#FFFFFF)にする
- 輪郭線は黒(#000000)にする
- 輪郭線は滑らかにし、均等な太さにする

画像のみを出力してください。"""

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        prompt,
    ]

    image_data = _generate_image(client, contents, model_name)
    return _save_image(image_data, output_path)


def preprocess_step3_stamp(
    image_path: str,
    contour_image_path: str,
    output_path: str,
    model_name: str = "gemini-2.5-flash-image"
) -> str:
    """画像下処理3: スタンプ用

    - クッキー型用枠を削除
    - それ以外の全ての要素（文字、内部の模様など）を残す

    Args:
        image_path: 入力画像のパス（step1で処理済みの画像）
        contour_image_path: クッキー型用輪郭画像のパス（step2で生成した画像）
        output_path: 出力画像のパス
        model_name: 使用するモデル名

    Returns:
        出力画像のパス
    """
    path = Path(image_path)
    contour_path = Path(contour_image_path)
    if not path.exists():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
    if not contour_path.exists():
        raise FileNotFoundError(f"輪郭画像ファイルが見つかりません: {contour_image_path}")

    client = init_genai()

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    with open(contour_image_path, "rb") as f:
        contour_bytes = f.read()

    mime_type = _get_mime_type(path)
    contour_mime_type = _get_mime_type(contour_path)

    prompt = """画像下処理: スタンプ用

2つの画像を提供します。

画像1: 処理済みの手書きイラスト（文字やディテールを含む）
画像2: クッキー型用の輪郭線（削除すべき外側の輪郭）

タスク:
- 画像1から画像2に含まれる輪郭線を削除する
- それ以外の全ての要素（文字、内部の模様など）を残す
- 細すぎる線（クッキーに押すには不向きな線）は削除する
- 線の端の跳ねやはみ出しを処理して、線の端を綺麗にする

出力形式:
- 背景: 純粋な白(#FFFFFF)
- 残す線: 黒(#000000)

画像のみを出力してください。"""

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        types.Part.from_bytes(data=contour_bytes, mime_type=contour_mime_type),
        prompt,
    ]

    image_data = _generate_image(client, contents, model_name)
    return _save_image(image_data, output_path)
