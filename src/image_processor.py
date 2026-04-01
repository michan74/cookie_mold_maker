"""画像処理モジュール - グレースケール変換と輪郭検出"""

import cv2
import numpy as np
from pathlib import Path


def load_image(image_path: str) -> np.ndarray:
    """画像を読み込む"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")

    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """画像をグレースケールに変換"""
    if len(image.shape) == 2:
        # 既にグレースケール
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def save_image(image: np.ndarray, output_path: str) -> None:
    """画像を保存"""
    cv2.imwrite(output_path, image)


def to_binary(gray_image: np.ndarray, threshold: int = 180, blur_size: int = 9) -> np.ndarray:
    """グレースケール画像を二値化

    Args:
        gray_image: グレースケール画像
        threshold: 二値化の閾値（デフォルト180）
        blur_size: ガウシアンブラーのカーネルサイズ（デフォルト9）

    処理内容:
        1. ガウシアンブラーでノイズ除去
        2. 二値化
        3. モルフォロジー演算（クロージング→オープニング）でノイズ除去
    """
    # 強めのぼかしでノイズ除去
    blurred = cv2.GaussianBlur(gray_image, (blur_size, blur_size), 0)

    # 二値化
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    # モルフォロジー演算でノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 小さな穴を埋める
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # 小さなノイズを除去

    return binary


def find_contours(binary_image: np.ndarray) -> list:
    """二値画像から輪郭を検出"""
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,  # 最外輪郭のみ
        cv2.CHAIN_APPROX_SIMPLE  # 輪郭を圧縮
    )
    return contours


def simplify_contour(contour: np.ndarray, epsilon_ratio: float = 0.01) -> np.ndarray:
    """輪郭を簡略化（点の数を減らす）"""
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def get_largest_contour(contours: list) -> np.ndarray:
    """最大の輪郭を取得"""
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def calculate_control_points(points: np.ndarray, smoothing: float = 0.2) -> list:
    """Catmull-Rom風のアルゴリズムで制御点を計算

    Args:
        points: 輪郭の点の配列 [(x1,y1), (x2,y2), ...]
        smoothing: 滑らかさの係数（0.1〜0.3が一般的）

    Returns:
        各点の制御点リスト [(ctrl1_x, ctrl1_y, ctrl2_x, ctrl2_y), ...]
    """
    n = len(points)
    control_points = []

    for i in range(n):
        # 前の点、現在の点、次の点を取得（ループするので % n）
        p_prev = points[(i - 1) % n]
        p_curr = points[i]
        p_next = points[(i + 1) % n]

        # 方向ベクトル（前の点から次の点へ）
        direction = p_next - p_prev

        # 角度に応じて係数を調整
        # 急カーブほど係数を小さく
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 > 0 and len2 > 0:
            # 内積からcos(角度)を計算
            cos_angle = np.dot(v1, v2) / (len1 * len2)
            cos_angle = np.clip(cos_angle, -1, 1)
            # 角度が急なほど係数を小さく（0.5〜1.0の範囲で調整）
            angle_factor = (1 + cos_angle) / 2  # 0〜1の範囲
            adjusted_smoothing = smoothing * (0.5 + 0.5 * angle_factor)
        else:
            adjusted_smoothing = smoothing

        # 制御点を計算
        ctrl1 = p_curr - direction * adjusted_smoothing
        ctrl2 = p_curr + direction * adjusted_smoothing

        control_points.append((ctrl1[0], ctrl1[1], ctrl2[0], ctrl2[1]))

    return control_points


def points_to_bezier_path(points: np.ndarray, smoothing: float = 0.2) -> str:
    """点列からベジェ曲線のSVGパスを生成

    Args:
        points: 輪郭の点の配列
        smoothing: 滑らかさの係数

    Returns:
        SVGのpath d属性の文字列
    """
    if len(points) < 3:
        return ""

    control_points = calculate_control_points(points, smoothing)
    n = len(points)

    # 開始点
    d = f"M {points[0][0]:.1f} {points[0][1]:.1f}"

    # 各点をベジェ曲線で接続
    for i in range(n):
        # 現在の点の「出口」制御点
        _, _, ctrl1_x, ctrl1_y = control_points[i]
        # 次の点の「入口」制御点
        next_i = (i + 1) % n
        ctrl2_x, ctrl2_y, _, _ = control_points[next_i]
        # 次の点
        end_x, end_y = points[next_i]

        # C = 3次ベジェ曲線（制御点2つ + 終点）
        d += f" C {ctrl1_x:.1f} {ctrl1_y:.1f}, {ctrl2_x:.1f} {ctrl2_y:.1f}, {end_x:.1f} {end_y:.1f}"

    return d


def export_contours_to_svg(
    contours: list,
    output_path: str,
    width: int,
    height: int,
    epsilon_ratio: float = 0.01,
    smoothing: float = 0.2,
) -> None:
    """輪郭をSVGファイルに書き出す（ベジェ曲線で滑らかに）

    Args:
        contours: 輪郭のリスト
        output_path: 出力ファイルパス
        width: SVGの幅
        height: SVGの高さ
        epsilon_ratio: 輪郭簡略化の係数（0またはNoneで簡略化を無効化）
        smoothing: ベジェ曲線の滑らかさ係数（0.1〜0.3）
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    paths_d = []
    for contour in contours:
        if len(contour) < 3:
            continue

        # epsilon_ratioが0またはNoneの場合は簡略化をスキップ
        if epsilon_ratio:
            simplified = simplify_contour(contour, epsilon_ratio)
            points = simplified.reshape(-1, 2)
        else:
            points = contour.reshape(-1, 2)

        if len(points) >= 3:
            d = points_to_bezier_path(points, smoothing)
        else:
            d = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
            for i in range(1, len(points)):
                d += f" L {points[i][0]:.1f} {points[i][1]:.1f}"
            d += " Z"

        paths_d.append(d)

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <g fill="none" stroke="black" stroke-width="1">
'''
    for d in paths_d:
        svg += f'    <path d="{d}"/>\n'
    svg += "  </g>\n</svg>\n"

    path.write_text(svg, encoding="utf-8")


def extract_contour_cv(
    image_path: str,
    output_path: str,
) -> str:
    """外枠抽出（クッキー型用）

    内側エッジからのflood fillで外枠を抽出する。

    処理:
    1. 外側からflood fillで外側領域をマーク
    2. 内側領域を特定（マークされなかった白い部分）
    3. 内側領域を白で塗りつぶし → 外枠だけ残る

    Args:
        image_path: 入力画像のパス（線の正規化済みの画像）
        output_path: 出力画像のパス

    Returns:
        出力画像のパス
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    # 画像を読み込み
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = image.shape

    # 二値化（白背景=255、黒線=0）
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Step 1: 外側からflood fillで外側領域をマーク
    flood_mask = binary.copy()

    # 4隅からflood fill（外側の白領域を灰色=128でマーク）
    corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
    for x, y in corners:
        if flood_mask[y, x] == 255:
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

    # Step 2: 内側領域を特定（128=外側、0=線、255=内側領域）
    inside_mask = (flood_mask == 255).astype(np.uint8) * 255

    # Step 3: 内側領域の輪郭を検出し、その内部を全て白で塗りつぶす
    contours, _ = cv2.findContours(inside_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = binary.copy()
    if contours:
        # 一番大きい内側領域（外枠の内側エッジ）で塗りつぶす
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(result, [largest], -1, 255, -1)

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
) -> str:
    """スタンプ用画像の抽出（外枠を除去）

    元画像から外枠を差し引いて、内部の要素のみを残す。

    処理:
    1. 外枠画像を膨張（dilate）させて太くする
    2. 元画像から膨張した外枠を差し引く
    3. 内部の線だけが残る

    Args:
        image_path: 入力画像のパス（線の正規化済みの画像）
        contour_image_path: 外枠画像のパス（extract_contour_cvで生成した画像）
        output_path: 出力画像のパス

    Returns:
        出力画像のパス
    """
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
    result = cv2.bitwise_or(original_bin, contour_dilated)

    # 出力
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), result)
    print(f"保存: {output_file}")

    return str(output_file)