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


def to_binary(gray_image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """グレースケール画像を二値化（ガウシアンブラーでノイズ除去のみ）"""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
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


def export_contours_to_svg(
    contours: list,
    output_path: str,
    width: int,
    height: int,
    epsilon_ratio: float = 0.01,
) -> None:
    """輪郭をSVGファイルに書き出す"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    paths_d = []
    for contour in contours:
        if len(contour) < 3:
            continue
        simplified = simplify_contour(contour, epsilon_ratio)
        points = simplified.reshape(-1, 2)
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