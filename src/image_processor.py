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


def remove_colored_lines(image: np.ndarray, saturation_threshold: int = 30) -> np.ndarray:
    """色付きの線を除去し、黒い線（鉛筆・ボールペン）だけ残す

    HSV色空間で:
    - 黒い線 = 彩度(S)が低い + 明度(V)が低い
    - 色付き線 = 彩度(S)が高い（ピンク、青など）
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 彩度が低い（黒/グレー）ピクセルだけマスク
    # 彩度が高い = 色付き = 除去対象
    gray_mask = s < saturation_threshold

    # 結果画像（白背景）
    result = np.full_like(v, 255)

    # 黒い線の部分だけ元の明度を使用
    result[gray_mask] = v[gray_mask]

    return result


def save_image(image: np.ndarray, output_path: str) -> None:
    """画像を保存"""
    cv2.imwrite(output_path, image)


def to_binary(gray_image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """グレースケール画像を二値化"""
    # ガウシアンブラーでノイズ除去
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 適応的閾値処理（照明ムラに強い）
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
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