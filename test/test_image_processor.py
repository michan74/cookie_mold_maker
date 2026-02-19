"""画像処理モジュールのテスト"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_processor import load_image, to_grayscale, save_image


def test_to_grayscale_from_color():
    """カラー画像からグレースケール変換をテスト"""
    # 100x100のカラー画像を作成 (BGR)
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    color_image[:, :, 0] = 255  # 青チャンネル

    gray = to_grayscale(color_image)

    assert len(gray.shape) == 2, "グレースケール画像は2次元であるべき"
    assert gray.shape == (100, 100), "サイズが維持されるべき"


def test_to_grayscale_already_gray():
    """既にグレースケールの画像はそのまま返す"""
    gray_image = np.zeros((100, 100), dtype=np.uint8)
    gray_image[50, 50] = 128

    result = to_grayscale(gray_image)

    assert np.array_equal(result, gray_image), "同じ画像が返されるべき"


if __name__ == "__main__":
    test_to_grayscale_from_color()
    print("test_to_grayscale_from_color: OK")

    test_to_grayscale_already_gray()
    print("test_to_grayscale_already_gray: OK")

    print("\nすべてのテストが成功しました!")
