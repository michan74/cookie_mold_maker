"""Cookie Mold Maker - 手書き画像からクッキー型のSTLファイルを生成"""

import sys
from pathlib import Path

from src.image_processor import (
    load_image,
    to_grayscale,
    to_binary,
    find_contours,
    get_largest_contour,
    export_contours_to_svg,
)


def main():
    print("Cookie Mold Maker")
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [--all-contours]")
        print("  --all-contours  すべての輪郭をSVGに出力（指定なしの場合は最大輪郭のみ）")
        return

    image_path = sys.argv[1]
    all_contours = "--all-contours" in sys.argv

    image = load_image(image_path)
    gray = to_grayscale(image)
    binary = to_binary(gray, threshold=150)
    contours = find_contours(binary)

    if not contours:
        print("輪郭が検出されませんでした")
        return

    if all_contours:
        contours_to_export = contours
    else:
        largest = get_largest_contour(contours)
        contours_to_export = [largest] if largest is not None else []

    name = Path(image_path).stem
    output_dir = Path("output")
    svg_path = output_dir / f"{name}.svg"
    export_contours_to_svg(
        contours_to_export,
        str(svg_path),
        width=image.shape[1],
        height=image.shape[0],
    )
    print(f"SVG saved: {svg_path}")


if __name__ == "__main__":
    main()
