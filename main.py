"""Cookie Mold Maker - 手書き画像からクッキー型のSTLファイルを生成"""

import argparse
from pathlib import Path

from src.image_processor import (
    load_image,
    to_grayscale,
    to_binary,
    find_contours,
    get_largest_contour,
    export_contours_to_svg,
)
from src.stl_generator import svg_to_stl


def process_image(
    image_path: str,
    output_dir: str = "output",
    use_ai: bool = False,
    all_contours: bool = False,
    generate_stl: bool = True,
    stl_height: float = 10.0,
    stl_wall_thickness: float = 1.0,
    stl_target_size: float = 50.0,
    smooth: bool = True,
    smoothing: float = 0.2,
):
    """画像からクッキー型を生成

    Args:
        image_path: 入力画像のパス
        output_dir: 出力ディレクトリ
        use_ai: AI前処理を使用するか（方眼紙除去）
        all_contours: すべての輪郭を出力するか
        generate_stl: STLファイルを生成するか
        stl_height: STLの高さ(mm)
        stl_wall_thickness: STLの壁の厚さ(mm)
        stl_target_size: STLの目標サイズ(mm)
        smooth: ベジェ曲線でスムージングするか
        smoothing: スムージングの強さ
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    name = Path(image_path).stem

    # Step 1: AI前処理（オプション）
    if use_ai:
        print("=== AI前処理（方眼紙除去）===")
        from src.ai_preprocessor import preprocess_image
        cleaned_path = output_path / f"{name}_cleaned.png"
        preprocess_image(image_path, str(cleaned_path))
        image_path = str(cleaned_path)

    # Step 2: 画像処理
    print("=== 画像処理 ===")
    image = load_image(image_path)
    gray = to_grayscale(image)
    binary = to_binary(gray, threshold=200, blur_size=5)
    contours = find_contours(binary)

    if not contours:
        print("輪郭が検出されませんでした")
        return None

    print(f"検出した輪郭: {len(contours)}個")

    if all_contours:
        contours_to_export = contours
    else:
        largest = get_largest_contour(contours)
        contours_to_export = [largest] if largest is not None else []

    # Step 3: SVG出力
    print("=== SVG出力 ===")
    svg_path = output_path / f"{name}.svg"
    export_contours_to_svg(
        contours_to_export,
        str(svg_path),
        width=image.shape[1],
        height=image.shape[0],
        smooth=smooth,
        smoothing=smoothing,
    )
    print(f"SVG保存: {svg_path}")

    # Step 4: STL生成（オプション）
    if generate_stl:
        print("=== STL生成 ===")
        stl_path = output_path / f"{name}.stl"
        svg_to_stl(
            str(svg_path),
            str(stl_path),
            height=stl_height,
            wall_thickness=stl_wall_thickness,
            target_size=stl_target_size,
        )
        print(f"STL保存: {stl_path}")
        return str(stl_path)

    return str(svg_path)


def main():
    parser = argparse.ArgumentParser(
        description="Cookie Mold Maker - 手書き画像からクッキー型を生成"
    )
    parser.add_argument("image_path", help="入力画像のパス")
    parser.add_argument("-o", "--output", default="output", help="出力ディレクトリ")
    parser.add_argument("--ai", action="store_true", help="AI前処理を使用（方眼紙除去）")
    parser.add_argument("--all-contours", action="store_true", help="すべての輪郭を出力")
    parser.add_argument("--no-stl", action="store_true", help="STL生成をスキップ")
    parser.add_argument("--height", type=float, default=10.0, help="STLの高さ(mm)")
    parser.add_argument("--wall", type=float, default=1.0, help="壁の厚さ(mm)")
    parser.add_argument("--size", type=float, default=50.0, help="目標サイズ(mm)")
    parser.add_argument("--no-smooth", action="store_true", help="スムージングを無効化")
    parser.add_argument("--smoothing", type=float, default=0.2, help="スムージングの強さ")

    args = parser.parse_args()

    print("Cookie Mold Maker")
    print(f"入力: {args.image_path}")
    print(f"AI前処理: {'有効' if args.ai else '無効'}")
    print()

    result = process_image(
        image_path=args.image_path,
        output_dir=args.output,
        use_ai=args.ai,
        all_contours=args.all_contours,
        generate_stl=not args.no_stl,
        stl_height=args.height,
        stl_wall_thickness=args.wall,
        stl_target_size=args.size,
        smooth=not args.no_smooth,
        smoothing=args.smoothing,
    )

    if result:
        print()
        print(f"=== 完了 ===")
        print(f"出力: {result}")


if __name__ == "__main__":
    main()
