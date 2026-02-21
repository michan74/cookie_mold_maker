"""STL生成モジュール - SVGからクッキー型のSTLを生成"""

import re
import numpy as np
import trimesh
from pathlib import Path


def parse_svg_path(svg_path: str) -> list:
    """SVGのpath要素のd属性をパースして座標リストに変換

    対応コマンド: M (moveto), L (lineto), Z (closepath)

    Args:
        svg_path: SVGのd属性値 (例: "M 100 100 L 200 100 L 200 200 Z")

    Returns:
        座標のリスト [(x1, y1), (x2, y2), ...]
    """
    # 数値を抽出（M, L, Zなどのコマンドは無視）
    numbers = re.findall(r'[-+]?\d*\.?\d+', svg_path)

    # 2つずつペアにして座標に変換
    coords = []
    for i in range(0, len(numbers) - 1, 2):
        x = float(numbers[i])
        y = float(numbers[i + 1])
        coords.append((x, y))

    return coords


def load_svg_paths(svg_file: str) -> list:
    """SVGファイルからすべてのpathのd属性を読み込む

    Args:
        svg_file: SVGファイルのパス

    Returns:
        各pathの座標リストのリスト [[(x1,y1),...], [(x1,y1),...], ...]
    """
    path = Path(svg_file)
    if not path.exists():
        raise FileNotFoundError(f"SVGファイルが見つかりません: {svg_file}")

    content = path.read_text(encoding='utf-8')

    # path要素のd属性を抽出
    d_values = re.findall(r'<path[^>]*d="([^"]*)"', content)

    paths = []
    for d in d_values:
        coords = parse_svg_path(d)
        if len(coords) >= 3:  # 最低3点必要
            paths.append(coords)

    return paths


def create_cookie_cutter(
    outline: list,
    height: float = 10.0,
    wall_thickness: float = 1.0
) -> trimesh.Trimesh:
    """2D輪郭からクッキー型（抜き型）の3Dメッシュを生成

    Args:
        outline: 輪郭の座標リスト [(x1, y1), (x2, y2), ...]
        height: 型の高さ (mm)
        wall_thickness: 壁の厚さ (mm)

    Returns:
        trimesh.Trimesh オブジェクト
    """
    from shapely.geometry import Polygon

    # 元の輪郭からポリゴンを作成
    base_polygon = Polygon(outline)

    # bufferで外側と内側の輪郭を作成
    outer = base_polygon.buffer(wall_thickness / 2)
    inner = base_polygon.buffer(-wall_thickness / 2)

    # 外側から内側を引いた形状を作成（リング状）
    ring = outer.difference(inner)

    # 2Dポリゴンを3Dに押し出し
    mesh = trimesh.creation.extrude_polygon(ring, height)

    return mesh


def export_stl(mesh: trimesh.Trimesh, output_path: str) -> None:
    """メッシュをSTLファイルとして保存

    Args:
        mesh: trimeshメッシュオブジェクト
        output_path: 出力ファイルパス
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def svg_to_stl(
    svg_file: str,
    output_file: str,
    height: float = 10.0,
    wall_thickness: float = 1.0,
    target_size: float = 50.0
) -> None:
    """SVGファイルからクッキー型のSTLを生成するメイン関数

    Args:
        svg_file: 入力SVGファイル
        output_file: 出力STLファイル
        height: 型の高さ (mm)
        wall_thickness: 壁の厚さ (mm)
        target_size: 型の最大サイズ (mm) - 長辺がこのサイズになるようスケール
    """
    # SVGからパスを読み込み
    paths = load_svg_paths(svg_file)

    if not paths:
        raise ValueError("SVGファイルにパスが見つかりません")

    # 最大のパス（メインの輪郭）を使用
    main_path = max(paths, key=len)

    # スケール計算（長辺がtarget_sizeになるように）
    coords = np.array(main_path)
    width = coords[:, 0].max() - coords[:, 0].min()
    height_2d = coords[:, 1].max() - coords[:, 1].min()
    max_dim = max(width, height_2d)
    scale = target_size / max_dim

    # 座標をスケール＆中心に移動
    center_x = (coords[:, 0].max() + coords[:, 0].min()) / 2
    center_y = (coords[:, 1].max() + coords[:, 1].min()) / 2
    scaled_path = [
        ((x - center_x) * scale, (y - center_y) * scale)
        for x, y in main_path
    ]

    # クッキー型を生成
    mesh = create_cookie_cutter(scaled_path, height, wall_thickness)

    # STLとして保存
    export_stl(mesh, output_file)
