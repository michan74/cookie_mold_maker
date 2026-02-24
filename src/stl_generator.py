"""STL生成モジュール - SVGからクッキー型のSTLを生成"""

import re
import numpy as np
import trimesh
from pathlib import Path


def cubic_bezier_point(p0, p1, p2, p3, t):
    """3次ベジェ曲線上の点を計算

    Args:
        p0, p1, p2, p3: 制御点 (x, y)
        t: パラメータ (0.0 ~ 1.0)

    Returns:
        (x, y) 曲線上の点
    """
    x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
    y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
    return (x, y)


def sample_bezier_curve(p0, p1, p2, p3, num_samples=10):
    """ベジェ曲線を複数の点でサンプリング

    Args:
        p0, p1, p2, p3: 制御点
        num_samples: サンプリング数

    Returns:
        曲線上の点のリスト
    """
    points = []
    for i in range(num_samples):
        t = i / num_samples
        points.append(cubic_bezier_point(p0, p1, p2, p3, t))
    return points


def parse_svg_path(svg_path: str, bezier_samples: int = 30) -> list:
    """SVGのpath要素のd属性をパースして座標リストに変換

    対応コマンド: M (moveto), L (lineto), C (cubic bezier), Z (closepath)

    Args:
        svg_path: SVGのd属性値
        bezier_samples: ベジェ曲線1つあたりのサンプリング数

    Returns:
        座標のリスト [(x1, y1), (x2, y2), ...]
    """
    coords = []
    current_pos = (0, 0)

    # コマンドと数値を分離
    # M, L, C, Z とその後の数値をマッチ
    tokens = re.findall(r'([MLCZ])|(-?\d+\.?\d*)', svg_path)

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token[0] == 'M':  # MoveTo
            i += 1
            x = float(tokens[i][1])
            i += 1
            y = float(tokens[i][1])
            current_pos = (x, y)
            coords.append(current_pos)
            i += 1

        elif token[0] == 'L':  # LineTo
            i += 1
            x = float(tokens[i][1])
            i += 1
            y = float(tokens[i][1])
            current_pos = (x, y)
            coords.append(current_pos)
            i += 1

        elif token[0] == 'C':  # Cubic Bezier
            i += 1
            # 制御点1
            c1x = float(tokens[i][1])
            i += 1
            c1y = float(tokens[i][1])
            i += 1
            # 制御点2
            c2x = float(tokens[i][1])
            i += 1
            c2y = float(tokens[i][1])
            i += 1
            # 終点
            ex = float(tokens[i][1])
            i += 1
            ey = float(tokens[i][1])

            # ベジェ曲線をサンプリング
            p0 = current_pos
            p1 = (c1x, c1y)
            p2 = (c2x, c2y)
            p3 = (ex, ey)
            sampled = sample_bezier_curve(p0, p1, p2, p3, bezier_samples)
            coords.extend(sampled)

            current_pos = (ex, ey)
            i += 1

        elif token[0] == 'Z':  # ClosePath
            i += 1

        else:
            i += 1

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


def create_stamp_base(
    outline: list,
    plate_thickness: float = 3.0,
    plate_offset: float = 1.0,
    handle_size: float = 10.0,
    handle_height: float = 15.0
) -> trimesh.Trimesh:
    """スタンプの基礎（板＋取手）を作成

    Args:
        outline: 輪郭の座標リスト [(x1, y1), (x2, y2), ...]
        plate_thickness: 板の厚さ (mm)
        plate_offset: 輪郭から内側にオフセットする距離 (mm)
        handle_size: 取手の一辺のサイズ (mm)
        handle_height: 取手の高さ (mm)

    Returns:
        trimesh.Trimesh オブジェクト
    """
    from shapely.geometry import Polygon, box

    # 輪郭からポリゴンを作成し、内側にオフセット
    base_polygon = Polygon(outline)
    plate_polygon = base_polygon.buffer(-plate_offset)

    if plate_polygon.is_empty:
        raise ValueError("オフセット後のポリゴンが空になりました。オフセット値が大きすぎます。")

    # 板を押し出し
    plate_mesh = trimesh.creation.extrude_polygon(plate_polygon, plate_thickness)

    # 取手を作成（板の重心位置に配置）
    centroid = plate_polygon.centroid
    half_size = handle_size / 2
    handle_polygon = box(
        centroid.x - half_size,
        centroid.y - half_size,
        centroid.x + half_size,
        centroid.y + half_size
    )
    handle_mesh = trimesh.creation.extrude_polygon(handle_polygon, handle_height)

    # 取手を板の上に移動
    handle_mesh.apply_translation([0, 0, plate_thickness])

    # 板と取手を結合
    combined = trimesh.util.concatenate([plate_mesh, handle_mesh])

    return combined


def create_stamp_relief(
    paths: list,
    relief_height: float = 1.5,
    line_width: float = 1.0
) -> trimesh.Trimesh:
    """スタンプの凸部分（レリーフ）を作成

    Args:
        paths: パスのリスト [[(x1,y1),...], [(x1,y1),...], ...]
        relief_height: 凸部分の高さ (mm)
        line_width: 線の幅 (mm)

    Returns:
        trimesh.Trimesh オブジェクト
    """
    from shapely.geometry import LineString, MultiPolygon
    from shapely.ops import unary_union

    relief_polygons = []

    for path in paths:
        if len(path) < 2:
            continue

        # パスをLineStringとして作成し、バッファで幅を持たせる
        line = LineString(path)
        buffered = line.buffer(line_width / 2, cap_style=2, join_style=2)

        if not buffered.is_empty:
            relief_polygons.append(buffered)

    if not relief_polygons:
        return None

    # すべてのポリゴンを結合
    combined_polygon = unary_union(relief_polygons)

    if combined_polygon.is_empty:
        return None

    # MultiPolygonの場合は個別に処理
    if isinstance(combined_polygon, MultiPolygon):
        meshes = []
        for poly in combined_polygon.geoms:
            if not poly.is_empty:
                mesh = trimesh.creation.extrude_polygon(poly, relief_height)
                meshes.append(mesh)
        if meshes:
            return trimesh.util.concatenate(meshes)
        return None
    else:
        return trimesh.creation.extrude_polygon(combined_polygon, relief_height)


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


def svg_to_stamp_stl(
    contour_svg_file: str,
    stamp_svg_file: str,
    output_file: str,
    target_size: float = 50.0,
    plate_thickness: float = 3.0,
    plate_offset: float = 1.0,
    handle_size: float = 10.0,
    handle_height: float = 15.0,
    relief_height: float = 1.5,
    line_width: float = 1.0
) -> None:
    """SVGファイルからスタンプのSTLを生成

    Args:
        contour_svg_file: 輪郭SVGファイル（スタンプの板の形状）
        stamp_svg_file: スタンプ用SVGファイル（凸部分のパターン）
        output_file: 出力STLファイル
        target_size: スタンプの最大サイズ (mm)
        plate_thickness: 板の厚さ (mm)
        plate_offset: 輪郭から内側にオフセットする距離 (mm)
        handle_size: 取手の一辺のサイズ (mm)
        handle_height: 取手の高さ (mm)
        relief_height: 凸部分の高さ (mm)
        line_width: 凸部分の線幅 (mm)
    """
    # 輪郭SVGを読み込み
    contour_paths = load_svg_paths(contour_svg_file)
    if not contour_paths:
        raise ValueError("輪郭SVGファイルにパスが見つかりません")

    # スタンプ用SVGを読み込み
    stamp_paths = load_svg_paths(stamp_svg_file)

    # メインの輪郭を取得
    main_contour = max(contour_paths, key=len)

    # スケール計算
    coords = np.array(main_contour)
    width = coords[:, 0].max() - coords[:, 0].min()
    height_2d = coords[:, 1].max() - coords[:, 1].min()
    max_dim = max(width, height_2d)
    scale = target_size / max_dim

    # 中心座標
    center_x = (coords[:, 0].max() + coords[:, 0].min()) / 2
    center_y = (coords[:, 1].max() + coords[:, 1].min()) / 2

    # 輪郭をスケール＆中心に移動
    scaled_contour = [
        ((x - center_x) * scale, (y - center_y) * scale)
        for x, y in main_contour
    ]

    # スタンプ用パスをスケール＆中心に移動
    scaled_stamp_paths = []
    for path in stamp_paths:
        scaled_path = [
            ((x - center_x) * scale, (y - center_y) * scale)
            for x, y in path
        ]
        scaled_stamp_paths.append(scaled_path)

    # スタンプ基礎を作成
    base_mesh = create_stamp_base(
        scaled_contour,
        plate_thickness=plate_thickness,
        plate_offset=plate_offset,
        handle_size=handle_size,
        handle_height=handle_height
    )

    # スタンプ凸部分を作成
    meshes = [base_mesh]

    if scaled_stamp_paths:
        relief_mesh = create_stamp_relief(
            scaled_stamp_paths,
            relief_height=relief_height,
            line_width=line_width
        )
        if relief_mesh is not None:
            # 凸部分をZ=0の位置に配置（板の下面）
            # スタンプは押すときに下向きになるので、凸部分は板の下に配置
            relief_mesh.apply_translation([0, 0, -relief_height])
            meshes.append(relief_mesh)

    # すべてのメッシュを結合
    combined = trimesh.util.concatenate(meshes)

    # STLとして保存
    export_stl(combined, output_file)
