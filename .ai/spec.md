## クッキー型メーカー

## input 

- 白黒の手書き絵の画像


## output

- .stlファイル

## 処理
### 画像 -> SVG変換

- pythonライブラリ: openCV

- 1. グレースケール変換
- 2. 輪郭の検出
- 3. SVGファイル書き出し

- タスク: .ai/tasks/20260219_image_to_svg.md

### SVG画像 -> .stl

- pythonライブラリ: trimesh
