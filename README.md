# Respiration Rate Estimator

Webカメラまたは動画ファイルから顔を認識し、その下の胸領域を切り出して、顔と胸の動き信号を融合して呼吸数を推定する Python プロトタイプです。表示ウインドウには顔ROI・胸ROIと、呼吸帯域だけを通した波形グラフを出します。

波形グラフは横幅あたりの表示時間を短めにしてあるので、呼吸の波が詰まりすぎず、少ない波数で見やすく表示されます。

顔と胸の位置が揺れやすい場合は、内部で時系列の平滑化をかけています。必要なら `--smooth-alpha` で追従の速さと安定性のバランスを調整できます。小さいほど安定し、大きいほど素早く追従します。

## 使い方

```bash
pip install -r requirements.txt
python respiration_rate.py
```

表示ウインドウを無効化したい場合は `--no-show` を付けます。

```bash
python respiration_rate.py --no-show
```

動画ファイルを使う場合は `--source` にパスを指定します。

```bash
python respiration_rate.py --source sample.mp4 --show
```

ROI のブレを少し強めに抑えたい場合は、たとえば次のようにします。

```bash
python respiration_rate.py --smooth-alpha 0.12
```

## Nix で管理する

このプロジェクトは Nix の開発シェルで依存関係を揃えられます。

```bash
nix develop
python respiration_rate.py
```

`flake` を使わない場合は次でも入れます。

```bash
nix-shell
python respiration_rate.py
```

## 補足

この実装は実用上の出発点です。照明条件、姿勢、服装、カメラ品質で精度は大きく変わります。必要なら次の段階で、胸部ROIの改善、信号前処理の強化、安定化、GUI化を追加できます。
