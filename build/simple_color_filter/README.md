# simple_color_filter

ピクセル毎の単純な色変換を行うフィルターのサンプルです。  
学習用の画像サンプルとしてセピア調フィルタ学習用データも同梱しています。


## ディレクトリ

- build/simlpe_color_filter/ - プロジェクト
- samples/simple_color_filter/ - 実行用
- samples/simple_color_filter/learn.bat - 学習実行
- samples/simple_color_filter/predict.bat - 予測実行
- samples/simple_color_filter/learn/source/ - 学習用ソース
- samples/simple_color_filter/learn/teacher/ - 学習用教師データ
- samples/simple_color_filter/predict/src/ - 予測用ソース
- samples/simple_color_filter/predict/out/ - 予測出力


## モデル

```cpp
mlp.AppendLayer( 3, 3, 1, activLinear ) ;
```
