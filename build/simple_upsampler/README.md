# simple_upsampler

画像を2倍サイズにアップサンプリングするサンプルです。  
拡大の際に多少いい感じに補間します。


## ディレクトリ

- build/simple_upsampler/ - プロジェクト
- samples/simple_upsampler/ - 実行用
- samples/simple_upsampler/learn.bat - 学習実行
- samples/simple_upsampler/predict.bat - 予測実行
- samples/simple_upsampler/learn/source/ - 学習用ソース
- samples/simple_upsampler/learn/teacher/ - 学習用教師データ
- samples/simple_upsampler/predict/src/ - 予測ソース
- samples/simple_upsampler/predict/out/ - 予測出力


## モデル

```cpp
mlp.AppendConvLayer( 64, 3, 3, 3, convPadBorder, 1, activReLU ) ;
mlp.AppendUp2x2Layer( 16, 64, 1, activReLU ) ;
mlp.AppendConvLayer( 3, 16, 3, 3, convPadZero, 1, activLinear ) ;
```


## 学習用データの準備

samples/simple_upsampler/make_learn/src/ 内に学習用画像（PNG|JPG|BMP）ファイルを設置し、

```bat
> cd samples/simple_upsampler
> make_learn.bat
```

を実行すると samples/simple_upsampler/learn/ 内に学習用画像ファイルが生成されます。  
実行には Python が必要です。


## メモ

より実用的なモデルのために、畳み込み範囲を広げる（ex.3→5）、チャネル数を増やす（ex.64→256）、L2正則化、学習時の適応的最適化の設定などを行うとよいかもしれません。

