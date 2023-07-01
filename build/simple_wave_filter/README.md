# simple_wave_filter

単純な RNN での音声フィルタです。イコライザを模擬します。
入力を4サンプル毎にしているのは処理を少し軽くするためで、本当は1サンプル毎のほうがいいでしょう。
少しづつチャネル数を増やして出来るだけ広い周波数帯の特性に対応できることを期待しています。


## ディレクトリ

- build/simple_wave_filter/ - プロジェクト
- samples/simple_wave_filter/ - 実行用
- samples/simple_wave_filter/learn.bat - 学習実行
- samples/simple_wave_filter/predict.bat - 予測実行
- samples/simple_wave_filter/learn/source/ - 学習用ソース
- samples/simple_wave_filter/learn/teacher/ - 学習用教師データ
- samples/simple_wave_filter/predict/src/ - 予測ソース
- samples/simple_wave_filter/predict/out/ - 予測出力


## モデル

```cpp
mlp.SetInputShape
    ( NNMultiLayerPerceptron::mlpFlagStream,
            NNBufDim( 5, 1, 4 ), NNBufDim( 1, 1, 4 ) ) ;
mlp.AppendLayer( 4, 4, 0, activLinear ) ;
mlp.AppendLayer( 8, 12, 0, activLinear )
    ->AddConnection( 1, 0, 0, 4 )            // 直前レイヤーから  : 4チャネル入力
    ->AddConnection( 0, 1, 0, 8 ) ;          // このレイヤー(t-1) : 8チャネル入力
mlp.AppendLayer( 16, 24, 0, activLinear )
    ->AddConnection( 1, 0, 0, 8 )
    ->AddConnection( 0, 1, 0, 16 ) ;
mlp.AppendLayer( 32, 48, 0, activLinear )
    ->AddConnection( 1, 0, 0, 16 )
    ->AddConnection( 0, 1, 0, 32 ) ;
mlp.AppendLayer( 64, 96, 0, activLinear )
    ->AddConnection( 1, 0, 0, 32 )
    ->AddConnection( 0, 1, 0, 64 ) ;
mlp.AppendLayer( 4, 64, 0, activLinear ) ;
```


## 学習用データの準備

samples/simple_wave_filter/learn/source/ に元の wave ファイル、samples/simple_wave_filter/learn/teacher/ に何らかのハイパスやローパス、エコライザなど（同じ効果）を掛けた wave ファイルを置いてください。  
1曲（1ファイル）あればおおよそ学習できると思います。  
但し、エコーなど時間方向に長い効果を及ぼすような効果には対応できないと思います。

