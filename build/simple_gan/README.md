# simple_gan

分類器を使って 16x16, Grayscale 画像の生成器を GAN (敵対的生成ネットワーク) で学習するサンプルです。  
samples/simple_gan/classes/ には適当に描いた文字画像を入れていますので、そのまま学習を実行できますが、学習用画像はもっとあったほうが良いでしょう。



## ディレクトリ

- build/simple_gan/ - プロジェクト
- samples/simple_gan/ - 実行用
- samples/simple_gan/learn.bat - 学習実行
- samples/simple_gan/predict.bat - 予測実行
- samples/simple_gan/classes/ - 学習用分類済み画像
- samples/simple_gan/predict/out/ - 予測用出力
- samples/simple_gan/predict/src/ - 予測用ソース


## モデル

```cpp
mlp.SetInputShape
    ( 0, NNBufDim( 1, 1, nClassCount ), NNBufDim( 1, 1, nClassCount )  ) ;

#ifdef    __CLASS_INDEX_FORMAT__
    mlp.AppendLayerAsOneHot( 128, nClassCount, activLinear ) ;  // 1x1
#else
    mlp.AppendLayer( 128, nClassCount, 1, activLinear ) ;       // 1x1
#endif
mlp.AppendUp2x2Layer( 64, 128, 1, activLinear ) ;               // 2x2
mlp.AppendUp2x2Layer( 32, 64, 1, activLinear ) ;                // 4x4
mlp.AppendUp2x2Layer( 16, 32, 1, activLinear ) ;                // 8x8
mlp.AppendUp2x2Layer( 1, 16, 1, activLinear ) ;                 // 16x16
```

AppendLayerAsOneHot は1チャネルの分類番号のみを入力しますが、行列サイズの列数は分類数（＋バイアス項）になる事に注意してください。  
生成条件はなく、分類番号（あるいは one-hot ベクトル）のみから画像を生成するため、常に同じ分類に対して同じ画像を生成する単純な生成器です。  
4層の AppendUp2x2Layer で 16x16 画像を生成していますが、1層の AppendUp16x16Layer で生成しても問題ないはずです。

