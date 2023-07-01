# simple_classifier

16x16, Grayscale 画像を分類する分類器のサンプルです。  
samples/simple_classifier/classes/ には適当に描いた文字画像を入れていますので、そのまま分類器の学習を実行できますが、本来、学習用画像はもっとあったほうが良いでしょう。



## ディレクトリ

- build/simple_classifier/ - プロジェクト
- samples/simple_classifier/ - 実行用
- samples/simple_classifier/learn.bat - 学習実行
- samples/simple_classifier/predict.bat - 予測実行
- samples/simple_classifier/classes/ - 学習用分類済み画像
- samples/simple_classifier/predict/ - 分類予測用ソース


## モデル

```cpp
mlp.SetInputShape( 0, NNBufDim( 16, 16, 1 ), NNBufDim( 0, 0, 1 )  ) ;

mlp.AppendConvLayer( 16, 1, 3, 3, convPadBorder, 1, activReLU ) ;       // 16x16
mlp.AppendMaxPoolLayer( 16, 2, 2 ) ;                                    // 8x8
mlp.AppendConvLayer( 32, 16, 3, 3, convPadZero, 1, activReLU ) ;
mlp.AppendMaxPoolLayer( 32, 2, 2 ) ;                                    // 4x4
mlp.AppendConvLayer( 64, 32, 3, 3, convPadZero, 1, activReLU, 2, 2 ) ;  // 2x2
mlp.AppendConvLayer( 128, 64, 2, 2, convNoPad, 0, activReLU, 2, 2 ) ;   // 1x1
#ifdef    __FAST_SOFTMAX__
    #ifdef    __CLASS_INDEX_FORMAT__
        mlp.AppendFastSoftmax( nClassCount, 128, 1, activFastArgmax ) ;
    #else
        mlp.AppendFastSoftmax( nClassCount, 128, 1, activFastSoftmax ) ;
    #endif
    #else
    #ifdef    __CLASS_INDEX_FORMAT__
        mlp.AppendLayer( nClassCount, 128, 1, activArgmax ) ;
    #else
        mlp.AppendLayer( nClassCount, 128, 1, activSoftmax ) ;
    #endif
#endif

```

最終的な出力は argmax でも softmax でも分類器として学習できます。  
しかし、argmax は出力レイヤーにしか使用できないこと、分類予測として softmax は各分類ごとの確率を得ることができる一方、argmax はもっとも確からしい1つの確率しか得ることが出来ない事に注意してください。
AppendFastSoftmax は activFastArgmax 又は activFastSoftmax と組み合わせて使用し、特に分類数が千を超えるような大きな場合に高速化されます。

