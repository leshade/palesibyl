# simple_vae

16x16, Grayscale 画像の変分オートエンコーダーです。  
samples/simple_vae/learn/source/ には適当に描いた文字画像を入れていますので、そのままでも学習を実行できます。



## ディレクトリ

- build/simple_vae/ - プロジェクト
- samples/simple_vae/ - 実行用
- samples/simple_vae/learn.bat - 学習実行
- samples/simple_vae/predict.bat - 予測実行
- samples/simple_vae/learn/source/ - 学習用画像
- samples/simple_vae/predict/out/ - 予測用出力
- samples/simple_gan/predict/src/ - 予測用ソース


## モデル

```cpp
// エンコーダー
mlp.AppendConvLayer( 16, 1, 3, 3, convPadBorder, bias, activReLU, 2, 2 ) ; // ->8x8
mlp.AppendConvLayer( 32, 16, 3, 3, convPadZero, bias, activReLU, 2, 2 ) ;  // ->4x4
mlp.AppendConvLayer( 64, 32, 3, 3, convPadZero, bias, activReLU, 2, 2 ) ;  // ->2x2
mlp.AppendConvLayer( 128, 64, 2, 2, convNoPad, bias, activReLU, 2, 2 ) ;   // ->1x1

#if	__SIMPLE_AUDOENCODER__
    mlp.AppendLayer( nLatentChannels, 128, bias, activLinear )
        ->SetIdentity( idEncoderOutLayer ) ;
#else
    NNPerceptronPtr	pLayerMean =
        mlp.AppendLayer( nLatentChannels, 128, bias, activLinear ) ;
    NNPerceptronPtr	pLayerLnVar =
        mlp.AppendLayer( nLatentChannels, 128, bias, activLinear ) ;
    pLayerLnVar->AddConnection( 2, 0, 0, 128 ) ;
  
    mlp.AppendGaussianLayer( nLatentChannels, pLayerMean, pLayerLnVar )
        ->SetIdentity( idEncoderOutLayer ) ;
#endif

// デコーダー
mlp.AppendUp2x2Layer( 128, nLatentChannels, bias, activReLU ) ;  // ->2x2
mlp.AppendUp2x2Layer( 64, 128, bias, activReLU ) ;               // ->4x4
mlp.AppendUp2x2Layer( 32, 64, bias, activReLU ) ;                // ->8x8
mlp.AppendUp2x2Layer( 16, 32, bias, activReLU ) ;                // ->16x16
mlp.AppendConvLayer( 1, 16, 3, 3, convPadZero, bias, activSigmoid ) ;

// 損失関数
mlp.SetLossFunction( std::make_shared<NNLossBernoulliNLL>() ) ;

#if	!__SIMPLE_AUDOENCODER__
    mlp.AddLossGaussianKLDivergence
        ( pLayerMean, pLayerLnVar, 1.0f/(16.0f*16.0f), 1.0f ) ;
#endif

// 評価値設定
mlp.SetEvaluationFunction( std::make_shared<NNEvaluationMSE>() ) ;
```

\_\_SIMPLE_AUDOENCODER\_\_ マクロを 0 に設定するとオートエンコーダー、1 にすると変分オートエンコーダーとなります。  
nLatentChannels は潜在変数次元（チャネル数）です。サンプルでは（補間出力を興味深いものにするため） 2 としていますが、通常は 20 などにすると良いと思います。

predict_inter.bat （又は predict_inter.sh）で補間出力ができます。



