# sample_unet

モノクロ画像への着色を学習させたい U-Net を使ったサンプルです。  


## ディレクトリ

- build/sample_unet/ - プロジェクト
- samples/sample_unet/ - 実行用
- samples/sample_unet/learn.bat - 学習実行
- samples/sample_unet/predict.bat - 予測実行
- samples/sample_unet/learn/source/ - 学習用ソース
- samples/sample_unet/learn/teacher/ - 学習用教師データ
- samples/sample_unet/predict/src/ - 予測ソース
- samples/sample_unet/predict/out/ - 予測出力


## モデル

[sample_unet.cpp](./sample_unet.cpp)
```cpp
// エンコーダー部
NNPerceptronPtr    pSkip0, pSkip1, pSkip2, pSkip3 ;
mlp.AppendConvLayer( 64, 4, 3, 3, convPadBorder, bias, activReLU ) ;
pSkip0 = mlp.AppendConvLayer( 64, 64, 3, 3, convPadZero, bias, activReLU ) ;
mlp.AppendMaxPoolLayer( 64, 2, 2 ) ;                            // 1/2
mlp.AppendConvLayer( 128, 64, 3, 3, convPadZero, bias, activReLU ) ;
pSkip1 = mlp.AppendConvLayer( 128, 128, 3, 3, convPadZero, bias, activReLU ) ;

mlp.AppendMaxPoolLayer( 128, 2, 2 ) ;                           // 1/4
mlp.AppendConvLayer( 256, 128, 3, 3, convPadZero, bias, activReLU )
    ->SetDropoutRate( dropout ) ;
pSkip2 = mlp.AppendConvLayer( 256, 256, 3, 3, convPadZero, bias, activReLU ) ;

mlp.AppendMaxPoolLayer( 256, 2, 2 ) ;                           // 1/8
mlp.AppendConvLayer( 512, 256, 3, 3, convPadZero, bias, activReLU )
    ->SetDropoutRate( dropout ) ;
pSkip3 = mlp.AppendConvLayer( 512, 512, 3, 3, convPadZero, bias, activReLU ) ;

mlp.AppendMaxPoolLayer( 512, 2, 2 ) ;                           // 1/16
mlp.AppendConvLayer( 1024, 512, 3, 3, convNoPad, bias, activReLU )
    ->SetDropoutRate( dropout ) ;

// デコーダー部
mlp.AppendConvLayer( 1024, 1024, 3, 3, convNoPad, bias, activReLU ) ;

mlp.AppendUp2x2Layer( 512, 1024, 1, activReLU ) ;               // 1/8
mlp.AppendConvLayer( 512, 512+512, 3, 3, convPadZero, bias, activReLU )
    ->AddConnection( 1, 0, 0, 512 )                             // ※１つ前のレイヤーと
    ->AddConnection( mlp.LayerOffsetOf(pSkip3),
                            0, 0, 512, trim3, trim3 ) ;         //   pSkip3 から入力する
mlp.AppendConvLayer( 512, 512, 3, 3, convPadZero, bias, activReLU ) ;

mlp.AppendUp2x2Layer( 256, 512, 1, activReLU ) ;                // 1/4
mlp.AppendConvLayer( 256, 256+256, 3, 3, convPadZero, bias, activReLU )
    ->AddConnection( 1, 0, 0, 256 )
    ->AddConnection( mlp.LayerOffsetOf(pSkip2), 0, 0, 256, trim2, trim2 ) ;
mlp.AppendConvLayer( 256, 256, 3, 3, convPadZero, bias, activReLU ) ;

mlp.AppendUp2x2Layer( 128, 256, 1, activReLU ) ;                // 1/2
mlp.AppendConvLayer( 128, 128+128, 3, 3, convPadZero, bias, activReLU )
    ->AddConnection( 1, 0, 0, 128 )
    ->AddConnection( mlp.LayerOffsetOf(pSkip1), 0, 0, 128, trim1, trim1 ) ;
mlp.AppendConvLayer( 128, 128, 3, 3, convPadZero, bias, activReLU ) ;

mlp.AppendUp2x2Layer( 64, 128, 1, activReLU ) ;                 // 1/1
mlp.AppendConvLayer( 64, 64+64, 3, 3, convPadZero, bias, activReLU )
    ->AddConnection( 1, 0, 0, 64 )
    ->AddConnection( mlp.LayerOffsetOf(pSkip0), 0, 0, 64, trim0, trim0 ) ;
mlp.AppendConvLayer( 64, 64, 3, 3, convPadZero, bias, activReLU )
    ->SetDropoutRate( dropout ) ;
NNPerceptronPtr    pResidual =
    mlp.AppendLayer( 4, 64, bias, activTanh ) ;
```

コード上ドロップアウトを設定していますが、学習に時間がかかるので dropout = 0.0f とし、ドロップアウトは行っていません。またドロップアウトするレイヤーも雰囲気で入れていて、どこに入れるのがいいのかは良く知りません。  
畳み込みは一番内側のレイヤーのみパッディング無し（convNoPad）で行っていて、出力サイズは64ドット小さくなります。スキップ結合でサイズが異なるため、AddConnection でトリミング位置を指定していることに注意してください。


## 学習用データの準備

samples/sample_unet/make_learn/src/ 内に学習用 PNG 画像ファイルを設置し、

```bat
> cd samples/sample_unet
> make_learn.bat
```

を実行すると samples/sample_unet/learn/ 内に学習用画像ファイルが生成されます。  
実行には Python が必要です。

画像は320x320に縮小＆トリミングした上で、教師画像の彩度を2倍にしています。  
彩度を上げなくていい場合には grayscale_image.py の sat.enhance( 2.0 ) の個所を書き換えてください。  

画像サイズを320より小さくしたい場合には [sample_unet.cpp](./sample_unet.cpp) の
```cpp
return    std::make_shared<NNMLPShellImageCropper>
    ( "learn\\source", "learn\\teacher",
        NNBufDim( 320, 320, 3 ), trim0, trim0, trim0, trim0 ) ;
```
の箇所の NNBufDim( 320, 320, 3 ) を小さくしておく必要があります。  
（学習データがより大きい場合には（プログラム上は）問題ありません）  
しかしサンプルコードの U-Net の都合上、128x128 未満にはしないほうが良いでしょう。出来れば 256x256 以上が良いと思います。


## メモ

このサンプルは [sample_resnet](../sample_resnet/) と同様に着色させようとしていますが、しっかり学習するためにはエポック数を増やすか画像を多め（と言っても100枚程度以上）に用意する必要があるかと思います。少し遊びで学習させるにはコスト高めなサンプルかと思います。

learn.bat を実行すると学習が開始されます。  
/vio で predict\out\valid_image.bmp にエポックごとに検証用画像が出力されますので、進捗を目視で確認できます。暇なら /tio でも画像を出力させて眺めているといい感じに時間が溶けていきます。  
/delta 0.03～0.02 あたりでしっかり目に学習させてから小さくしていかないと色分けも中途半端で、アップサンプリングでのノイズも残ったような仕上がりになりがちな気がします。

