# sample_resnet

モノクロ画像への着色を学習させたい ResNet を使ったサンプルです。  
CNN 層を重ねて色差を学習させます。  
CNN は2層ごとにスキップ接続していますが、全体でも色差を学習させるためにスキップしています。全体のスキップは必ずしも必要ないかもしれませんが、あったほうが学習させやすいと思います。  
色差は YUV などでも良いと思いますが、YUV は視覚特性に合わせるために色ごとに重みが異なっており、色ごとに学習速度に差が出るかと思いますので、単純な輝度とRGBの差の計4チャンネルの形式に変換しています。


## ディレクトリ

- build/sample_resnet/ - プロジェクト
- samples/sample_resnet/ - 実行用
- samples/sample_resnet/learn.bat - 学習実行
- samples/sample_resnet/predict.bat - 予測実行
- samples/sample_resnet/learn/source/ - 学習用ソース
- samples/sample_resnet/learn/teacher/ - 学習用教師データ
- samples/sample_resnet/predict/src/ - 予測ソース
- samples/sample_resnet/predict/out/ - 予測出力


## モデル

[sample_resnet.cpp](./sample_resnet.cpp)
```cpp
for ( int i = 0; i < 10; i ++ )
{
    mlp.AppendConvLayer( 64, 64, 3, 3, convPadZero, 0, activReLU )
        ->SetRidgeParameter( 0.01f )
        ->SetAdaptiveOptimization( NNPerceptron::adaOptMomentum, adaParam )
        ->SetNormalization( std::make_shared<NNInstanceNormalization>(normParam) ) ;

    NNPerceptronPtr    pResidual =
        mlp.AppendConvLayer( 64, 64, 3, 3, convPadZero, 0, activLinear ) ;
    pResidual
        ->SetRidgeParameter( 0.01f )
        ->SetAdaptiveOptimization( NNPerceptron::adaOptMomentum, adaParam )
        ->SetNormalization( std::make_shared<NNInstanceNormalization>(normParam) ) ;

    pPrevLayer =
        mlp.AppendPointwiseAdd
            ( 64, pResidual, 0, pPrevLayer, 0, 0, 0, activReLU ) ;
}
```


## 学習用データの準備

samples/sample_resnet/make_learn/src/ 内に学習用 PNG 画像ファイルを設置し、

```bat
> cd samples/sample_resnet
> make_learn.bat
```

を実行すると samples/sample_resnet/learn/ 内に学習用画像ファイルが生成されます。  
実行には Python が必要です。

画像は学習しやすいようにサイズを一定サイズまで縮小した上で、教師画像の彩度を2倍にしています。  
彩度を上げなくていい場合には grayscale_image.py の sat.enhance( 2.0 ) の個所を書き換えてください。  


## メモ

このサンプルで汎用的に着色を学習させるのは無理があると思いますので、ある程度種類を絞ったほうが良いと思います。 

learn.bat を実行すると学習が開始されます。  
/vio で predict\out\valid_image.bmp にエポックごとに検証用画像が出力されますので、進捗を目視で確認できます。

ミニバッチサイズは全体の学習用画像ファイル数に合わせて調整したほうが良いかもしれません。またその場合、適応最適化や正規化のハイパーパラメータも調整したほうが良い場合があるかもしれません。  

初めのうちはノイズの状態からモノクロ画像になっていき、その後で徐々に着色される場所が出現し、途中判断に迷って変な色を付けたり紆余曲折しながら判別できるパーツが増えていったあと、ある程度落ち着いてくると思います。  
落ち着かない場合にはミニバッチサイズを大きくしたり、/delta の値を少し小さくして追加で学習させれば落ち着く（灰色になっていく）と思います。

