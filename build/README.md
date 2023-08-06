# Palesibyl, the deep learning library

## ディレクトリ

- `README.md` - このファイル
- `Palesibyl/Palesibyl.sln` - Visual Studio 用ソリューション
- `common/include/` - サンプル用共通ヘッダ
- `common/source/` - サンプル用共通ソースコード
- `PalesibylTest/` - 申し訳程度のテストプロジェクト
- [`simple_color_filter/`](./simple_color_filter/) - 単純色変換サンプルプロジェクト
- [`simple_upsampler/`](./simple_upsampler/) - 簡単なアップサンプラープロジェクト
- [`simple_classifier/`](./simple_classifier/) - 簡単な分類器サンプルプロジェクト
- [`simple_gan/`](./simple_gan/) - 簡単な GAN サンプルプロジェクト
- [`simple_vae/`](./simple_vae/) - 簡単な VAE サンプルプロジェクト
- [`simple_wave_filter/`](./simple_wave_filter/) - 簡単な RNN 音声フィルタープロジェクト
- [`sample_resnet/`](./sample_resnet/) - ResNet サンプルプロジェクト
- [`sample_unet/`](./sample_unet/) - U-Net サンプルプロジェクト


## ビルド環境

* Visual Studio (Windows)  
  開発時点では Visual Studio Professional 2019 を使用  
  C++17 標準ライブラリ (std::filesystem を使用のため／それ以外は C++14 でも可) 

* NVIDIA CUDA Toolkit  
  開発時点では NVIDIA CUDA Toolkit 12 を使用

* OpenCV (Linux)  
  Linux では画像コーデックとして使用するため。  
  （Windows では GDI+ を画像コーデックとして使用するため必要ありません）

## 使用法

* ヘッダファイル  
  Palesibyl を使用する C++ ソース、又はヘッダで `palesibyl.h` をインクルードします。  
  ```cpp
  #include <palesibyl.h>
  ``` 

* 名前空間  
  クラス等は名前空間 `Palesibyl` に含まれます。  
  ```cpp
  using namespace Palesibyl ;
  ```
  とするか、宗派によっては  
  ```cpp
  namespace psl = Palesibyl ;
  ```
  等とすると便利です。

* ライブラリ（Windows）  
  デバッグ用は `palesibyl_db.lib`、リリース用は `palesibyl.lib` をリンクします。  
  以下のように `palesibyl_lib.h` をインクルードすると便利です。  
  ```cpp
  #include <palesibyl_lib.h>
  ```

* 初期化  
  ライブラリの静的な初期化処理は Palesibyl::NMLPShell::StaticInitialize 関数を呼び出します。  
  CUDA を使用する場合には Palesibyl::cudaInit 関数も呼び出します。
  ```cpp
  Palesibyl::cudaInit() ;
  NNMLPShell::StaticInitialize() ;
  ```
  終了時には Palesibyl::NNMLPShell::StaticRelase 関数を呼び出します。
  ```cpp
  NNMLPShell::StaticRelase() ;
  ```

* チュートリアル  
  多層ニューラルネットワークは NNMultiLayerPerceptron クラスで実装されていますが、ファイルIO、学習、予測処理等はそのラッパークラスである NNMLPShell を利用すると便利です。

  モデルの構築は以下のように NNMultiLayerPerceptron に対して行います。
  ```cpp
  NNMLPShell shell ;
  NNMultiLayerPerceptron& mlp = shell.MLP() ;
  mlp.AppendLayer(...) ;				// モデルの構築
  ...
  ```
  モデルは SaveModel で保存、LoadModel で読み込むことができます。
  ```cpp
  shell.LoadModel( "model.mlp" ) ;		// model.mlp の読み込み
  shell.SaveModel( "model.mlp" ) ;		// model.mlp へ保存
  ```
  学習や予測に使うデータは NNMLPShell::Iterator で入出力します。  
  画像や音声ファイルの入出力には NNMLPShellImageIterator や NNMLPShellWaveIterator などが便利です。  
  例えば、訓練データとして source ディレクトリと、対応する教師データとして teacher ディレクトリに同名の画像ファイルを用意しておき、
  ```cpp
  NNMLPShell::LearningParameter param ;	// パラメータを設定しておく
  NNMLPShellImageCropper iter( "source", "teacher", NNBufDim( 128, 128, 3 ) ) ;
  shell.DoLearning( iter, param ) ;
  ```
  とすればモデルを学習させることができます。  
  この例のように NNMLPShellImageCropper を使用する場合、ソース画像をエポック毎にランダムに任意サイズ（この場合 128x128, 3 チャネル）で切り出して学習します（予め画像を決まったサイズに切り出したファイルを用意したほうが良いかもしれませんが手抜き用です）。

  同じように source ディレクトリに予測元画像を用意しておき、
  ```cpp
  NNMLPShellImageIterator iter( "source", "output", true, 3 ) ;
  shell.DoPrediction( iter ) ;
  ```
  とすれば、output ディレクトリに予測画像を出力できます。


## 機能概覧

* レイヤー追加関数（NNMultiLayerPerceptron クラスメンバ）  
  - `AppendLayer` - 全結合、又は任意のレイヤー
  - `AppendConvLayer` - 畳み込みレイヤー
  - `AppendDepthwiseLayer` - 疎なチャネル別結合レイヤー
  - `AppendDepthwiseConv` - 畳み込みレイヤー（チャネル別）
  - `AppendUpsamplingLayer` - アップサンプリングレイヤー
  - `AppendUp2x2Layer` - 2x2 アップサンプリングレイヤー
  - `AppendUpsamplingFixLayer` - アップサンプリングレイヤー（変換無し）
  - `AppendLayerAsOneHot` - インデックス値を one-hot ベクトルとして入力するレイヤー
  - `AppendFastSoftmax` - softmax, 又は argmax 高速化レイヤー
  - `AppendMaxPoolLayer` - MaxPooling 畳み込みレイヤー
  - `AppendGatedLayer` - ゲート付きレイヤー
  - `AppendPointwiseAdd` - 加算結合レイヤー
  - `AppendPointwiseMul` - 乗算結合レイヤー
  - `AppendGaussianLayer` - N(μ,σ) 乱数発生レイヤー

* 活性化関数（識別子と実装クラス）  
  - `activLinear` - NNActivationLinear - 線形 (平均二乗誤差)
  - `activLinearMAE` - NNActivationLinearMAE - 線形 (平均絶対誤差)
  - `activReLU` - NNActivationReLU - 整流関数
  - `activSigmoid` - NNActivationSigmoid - シグモイド関数
  - `activTanh` - NNActivationTanh - tanh 関数
  - `activSoftmax` - NNActivationSoftmax - softmax 関数
  - `activFastSoftmax` - NNActivationFastSoftmax - softmax 関数（高速化）
  - `activArgmax` - NNActivationArgmax - argmax 関数
  - `activFastArgmax` - NNActivationFastArgmax - argmax 関数（高速化）
  - `activMaxPool` - NNActivationMaxPool - max 関数
  - `activMultiply` - NNActivationMultiply - 乗算結合

* 正規化（実装クラス）  
  - `NNLayerNormalization` - レイヤー正規化
  - `NNGroupNormalization` - グループ正規化
  - `NNInstanceNormalization` - インスタンス正規化

* 適応的最適化（NNPerceptron::AdaptiveOptimization 列挙子）
  - `adaOptNo` - SGD (適応的最適化無し)
  - `adaOptMomentum` - モメンタム
  - `adaOptRMSProp` - RMSProp
  - `adaOptAdam` - Adam

* ドロップアウト - NNPerceptron::SetDropoutRate 関数

* L2 正則化 - NNPerceptron::SetRidgeParameter 関数

* 評価関数（実装クラス）  
  - `NNEvaluationMSE` - 平均二乗誤差
  - `NNEvaluationR2Score` - 決定係数 R2
  - `NNEvaluationArgmaxAccuracy` - argmax 正解率


## Palesibyl 固有の留意点

* レイヤー  
  * 全結合レイヤーは各サンプル毎のチャネルの全結合となります。  
    Palesibyl では Conv1x1 と同等です。
  * AppendLayerAsOneHot は入力レイヤーのみに使用できます。
  * AvgPool は未実装です。
  
* 活性化関数と損失関数  
  * 活性化関数の実装クラスには損失関数も実装されており、デフォルトは活性化関数の交差エントロピーが損失関数として実装されています。  
    NNActivationLinearMAE だけは交差エントロピーではない平均絶対誤差が損失関数として実装されています。  
	これ以外の損失関数を使用する場合には、損失関数を実装した活性化関数クラスを実装するか、SetLossFunction 関数で任意の損失関数を設定する必要があります。  
	（追加的な損失関数の設定例は [VAE サンプル](./simple_vae/)を参照）
  * 現在の実装では（argmax を除き）CUDA で処理できる活性化関数の最大チャネル数は 3072 です。  
    （CPU 及び、CUDA でも行列サイズには制限はありません）
  * softmax の高速化は CPU のみの実装となっています（CUDA では高速化されずに実行します）
  * softmax, argmax の高速化法はポジティブサンプリング＋ネガティブサンプリングです。
  * argmax は出力レイヤーのみに使用でき、隠れ層には使用できません。  
    （つまり膨大な分類数の分類器の出力には softmax ではなく argmax の使用が推奨されます）

* 正規化  
  * バッチ正規化は未実装です。  
    正規化のための平均・分散はインスタンス毎に集計されますが、実際には以前の値も重みを付けて反映されます。つまり以前の値を縮小した後に新しいインスタンスの値を加算し、正規化のための平均と分散を計算しています。  
	以前の値をどれだけ縮小するかは NNNormalizationFilter::Hyperparameter の alpha と beta で指定できます。alpha はミニバッチ毎の、beta はエポックごとの係数で、0.0f を指定すると完全にミニバッチ毎、又はエポック毎（バッチ毎）にリセットされることになります（ミニバッチ内のインスタンスはスケールされずに合計されます）。  
	従って、NNInstanceNormalization は実質的にはバッチ正規化とほぼ同等となることを期待した実装となっています（間違えていたら教えてください）。  


## サンプルプログラム（共通）

* [sample_basic_app.cpp](./common/source/sample_basic_app.cpp)
  
  サンプルプログラム共通の処理は sample_basic_app.cpp に記述されています。  
  ほぼ CLI から NNMLPShell クラスへパラメータを受け渡しているだけです。


* 学習の実行と損失グラフ表示  
  ```bat
  > learn.bat
  ```
  又は
  ```bash
  $ sh learn.sh
  ```
  各サンプル実行用ディレクトリの `learn.bat` (Windows)、又は `learn.sh` (Linux) を実行すると学習が開始されます。  
  learn.bat / learn.sh では、ログファイルを出力するとともに Python スクリプトでログファイルを逐次リアルタイムにグラフ表示させることができます。

  1. Python をインストールし、python.exe のファイルパスを環境変数 PYTHON_BIN_PATH に設定 (Windows)  
     ex. C:\Users\user_name\AppData\Local\Programs\Python\Python39\python.exe
  2. numpy, matplotlib, pandas, watchdog をインストール
     ```bat
	 > pip install numpy
	 > pip install matplotlib
	 > pip install pandas
	 > pip install watchdog
	 ```
  3. learn.bat 又は learn.sh を実行


* 学習の中断と継続
  
  学習の途中で中断してそれまでの結果をモデルに保存したい場合には ESC キーを押してください。  
  また、モデルファイルが存在している状態で再度学習を開始すると追加で学習をします。  
  初期状態から学習させたい場合にはモデルファイルを削除してください。


* コマンドライン引数  
  - /l *\<model-file\>*  
    モデルを訓練します。  
    指定されたモデルファイルが既に存在している場合、そのモデルファイルを読み込んで追加的に学習します。

  - /p *\<model-file\>*  
    指定されたモデルファイルを読み込んで予測します。

  - /cuda  
    CUDA を利用します。

  - /loop *\<epoch-count\>*  
    訓練エポック数を指定します。

  - /subloop *\<count\>*  
    1つのミニバッチの反復学習回数（学習勾配反映回数）を指定します。  
	デフォルトでは1回ですが、ミニバッチのデータはメモリ上に保持するので、ミニバッチサイズが十分に大きい場合や過学習が問題にならないような場合、この回数を増やすことによってデータの読み込み回数を減らす等、多少高速化できます。

  - /batch *\<count\>*  
    ミニバッチサイズを指定します。  
	ミニバッチのデータは一度にメモリ上に保持されます。
	但しニューラルネットワークの隠れ層メモリは共有されるので、ミニバッチサイズが増えてもメモリ使用量（CUDA 用 VRAM 含む）は増加せず、モデルの規模と1つのデータのサンプル数（ピクセル数）に依存します。

  - /delta *\<rate\>*[,*\<end-rate\>*]  
    学習勾配を反映させる係数（学習速度）を指定します。  
	*end-rate* を指定した場合（,前後はスペースを空けずにCLI上一つの引数として）、係数はエポックの進捗に伴って対数的に補完されます。

  - /thread *\<count\>*  
    CPU で処理する場合のスレッド数を指定します。  
	デフォルトは論理プロセッサ数になります。

  - /batch_thread *\<count\>*  
    ミニバッチの各データを並列処理するスレッド数を指定します。  
	デフォルトは 1 です。  
	これは 1 つのデータが小さすぎてパフォーマンスを十分に発揮できないような場合、これらを並列化することによって高速化を図ります。  
	但し、隠れ層のメモリ（CUDA メモリ含む）はスレッド数分だけ増加します。  

  - /log *\<csv-file\>*  
    学習のログ（損失値）を CSV 形式でファイルに出力します。  

  - /lgrd  
    学習ログに各レイヤーの勾配ノルム（Frobenius ノルム）も出力します。

  - /tio *\<image-file\>*  
    学習時の出力のうちの１枚をミニバッチ毎に画像ファイルとして書き出します。  
	出力した画像ファイルをモニタリングすることで学習の経過を視覚的に確認することができます。

  - /vio *\<image-file\>*  
    学習時の検証用出力のうちの初めの１枚を画像ファイルとしてエポックごとに逐次書き出します。  
	出力した画像ファイルをモニタリングすることで学習の経過を視覚的に確認することができます。

  - /pbs  
    学習時に使用する中間バッファのサイズを表示します。

  - /cubs  
    学習時に使用した中間バッファの CUDA メモリサイズを表示します。


* コマンドライン引数（GAN のみ）

  - /clsf *\<model-file\>*  
    分類器のモデルファイルを指定します。

  - /ganloop *\<count\>*  
    GAN のループ回数を指定します。


## 新規プロジェクト作成時メモ等 (VS2019 GUI)

* 環境変数の設定  
  Paysibyl を設置した場所を環境変数に設定しておくと便利です。  
  ここでは PALESIBYL_HOME にパスを設定するものとします。

* CUDA Toolkit の設定  
  ソリューションエクスプローラー上でプロジェクトを選択。  
  コンテキストメニュー「ビルド依存関係」＞「ビルドのカスタマイズ」（**分かりにくい！**）、又はメニュー「プロジェクト」＞「ビルドのカスタマイズ」から「CUDA」をチェック。

* インクルードディレクトリの設定  
  プロジェクトプロパティ「C/C++」の「全般」＞「追加のインクルードディレクトリ」に `$(PALESIBYL_HOME)/Palesibyl/include` と \$(CudaToolkitIncludeDir) を追加。  
  ※\$(CudaToolkitIncludeDir) は CUDA Tookkit の設定で自動的に追加されます。

* ライブラリディレクトリの設定  
  プロジェクトプロパティ「リンカー」の「追加のライブラリディレクトリ」に `$(PALESIBYL_HOME)/Palesibyl/library` と \$(CudaToolkitLibDir) を追加。  
  ※\$(CudaToolkitLibDir) は CUDA Tookkit の設定で自動的に追加されます。

* C++ 標準ライブラリ  
  プロジェクトプロパティ「全般」の「C++ 言語標準」に「ISO C++17 標準 (/std:c++17)」を選択。

* 文字セット  
  プロジェクトプロパティ「詳細」の「文字セット」に「マルチ バイト文字セットを使用する」を選択。  
  （ライブラリのビルド時の設定と合わせればこれ以外でも構いません）

* ランライムライブラリ  
  プロジェクトプロパティ「C/C++」の「コード生成」＞「ランライムライブラリ」に、Debug では「マルチスレッド デバッグ (/MTd)」、Release では「マルチスレッド (/MT)」を選択。  
  （ライブラリのビルド時の設定と合わせればこれ以外でも構いません）


## インストールメモ (Linux/WSL)

* OpenCV  
  [OpenCV](https://github.com/opencv/opencv) を git clone するか、zip をダウンロードし、以下のように実行するとインストール出来ました。

  ```bash
  $ unzip opencv-4.8.0.zip
  $ cd opencv-4.8.0
  $ mkdir build
  $ cd build
  $ cmake -DOPENCV_GENERATE_PKGCONFIG=ON ..
  $ cmake --build .
  $ make
  $ sudo make install
  $ sudo ldconfig
  ```

  OPENCV_GENERATE_PKGCONFIG=ON としておくのがポイントらしいです。


