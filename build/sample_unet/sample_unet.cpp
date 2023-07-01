
#include "sample_app_stub.h"


constexpr static const int	trim4	= 1+1 ;		// 一番内側で 3x3 Conv NoPad を2回
constexpr static const int	trim3	= trim4*2 ;	// 2x2 Upsampling
constexpr static const int	trim2	= trim3*2 ;
constexpr static const int	trim1	= trim2*2 ;
constexpr static const int	trim0	= trim1*2 ;


// アプリ初期化
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::Initialize( void )
{
}

// アプリ固有の説明（ファイルの配置など）
//////////////////////////////////////////////////////////////////////////////
const char *	PalesibylApp::s_pszSpecificDescription =
	"ディレクトリ構成;\n"
	"[learn\\]\n"
	"  + [source\\]  : 学習元画像ファイル\n"
	"  + [teacher\\] : 教師画像ファイル（学習元と同名ファイル）\n"
	"[predict\\]\n"
	"  + [src\\]     : 予測入力画像ファイル\n"
	"  + [out\\]     : 予測出力先\n" ;


// モデルを作成
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::BuildModel( NNMLPShell::Iterator * pIter )
{
	NNMultiLayerPerceptron&	mlp = m_shell.MLP() ;

	NNPerceptron::AdaptiveHyperparameter	adaParam ;
	adaParam.alpha = 0.9f ;
	adaParam.delta = 0.1f ;

	NNNormalizationFilter::Hyperparameter	normParam ;
	normParam.delta = 0.3f ;
	normParam.deltac = 0.000001f ;

	const int	bias = 0 ;
	const float	dropout = 0.0f ;


	// RGB -> 輝度＋色差
	static const float	rcp3 = 1.0f / 3.0f ;
	static const float	rgb2ycrgb[12] =
	{
		// Blue      Green      Red
		   rcp3,     rcp3,     rcp3,	// Y
		1.0f-rcp3,  -rcp3,    -rcp3,	// Cb
		  -rcp3,  1.0f-rcp3,  -rcp3,	// Cg
		  -rcp3,    -rcp3,  1.0f-rcp3,	// Cr
	} ;
	NNPerceptronPtr	pRGB2YCrgb =
		std::make_shared<NNFixedPerceptron>
			( 4, 3, 1, 0,
				std::make_shared<NNSamplerInjection>(),
				std::make_shared<NNActivationLinear>() ) ;
	for ( size_t i = 0; i < 12; i ++ )
	{
		pRGB2YCrgb->Matrix().ArrayAt(i) = rgb2ycrgb[i] ;
	}
	mlp.AppendLayer( pRGB2YCrgb ) ;

	// エンコーダー部
	NNPerceptronPtr	pSkip0, pSkip1, pSkip2, pSkip3 ;
	mlp.AppendConvLayer( 64, 4, 3, 3, convPadBorder, bias, activReLU ) ;
	pSkip0 = mlp.AppendConvLayer( 64, 64, 3, 3, convPadZero, bias, activReLU ) ;
	mlp.AppendMaxPoolLayer( 64, 2, 2 ) ;							// 1/2
	mlp.AppendConvLayer( 128, 64, 3, 3, convPadZero, bias, activReLU ) ;
	pSkip1 = mlp.AppendConvLayer( 128, 128, 3, 3, convPadZero, bias, activReLU ) ;

	mlp.AppendMaxPoolLayer( 128, 2, 2 ) ;							// 1/4
	mlp.AppendConvLayer( 256, 128, 3, 3, convPadZero, bias, activReLU )
		->SetDropoutRate( dropout ) ;
	pSkip2 = mlp.AppendConvLayer( 256, 256, 3, 3, convPadZero, bias, activReLU ) ;

	mlp.AppendMaxPoolLayer( 256, 2, 2 ) ;							// 1/8
	mlp.AppendConvLayer( 512, 256, 3, 3, convPadZero, bias, activReLU )
		->SetDropoutRate( dropout ) ;
	pSkip3 = mlp.AppendConvLayer( 512, 512, 3, 3, convPadZero, bias, activReLU ) ;

	mlp.AppendMaxPoolLayer( 512, 2, 2 ) ;							// 1/16
	mlp.AppendConvLayer( 1024, 512, 3, 3, convNoPad, bias, activReLU )
		->SetDropoutRate( dropout ) ;

	// デコーダー部
	mlp.AppendConvLayer( 1024, 1024, 3, 3, convNoPad, bias, activReLU ) ;

	mlp.AppendUp2x2Layer( 512, 1024, 1, activReLU ) ;				// 1/8
	mlp.AppendConvLayer( 512, 512+512, 3, 3, convPadZero, bias, activReLU )
		->AddConnection( 1, 0, 0, 512 )								// ※１つ前のレイヤーと
		->AddConnection( mlp.LayerOffsetOf(pSkip3),
								0, 0, 512, trim3, trim3 ) ;			//   pSkip3 から入力する
	mlp.AppendConvLayer( 512, 512, 3, 3, convPadZero, bias, activReLU ) ;

	mlp.AppendUp2x2Layer( 256, 512, 1, activReLU ) ;				// 1/4
	mlp.AppendConvLayer( 256, 256+256, 3, 3, convPadZero, bias, activReLU )
		->AddConnection( 1, 0, 0, 256 )
		->AddConnection( mlp.LayerOffsetOf(pSkip2), 0, 0, 256, trim2, trim2 ) ;
	mlp.AppendConvLayer( 256, 256, 3, 3, convPadZero, bias, activReLU ) ;

	mlp.AppendUp2x2Layer( 128, 256, 1, activReLU ) ;				// 1/2
	mlp.AppendConvLayer( 128, 128+128, 3, 3, convPadZero, bias, activReLU )
		->AddConnection( 1, 0, 0, 128 )
		->AddConnection( mlp.LayerOffsetOf(pSkip1), 0, 0, 128, trim1, trim1 ) ;
	mlp.AppendConvLayer( 128, 128, 3, 3, convPadZero, bias, activReLU ) ;

	mlp.AppendUp2x2Layer( 64, 128, 1, activReLU ) ;				// 1/1
	mlp.AppendConvLayer( 64, 64+64, 3, 3, convPadZero, bias, activReLU )
		->AddConnection( 1, 0, 0, 64 )
		->AddConnection( mlp.LayerOffsetOf(pSkip0), 0, 0, 64, trim0, trim0 ) ;
	mlp.AppendConvLayer( 64, 64, 3, 3, convPadZero, bias, activReLU )
		->SetDropoutRate( dropout ) ;
	NNPerceptronPtr	pResidual =
		mlp.AppendLayer( 4, 64, bias, activTanh ) ;

	mlp.AppendPointwiseAdd( 4, pResidual, 0, pRGB2YCrgb, 0, trim0, trim0, activLinear ) ;


	// 輝度＋色差 -> RGB
	static const float	crgb2rgb[12] =
	{
		// Y    Cb    Cg    Cr
		 1.0f, 1.0f, 0.0f, 0.0f,	// Blue
		 1.0f, 0.0f, 1.0f, 0.0f,	// Green
		 1.0f, 0.0f, 0.0f, 1.0f,	// Red
	} ;

	NNPerceptronPtr	pYCrgb2RGB =
		std::make_shared<NNFixedPerceptron>
			( 3, 4, 1, 0,
				std::make_shared<NNSamplerInjection>(),
				std::make_shared<NNActivationLinear>() ) ;
	for ( size_t i = 0; i < 12; i ++ )
	{
		pYCrgb2RGB->Matrix().ArrayAt(i) = crgb2rgb[i] ;
	}
	mlp.AppendLayer( pYCrgb2RGB ) ;


	// 各レイヤー（PointwiseAdd, MaxPool レイヤー除く）に
	// L2 正則化・適応的最適化・正規化設定
	for ( size_t iLayer = 0; iLayer < mlp.GetLayerCount() - 1; iLayer ++ )
	{
		NNPerceptronPtr	pLayer = mlp.GetLayerAt(iLayer) ;
		if ( dynamic_cast<NNFixedPerceptron*>( pLayer.get() ) != nullptr )
		{
			// 固定パラメータ（MaxPool, PointwiseAdd）レイヤーは除外
			continue ;
		}
		pLayer->SetRidgeParameter( 0.01f ) ;
		pLayer->SetAdaptiveOptimization( NNPerceptron::adaOptMomentum, adaParam ) ;
		pLayer->SetNormalization( std::make_shared<NNInstanceNormalization>(normParam) ) ;
	}
}

// 学習実行前
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::BeforeLearning( void )
{
}

// 学習用イテレーター作成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNMLPShell::Iterator> PalesibylApp::MakeLearningIter( void )
{
	return	std::make_shared<NNMLPShellImageCropper>
		( "learn\\source", "learn\\teacher",
			NNBufDim( 320, 320, 3 ), trim0, trim0, trim0, trim0 ) ;
}

// 予測用イテレーター作成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNMLPShell::Iterator> PalesibylApp::MakePredictiveIter( void )
{
	return	std::make_shared<NNMLPShellImageIterator>
					( "predict\\src", "predict\\out", true, 3 ) ;
}

