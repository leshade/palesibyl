
#include "sample_app_stub.h"


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

	NNPerceptron::AdaptiveHyperparameter	adaParam ;
	adaParam.alpha = 0.9f ;
	adaParam.delta = 0.1f ;

	NNNormalizationFilter::Hyperparameter	normParam ;
	normParam.delta = 0.3f ;
	normParam.deltac = 0.00001f ;

	NNPerceptronPtr	pPrevLayer =
		mlp.AppendConvLayer( 64, 4, 3, 3, convPadBorder, 0, activReLU ) ;
	pPrevLayer
		->SetRidgeParameter( 0.01f )
		->SetAdaptiveOptimization( NNPerceptron::adaOptMomentum, adaParam )
		->SetNormalization( std::make_shared<NNInstanceNormalization>(normParam) ) ;

	for ( int i = 0; i < 10; i ++ )
	{
		mlp.AppendConvLayer( 64, 64, 3, 3, convPadZero, 0, activReLU )
			->SetRidgeParameter( 0.01f )
			->SetAdaptiveOptimization( NNPerceptron::adaOptMomentum, adaParam )
			->SetNormalization( std::make_shared<NNInstanceNormalization>(normParam) ) ;

		NNPerceptronPtr	pResidual =
			mlp.AppendConvLayer( 64, 64, 3, 3, convPadZero, 0, activLinear ) ;
		pResidual
			->SetRidgeParameter( 0.01f )
			->SetAdaptiveOptimization( NNPerceptron::adaOptMomentum, adaParam )
			->SetNormalization( std::make_shared<NNInstanceNormalization>(normParam) ) ;

		pPrevLayer =
			mlp.AppendPointwiseAdd
				( 64, pResidual, 0, pPrevLayer, 0, 0, 0, activReLU ) ;
	}

	NNPerceptronPtr	pResidual = mlp.AppendLayer( 4, 64, 0, activTanh ) ;
	pResidual
		->SetRidgeParameter( 0.01f )
		->SetAdaptiveOptimization( NNPerceptron::adaOptMomentum, adaParam )
		->SetNormalization( std::make_shared<NNInstanceNormalization>(normParam) ) ;

	mlp.AppendPointwiseAdd( 4, pResidual, 0, pRGB2YCrgb, 0 ) ;

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
					( "learn\\source", "learn\\teacher", NNBufDim( 128, 128, 3 ) ) ;
}

// 予測用イテレーター作成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNMLPShell::Iterator> PalesibylApp::MakePredictiveIter( void )
{
	return	std::make_shared<NNMLPShellImageIterator>
					( "predict\\src", "predict\\out", true, 3 ) ;
}

