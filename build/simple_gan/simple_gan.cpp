
#include "sample_app_gan_stub.h"

#define	__CLASS_INDEX_FORMAT__	1		// one-hot をインデックス表現する
//#define	__FAST_SOFTMAX__		1		// softmax, argmax 高速化



// アプリ初期化
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::Initialize( void )
{
}

// アプリ固有の説明（ファイルの配置など）
//////////////////////////////////////////////////////////////////////////////
const char *	PalesibylApp::s_pszSpecificDescription =
	"ファイル・ディレクトリ構成;\n"
	"classifier.mlp : 分類器モデルファイル\n"
	"[classes\\]\n"
	"  + [xxxxxx\\] : 分類別画像ファイル（フォルダ名は分類名）\n"
	"[predict\\]\n"
	"  + [src\\]    : 分類名記述ファイル\n"
	"  + [out\\]    : 生成画像出力先\n" ;


// 生成器モデルを作成
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::BuildModel( NNMLPShell::Iterator * pIter )
{
	NNMLPShell::GANIterator *
		pGanIter = dynamic_cast<NNMLPShell::GANIterator*>( pIter ) ;
	assert( pGanIter != nullptr ) ;

	NNMLPShell::Classifier *
		pClassifier = dynamic_cast<NNMLPShell::Classifier*>
							( pGanIter->GetClassifierIterator().get() ) ;
	assert( pClassifier != nullptr ) ;

	const size_t	nClassCount = (pClassifier != nullptr)
									? pClassifier->GetClassCount() : 11 ;
#ifdef	__CLASS_INDEX_FORMAT__
	const size_t	nInChannels = 1 ;
#else
	const size_t	nInChannels = nClassCount ;
#endif

	NNMultiLayerPerceptron&	mlp = m_shell.MLP() ;
	mlp.SetInputShape
		( 0, NNBufDim( 1, 1, nInChannels ), NNBufDim( 1, 1, nInChannels )  ) ;

#ifdef	__CLASS_INDEX_FORMAT__
	mlp.AppendLayerAsOneHot( 128, nClassCount, activLinear ) ;	// 1x1
#else
	mlp.AppendLayer( 128, nClassCount, 1, activLinear ) ;			// 1x1
#endif
	mlp.AppendUp2x2Layer( 64, 128, 1, activLinear ) ;				// 2x2
	mlp.AppendUp2x2Layer( 32, 64, 1, activLinear ) ;				// 4x4
	mlp.AppendUp2x2Layer( 16, 32, 1, activLinear ) ;				// 8x8
	mlp.AppendUp2x2Layer( 1, 16, 1, activLinear ) ;					// 16x16
}

// 分類器モデルを作成
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::BuildClassifier( NNMLPShell::Iterator * pIter )
{
	NNMLPShell::GANIterator *
		pGanIter = dynamic_cast<NNMLPShell::GANIterator*>( pIter ) ;
	assert( pGanIter != nullptr ) ;

	NNMLPShell::Classifier *
		pClassifier = dynamic_cast<NNMLPShell::Classifier*>
							( pGanIter->GetClassifierIterator().get() ) ;
	assert( pClassifier != nullptr ) ;

	const size_t	nClassCount = (pClassifier != nullptr)
									? pClassifier->GetClassCount() : 11 ;

	NNMultiLayerPerceptron&	mlp = m_classifier.MLP() ;
	mlp.SetInputShape( 0, NNBufDim( 16, 16, 1 ), NNBufDim( 0, 0, 1 )  ) ;

	mlp.AppendConvLayer( 16, 1, 3, 3, convPadBorder, 1, activReLU ) ;		// 16x16
	mlp.AppendMaxPoolLayer( 16, 2, 2 ) ;									// 8x8
	mlp.AppendConvLayer( 32, 16, 3, 3, convPadZero, 1, activReLU ) ;
	mlp.AppendMaxPoolLayer( 32, 2, 2 ) ;									// 4x4
	mlp.AppendConvLayer( 64, 32, 3, 3, convPadZero, 1, activReLU, 2, 2 ) ;	// 2x2
	mlp.AppendConvLayer( 128, 64, 2, 2, convNoPad, 0, activReLU, 2, 2 ) ;	// 1x1
#ifdef	__FAST_SOFTMAX__
	#ifdef	__CLASS_INDEX_FORMAT__
		mlp.AppendFastSoftmax( nClassCount, 128, 1, activFastArgmax ) ;
	#else
		mlp.AppendFastSoftmax( nClassCount, 128, 1, activSoftmax ) ;
	#endif
	#else
	#ifdef	__CLASS_INDEX_FORMAT__
		mlp.AppendLayer( nClassCount, 128, 1, activArgmax ) ;
	#else
		mlp.AppendLayer( nClassCount, 128, 1, activSoftmax ) ;
	#endif
#endif
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
	std::shared_ptr<NNMLPShellImageClassifier>	iter =
			std::make_shared<NNMLPShellImageClassifier>
								( "classes", false, nullptr, 1 ) ;
#ifdef	__CLASS_INDEX_FORMAT__
	iter->SetOneHotToIndexFormat( true ) ;
#endif
	return	std::make_shared<NNMLPShellGANIterator>( iter ) ;
}

// 予測用イテレーター作成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNMLPShell::Iterator> PalesibylApp::MakePredictiveIter( void )
{
	std::shared_ptr<NNMLPShellImageGenerativeIterator>	iter =
		std::make_shared<NNMLPShellImageGenerativeIterator>
						( "predict\\src", "predict\\out", "classes" ) ;
#ifdef	__CLASS_INDEX_FORMAT__
	iter->SetOneHotToIndexFormat( true ) ;
#endif
	return	iter ;
}


