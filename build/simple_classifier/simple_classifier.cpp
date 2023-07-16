
#include "sample_app_stub.h"

//#define	__CLASS_INDEX_FORMAT__	1		// one-hot をインデックス表現する
//#define	__FAST_SOFTMAX__		1		// softmax, argmax 高速化


// アプリ初期化
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::Initialize( void )
{
}

// アプリ固有の説明（ファイルの配置など）
//////////////////////////////////////////////////////////////////////////////
const char *	PalesibylApp::s_pszSpecificDescription =
	"ディレクトリ構成;\r\n"
	"[classes/]\r\n"
	"  + [xxxxxx/] : 分類別画像ファイル（フォルダ名は分類名）\r\n"
	"[predict/]    : 予測分類元画像ファイル\r\n" ;


// モデルを作成
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::BuildModel( NNMLPShell::Iterator * pIter )
{
	NNMLPShell::Classifier *	
					pClassifier = dynamic_cast<NNMLPShell::Classifier*>( pIter ) ;
	assert( pClassifier != nullptr ) ;

	const size_t	nClassCount = (pClassifier != nullptr)
									? pClassifier->GetClassCount() : 11 ;

	NNMultiLayerPerceptron&	mlp = m_shell.MLP() ;

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
		mlp.AppendFastSoftmax( nClassCount, 128, 1, activFastSoftmax ) ;
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
		std::make_shared<NNMLPShellImageClassifier>( "classes", false, nullptr, 1 ) ;
#ifdef	__CLASS_INDEX_FORMAT__
	iter->SetOneHotToIndexFormat( true ) ;
#endif
	return	iter ;
}

// 予測用イテレーター作成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNMLPShell::Iterator> PalesibylApp::MakePredictiveIter( void )
{
	std::shared_ptr<NNMLPShellImageClassifier>	iter =
		std::make_shared<NNMLPShellImageClassifier>( "predict", true, "classes", 1 ) ;
#ifdef	__CLASS_INDEX_FORMAT__
	iter->SetOneHotToIndexFormat( true ) ;
#endif
	return	iter ;
}


