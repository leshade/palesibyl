
#include "sample_app_stub.h"
#include "simple_upsampler.h"


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

	mlp.AppendConvLayer( 64, 3, 3, 3, convPadBorder, 1, activReLU ) ;
	mlp.AppendUp2x2Layer( 16, 64, 1, activReLU ) ;
	mlp.AppendConvLayer( 3, 16, 3, 3, convPadZero, 1, activLinear ) ;
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
	return	std::make_shared<UpsamplerImageCropper>
				( "learn\\source", "learn\\teacher", NNBufDim( 128, 128, 3 ), 2 ) ;
}

// 予測用イテレーター作成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNMLPShell::Iterator> PalesibylApp::MakePredictiveIter( void )
{
	return	std::make_shared<NNMLPShellImageIterator>
					( "predict\\src", "predict\\out", true, 3 ) ;
}




//////////////////////////////////////////////////////////////////////////////
// アップサンプラー画像ファイル固定サイズ切り出し（学習専用）
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
UpsamplerImageCropper::UpsamplerImageCropper
	( const char * pszSourceDir,
		const char * pszPairDir,
		const NNBufDim& dimCrop, size_t nUpScale )
	: NNMLPShellImageCropper( pszSourceDir, pszPairDir, dimCrop ),
		m_nUpScale( nUpScale )
{
}

// ソースの切り出し位置に対応する教師データを切り出す
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> UpsamplerImageCropper::CropTeachingData
	( std::shared_ptr<NNBuffer> pTeaching, const NNBufDim& dimCropOffset )
{
	NNBufDim	dimUpCropSize( m_dimCrop.x * m_nUpScale,
								m_dimCrop.y * m_nUpScale, m_dimCrop.z ) ;
	NNBufDim	dimUpCropOffset( dimCropOffset.x * m_nUpScale,
								dimCropOffset.y * m_nUpScale, dimCropOffset.z ) ;

	std::shared_ptr<NNBuffer>	pCropTeaching = std::make_shared<NNBuffer>() ;
	pCropTeaching->Create( dimUpCropSize ) ;
	pCropTeaching->CopyFrom( *pTeaching, dimUpCropOffset ) ;
	return	pCropTeaching ;
}


