
#include "sample_app_stub.h"


// アプリ初期化
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::Initialize( void )
{
}

// アプリ固有の説明（ファイルの配置など）
//////////////////////////////////////////////////////////////////////////////
const char *	PalesibylApp::s_pszSpecificDescription =
	"ディレクトリ構成;\r\n"
	"[learn/]\r\n"
	"  + [source/]  : 学習元画像ファイル\r\n"
	"  + [teacher/] : 教師画像ファイル（学習元と同名ファイル）\r\n"
	"[predict/]\r\n"
	"  + [src/]     : 予測入力画像ファイル\r\n"
	"  + [out/]     : 予測出力先\r\n" ;


// モデルを作成
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::BuildModel( NNMLPShell::Iterator * pIter )
{
	NNMultiLayerPerceptron&	mlp = m_shell.MLP() ;
	mlp.AppendLayer( 3, 3, 1, activLinear ) ;
}

// 学習実行前
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::BeforeLearning( void )
{
}

// 予測実行前
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::BeforePrediction( void )
{
}

// 学習用イテレーター作成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNMLPShell::Iterator> PalesibylApp::MakeLearningIter( void )
{
	return	std::make_shared<NNMLPShellImageIterator>
					( "learn/source", "learn/teacher", false, 3 ) ;
}

// 予測用イテレーター作成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNMLPShell::Iterator> PalesibylApp::MakePredictiveIter( void )
{
	return	std::make_shared<NNMLPShellImageIterator>
					( "predict/src", "predict/out", true, 3 ) ;
}

