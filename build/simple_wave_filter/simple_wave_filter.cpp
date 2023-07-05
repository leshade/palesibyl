
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
	"  + [source\\]  : 学習元 WAVE ファイル\n"
	"  + [teacher\\] : 教師 WAVE ファイル（学習元と同名ファイル）\n"
	"[predict\\]\n"
	"  + [src\\]     : 予測入力 WAVE ファイル\n"
	"  + [out\\]     : 予測出力先\n" ;


// モデルを作成
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::BuildModel( NNMLPShell::Iterator * pIter )
{
	NNMultiLayerPerceptron&	mlp = m_shell.MLP() ;
	mlp.SetInputShape
		( NNMultiLayerPerceptron::mlpFlagStream,
				NNBufDim( 5, 1, 4 ), NNBufDim( 1, 1, 4 ) ) ;
	mlp.AppendLayer( 4, 4, 0, activLinear ) ;
	mlp.AppendLayer( 8, 12, 0, activLinear )
		->AddConnection( 1, 0, 0, 4 )			// 直前レイヤーから  : 4チャネル入力
		->AddConnection( 0, 1, 0, 8 ) ;			// このレイヤー(t-1) : 8チャネル入力
	mlp.AppendLayer( 16, 24, 0, activLinear )
		->AddConnection( 1, 0, 0, 8 )
		->AddConnection( 0, 1, 0, 16 ) ;
	mlp.AppendLayer( 32, 48, 0, activLinear )
		->AddConnection( 1, 0, 0, 16 )
		->AddConnection( 0, 1, 0, 32 ) ;
	mlp.AppendLayer( 64, 96, 0, activLinear )
		->AddConnection( 1, 0, 0, 32 )
		->AddConnection( 0, 1, 0, 64 ) ;
	mlp.AppendLayer( 4, 64, 0, activLinear ) ;
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
	return	std::make_shared<NNMLPShellWaveCropper>
			( "learn\\source", "learn\\teacher",
					NNBufDim( 1024/4, 256, 1 ),
					NNMLPShellWaveCropper::cropPadZero, 4, 4 ) ;
}

// 予測用イテレーター作成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNMLPShell::Iterator> PalesibylApp::MakePredictiveIter( void )
{
	return	std::make_shared<NNMLPShellWaveIterator>
			( "predict\\src", "predict\\out", true, 1, 4, 4 ) ;
}

