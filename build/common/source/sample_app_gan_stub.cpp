
#include "sample_app_gan_stub.h"


// エントリポイント
//////////////////////////////////////////////////////////////////////////////
int main( int argc, char *argv[], char *envp[] )
{
	cudaInit() ;
	NNMLPShell::StaticInitialize() ;

	int	codeExit = 0 ;
	{
		PalesibylApp	app ;
		app.Initialize() ;

		codeExit = app.ParseArguments( argc, argv ) ;
		if ( codeExit == 0 )
		{
			codeExit = app.Run() ;
		}
	}

	NNMLPShell::StaticRelase() ;
	return	codeExit ;
}



//////////////////////////////////////////////////////////////////////////////
// アプリケーション
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
PalesibylApp::PalesibylApp( void )
	: m_strClassModelFile( "classifier.mlp" ), m_nGANLoopCount( 50 )
{
}

// 引数解釈
//////////////////////////////////////////////////////////////////////////////
bool PalesibylApp::ParseArgumentAt( int& iArg, int nArgs, char * pszArgs[] )
{
	if ( PalesibylBasicApp::ParseArgumentAt( iArg, nArgs, pszArgs ) )
	{
		return	true ;
	}
	enum	ParameterType
	{
		paramNull,
		paramClassifier,
		paramGANLoopCount,
	} ;
	static const struct
	{
		const char *	pszOpt ;
		ParameterType	paramType ;
	}	s_ParamOpt[] =
	{
		{ "/clsf", paramClassifier },
		{ "/ganloop", paramGANLoopCount },
		{ nullptr, paramNull },
	} ;
	const char *	pszArg = pszArgs[iArg] ;
	bool			flagSuccess = true ;
	ParameterType	paramNext = paramNull ;
	for ( int i = 0; s_ParamOpt[i].pszOpt != nullptr; i ++ )
	{
		if ( strcmp( pszArg, s_ParamOpt[i].pszOpt ) == 0 )
		{
			paramNext = s_ParamOpt[i].paramType ;
			break ;
		}
	}
	switch ( paramNext )
	{
	case	paramClassifier:
		if ( IsValidNextArgument( iArg, nArgs, pszArgs ) )
		{
			m_strClassModelFile = pszArgs[++ iArg] ;
		}
		else
		{
			flagSuccess = false ;
		}
		break ;

	case	paramGANLoopCount:
		m_nGANLoopCount =
			(size_t) NextArgumentInt( flagSuccess, iArg, nArgs, pszArgs ) ;
		break ;

	case	paramNull:
	default:
		flagSuccess = false ;
		break ;
	}
	return	flagSuccess ;
}

// ヘルプ表示
//////////////////////////////////////////////////////////////////////////////
int PalesibylApp::RunHelp( void )
{
	std::cout << "usage: [options]..." << std::endl ;
	std::cout << "option;" << std::endl ;
	std::cout << "/l <mode-file>    : 学習を実行します" << std::endl ;
	std::cout << "/p <mode-file>    : 予測を実行します" << std::endl ;
	std::cout << "/cuda             : CUDA を利用します" << std::endl ;
	std::cout << "/clsf <file>      : 分類器モデルファイル名を指定します" << std::endl ;
	std::cout << "/ganloop <count>  : GAN ループ回数を指定します" << std::endl ;
	std::cout << "/loop <count>     : 学習ループ回数を指定します" << std::endl ;
	std::cout << "/subloop <count>  : ミニバッチの反復回数を指定します" << std::endl ;
	std::cout << "/batch <count>    : ミニバッチサイズを指定します" << std::endl ;
	std::cout << "/delta <rate>[,<end-rate>]" << std::endl ;
	std::cout << "                  : 学習係数を指定します" << std::endl ;
	std::cout << "/thread <count>   : 最大のスレッド数を指定します" << std::endl ;
	std::cout << "/batch_thread <count>" << std::endl ;
	std::cout << "                  : ミニバッチの並列スレッド数を指定します" << std::endl ;
	std::cout << "/log <csv-file>   : 学習ログファイルを出力します" << std::endl ;
	std::cout << "/lgrd             : 学習ログにレイヤー毎の勾配ノルムを出力します" << std::endl ;
	std::cout << "/tio <image-file> : 訓練画像の予測をミニバッチ毎に出力します" << std::endl ;
	std::cout << "/vio <image-file> : 検証用画像の予測を逐次出力します" << std::endl ;
	std::cout << "/ndo              : ドロップアウトは行わない" << std::endl ;
	std::cout << "/nlfb             : ミニバッチ毎に進捗表示を改行しない" << std::endl ;
	std::cout << "/pbs              : 学習中間バッファのサイズを表示します" << std::endl ;
	std::cout << "/cubs             : 学習中間 CUDA バッファのサイズを表示します" << std::endl ;
	std::cout << std::endl ;
	std::cout << "※学習／予測処理は ESC キーで中断できます" << std::endl ;
	std::cout << std::endl ;
	std::cout << s_pszSpecificDescription ;
	return	0 ;
}

// 学習実行
//////////////////////////////////////////////////////////////////////////////
int PalesibylApp::RunLearning( void )
{
	m_classifier.SetShellConfig( m_cfgShell ) ;
	m_classifier.SetMLPConfig( m_cfgBuf ) ;

	std::shared_ptr<NNMLPShell::Iterator>	pIter = MakeLearningIter() ;
	NNMLPShell::GANIterator *
		pGanIter = dynamic_cast<NNMLPShell::GANIterator*>( pIter.get() ) ;
	assert( pGanIter != nullptr ) ;
	if ( pGanIter == nullptr )
	{
		return	1 ;
	}

	// 生成モデル
	if ( !m_shell.LoadModel( m_strModelFile.c_str() ) )
	{
		m_shell.MLP().ClearAll() ;
		BuildModel( pIter.get() ) ;
		m_shell.SaveModel( m_strModelFile.c_str() ) ;
	}

	// 分類器モデル
	if ( !m_classifier.LoadModel( m_strClassModelFile.c_str() ) )
	{
		m_classifier.MLP().ClearAll() ;
		BuildClassifier( pIter.get() ) ;
		m_classifier.SaveModel( m_strClassModelFile.c_str() ) ;
	}
	BeforeLearning() ;

	// ログファイル
	if ( !m_strLearnLogFile.empty() )
	{
		if ( !m_shell.MakeOutputCSV( m_strLearnLogFile.c_str() ) )
		{
			std::cout << m_strLearnLogFile << " を開けませんでした" << std::endl ;
		}
	}

	// GAN 訓練
	m_shell.DoLearningGAN
		( *pIter, m_classifier,
			*(pGanIter->GetClassifierIterator()),
			m_nGANLoopCount, m_param ) ;

	// モデル保存
	if ( !m_shell.SaveModel( m_strModelFile.c_str() ) )
	{
		std::cout << m_strModelFile << " への書き出しに失敗しました" << std::endl ;
	}
	if ( !m_classifier.SaveModel( m_strClassModelFile.c_str() ) )
	{
		std::cout << m_strClassModelFile << " への書き出しに失敗しました" << std::endl ;
	}
	return	0 ;
}
