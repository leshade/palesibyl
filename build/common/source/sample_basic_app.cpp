﻿
#include "sample_basic_app.h"
#include <string.h>


//////////////////////////////////////////////////////////////////////////////
// アプリケーション
//////////////////////////////////////////////////////////////////////////////

// 訓練データ出力を画像ファイルとして出力する
//////////////////////////////////////////////////////////////////////////////
void PalesibylBasicApp::AppShell::SetTrainingImageFile( const char * pszFilePath )
{
	m_pathTrainImage = pszFilePath ;
}

// 検証データ出力を画像ファイルとして出力する
//////////////////////////////////////////////////////////////////////////////
void PalesibylBasicApp::AppShell::SetValidationImageFile( const char * pszFilePath )
{
	m_pathValidImage = pszFilePath ;
}

// 出力 CSV ファイルを開く
//////////////////////////////////////////////////////////////////////////////
bool PalesibylBasicApp::AppShell::MakeOutputCSV( const char * pszFilePath )
{
	m_strLogFile = pszFilePath ;
	std::unique_ptr<std::ofstream>	ofs = OpenLogFile( true ) ;
	if ( ofs == nullptr )
	{
		return	false ;
	}
	std::shared_ptr<NNEvaluationFunction>
			pEval = MLP().GetEvaluationFunction() ;
	if ( pEval != nullptr )
	{
		*ofs << "\"traning " << pEval->GetDisplayName()
				<< "\",\"validation " << pEval->GetDisplayName() << "\"," ;
	}
	*ofs << "\"traning loss\",\"validation loss\"" ;
	if ( m_logGradient )
	{
		*ofs << "," ;
		for ( size_t i = 0; i < MLP().GetLayerCount(); i ++ )
		{
			NNPerceptronPtr	pLayer = MLP().GetLayerAt(i) ;
			if ( !pLayer->IsMatrixFixed() )
			{
				if ( pLayer->GetIdentity().empty() )
				{
					*ofs << ",g" << i ;
				}
				else
				{
					*ofs << ",\"" << pLayer->GetIdentity() << "\"" ;
				}
			}
		}
	}
	*ofs << std::endl ;
	m_logOutput = true ;
	return	true ;
}

// 勾配をログ出力させるか
//////////////////////////////////////////////////////////////////////////////
void PalesibylBasicApp::AppShell::SetLogGradient( bool log )
{
	m_logGradient = log ;
}

// ログファイルを開く
//////////////////////////////////////////////////////////////////////////////
std::unique_ptr<std::ofstream>
	PalesibylBasicApp::AppShell::OpenLogFile( bool flagCreate ) const
{
	try
	{
		return	std::make_unique<std::ofstream>
					( m_strLogFile, flagCreate ? std::ios::out : std::ios::app ) ;
	}
	catch ( const std::exception& e )
	{
		TRACE( "exception at PalesibylBasicApp::AppShell::OpenLogFile: %s\r\n", e.what() ) ;
	}
	return	nullptr ;
}

// 自動保存ファイル名を設定し、自動保存を有効にする
//////////////////////////////////////////////////////////////////////////////
void PalesibylBasicApp::AppShell::SetAutoSaveFile( const char * pszFilePath )
{
	m_flagAutoSave = true ;
	m_strAutoSave = pszFilePath ;
	m_timeAutoSave.Start() ;
}

// 学習進捗
//////////////////////////////////////////////////////////////////////////////
void PalesibylBasicApp::AppShell::OnLearningProgress
	( LearningEvent le, const LearningProgressInfo& lpi )
{
	NNMLPShell::OnLearningProgress( le, lpi ) ;

	if ( (le == learningEndMiniBatch)
		&& !m_pathTrainImage.empty() && (lpi.pTraining != nullptr) )
	{
		NNImageCodec::SaveToFile( m_pathTrainImage, *(lpi.pTraining) ) ;
	}
	if ( (le == learningValidation) && !m_pathValidImage.empty()
		&& (lpi.iValidation == 0) && (lpi.pValidation != nullptr) )
	{
		NNImageCodec::SaveToFile( m_pathValidImage, *(lpi.pValidation) ) ;
	}
	if ( (le == learningEndEpoch) && m_logOutput )
	{
		std::unique_ptr<std::ofstream>	ofs = OpenLogFile( false ) ;
		if ( ofs != nullptr )
		{
			std::shared_ptr<NNEvaluationFunction>
					pEval = MLP().GetEvaluationFunction() ;
			if ( pEval != nullptr )
			{
				*ofs << lpi.evalLearn << "," << lpi.evalValid << "," ;
			}
			*ofs << lpi.lossLearn << "," << lpi.lossValid ;
			if ( m_logGradient )
			{
				*ofs << "," ;
				for ( size_t i = 0; i < lpi.gradNorms.size(); i ++ )
				{
					if ( !MLP().GetLayerAt(i)->IsMatrixFixed() )
					{
						*ofs << "," << (lpi.gradNorms.at(i) / (float) lpi.nGradNorm) ;
					}
				}
			}
			*ofs << std::endl ;
		}
	}
	if ( (le == learningEndEpoch) && m_flagAutoSave )
	{
		if ( m_timeAutoSave.MeasureMilliSec() > 60 * 60 * 1000 )
		{
			SaveModel( m_strAutoSave.c_str() ) ;
			m_timeAutoSave.Start() ;
		}
	}
}


// PalesibylBasicApp 構築
//////////////////////////////////////////////////////////////////////////////
PalesibylBasicApp::PalesibylBasicApp( void )
	: m_flagsArg(0)
{
	m_cfgShell = m_shell.GetShellConfig() ;
	m_cfgBuf = m_shell.GetMLPConfig() ;
	m_cfgBuf.flagUseCuda = false ;
}

// アプリ初期化
//////////////////////////////////////////////////////////////////////////////
void PalesibylBasicApp::Initialize( void )
{
}

// 引数解釈
//////////////////////////////////////////////////////////////////////////////
int PalesibylBasicApp::ParseArguments( int nArgs, char * pszArgs[] )
{
	for ( int iArg = 1; iArg < nArgs; iArg ++ )
	{
		if ( !ParseArgumentAt( iArg, nArgs, pszArgs ) )
		{
			std::cout << pszArgs[iArg] << " は不正なパラメータです" << std::endl ;
			return	1 ;
		}
	}
	if ( !CheckArgumentsAfterParse() )
	{
		return	1 ;
	}
	return	0 ;
}

bool PalesibylBasicApp::ParseArgumentAt( int& iArg, int nArgs, char * pszArgs[] )
{
	enum	ParameterType
	{
		paramNull,
		paramCuda,
		paramHelp,
		paramLearn,
		paramPredict,
		paramEpochCount,
		paramSubLoopCount,
		paramBatchCount,
		paramDeltaRate,
		paramThreadCount,
		paramBatchThreadCount,
		paramLearningLog,
		paramLogGradient,
		paramTrainImageOut,
		paramValidImageOut,
		paramAutoSaveModel,
		paramLineFeedByMiniBatch,
		paramPrintBufferSize,
		paramPrintCudaBufferSize,
		paramNoDropout,
	} ;
	static const struct
	{
		const char *	pszOpt ;
		ParameterType	paramType ;
	}	s_ParamOpt[] =
	{
		{ "/cuda", paramCuda },
		{ "/help", paramHelp },
		{ "/?", paramHelp },
		{ "/l", paramLearn },
		{ "/p", paramPredict },
		{ "/loop", paramEpochCount },
		{ "/subloop", paramSubLoopCount },
		{ "/batch", paramBatchCount },
		{ "/delta", paramDeltaRate },
		{ "/thread", paramThreadCount },
		{ "/batch_thread", paramBatchThreadCount },
		{ "/log", paramLearningLog },
		{ "/lgrd", paramLogGradient },
		{ "/tio", paramTrainImageOut },
		{ "/vio", paramValidImageOut },
		{ "/asm", paramAutoSaveModel },
		{ "/nlfb", paramLineFeedByMiniBatch },
		{ "/pbs", paramPrintBufferSize },
		{ "/cubs", paramPrintCudaBufferSize },
		{ "/ndo", paramNoDropout },
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
	case	paramCuda:
		if ( cudaIsAvailable() )
		{
			m_cfgBuf.flagUseCuda = true ;
		}
		else
		{
			std::cout << "CUDA は利用できません" << std::endl ;
		}
		break ;

	case	paramHelp:
		m_flagsArg |= argumentHelp ;
		break ;

	case	paramLearn:
		flagSuccess = IsValidNextArgument( iArg, nArgs, pszArgs ) ;
		if ( flagSuccess )
		{
			m_flagsArg |= argumentLearn ;
			m_strModelFile = pszArgs[++ iArg] ;
		}
		break ;

	case	paramPredict:
		flagSuccess = IsValidNextArgument( iArg, nArgs, pszArgs ) ;
		if ( flagSuccess )
		{
			m_flagsArg |= argumentPrediction ;
			m_strModelFile = pszArgs[++ iArg] ;
		}
		break ;

	case	paramEpochCount:
		m_param.nEpochCount =
			(size_t) NextArgumentInt( flagSuccess, iArg, nArgs, pszArgs ) ;
		break ;

	case	paramSubLoopCount:
		m_param.nSubLoopCount =
			(size_t) NextArgumentInt( flagSuccess, iArg, nArgs, pszArgs ) ;
		break ;

	case	paramBatchCount:
		m_param.nMiniBatchCount =
			(size_t) NextArgumentInt( flagSuccess, iArg, nArgs, pszArgs ) ;
		break ;

	case	paramDeltaRate:
		m_param.deltaRate0 =
			(float) NextArgumentFloat( flagSuccess, iArg, nArgs, pszArgs ) ;
		if ( flagSuccess )
		{
			m_param.deltaRate1 = m_param.deltaRate0 ;
			//
			pszArg = pszArgs[iArg] ;
			for ( size_t i = 0; pszArg[i] != 0; i ++ )
			{
				if ( pszArg[i] == ',' )
				{
					m_param.deltaRate1 = (float) atof( pszArg + (i + 1) ) ;
					break ;
				}
			}
		}
		break ;

	case	paramThreadCount:
		m_cfgBuf.nThreadCount =
			(size_t) NextArgumentInt( flagSuccess, iArg, nArgs, pszArgs ) ;
		break ;

	case	paramBatchThreadCount:
		m_cfgShell.nBatchThreads =
			(size_t) NextArgumentInt( flagSuccess, iArg, nArgs, pszArgs ) ;
		m_cfgBuf.nThreadCount = 1 ;
		break ;

	case	paramLearningLog:
		flagSuccess = IsValidNextArgument( iArg, nArgs, pszArgs ) ;
		if ( flagSuccess )
		{
			m_strLearnLogFile = pszArgs[++ iArg] ;
		}
		break ;

	case	paramLogGradient:
		m_shell.SetLogGradient( true ) ;
		break ;

	case	paramTrainImageOut:
		flagSuccess = IsValidNextArgument( iArg, nArgs, pszArgs ) ;
		if ( flagSuccess )
		{
			m_shell.SetTrainingImageFile( pszArgs[++ iArg] ) ;
		}
		break ;

	case	paramValidImageOut:
		flagSuccess = IsValidNextArgument( iArg, nArgs, pszArgs ) ;
		if ( flagSuccess )
		{
			m_shell.SetValidationImageFile( pszArgs[++ iArg] ) ;
		}
		break ;

	case	paramAutoSaveModel:
		flagSuccess = IsValidNextArgument( iArg, nArgs, pszArgs ) ;
		if ( flagSuccess )
		{
			m_shell.SetAutoSaveFile( pszArgs[++ iArg] ) ;
		}
		break ;

	case	paramLineFeedByMiniBatch:
		m_cfgShell.flagsBehavior &= ~NNMLPShell::behaviorLineFeedByMiniBatch ;
		break ;

	case	paramPrintBufferSize:
		m_cfgShell.flagsBehavior |= NNMLPShell::behaviorPrintBufferSize ;
		break ;

	case	paramPrintCudaBufferSize:
		m_cfgShell.flagsBehavior |= NNMLPShell::behaviorPrintCudaBufferSize ;
		break ;

	case	paramNoDropout:
		m_cfgShell.flagsBehavior &= ~NNMLPShell::behaviorNoDropout ;
		break ;

	case	paramNull:
	default:
		flagSuccess = false ;
		break ;
	}
	return	flagSuccess ;
}

bool PalesibylBasicApp::CheckArgumentsAfterParse( void )
{
	if ( (m_flagsArg
				& (argumentHelp | argumentLearn
								| argumentPrediction)) == 0 )
	{
		m_flagsArg |= argumentHelp ;
	}
	else if ( (m_flagsArg & (argumentLearn | argumentPrediction))
							== (argumentLearn | argumentPrediction) )
	{
		std::cout << "/l と /p が同時に指定されています" << std::endl ;
		return	false ;
	}
	else if ( m_strModelFile.empty() && !(m_flagsArg & argumentHelp) )
	{
		std::cout << "モデルファイル名が指定されていません" << std::endl ;
		m_flagsArg |= argumentHelp ;
	}
	m_shell.SetShellConfig( m_cfgShell ) ;
	m_shell.SetMLPConfig( m_cfgBuf ) ;
	return	true ;
}

bool PalesibylBasicApp::IsValidNextArgument( int& iArg, int nArgs, char * pszArgs[] )
{
	if ( (iArg + 1 >= nArgs)
		|| (pszArgs[iArg + 1][0] == '/') )
	{
		std::cout << pszArgs[iArg] << " 引数のパラメータが指定されていません" << std::endl ;
		return	false ;
	}
	return	true ;
}

int PalesibylBasicApp::NextArgumentInt
	( bool& validResult, int& iArg, int nArgs, char * pszArgs[] )
{
	if ( !IsValidNextArgument( iArg, nArgs, pszArgs ) )
	{
		validResult = false ;
		return	0 ;
	}
	validResult = true ;
	return	atoi( pszArgs[++ iArg] ) ;
}

double PalesibylBasicApp::NextArgumentFloat
	( bool& validResult, int& iArg, int nArgs, char * pszArgs[] )
{
	if ( !IsValidNextArgument( iArg, nArgs, pszArgs ) )
	{
		validResult = false ;
		return	0 ;
	}
	validResult = true ;
	return	atof( pszArgs[++ iArg] ) ;
}

// 実行
//////////////////////////////////////////////////////////////////////////////
int PalesibylBasicApp::Run( void )
{
	if ( m_flagsArg & argumentHelp )
	{
		return	RunHelp() ;
	}
	if ( m_flagsArg & argumentLearn )
	{
		return	RunLearning() ;
	}
	if ( m_flagsArg & argumentPrediction )
	{
		return	RunPrediction() ;
	}
	return	0 ;
}

// ヘルプ表示
//////////////////////////////////////////////////////////////////////////////
int PalesibylBasicApp::RunHelp( void )
{
	std::cout << "\rusage: [options]...\r" << std::endl ;
	std::cout << "option;\r" << std::endl ;
	std::cout << "/l <mode-file>    : 学習を実行します\r" << std::endl ;
	std::cout << "/p <mode-file>    : 予測を実行します\r" << std::endl ;
	std::cout << "/cuda             : CUDA を利用します\r" << std::endl ;
	std::cout << "/loop <count>     : 学習ループ回数を指定します\r" << std::endl ;
	std::cout << "/subloop <count>  : ミニバッチの反復回数を指定します\r" << std::endl ;
	std::cout << "/batch <count>    : ミニバッチサイズを指定します\r" << std::endl ;
	std::cout << "/delta <rate>[,<end-rate>]\r" << std::endl ;
	std::cout << "                  : 学習係数を指定します\r" << std::endl ;
	std::cout << "/thread <count>   : 最大のスレッド数を指定します\r" << std::endl ;
	std::cout << "/batch_thread <count>\r" << std::endl ;
	std::cout << "                  : ミニバッチの並列スレッド数を指定します\r" << std::endl ;
	std::cout << "/log <csv-file>   : 学習ログファイルを出力します\r" << std::endl ;
	std::cout << "/lgrd             : 学習ログにレイヤー毎の勾配ノルムを出力します\r" << std::endl ;
	std::cout << "/tio <image-file> : 訓練画像の予測をミニバッチ毎に出力します\r" << std::endl ;
	std::cout << "/vio <image-file> : 検証用画像の予測を逐次出力します\r" << std::endl ;
	std::cout << "/asm <model-file> : 学習時に1時間毎にモデルを自動保存します\r" << std::endl ;
	std::cout << "/ndo              : ドロップアウトは行わない\r" << std::endl ;
	std::cout << "/nlfb             : ミニバッチ毎に進捗表示を改行しない\r" << std::endl ;
	std::cout << "/pbs              : 学習中間バッファのサイズを表示します\r" << std::endl ;
	std::cout << "/cubs             : 学習中間 CUDA バッファのサイズを表示します\r" << std::endl ;
	std::cout << std::endl ;
	std::cout << "※学習／予測処理は ESC キーで中断できます\r" << std::endl ;
	std::cout << std::endl ;
	return	0 ;
}

// 学習実行
//////////////////////////////////////////////////////////////////////////////
int PalesibylBasicApp::RunLearning( void )
{
	std::shared_ptr<NNMLPShell::Iterator>	pIter = MakeLearningIter() ;
	pIter->InitializeIterator() ;

	// モデル
	if ( !m_shell.LoadModel( m_strModelFile.c_str() ) )
	{
		m_shell.MLP().ClearAll() ;
		BuildModel( pIter.get() ) ;
		m_shell.SaveModel( m_strModelFile.c_str() ) ;
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

	// 訓練
	m_shell.DoLearning( *pIter, m_param ) ;

	// モデル保存
	if ( !m_shell.SaveModel( m_strModelFile.c_str() ) )
	{
		std::cout << m_strModelFile << " への書き出しに失敗しました" << std::endl ;
	}
	return	0 ;
}

// 予測実行
//////////////////////////////////////////////////////////////////////////////
int PalesibylBasicApp::RunPrediction( void )
{
	if ( !m_shell.LoadModel( m_strModelFile.c_str() ) )
	{
		std::cout << m_strModelFile << " の読み込みに失敗しました" << std::endl ;
		return	1 ;
	}
	BeforePrediction() ;

	std::shared_ptr<NNMLPShell::Iterator>	pIter = MakePredictiveIter() ;
	pIter->InitializeIterator() ;

	m_shell.DoPrediction( *pIter ) ;

	return	0 ;
}




