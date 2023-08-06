
#include "simple_vae.h"
#include <cstring>

#define	__SIMPLE_AUDOENCODER__	0

constexpr static const char	idEncoderOutLayer[]	= "encoder_out" ;


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
	: m_flagInterpolate( false ),
		m_flagMeanVariance( false ),
		m_flagEncodedCsvFile( false )
{
	m_aggrEncoded.sum = 0.0f ;
	m_aggrEncoded.sum2 = 0.0f ;
	m_aggrEncoded.num = 0.0f ;
}

// アプリ初期化
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::Initialize( void )
{
	m_shell.AttachProgressListener( this ) ;
}

// 引数解釈
//////////////////////////////////////////////////////////////////////////////
bool PalesibylApp::ParseArgumentAt( int& iArg, int nArgs, char * pszArgs[] )
{
	enum	ParameterType
	{
		paramNull,
		paramEncodedVariance,
		paramEncodedCsvFile,
		paramInterpolate,
	} ;
	static const struct
	{
		const char *	pszOpt ;
		ParameterType	paramType ;
	}	s_ParamOpt[] =
	{
		{ "/encvar", paramEncodedVariance },
		{ "/enccsv", paramEncodedCsvFile },
		{ "/inter", paramInterpolate },
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
	case	paramEncodedVariance:
		m_flagMeanVariance = true ;
		return	true ;

	case	paramEncodedCsvFile:
		flagSuccess = IsValidNextArgument( iArg, nArgs, pszArgs ) ;
		if ( flagSuccess )
		{
			m_flagEncodedCsvFile = true ;
			m_strEncodedCsvFile = pszArgs[++ iArg] ;
		}
		return	true ;

	case	paramInterpolate:
		if ( IsValidNextArgument( iArg, nArgs, pszArgs ) )
		{
			m_strInterpolateSrc1 = pszArgs[++ iArg] ;
			if ( IsValidNextArgument( iArg, nArgs, pszArgs ) )
			{
				m_strInterpolateSrc2 = pszArgs[++ iArg] ;
				m_flagInterpolate = true ;
				return	true ;
			}
		}
		return	false ;

	default:
		break ;
	}
	return	PalesibylBasicApp::ParseArgumentAt( iArg, nArgs, pszArgs ) ;
}

// 実行
//////////////////////////////////////////////////////////////////////////////
int PalesibylApp::Run( void )
{
	if ( (m_flagsArg & argumentPrediction) && m_flagInterpolate )
	{
		return	RunInterpolation() ;
	}
	return	PalesibylBasicApp::Run() ;
}

// アプリ固有の説明（ファイルの配置など）
//////////////////////////////////////////////////////////////////////////////
const char *	PalesibylApp::s_pszSpecificDescription =
	"/encvar            : エンコード値の平均と分散を計算して表示します\r\n"
	"/enccsv <csv-file> : エンコード値を CSV ファイルへ出力するします\r\n"
	"/inter <image1> <image2>\r\n"
	"                   : image1 と image2 画像の中間画像を生成します\r\n"
	"\r\n"
	"ディレクトリ構成;\r\n"
	"[learn/]\r\n"
	"  + [source/]  : 学習元画像ファイル\r\n"
	"[predict/]\r\n"
	"  + [src/]     : 予測入力画像ファイル\r\n"
	"  + [out/]     : 予測出力先\r\n" ;

int PalesibylApp::RunHelp( void )
{
	PalesibylBasicApp::RunHelp() ;
	std::cout << s_pszSpecificDescription ;
	return	0 ;
}

// 予測実行
//////////////////////////////////////////////////////////////////////////////
int PalesibylApp::RunPrediction( void )
{
	if ( m_flagEncodedCsvFile )
	{
		try
		{
			m_ofsEncodedCsvFile = std::make_unique<std::ofstream>
										( m_strEncodedCsvFile, std::ios::out ) ;
		}
		catch ( const std::exception& e )
		{
			TRACE( "exception: %s\r\n", e.what() ) ;
			std::cout << m_strEncodedCsvFile << " を開けませんでした\r" << std::endl ;
			return	1 ;
		}
	}

	int	nExitCode = PalesibylBasicApp::RunPrediction() ;

	if ( m_flagMeanVariance )
	{
		float	mean = m_aggrEncoded.sum / m_aggrEncoded.num ;
		float	variance = m_aggrEncoded.sum2 / m_aggrEncoded.num - mean * mean ;
		m_shell.Print( "\r\nmean = %f, variance = %f\r\n", mean, variance ) ;
	}
	return	nExitCode ;
}

// 補間予測実行
//////////////////////////////////////////////////////////////////////////////
int PalesibylApp::RunInterpolation( void )
{
	// モデル読み込み
	if ( !m_shell.LoadModel( m_strModelFile.c_str() ) )
	{
		std::cout << m_strModelFile << " の読み込みに失敗しました\r" << std::endl ;
		return	1 ;
	}
	BeforePrediction() ;

	// 画像読み込み
	std::shared_ptr<NNBuffer>	pSrc1 =
		NNImageCodec::LoadFromFile( std::filesystem::path(m_strInterpolateSrc1), 1 ) ;
	if ( pSrc1 == nullptr )
	{
		std::cout << m_strInterpolateSrc1 << " の読み込みに失敗しました\r" << std::endl ;
		return	1 ;
	}
	std::shared_ptr<NNBuffer>	pSrc2 =
		NNImageCodec::LoadFromFile( std::filesystem::path(m_strInterpolateSrc2), 1 ) ;
	if ( pSrc2 == nullptr )
	{
		std::cout << m_strInterpolateSrc2 << " の読み込みに失敗しました\r" << std::endl ;
		return	1 ;
	}
	if ( pSrc1->GetSize() != pSrc2->GetSize() )
	{
		std::cout << "画像サイズが一致しません\r" << std::endl ;
		return	1 ;
	}

	// 処理準備
	NNMultiLayerPerceptron::BufferArrays	bufArrays ;
	const uint32_t	flagsBuf = 0 ;

	NNMultiLayerPerceptron&	mlp = m_shell.MLP() ;
	mlp.PrepareBuffer( bufArrays, pSrc1->GetSize(), flagsBuf, m_cfgBuf ) ;

	// エンコード
	NNBuffer	bufEncoded1, bufEncoded2 ;
	EncodeForInterpolation( bufEncoded1, *pSrc1, bufArrays ) ;
	EncodeForInterpolation( bufEncoded2, *pSrc2, bufArrays ) ;

	const NNBufDim	dimEncoded = bufEncoded1.GetSize() ;
	assert( bufEncoded2.GetSize() == dimEncoded ) ;
	NNBuffer	bufInter ;
	bufInter.Create
		( dimEncoded, (bufArrays.stream.m_useCuda
							? NNBuffer::cudaAllocate : 0) ) ;

	// 補間した値をデコード
	int	nError = 0 ;
	for ( int i = 0; i <= 10; i ++ )
	{
		const float		t = (float) i / 10.0f ;
		const float *	pEncoded1 = bufEncoded1.GetConstBuffer() ;
		const float *	pEncoded2 = bufEncoded2.GetConstBuffer() ;
		float *			pInter = bufInter.GetBuffer() ;
		const size_t	nCount = dimEncoded.n * dimEncoded.z ;
		for ( size_t j = 0; j < nCount; j ++ )
		{
			pInter[j] = pEncoded1[j] * (1.0f - t) + pEncoded2[j] * t ;
		}
		if ( bufArrays.stream.m_useCuda )
		{
			bufInter.CudaAsyncToDevice( bufArrays.stream.m_cudaStream ) ;
		}

		NNBuffer *	pDecodedBuf =
						DecodeForInterpolation( bufInter, bufArrays ) ;
		assert( pDecodedBuf != nullptr ) ;
		if ( pDecodedBuf != nullptr )
		{
			char	szBuf[0x100] ;
			#if	defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
				sprintf_s( szBuf, "predict/out/interpolated%d.bmp", i ) ;
			#else
				sprintf( szBuf, "predict/out/interpolated%d.bmp", i ) ;
			#endif
			std::cout << szBuf << "\r" << std::endl ;

			if ( !NNImageCodec::SaveToFile
				( std::filesystem::path(szBuf), *pDecodedBuf ) )
			{
				std::cout << szBuf << " への書き出しに失敗しました\r" << std::endl ;
				nError ++ ;
			}
		}
	}
	return	nError ;
}

void PalesibylApp::EncodeForInterpolation
	( NNBuffer& bufEncoded, NNBuffer& bufSource,
		NNMultiLayerPerceptron::BufferArrays& bufArrays )
{
	NNMultiLayerPerceptron&	mlp = m_shell.MLP() ;
	const int	iEncOutLayer = mlp.FindLayerAs( idEncoderOutLayer ) ;
	assert( iEncOutLayer >= 0 ) ;
	bufArrays.iFirstLayer = 0 ;
	bufArrays.iEndLayer = (size_t) iEncOutLayer + 1 ;

	NNBuffer *	pOutput = mlp.Prediction( bufArrays, bufSource, false, false ) ;
	assert( pOutput != nullptr ) ;
	if ( pOutput != nullptr )
	{
		const NNBufDim	dimOutput = pOutput->GetSize() ;
		bufEncoded.Create( dimOutput ) ;
		bufEncoded.CopyFrom( *pOutput ) ;
	}
}

NNBuffer * PalesibylApp::DecodeForInterpolation
	( NNBuffer& bufInter,
		NNMultiLayerPerceptron::BufferArrays& bufArrays )
{
	NNMultiLayerPerceptron&	mlp = m_shell.MLP() ;
	const int	iEncOutLayer = mlp.FindLayerAs( idEncoderOutLayer ) ;
	assert( iEncOutLayer >= 0 ) ;
	bufArrays.iFirstLayer = (size_t) iEncOutLayer + 1 ;
	bufArrays.iEndLayer = mlp.GetLayerCount() ;

	return	mlp.Prediction( bufArrays, bufInter, false, false ) ;
}

// 予測実行完了
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::OnProcessedPrediction
	( NNMLPShell& shell,
		const char * pszSourcePath,
		NNBuffer * pSource, NNBuffer * pOutput,
		const NNMultiLayerPerceptron::BufferArrays& bufArrays )
{
	if ( !m_flagMeanVariance && !m_flagEncodedCsvFile )
	{
		return ;
	}
	NNMultiLayerPerceptron&	mlp = m_shell.MLP() ;
	const int	iEncOutLayer = mlp.FindLayerAs( idEncoderOutLayer ) ;
	assert( iEncOutLayer >= 0 ) ;
	if ( iEncOutLayer < 0 )
	{
		return ;
	}

	NNBuffer&	bufEncOutput = bufArrays.buffers.at(iEncOutLayer)->bufOutput ;
	if ( bufArrays.stream.m_useCuda )
	{
		bufEncOutput.CommitCudaWithHost() ;
		bufEncOutput.CudaAsyncFromDevice( bufArrays.stream.m_cudaStream ) ;
		bufArrays.stream.m_cudaStream.Synchronize() ;
	}
	if ( m_flagMeanVariance )
	{
		OutputAggregation( m_aggrEncoded, bufEncOutput ) ;
	}
	if ( m_flagEncodedCsvFile )
	{
		const NNBufDim	dimOutput = bufEncOutput.GetSize() ;
		if ( dimOutput.n >= 1 )
		{
			const float *	pEncOutput = bufEncOutput.GetConstBufferAt(0,0) ;
			*m_ofsEncodedCsvFile << pszSourcePath ;
			for ( size_t i = 0; i < dimOutput.z; i ++ )
			{
				*m_ofsEncodedCsvFile << "," << pEncOutput[i] ;
			}
			*m_ofsEncodedCsvFile << std::endl ;
		}
	}
}

// モデルを作成
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::BuildModel( NNMLPShell::Iterator * pIter )
{
	NNMultiLayerPerceptron&	mlp = m_shell.MLP() ;

	mlp.SetInputShape( 0, NNBufDim( 16, 16, 1 ), NNBufDim( 0, 0, 1 )  ) ;

	const size_t	nLatentChannels = 2 ;		// 潜在変数次元
	const int		bias = 1 ;

	// エンコーダー
	mlp.AppendConvLayer( 16, 1, 3, 3, convPadBorder, bias, activReLU, 2, 2 ) ;	// ->8x8
	mlp.AppendConvLayer( 32, 16, 3, 3, convPadZero, bias, activReLU, 2, 2 ) ;	// ->4x4
	mlp.AppendConvLayer( 64, 32, 3, 3, convPadZero, bias, activReLU, 2, 2 ) ;	// ->2x2
	mlp.AppendConvLayer( 128, 64, 2, 2, convNoPad, bias, activReLU, 2, 2 ) ;	// ->1x1

#if	__SIMPLE_AUDOENCODER__
	mlp.AppendLayer( nLatentChannels, 128, bias, activLinear )
		->SetIdentity( idEncoderOutLayer ) ;
#else
	NNPerceptronPtr	pLayerMean =
		mlp.AppendLayer( nLatentChannels, 128, bias, activLinear ) ;
	NNPerceptronPtr	pLayerLnVar =
		mlp.AppendLayer( nLatentChannels, 128, bias, activLinear ) ;
	pLayerLnVar->AddConnection( 2, 0, 0, 128 ) ;

	mlp.AppendGaussianLayer( nLatentChannels, pLayerMean, pLayerLnVar )
		->SetIdentity( idEncoderOutLayer ) ;
#endif

	// デコーダー
	mlp.AppendUp2x2Layer( 128, nLatentChannels, bias, activReLU ) ;	// ->2x2
	mlp.AppendUp2x2Layer( 64, 128, bias, activReLU ) ;				// ->4x4
	mlp.AppendUp2x2Layer( 32, 64, bias, activReLU ) ;				// ->8x8
	mlp.AppendUp2x2Layer( 16, 32, bias, activReLU ) ;				// ->16x16
	mlp.AppendConvLayer( 1, 16, 3, 3, convPadZero, bias, activSigmoid ) ;

	// 損失関数
	mlp.SetLossFunction( std::make_shared<NNLossBernoulliNLL>() ) ;

#if	!__SIMPLE_AUDOENCODER__
	mlp.AddLossGaussianKLDivergence
		( pLayerMean, pLayerLnVar, 1.0f/(16.0f*16.0f), 1.0f ) ;
#endif

	// 評価値設定
	mlp.SetEvaluationFunction( std::make_shared<NNEvaluationMSE>() ) ;
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
					( "learn/source", "learn/source", false, 1, true, 0.05 ) ;
}

// 予測用イテレーター作成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNMLPShell::Iterator> PalesibylApp::MakePredictiveIter( void )
{
	return	std::make_shared<NNMLPShellImageIterator>
					( "predict/src", "predict/out", true, 1 ) ;
}

// 出力値の集計
//////////////////////////////////////////////////////////////////////////////
void PalesibylApp::OutputAggregation
	( NNNormalizationFilter::Aggregation& aggr, const NNBuffer& bufOutput )
{
	const NNBufDim	dimOutput = bufOutput.GetSize() ;
	const size_t	nCount = dimOutput.n * dimOutput.z ;
	const float *	pOutput = bufOutput.GetConstBuffer() ;
	for ( size_t i = 0; i < nCount; i ++ )
	{
		const float	x = pOutput[i] ;
		aggr.sum += x ;
		aggr.sum2 += x * x ;
	}
	aggr.num += (float) nCount ;
}

