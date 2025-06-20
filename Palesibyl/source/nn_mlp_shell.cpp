
#include "nn_mlp_shell.h"
#include "nn_shell_image_file.h"

#include <random>

#include <stdarg.h>
#include <stdio.h>

#if	defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
#include <conio.h>

#else
#include <unistd.h>
#include <termios.h>

inline bool _kbhit(void)
{
	fd_set	fds ;
	timeval	tv ;

	FD_ZERO( &fds ) ;
	FD_SET( 0, &fds ) ;

	tv.tv_sec = 0 ;
	tv.tv_usec = 0 ;

	int	r = select( 1, &fds, nullptr, nullptr, &tv ) ;
	return	(r != -1) && FD_ISSET( 0, &fds ) ;
}

inline int _getch(void)
{
	return	getchar() ;
}

#endif

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// 処理ラッパ基底
//////////////////////////////////////////////////////////////////////////////

#if	defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
static DWORD	s_dwConMode ;
#else
static termios	s_termiosCooked ;
#endif

// 関連初期化処理
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::StaticInitialize( void )
{
#if	defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
	HANDLE	hStd = ::GetStdHandle( STD_OUTPUT_HANDLE ) ;
	DWORD	dwMode = 0 ;
	::GetConsoleMode( hStd, &dwMode ) ;
	s_dwConMode = dwMode ;
	dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING ;
	::SetConsoleMode( hStd, dwMode ) ;
#else
	tcgetattr( STDIN_FILENO, &s_termiosCooked ) ;

	termios termiosRaw ;
	termiosRaw = s_termiosCooked ;
	cfmakeraw( &termiosRaw ) ;
	tcsetattr( STDIN_FILENO, 0, &termiosRaw ) ;
#endif

	NNNormalizationFilter::InitMake() ;
	NNSamplingFilter::InitMake() ;
	NNLossFunction::InitMakeLoss() ;
	NNActivationFunction::InitMake() ;
	NNEvaluationFunction::InitMake() ;
	NNGeneratorFunction::InitMake() ;
	NNPerceptron::InitMake() ;
	NNImageCodec::InitializeLib() ;
}

// 関連初期化処理
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::StaticRelase( void )
{
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
	HANDLE	hStd = ::GetStdHandle( STD_OUTPUT_HANDLE ) ;
	::SetConsoleMode( hStd, s_dwConMode ) ;
#else
	tcsetattr( STDIN_FILENO, 0, &s_termiosCooked ) ;
#endif

	NNImageCodec::ReleaseLib() ;
}


// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShell::NNMLPShell( void )
{
}

// モデル読み込み
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShell::LoadModel( const char * pszFilePath )
{
	try
	{
		std::ifstream	ifs( pszFilePath, std::ios_base::in | std::ios_base::binary ) ;
		if ( !ifs )
		{
			return	false ;
		}
		NNDeserializer	dsr( ifs ) ;
		return	m_mlp.Deserialize( dsr ) ;
	}
	catch ( const std::exception& e )
	{
		TRACE( "exception at LoadModel: %s\r\n", e.what() ) ;
		return	false ;
	}
}

// モデル書き出し
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShell::SaveModel( const char * pszFilePath )
{
	try
	{
		std::ofstream	ofs( pszFilePath, std::ios_base::out | std::ios_base::binary ) ;
		if ( !ofs )
		{
			return	false ;
		}
		NNSerializer	ser( ofs ) ;
		m_mlp.Serialize( ser ) ;
	}
	catch ( const std::exception& e )
	{
		TRACE( "exception at SaveModel: %s\r\n", e.what() ) ;
		return	false ;
	}
	return	true ;
}

// モデル取得
//////////////////////////////////////////////////////////////////////////////
NNMultiLayerPerceptron& NNMLPShell::MLP( void )
{
	return	m_mlp ;
}

const NNMultiLayerPerceptron& NNMLPShell::GetMLP( void ) const
{
	return	m_mlp ;
}

// コンフィグ
//////////////////////////////////////////////////////////////////////////////
const NNMLPShell::ShellConfig& NNMLPShell::GetShellConfig( void ) const
{
	return	m_config ;
}

void NNMLPShell::SetShellConfig( const NNMLPShell::ShellConfig& cfg )
{
	m_config = cfg ;
}

const NNMultiLayerPerceptron::BufferConfig& NNMLPShell::GetMLPConfig( void ) const
{
	return	m_bufConfig ;
}

void NNMLPShell::SetMLPConfig( const NNMultiLayerPerceptron::BufferConfig& cfg )
{
	m_bufConfig = cfg ;
}

// 学習
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::DoLearning
	( NNMLPShell::Iterator& iter, const LearningParameter& param )
{
	CudaErrorHandler	cudaError
		( [this](cudaError_t error,const char* pszError)
			{ Print( "\r\nCUDA error: %08X : %s\r\n", error, pszError ) ; } ) ;

	// 学習用バッファ配列を準備
	uint32_t	flagsBuf = NNMultiLayerPerceptron::bufferForLearning ;
	if ( m_config.flagsBehavior & behaviorNoDropout )
	{
		flagsBuf |= NNMultiLayerPerceptron::bufferNoDropout ;
	}
	LearningContext	context ;
	PrepareLearning( context, param, false ) ;

	// ループ情報
	LearningProgressInfo	lpi ;
	lpi.nEpochCount = param.nEpochCount ;
	lpi.nCountInBatch = 0 ;

	// 学習ループ
	for ( lpi.iLoopEpoch = 0; lpi.iLoopEpoch < param.nEpochCount; lpi.iLoopEpoch ++ )
	{
		DoLearningEpoch( context, iter, lpi, param ) ;

		if ( context.flagCanceled )
		{
			Print( "\r\ncanceled.\r\n" ) ;
			return ;
		}
	}
}

// 学習（GAN）
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::DoLearningGAN
	( NNMLPShell::Iterator& iterGAN,
		NNMLPShell& mlpClassifier,
		NNMLPShell::Iterator& iterClassifier,
		size_t nGANLoop, const LearningParameter& param )
{
	CudaErrorHandler	cudaError
		( [this](cudaError_t error,const char* pszError)
			{ Print( "\r\nCUDA error: %08X : %s\r\n", error, pszError ) ; } ) ;

	// 学習用バッファ配列を準備
	LearningContext	lcClassifier ;
	LearningContext	lcGAN ;
	assert( mlpClassifier.m_config.nBatchThreads == m_config.nBatchThreads ) ;
	assert( mlpClassifier.m_bufConfig.nThreadCount == m_bufConfig.nThreadCount ) ;
	uint32_t	flagsBuf = NNMultiLayerPerceptron::bufferForLearning ;
	if ( m_config.flagsBehavior & behaviorNoDropout )
	{
		flagsBuf |= NNMultiLayerPerceptron::bufferNoDropout ;
	}
	mlpClassifier.PrepareLearning
		( lcClassifier, param,
			(flagsBuf | NNMultiLayerPerceptron::bufferPropagateDelta) ) ;
	PrepareLearning( lcGAN, param, flagsBuf ) ;

	// ループ情報
	LearningProgressInfo	lpiClassifier ;
	LearningProgressInfo	lpiGAN ;
	lpiClassifier.flags = learningGANClassifier ;
	lpiClassifier.nGANCount = nGANLoop ;
	lpiClassifier.nEpochCount = param.nEpochCount ;
	lpiClassifier.nCountInBatch = 0 ;
	lpiGAN.flags = learningGAN ;
	lpiGAN.nGANCount = nGANLoop ;
	lpiGAN.nEpochCount = param.nEpochCount ;
	lpiGAN.nCountInBatch = 0 ;

	// 分類器学習用反復器
	NNMLPShellGANClassifierIterator
		iterGANClassifier( iterClassifier, m_mlp, m_bufConfig, lcGAN, iterGAN ) ;

	for ( lpiGAN.iGANLoop = 0; lpiGAN.iGANLoop < nGANLoop; lpiGAN.iGANLoop ++ )
	{
		// 分類器の訓練
		Print( "\r\n%d-th training of classifier.\r\n", lpiGAN.iGANLoop + 1 ) ;
		lpiClassifier.iGANLoop = lpiGAN.iGANLoop ;

		for ( size_t iLoopEpoch = 0; iLoopEpoch < param.nEpochCount; iLoopEpoch ++ )
		{
			lpiClassifier.iLoopEpoch = iLoopEpoch ;
			mlpClassifier.DoLearningEpoch
				( lcClassifier, iterGANClassifier,
					lpiClassifier, param ) ;

			if ( lcClassifier.flagCanceled )
			{
				Print( "\r\ncanceled.\r\n" ) ;
				return ;
			}
		}
		mlpClassifier.PreapreForMiniBatch( lcClassifier, false ) ;

		// 生成器の訓練
		Print( "\r\n%d-th training of generator.\r\n", lpiGAN.iGANLoop + 1 ) ;
		int	iTraining = 0 ;
		do
		{
			for ( size_t iLoopEpoch = 0; iLoopEpoch < param.nEpochCount; iLoopEpoch ++ )
			{
				lpiGAN.iLoopEpoch = iLoopEpoch ;
				DoLearningEpoch
					( lcGAN, iterGAN, lpiGAN, param,
						&mlpClassifier.m_mlp, &lcClassifier ) ;

				if ( lcGAN.flagCanceled )
				{
					Print( "\r\ncanceled.\r\n" ) ;
					return ;
				}
			}
		}
		while ( (iTraining ++ < 10)
			&& (lpiClassifier.lossLearn < lpiGAN.lossLearn) ) ;
	}
}

// 予測変換
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::DoPrediction( NNMLPShell::Iterator& iter )
{
	CudaErrorHandler	cudaError
		( [this](cudaError_t error,const char* pszError)
			{ Print( "\r\nCUDA error: %08X : %s\r\n", error, pszError ) ; } ) ;

	bool	flagCanceled = false ;
	while ( !flagCanceled )
	{
		std::shared_ptr<NNBuffer>	pSource = iter.NextSource() ;
		if ( pSource == nullptr )
		{
			break ;
		}
		Print( "\r%s -> \x1b[s", iter.GetSourcePath().c_str() ) ;

		NNMultiLayerPerceptron::BufferArrays	bufArrays ;
		const uint32_t	flagsBuf = 0 ;
		const NNBufDim	dimSource = pSource->GetSize() ;

		std::shared_ptr<NNStreamBuffer>	pStreamOut ;
		NNBuffer *						pOutput = nullptr ;

		if ( m_mlp.GetMLPFlags() & NNMultiLayerPerceptron::mlpFlagStream )
		{
			Streamer *	pStreamer = dynamic_cast<Streamer*>( &iter ) ;
			m_nLastProgress = -1 ;

			// ストリーミング
			const NNBufDim	dimOutput = m_mlp.CalcOutputSize( dimSource ) ;
			const NNBufDim	dimInUnit = m_mlp.GetInputUnit() ;
			const NNBufDim	dimOutUnit = m_mlp.CalcOutputSize( dimInUnit ) ;

			// バッファ準備
			NNStreamBuffer	bufStream ;
			const NNBufDim	dimInShape = m_mlp.GetInputShape() ;
			bufStream.Create( dimInShape ) ;

			m_mlp.PrepareBuffer( bufArrays, dimInShape, flagsBuf, m_bufConfig ) ;

			NNMultiLayerPerceptron::VerifyError
					verfError = VerifyDataShape( bufArrays, dimInShape ) ;
			bool	flagLowMemory = false ;
			if ( verfError != NNMultiLayerPerceptron::verifyNormal )
			{
				continue ;
			}

			pStreamOut = std::make_shared<NNStreamBuffer>() ;
			pStreamOut->Create( dimOutput ) ;

			// 始めの入力処理
			const size_t	xLead = __min( dimInShape.x, dimSource.x ) ;
			bufStream.Stream( *pSource, 0, xLead ) ;
			//
			size_t	nPreCount = m_mlp.CountOfPrePrediction() ;
			for ( size_t i = 0; i < nPreCount; i ++ )
			{
				pOutput = m_mlp.Prediction( bufArrays, bufStream, false ) ;
			}
			if ( pOutput != nullptr )
			{
				// 始めの出力
				const size_t	xCurrent = pStreamOut->GetCurrent() ;
				pStreamOut->Stream( *pOutput, 0, pOutput->GetSize().x ) ;

				OnStreamingProgress
					( pStreamer, xLead, dimSource.x, pStreamOut.get(), xCurrent ) ;
			}

			// 順次入力・予測処理
			for ( size_t xNextSrc = xLead;
					xNextSrc < dimSource.x; xNextSrc += dimInUnit.x )
			{
				// バッファストリーミング
				const size_t	xNextInput = __min( dimSource.x - xNextSrc, dimInUnit.x ) ;
				const size_t	xShift = bufStream.Stream( *pSource, xNextSrc, xNextInput ) ;
				m_mlp.ShiftBufferWithStreaming( bufArrays, xShift ) ;

				// 予測
				bufArrays.xBoundary = dimInShape.x - xShift ;
				pOutput = m_mlp.Prediction( bufArrays, bufStream, false ) ;

				if ( pOutput != nullptr )
				{
					// ストリーミング出力
					const size_t	xCurrent = pStreamOut->GetCurrent() ;
					pStreamOut->Stream
						( *pOutput, pOutput->GetSize().x - dimOutUnit.x, dimOutUnit.x ) ;

					OnStreamingProgress
						( pStreamer,
							xNextSrc + xNextInput, dimSource.x,
							pStreamOut.get(), xCurrent ) ;
				}
				if ( IsCancel() )
				{
					flagCanceled = true ;
					break ;
				}
			}
			pStreamOut->Trim() ;
			pOutput = pStreamOut.get() ;
		}
		else
		{
			// 通常の予測
			// バッファ準備
			m_mlp.PrepareBuffer( bufArrays, dimSource, flagsBuf, m_bufConfig ) ;

			NNMultiLayerPerceptron::VerifyError
					verfError = VerifyDataShape( bufArrays, dimSource ) ;
			bool	flagLowMemory = false ;
			if ( verfError != NNMultiLayerPerceptron::verifyNormal )
			{
				if ( verfError != NNMultiLayerPerceptron::lowCudaMemory )
				{
					continue ;
				}
				flagLowMemory = true ;
			}

			// 予測
			pOutput = m_mlp.Prediction( bufArrays, *pSource, false, flagLowMemory ) ;

			OnProcessedPrediction
				( iter.GetSourcePath().c_str(), pSource.get(), pOutput, bufArrays ) ;
		}

		if ( pOutput != nullptr )
		{
			Print( "%s", iter.GetOutputPath().c_str() ) ;
			if ( iter.OutputPrediction( *pOutput ) )
			{
				Print( " saved.\r\n" );
			}
			else
			{
				Print( " failed to save.\r\n" );
			}
		}
		else
		{
			Print( "failed\r\n" );
		}
		if ( flagCanceled || IsCancel() )
		{
			Print( "canceled.\r\n" ) ;
			break ;
		}
	}
}

// 学習準備
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::PrepareLearning
		( NNMLPShell::LearningContext& context,
			const NNMLPShell::LearningParameter& param, uint32_t flagsBuf )
{
	context.nBufCount = (m_config.nBatchThreads >= 1) ? m_config.nBatchThreads : 1 ;

	context.bufArraysArray.clear() ;
	context.dimSourceArray.resize( context.nBufCount ) ;
	for ( size_t i = 0; i < context.nBufCount; i ++ )
	{
		context.bufArraysArray.push_back
			( std::make_shared<NNMultiLayerPerceptron::BufferArrays>() ) ;
		context.dimSourceArray.at(i) = NNBufDim( 0, 0, 0 ) ;
	}
	m_mlp.PrepareLossAndGradientArrays( context.lagArrays ) ;

	context.vEvalArray.resize( context.nBufCount ) ;
	context.vEvalSummed.resize( context.nBufCount ) ;

	if ( context.nBufCount >= 2 )
	{
		context.ploop.BeginThreads( context.nBufCount ) ;
	}
	context.flagsBuf = flagsBuf | NNMultiLayerPerceptron::bufferForLearning ;

	context.flagsLearning = param.flagsLearning ;
	context.lossLearning = 0.0 ;
	context.lossCurLoop = 0.0 ;
	context.lossLastLoop = 0.0 ;
	context.lossSummed = 0.0 ;
	context.lossValidation = 0.0 ;
	context.deltaRate = param.deltaRate0 ;
	context.deltaCurRate = param.deltaRate0 ;
	context.flagEndOfIter = false ;
	context.flagCanceled = false ;
}

// １エポック学習
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::DoLearningEpoch
	( NNMLPShell::LearningContext& context,
		NNMLPShell::Iterator& iter,
		NNMLPShell::LearningProgressInfo& lpi,
		const LearningParameter& param,
		NNMultiLayerPerceptron * pForwardMLP,
		NNMLPShell::LearningContext * pForwardContext )
{
	const float	tEpoch = (float) lpi.iLoopEpoch / (float) lpi.nEpochCount ;
	context.deltaRate = exp( log(param.deltaRate0) * (1.0f - tEpoch)
								+ log(param.deltaRate1) * tEpoch ) ;
	context.deltaCurRate = context.deltaRate ;
	lpi.nSubLoopCount = param.nSubLoopCount ;

	OnBeginEpoch( context, lpi, iter ) ;

	lpi.iInBatch = 0 ;		// バッチごとの回数
	for ( ; ; )
	{
		// バッチ収集
		LoadTrainingData( context, iter, param.nMiniBatchCount ) ;

		lpi.nMiniBatchCount = context.sources.size() ;
		if ( context.sources.size() == 0 )
		{
			break ;
		}
		Print( "\x1b[s" ) ;

		// バッファ準備
		PrepareBuffer( context ) ;

		if ( pForwardMLP != nullptr )
		{
			assert( pForwardContext != nullptr ) ;
			PrepareForwardBuffer
				( *pForwardMLP, *pForwardContext, context ) ;
		}

		// 繰り返し
		OnBeginMiniBatch( context, true ) ;
		OnLearningProgress( learningStartMiniBatch, lpi ) ;

		for ( lpi.iSubLoop = 0; lpi.iSubLoop < param.nSubLoopCount; lpi.iSubLoop ++ )
		{
			// 学習
			LearnOnce( context, lpi, pForwardMLP, pForwardContext ) ;
			//
			if ( context.flagCanceled )
			{
				return ;
			}

			// 損失値と勾配を合計し、平均損失計算
			lpi.lossLearn = IntegrateLossAndGradient( context ) ;
			OnLearningProgress( learningEndMiniBatch, lpi ) ;

			// 勾配反映
			GradientReflection( context, lpi ) ;
		}

		// 次のバッチへ
		OnEndMiniBatch( context ) ;

		lpi.iInBatch ++ ;
		lpi.iSubLoop = param.nSubLoopCount ;
		lpi.lossLearn = context.lossLearning ;
		lpi.evalLearn = context.evalLearning ;
		OnLearningProgress( learningEndSubLoop, lpi ) ;

		if ( context.flagEndOfIter )
		{
			break ;
		}
	}
	OnEndEpoch( context, lpi ) ;

	lpi.nCountInBatch = lpi.iInBatch ;

	// 検証
	if ( ValidateLearning( context, lpi, iter ) )
	{
		if ( context.flagCanceled )
		{
			return ;
		}
		lpi.lossValid = context.lossValidation ;
	}
	OnLearningProgress( learningEndEpoch, lpi ) ;
}

// エポック開始時
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::OnBeginEpoch
	( NNMLPShell::LearningContext& context,
		NNMLPShell::LearningProgressInfo& lpi, NNMLPShell::Iterator& iter )
{
	iter.ResetIterator() ;

	context.lossLearning = 0.0 ;
	context.lossSummed = 0.0 ;
	context.nLossSummed = 0 ;
	context.evalLearning = 0.0 ;
	context.evalSummed = 0.0 ;
	context.nEvalSummed = 0 ;
	context.flagEndOfIter = false ;

	lpi.gradNorms.resize( m_mlp.GetLayerCount() ) ;
	for ( size_t i = 0; i < m_mlp.GetLayerCount(); i ++ )
	{
		lpi.gradNorms.at(i) = 0.0f ;
	}
	lpi.nGradNorm = 0 ;

	m_mlp.OnBeginEpoch() ;
}

// エポック終了時
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::OnEndEpoch
	( LearningContext& context, NNMLPShell::LearningProgressInfo& lpi )
{
	m_mlp.OnEndEpoch( *(context.bufArraysArray.at(0)) ) ;
}

// ミニバッチ訓練データ収集
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShell::LoadTrainingData
	( NNMLPShell::LearningContext& context,
		NNMLPShell::Iterator& iter, size_t nBatchCount )
{
	context.sources.clear() ;
	context.teachers.clear() ;

	for ( size_t i = 0; i < nBatchCount; i ++ )
	{
		std::shared_ptr<NNBuffer>	pSource = iter.NextSource() ;
		if ( pSource == nullptr )
		{
			context.flagEndOfIter = true ;
			break ;
		}
		std::shared_ptr<NNBuffer>	pTeaching = iter.GetTeachingData() ;
		if ( pTeaching != nullptr )
		{
			if ( m_config.flagsBehavior & behaviorPrintLearningFile )
			{
				Print( "\r\nsource: %s, %s",
						iter.GetSourcePath().c_str(),
						iter.GetTeachingDataPath().c_str() ) ;
			}
			context.sources.push_back( pSource ) ;
			context.teachers.push_back( pTeaching ) ;
		}
	}
	return	context.flagEndOfIter ;
}

// バッファ準備
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::PrepareBuffer( NNMLPShell::LearningContext& context )
{
	PrepareBuffer( m_mlp, context, context.sources.at(0)->GetSize() ) ;
}

void NNMLPShell::PrepareBuffer
	( NNMultiLayerPerceptron& mlp,
		NNMLPShell::LearningContext& context, const NNBufDim& dimSource )
{
	for ( size_t i = 0; i < context.nBufCount; i ++ )
	{
		if ( context.dimSourceArray.at(i) != dimSource )
		{
			context.dimSourceArray.at(i) = dimSource ;
			mlp.PrepareBuffer
				( *(context.bufArraysArray.at(i)),
					context.dimSourceArray.at(i), context.flagsBuf, m_bufConfig ) ;
		}
		mlp.ResetWorkInBatch( *(context.bufArraysArray.at(i)) ) ;
	}
	mlp.ResetLossAndGrandient( context.lagArrays ) ;
}

void NNMLPShell::PrepareForwardBuffer
	( NNMultiLayerPerceptron& mlpForward,
		NNMLPShell::LearningContext& lcForward,
		NNMLPShell::LearningContext& context )
{
	assert( context.bufArraysArray.size() >= 1 ) ;
	const NNBufDim	dimOutput = m_mlp.GetOutputSize( *(context.bufArraysArray.at(0)) ) ;
	PrepareBuffer( mlpForward, lcForward, dimOutput ) ;
}

// バッファのミニバッチ用準備処理（ドロップアウト用）
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::PreapreForMiniBatch
	( NNMLPShell::LearningContext& context, bool flagLearning )
{
	std::random_device	random ;
	auto				rndSeed = random() ;

	const uint32_t		flags = (flagLearning ? NNMultiLayerPerceptron::bufferForLearning : 0) ;

	for ( size_t i = 0; i < context.nBufCount; i ++ )
	{
		m_mlp.PrepareForMiniBatch
			( *(context.bufArraysArray.at(i)), flags, rndSeed ) ;

		context.vEvalArray.at(i) = 0.0 ;
		context.vEvalSummed.at(i) = 0 ;
	}
}

// ミニバッチループ開始時
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::OnBeginMiniBatch
	( NNMLPShell::LearningContext& context, bool flagLearning )
{
	context.lossLastLoop = context.lossLearning ;

	PreapreForMiniBatch( context, flagLearning ) ;
}

// ミニバッチループ終了時
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::OnEndMiniBatch( NNMLPShell::LearningContext& context )
{
	context.lossSummed += context.lossCurLoop ;
	context.nLossSummed ++ ;
	context.lossLearning = context.lossSummed / (double) context.nLossSummed ;

	assert( context.vEvalArray.size() == context.nBufCount ) ;
	assert( context.vEvalSummed.size() == context.nBufCount ) ;
	for ( size_t i = 0; i < context.nBufCount; i ++ )
	{
		context.evalSummed += context.vEvalArray.at(i) ;
		context.nEvalSummed += context.vEvalSummed.at(i) ;
	}
	if ( context.nEvalSummed > 0 )
	{
		context.evalLearning =
			context.evalSummed / (double) context.nEvalSummed ;
	}
}

// １回学習
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::LearnOnce
	( NNMLPShell::LearningContext& context,
		NNMLPShell::LearningProgressInfo& lpi,
		NNMultiLayerPerceptron * pForwardMLP,
		NNMLPShell::LearningContext * pForwardContext )
{
	std::shared_ptr<NNEvaluationFunction>
			pEvaluation = m_mlp.GetEvaluationFunction() ;

	lpi.msecLearn = 0 ;
	lpi.deltaRate = context.deltaCurRate ;

	TimeMeasure	tm ;
	context.ploop.Loop
		( 0, context.sources.size(),
			[&]( size_t iThread, size_t iMiniBatch )
	{
		if ( context.flagCanceled )
		{
			return ;
		}
		assert( iThread < context.bufArraysArray.size() ) ;
		NNMultiLayerPerceptron::BufferArrays&
					bufArrays = *(context.bufArraysArray.at( iThread )) ;
		NNBufDim&	dimSource = context.dimSourceArray.at( iThread ) ;
		//
		NNMultiLayerPerceptron::BufferArrays *
					pForwardBufArrays =
						(pForwardContext == nullptr) ? nullptr :
							pForwardContext->bufArraysArray.at(iThread).get() ;

		std::shared_ptr<NNBuffer>	pSource = context.sources.at(iMiniBatch) ;
		std::shared_ptr<NNBuffer>	pTeaching = context.teachers.at(iMiniBatch) ;
		if ( dimSource != pSource->GetSize() )
		{
			// データサイズが異なる場合、更新してバッファを作り直す
			{
				std::lock_guard<std::mutex>	lock(m_mutex) ;
				m_mlp.AddLossAndGradient( context.lagArrays, bufArrays ) ;
			}
			dimSource = pSource->GetSize() ;
			m_mlp.PrepareBuffer
				( bufArrays, dimSource, context.flagsBuf, m_bufConfig ) ;
			m_mlp.ResetWorkInBatch( bufArrays ) ;
			//
			if ( pForwardMLP != nullptr )
			{
				NNBufDim	dimOutput = m_mlp.GetOutputSize( bufArrays ) ;
				if ( dimOutput != pForwardContext->dimSourceArray.at(iThread) )
				{
					pForwardMLP->PrepareBuffer
						( *pForwardBufArrays,
							dimOutput, pForwardContext->flagsBuf, m_bufConfig ) ;
				}
			}
		}
		NNBufDim	dimTeaching( 0, 0, 0 ) ;
		if ( pForwardMLP == nullptr )
		{
			dimTeaching = pTeaching->GetSize() ;
		}
		if ( VerifyDataShape( bufArrays, dimTeaching, dimSource )
								!= NNMultiLayerPerceptron::verifyNormal )
		{
			return ;
		}
		if ( bufArrays.stream.m_useCuda )
		{
			std::lock_guard<std::mutex>	lock(m_mutex) ;
			pSource->CommitCuda() ;
			pTeaching->CommitCuda() ;
		}

		// １回学習
		lpi.lossLearn = m_mlp.Learning( bufArrays, *pTeaching, *pSource,
										pForwardMLP, pForwardBufArrays ) ;
		const long	msecLearn = tm.MeasureMilliSec() ;

		// 評価値計算
		double	evalLearn = 0.0 ;
		if ( pEvaluation != nullptr )
		{
			evalLearn = pEvaluation->Evaluate
							( bufArrays.buffers.back()->bufOutput, *pTeaching ) ;
			context.vEvalArray.at(iThread) += evalLearn ;
			context.vEvalSummed.at(iThread) += 1 ;
		}

		// 進捗
		{
			std::lock_guard<std::mutex>	lock(m_mutex) ;
			lpi.iMiniBatch = iMiniBatch ;
			lpi.evalLearn = evalLearn ;
			lpi.msecLearn = msecLearn ;
			lpi.pTraining = &(bufArrays.buffers.back()->bufOutput) ;
			lpi.nBufferBytes = bufArrays.buffers.GetTotalBufferBytes() ;
			lpi.nCudaBufferBytes = bufArrays.buffers.GetTotalCudaBufferBytes() ;
			OnLearningProgress( learningOneData, lpi ) ;
			//
			if ( IsCancel() )
			{
				context.flagCanceled = true ;
				return ;
			}
		}
	} ) ;
}

// 損失値と勾配を合計し、平均損失を返す
//////////////////////////////////////////////////////////////////////////////
double NNMLPShell::IntegrateLossAndGradient( NNMLPShell::LearningContext& context )
{
	for ( size_t i = 0; i < context.nBufCount; i ++ )
	{
		m_mlp.AddLossAndGradient( context.lagArrays, *(context.bufArraysArray.at(i)) ) ;
		m_mlp.ResetWorkInBatch( *(context.bufArraysArray.at(i)) ) ;
	}

	context.lossCurLoop = m_mlp.GetAverageLoss( context.lagArrays ) ;
	return	context.lossCurLoop ;
}

// 勾配反映
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::GradientReflection
	( NNMLPShell::LearningContext& context,
		NNMLPShell::LearningProgressInfo& lpi )
{
	context.lossLastLoop = context.lossCurLoop ;

	// 勾配ノルム（ログ用）
	assert( context.lagArrays.size() >= lpi.gradNorms.size() ) ;
	for ( size_t i = 0; i < lpi.gradNorms.size(); i ++ )
	{
		const NNPerceptron::LossAndGradient&	grad = context.lagArrays.at(i) ;
		if ( grad.nGradient > 0 )
		{
			NNPerceptronPtr	pLayer = m_mlp.GetLayerAt(i) ;
			const float	gradFactor = (pLayer != nullptr)
									? pLayer->GetGradientFactor() : 1.0f ;
			const float	norm = (grad.matGradient
									/ (float) grad.nGradient).FrobeniusNorm() ;
			lpi.gradNorms.at(i) += norm * gradFactor ;
		}
	}
	lpi.nGradNorm ++ ;

	// 勾配反映
	if ( !isinf( context.lossCurLoop ) )
	{
		m_mlp.GradientReflection( context.lagArrays, context.deltaCurRate ) ;
	}
	m_mlp.ResetLossAndGrandient( context.lagArrays ) ;
}

// 検証
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShell::ValidateLearning
	( NNMLPShell::LearningContext& context,
		NNMLPShell::LearningProgressInfo& lpi, NNMLPShell::Iterator& iter )
{
	std::shared_ptr<NNEvaluationFunction>
				pEvaluation = m_mlp.GetEvaluationFunction() ;

	NNMultiLayerPerceptron::BufferArrays&
				bufArrays0 = *(context.bufArraysArray.at(0)) ;
	NNBufDim&	dimSource0 = context.dimSourceArray.at(0) ;

	std::random_device	random ;
	m_mlp.ResetWorkInBatch( bufArrays0 ) ;
	m_mlp.PrepareForMiniBatch( bufArrays0, 0, random() ) ;

	bool	flagValidation = false ;
	double	evalValidation = 0.0 ;
	lpi.iValidation = 0 ;
	while ( !context.flagCanceled )
	{
		// 検証用データ取得
		std::shared_ptr<NNBuffer>	pSource = iter.NextValidation() ;
		if ( pSource == nullptr )
		{
			break ;
		}
		std::shared_ptr<NNBuffer>	pTeaching = iter.GetTeachingData() ;
		if ( pTeaching == nullptr )
		{
			continue ;
		}
		if ( m_config.flagsBehavior & behaviorPrintLearningFile )
		{
			Print( "\nvalidation: %s, %s",
					iter.GetSourcePath().c_str(),
					iter.GetTeachingDataPath().c_str() ) ;
		}
		if ( dimSource0 != pSource->GetSize() )
		{
			dimSource0 = pSource->GetSize() ;
			m_mlp.PrepareBuffer
				( bufArrays0, dimSource0, context.flagsBuf, m_bufConfig ) ;
		}

		// 予測して損失計算
		lpi.pValidation = m_mlp.Prediction( bufArrays0, *pSource, true ) ;

		m_mlp.CalcLoss( bufArrays0, *pTeaching ) ;
		lpi.lossValid = m_mlp.GetAverageLoss( bufArrays0 ) ;

		// 評価値計算
		if ( pEvaluation != nullptr )
		{
			evalValidation +=
				pEvaluation->Evaluate( *lpi.pValidation, *pTeaching ) ;
			lpi.evalValid = evalValidation / (double) (lpi.iValidation + 1) ;
		}

		// 進捗
		OnLearningProgress( learningValidation, lpi ) ;
		lpi.iValidation ++ ;
		flagValidation = true ;

		if ( IsCancel() )
		{
			context.flagCanceled = true ;
			return	flagValidation ;
		}
	}
	lpi.nValidationCount = lpi.iValidation ;
	context.lossValidation = m_mlp.GetAverageLoss( bufArrays0 ) ;
	return	flagValidation ;
}

// モデルとデータの形状の検証
//////////////////////////////////////////////////////////////////////////////
NNMultiLayerPerceptron::VerifyError
	NNMLPShell::VerifyDataShape
		( const NNMultiLayerPerceptron::BufferArrays& bufArrays,
			const NNBufDim& dimTeaching, const NNBufDim& dimSource0 ) const
{
	NNMultiLayerPerceptron::VerifyResult	verfResult ;
	if ( m_mlp.VerifyDataShape
		( verfResult, bufArrays, dimTeaching, dimSource0 ) )
	{
		return	NNMultiLayerPerceptron::verifyNormal ;
	}
	switch ( verfResult.verfError )
	{
	case	NNMultiLayerPerceptron::outOfRangeInputLayer:
		Print( "\r\nInput to layer[%d] (connection[%d]) out of range.\r\n\x1b[s",
				(int) verfResult.iLayer, (int) verfResult.iConnection ) ;
		break ;
	case	NNMultiLayerPerceptron::mustBeFirstInputLayer:
		Print( "Sampler is used that must be an input layer in layer[%d].\r\n\x1b[s",
				(int) verfResult.iLayer ) ;
		break ;
	case	NNMultiLayerPerceptron::mustBeLastLayer:
		Print( "Activation function is used that must be the final layer in layer[%d].\r\n\x1b[s",
				(int) verfResult.iLayer ) ;
		break ;
	case	NNMultiLayerPerceptron::mustNotBeLastLayer:
		Print( "Activation function is used that cannot be used in the final layer [%d].\r\n\x1b[s",
				(int) verfResult.iLayer ) ;
		break ;
	case	NNMultiLayerPerceptron::mismatchInputSize:
		Print( "\r\nInput size to layer[%d] (connection[%d]) is mismatched.\r\n\x1b[s",
				(int) verfResult.iLayer, (int) verfResult.iConnection ) ;
		break ;
	case	NNMultiLayerPerceptron::mismatchInputChannel:
		Print( "\r\nInput channel count to layer[%d] is mismatched.\r\n\x1b[s",
				(int) verfResult.iLayer ) ;
		break ;
	case	NNMultiLayerPerceptron::mismatchSourceSize:
		Print( "\r\nSource data size is mismatched.\r\n\x1b[s" ) ;
		break ;
	case	NNMultiLayerPerceptron::mismatchSourceChannel:
		Print( "\r\nSource channel count is mismatched.\r\n\x1b[s" ) ;
		break ;
	case	NNMultiLayerPerceptron::mismatchTeachingSize:
		Print( "\r\nTeaching data size is mismatched.\r\n\x1b[s" ) ;
		break ;
	case	NNMultiLayerPerceptron::mismatchTeachingChannel:
		Print( "\r\nTeaching data channel count is mismatched.\r\n\x1b[s" ) ;
		break ;
	case	NNMultiLayerPerceptron::lowCudaMemory:
		Print( "\r\nInsufficient CUDA memory.\r\n\x1b[s" ) ;
		break ;
	case	NNMultiLayerPerceptron::tooHugeMatrixForCuda:
		Print( "\r\nMatrix size too huge to run in CUDA.\r\n\x1b[s" ) ;
		break ;
	}
	return	verfResult.verfError ;
}

NNMultiLayerPerceptron::VerifyError
	NNMLPShell::VerifyDataShape
		( const NNMultiLayerPerceptron::BufferArrays& bufArrays,
			const NNBufDim& dimSource0 ) const
{
	NNBufDim	dimTeaching( 0, 0, 0 ) ;
	return	VerifyDataShape( bufArrays, dimTeaching, dimSource0 ) ;
}

// 学習進捗表示
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::OnLearningProgress
	( NNMLPShell::LearningEvent le, const NNMLPShell::LearningProgressInfo& lpi )
{
	switch ( le )
	{
	case	learningStartMiniBatch:
		if ( lpi.flags & (learningGAN | learningGANClassifier) )
		{
			Print( "\x1b[s"
					"loop: %d/%d, epoch: %d/%d, batch: %d/%d, rep: %d/%d (%d/%d)\x1b[0K",
					(int) lpi.iGANLoop+1, (int) lpi.nGANCount,
					(int) lpi.iLoopEpoch+1, (int) lpi.nEpochCount,
					(int) lpi.iInBatch+1, (int) lpi.nCountInBatch,
					0, (int) lpi.nSubLoopCount,
					0, (int) lpi.nMiniBatchCount ) ;
		}
		else
		{
			Print( "\x1b[s"
					"epoch: %d/%d, batch: %d/%d, rep: %d/%d (%d/%d)\x1b[0K",
					(int) lpi.iLoopEpoch+1, (int) lpi.nEpochCount,
					(int) lpi.iInBatch+1, (int) lpi.nCountInBatch,
					0, (int) lpi.nSubLoopCount,
					0, (int) lpi.nMiniBatchCount ) ;
		}
		break ;

	case	learningOneData:
	case	learningEndMiniBatch:
		TRACE( "epoch: %d/%d, batch: %d/%d, rep: %d/%d (%d/%d), loss=%f  [%ld ms]  (delta=%f)\r\n",
				(int) lpi.iLoopEpoch+1, (int) lpi.nEpochCount,
				(int) lpi.iInBatch+1, (int) lpi.nCountInBatch,
				(int) lpi.iSubLoop+1, (int) lpi.nSubLoopCount,
				(int) lpi.iMiniBatch+1, (int) lpi.nMiniBatchCount,
				lpi.lossLearn, lpi.msecLearn, lpi.deltaRate ) ;
		if ( lpi.flags & (learningGAN | learningGANClassifier) )
		{
			Print( "\x1b[s"
					"loop: %d/%d, epoch: %d/%d, batch: %d/%d, rep: %d/%d (%d/%d), loss=%f  [%ld ms]",
					(int) lpi.iGANLoop+1, (int) lpi.nGANCount,
					(int) lpi.iLoopEpoch+1, (int) lpi.nEpochCount,
					(int) lpi.iInBatch+1, (int) lpi.nCountInBatch,
					(int) lpi.iSubLoop+1, (int) lpi.nSubLoopCount,
					(int) lpi.iMiniBatch+1, (int) lpi.nMiniBatchCount,
					lpi.lossLearn, lpi.msecLearn, lpi.deltaRate ) ;
		}
		else
		{
			Print( "\x1b[s"
					"epoch: %d/%d, batch: %d/%d, rep: %d/%d (%d/%d), loss=%f  [%ld ms]",
					(int) lpi.iLoopEpoch+1, (int) lpi.nEpochCount,
					(int) lpi.iInBatch+1, (int) lpi.nCountInBatch,
					(int) lpi.iSubLoop+1, (int) lpi.nSubLoopCount,
					(int) lpi.iMiniBatch+1, (int) lpi.nMiniBatchCount,
					lpi.lossLearn, lpi.msecLearn, lpi.deltaRate ) ;
		}
		if ( m_config.flagsBehavior & behaviorPrintBufferSize )
		{
			Print( "  (buf=%dMB)\x1b[0K", (int) (lpi.nBufferBytes / (1024*1024)) ) ;
		}
		else if ( m_config.flagsBehavior & behaviorPrintCudaBufferSize )
		{
			Print( "  (CUDA=%dMB)\x1b[0K", (int) (lpi.nCudaBufferBytes / (1024*1024)) ) ;
		}
		else
		{
			Print( "  (delta=%f)\x1b[0K", lpi.deltaRate ) ;
		}
		break ;

	case	learningValidation:
		Print( "\x1b[s"
				"validation (%d/%d) loss = %f\x1b[0K",
				(int) lpi.iValidation + 1, (int) lpi.nValidationCount, lpi.lossValid ) ;
		break ;

	case	learningEndEpoch:
		Print( "validation loss = %f, rate = %f\x1b[0K\r\n",
						lpi.lossValid, lpi.lossValid/lpi.lossLearn ) ;
		break ;
	}
	switch ( le )
	{
	case	learningStartMiniBatch:
	case	learningOneData:
	case	learningValidation:
		Print( "\r\x1b[u" ) ;
		break ;

	case	learningEndMiniBatch:
		if ( m_config.flagsBehavior & behaviorLineFeedByMiniBatch )
		{
			Print( "\r\n\x1b[s" ) ;
		}
		else
		{
			Print( "\r\x1b[u" ) ;
		}
		break ;

	case	learningEndSubLoop:
		if ( !(m_config.flagsBehavior & behaviorLineFeedByMiniBatch) )
		{
			Print( "\r\n\x1b[s" ) ;
		}
		break ;
	}

	for ( auto l : m_listeners )
	{
		l->OnLearningProgress( *this, le, lpi ) ;
	}
}

// 予測出力完了
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::OnProcessedPrediction
	( const char * pszSourcePath,
		NNBuffer * pSource, NNBuffer * pOutput,
		const NNMultiLayerPerceptron::BufferArrays& bufArrays )
{
	for ( auto l : m_listeners )
	{
		l->OnProcessedPrediction
			( *this, pszSourcePath, pSource, pOutput, bufArrays ) ;
	}
}

// ストリーム出力進捗表示
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::OnStreamingProgress
	( NNMLPShell::Streamer * pStreamer,
		size_t current, size_t total,
		NNStreamBuffer * psbOutput, size_t xLastStream )
{
	if ( pStreamer != nullptr )
	{
		pStreamer->OnProgress
			( *this, current, total, psbOutput, xLastStream ) ;
	}
	if ( total > 0 )
	{
		int	nProgress = (int) (current * 100 / total) ;
		if ( nProgress != m_nLastProgress )
		{
			Print( "%d%%" "\x1b[0K" "\x1b[u", nProgress ) ;
			m_nLastProgress = nProgress ;
		}
	}

	for ( auto l : m_listeners )
	{
		l->OnStreamingProgress
			( *this, pStreamer, current, total, psbOutput, xLastStream ) ;
	}
}

// 進捗リスナ
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::AttachProgressListener( NNMLPShell::ProgressListener * pListener )
{
	if ( std::find( m_listeners.begin(), m_listeners.end(), pListener ) == m_listeners.end() )
	{
		m_listeners.push_back( pListener ) ;
	}
}

bool NNMLPShell::DetachProgressListener( NNMLPShell::ProgressListener * pListener )
{
	auto	iterFound = std::find( m_listeners.begin(), m_listeners.end(), pListener ) ;
	if ( iterFound == m_listeners.end() )
	{
		return	false ;
	}
	m_listeners.erase( iterFound ) ;
	return	true ;
}

// メッセージ出力
//////////////////////////////////////////////////////////////////////////////
void NNMLPShell::Print( const char * pszFormat, ... ) const
{
	va_list	vl ;
	va_start( vl, pszFormat ) ;
	vprintf( pszFormat, vl ) ;
	fflush( stdout ) ;
}

// 処理中断
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShell::IsCancel( void )
{
	if ( _kbhit() )
	{
		if ( _getch() == 0x1b )
		{
			Print( "\nDo you want to abort? (Y/N):" ) ;
			for ( ; ; )
			{
				char	c = getchar() ;
				if ( (c == 'N') || (c == 'n') )
				{
					break ;
				}
				if ( (c == 'Y') || (c == 'y') )
				{
					return	true ;
				}
			}
		}
	}
	return	false ;
}



//////////////////////////////////////////////////////////////////////////////
// GAN 分類器用反復器
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellGANClassifierIterator::NNMLPShellGANClassifierIterator
	( NNMLPShell::Iterator& iterClassifier,
		NNMultiLayerPerceptron& mlpGenerator,
		const NNMultiLayerPerceptron::BufferConfig& cfgGenerator,
		NNMLPShell::LearningContext& lcGenerator,
		NNMLPShell::Iterator& iterGAN )
	: m_iterClassifier( iterClassifier ),
		m_pClassifier( nullptr ),
		m_mlpGenerator( mlpGenerator ),
		m_cfgGenerator( cfgGenerator ),
		m_lcGenerator( lcGenerator ),
		m_iterGAN( iterGAN ), m_pGANIter( nullptr ),
		m_iIterNext( 0 ), m_flagDataGen( false )
{
	m_pClassifier = dynamic_cast<NNMLPShell::Classifier*>( &iterClassifier ) ;
	assert( m_pClassifier != nullptr ) ;

	m_pGANIter = dynamic_cast<NNMLPShell::GANIterator*>( &iterGAN ) ;
	assert( m_pGANIter != nullptr ) ;
}

// 初期化処理
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellGANClassifierIterator::InitializeIterator( void )
{
}

// 初めから
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellGANClassifierIterator::ResetIterator( void )
{
	m_iterClassifier.ResetIterator() ;
	m_iterGAN.ResetIterator() ;
	m_iIterNext = 0 ;
}

// 次の入力データを取得する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellGANClassifierIterator::NextSource( void )
{
	if ( (m_iIterNext++ % 2) == 0 )
	{
		// ２つに１つは分類器用の入力そのまま
		m_flagDataGen = false ;
		return	m_iterClassifier.NextSource() ;
	}

	// ２つに１つは生成器の出力
	std::shared_ptr<NNBuffer>	pSource = m_iterGAN.NextSource() ;
	if ( pSource == nullptr )
	{
		m_iterGAN.ResetIterator() ;
		pSource = m_iterGAN.NextSource() ;
		if ( pSource == nullptr )
		{
			return	nullptr ;
		}
	}

	// 分類器の訓練用に「偽物」が教師データとなる
	m_pTeachingGen = m_pGANIter->GetCounterfeitData() ;
	assert( m_pTeachingGen != nullptr ) ;
	m_flagDataGen = true ;

	// 生成器のバッファ準備
	NNMultiLayerPerceptron::BufferArrays&
				bufArrays0 = *(m_lcGenerator.bufArraysArray.at(0)) ;
	NNBufDim&	dimSource0 = m_lcGenerator.dimSourceArray.at(0) ;
	if ( dimSource0 != pSource->GetSize() )
	{
		dimSource0 = pSource->GetSize() ;
		m_mlpGenerator.PrepareBuffer
			( bufArrays0, dimSource0, m_lcGenerator.flagsBuf, m_cfgGenerator ) ;
		m_mlpGenerator.ResetWorkInBatch( bufArrays0 ) ;
	}

	// 生成器のネットワーク検証
	NNMultiLayerPerceptron::VerifyResult	verfResult ;
	bool	flagLowMemory = false ;
	if ( !m_mlpGenerator.VerifyDataShape
		( verfResult, bufArrays0, NNBufDim(0,0,0), dimSource0 ) )
	{
		if ( verfResult.verfError != NNMultiLayerPerceptron::lowCudaMemory )
		{
			return	nullptr ;
		}
		flagLowMemory = true ;
	}

	// 生成器での生成
	NNBuffer *	pOutputGen =
		m_mlpGenerator.Prediction( bufArrays0, *pSource, false, flagLowMemory ) ;
	assert( pOutputGen != nullptr ) ;
	if ( pOutputGen == nullptr )
	{
		return	nullptr ;
	}

	std::shared_ptr<NNBuffer>	pGenAsSource = std::make_shared<NNBuffer>() ;
	pGenAsSource->Create( pOutputGen->GetSize() ) ;
	pGenAsSource->CopyFrom( *pOutputGen ) ;
	return	pGenAsSource ;
}

// 次の検証データを取得する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellGANClassifierIterator::NextValidation( void )
{
	m_flagDataGen = false ;
	return	m_iterClassifier.NextValidation() ;
}

// 最後に取得した{入力|検証}データに対応する教師データを取得する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellGANClassifierIterator::GetTeachingData( void )
{
	return	m_flagDataGen ? m_pTeachingGen
							: m_iterClassifier.GetTeachingData() ;
}

// 最後に取得した入力データに対応する予測データを出力する
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellGANClassifierIterator::OutputPrediction( const NNBuffer& bufOutput )
{
	if ( !m_flagDataGen )
	{
		return	m_iterClassifier.OutputPrediction( bufOutput ) ;
	}
	return	true ;
}

// {NextSource|NextValidation} で取得した入力データのパスを取得する
//////////////////////////////////////////////////////////////////////////////
std::string NNMLPShellGANClassifierIterator::GetSourcePath( void )
{
	return	m_flagDataGen ? std::string()
							: m_iterClassifier.GetSourcePath() ;
}

// GetTeachingData で取得できる教師データのパスを取得する
//////////////////////////////////////////////////////////////////////////////
std::string NNMLPShellGANClassifierIterator::GetTeachingDataPath( void )
{
	return	m_flagDataGen ? std::string()
							: m_iterClassifier.GetSourcePath() ;
}

// OutputPrediction で出力するパスを取得する
//////////////////////////////////////////////////////////////////////////////
std::string NNMLPShellGANClassifierIterator::GetOutputPath( void )
{
	return	m_flagDataGen ? std::string()
							: m_iterClassifier.GetSourcePath() ;
}

// 分類名取得
//////////////////////////////////////////////////////////////////////////////
const std::string& NNMLPShellGANClassifierIterator::GetClassNameAt( size_t iClass ) const
{
	return	m_pClassifier->GetClassNameAt( iClass ) ;
}

size_t NNMLPShellGANClassifierIterator::GetClassCount( void ) const
{
	return	m_pClassifier->GetClassCount() ;
}

// 分類インデックス
//////////////////////////////////////////////////////////////////////////////
int NNMLPShellGANClassifierIterator::GetClassIndexOf( const char * pszClassName ) const
{
	return	m_pClassifier->GetClassIndexOf( pszClassName ) ;
}




//////////////////////////////////////////////////////////////////////////////
// ファイル入出力基底
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellGenFileIterator::NNMLPShellGenFileIterator( bool flagShuffle )
	: m_iValidation( 0 ),
		m_flagShuffle( flagShuffle ),
		m_iNextSource( 0 ),
		m_iNextValidation( 0 )
{
}

// ファイル列挙
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellGenFileIterator::EnumerateFiles
	( const char * pszDir, std::function<void(const std::filesystem::path&)> func )
{
	for ( auto x : std::filesystem::directory_iterator(pszDir) )
	{
		const std::filesystem::path&	pathFile = x.path() ;
		if ( std::filesystem::exists( pathFile )
			&& std::filesystem::is_regular_file( pathFile ) )
		{
			func( pathFile ) ;
		}
	}
}

// ディレクトリ列挙
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellGenFileIterator::EnumerateDirectories
	( const char * pszDir, std::function<void(const std::filesystem::path&)> func )
{
	for ( auto x : std::filesystem::directory_iterator(pszDir) )
	{
		const std::filesystem::path&	pathDir = x.path() ;
		if ( std::filesystem::is_directory( pathDir ) )
		{
			func( pathDir ) ;
		}
	}
}

// 汎用シャッフル関数
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellGenFileIterator::Shuffle
	( size_t iFirst, size_t nCount,
		std::function<void(size_t,size_t)> funcShuffle )
{
	std::random_device	random ;
	std::mt19937		engine( random() ) ;

	for ( size_t i = 0; i < nCount; i += 2 )
	{
		size_t	iShuffle0 = iFirst + (engine() % nCount) ;
		size_t	iShuffle1 = iFirst + (engine() % nCount) ;
		if ( iShuffle0 != iShuffle1 )
		{
			funcShuffle( iShuffle0, iShuffle1 ) ;
		}
	}
}


// 初めから
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellGenFileIterator::ResetIterator( void )
{
	m_iNextSource = 0 ;
	m_iNextValidation = m_iValidation ;

	if ( m_flagShuffle )
	{
		auto	funcShuffle = [this]( size_t iShuffle1, size_t iShuffle2 )
		{
			ShuffleFile( iShuffle1, iShuffle2 ) ;
		} ;
		assert( m_iValidation <= m_files.size() ) ;
		Shuffle( 0, m_iValidation, funcShuffle ) ;
	}
}

// 次の入力データを取得する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellGenFileIterator::NextSource( void )
{
	m_pSource = nullptr ;
	m_pTeaching = nullptr ;

	while ( m_iNextSource < m_iValidation )
	{
		if ( PrepareNextDataAt( m_iNextSource ++ ) )
		{
			break ;
		}
	}
	return	m_pSource ;
}

// 次の検証データを取得する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellGenFileIterator::NextValidation( void )
{
	m_pSource = nullptr ;
	m_pTeaching = nullptr ;

	while ( m_iNextValidation < m_files.size() )
	{
		if ( PrepareNextDataAt( m_iNextValidation ++ ) )
		{
			break ;
		}
	}
	return	m_pSource ;
}

// 最後に取得した{入力|検証}データに対応する教師データを取得する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellGenFileIterator::GetTeachingData( void )
{
	return	m_pTeaching ;
}

// {NextSource|NextValidation} で取得した入力データのパスを取得する
//////////////////////////////////////////////////////////////////////////////
std::string NNMLPShellGenFileIterator::GetSourcePath( void )
{
	return	m_pathSourceFile.string() ;
}

// 入力ファイルをシャッフルする（指定指標のファイルを入れ替える）
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellGenFileIterator::ShuffleFile( size_t iFile1, size_t iFile2 )
{
	assert( iFile1 < m_files.size() ) ;
	assert( iFile2 < m_files.size() ) ;
	std::filesystem::path	temp = m_files.at(iFile1) ;
	m_files.at(iFile1) = m_files.at(iFile2) ;
	m_files.at(iFile2) = temp ;
}

// 入力元からファイルを読み込んでバッファに変換する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer>
	NNMLPShellGenFileIterator::LoadSourceFromFile( const std::filesystem::path& path )
{
	return	LoadFromFile( path ) ;
}

// 教師データをファイルを読み込んでバッファに変換する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer>
	NNMLPShellGenFileIterator::LoadTeachingFromFile
		( const std::filesystem::path& path, std::shared_ptr<NNBuffer> pSource )
{
	return	LoadFromFile( path ) ;
}

// 読み込んだバッファを処理して次のデータとして設定する
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellGenFileIterator::SetNextDataOnLoaded
			( std::shared_ptr<NNBuffer> pSource,
				std::shared_ptr<NNBuffer> pTeaching )
{
	m_pSource = pSource ;
	m_pTeaching = pTeaching ;
	return	true ;
}



//////////////////////////////////////////////////////////////////////////////
// ファイル入力
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellFileIterator::NNMLPShellFileIterator
		( const char * pszSourceDir,
				const char * pszPairDir, bool flagOutputPair,
				bool flagRandValidation, double rateValidation )
	: NNMLPShellGenFileIterator( !flagOutputPair ),
		m_pathSourceDir( pszSourceDir ),
		m_pathPairDir( pszPairDir ),
		m_flagOutputPair( flagOutputPair ),
		m_flagRandValidation( flagRandValidation ),
		m_rateValidation( rateValidation )
{
}

// 初期化処理
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellFileIterator::InitializeIterator( void )
{
	EnumerateFiles
		( m_pathSourceDir.string().c_str(),
			[this]( const std::filesystem::path& pathFile )
	{
		std::filesystem::path	pathPair = m_flagOutputPair
												? MakeOutputPathOf(pathFile)
												:  MakeTeacherPathOf(pathFile) ;
		if ( m_flagOutputPair
			|| (std::filesystem::exists( pathPair )
				&& std::filesystem::is_regular_file( pathPair )) )
		{
			m_files.push_back( pathFile ) ;
		}
	} ) ;
	m_iValidation = m_files.size() ;
	if ( !m_flagOutputPair )
	{
		if ( m_flagRandValidation )
		{
			Shuffle( 0, m_files.size(),
						[this]( size_t iShuffle1, size_t iShuffle2 )
							{ ShuffleFile( iShuffle1, iShuffle2 ) ; }) ;
		}
		m_iValidation -= (size_t) floor( m_files.size() * m_rateValidation ) ;
	}

	ResetIterator() ;
}

// 最後に取得した入力データに対応する予測データを出力する
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellFileIterator::OutputPrediction( const NNBuffer& bufOutput )
{
	return	SaveToFile( m_pathPairFile, bufOutput ) ;
}

// GetTeachingData で取得できる教師データのパスを取得する
//////////////////////////////////////////////////////////////////////////////
std::string NNMLPShellFileIterator::GetTeachingDataPath( void )
{
	return	!m_flagOutputPair ? m_pathPairFile.string() : std::string() ;
}

// OutputPrediction で出力するパスを取得する
//////////////////////////////////////////////////////////////////////////////
std::string NNMLPShellFileIterator::GetOutputPath( void )
{
	return	m_flagOutputPair ? m_pathPairFile.string() : std::string() ;
}

// 次のデータを用意する
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellFileIterator::PrepareNextDataAt( size_t iFile )
{
	const std::filesystem::path&	pathSource = m_files.at(iFile) ;

	std::filesystem::path	pathPair = m_flagOutputPair
											? MakeOutputPathOf(pathSource)
											:  MakeTeacherPathOf(pathSource) ;

	std::shared_ptr<NNBuffer>	pSource = LoadSourceFromFile( pathSource ) ;
	std::shared_ptr<NNBuffer>	pTeaching ;
	if ( pSource == nullptr )
	{
		return	false ;
	}
	if ( !m_flagOutputPair )
	{
		pTeaching = LoadTeachingFromFile( pathPair, pSource ) ;
		if ( pTeaching == nullptr )
		{
			return	false ;
		}
	}
	if ( SetNextDataOnLoaded( pSource, pTeaching ) )
	{
		m_pathSourceFile = pathSource ;
		m_pathPairFile = pathPair ;
		return	true ;
	}
	return	false ;
}

// 教師ファイル名を生成する
//////////////////////////////////////////////////////////////////////////////
std::filesystem::path
	NNMLPShellFileIterator::MakeTeacherPathOf( const std::filesystem::path& pathSource )
{
	std::filesystem::path	pathTeacher = m_pathPairDir ;
	pathTeacher /= pathSource.filename() ;
	return	pathTeacher ;
}

// 出力ファイル名を生成する
//////////////////////////////////////////////////////////////////////////////
std::filesystem::path
	NNMLPShellFileIterator::MakeOutputPathOf( const std::filesystem::path& pathSource )
{
	std::filesystem::path	pathOutput = m_pathPairDir ;
	pathOutput /= pathSource.filename() ;
	return	pathOutput ;
}



//////////////////////////////////////////////////////////////////////////////
// ファイル分類器
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellFileClassifier::NNMLPShellFileClassifier
		( const char * pszClassDir, bool formatIndex )
	: m_formatIndex( formatIndex )
{
	if ( pszClassDir != nullptr )
	{
		DirectoryAsClassName( pszClassDir ) ;
	}
}

// 予測モードでサブディレクトリを分類名として設定する
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellFileClassifier::DirectoryAsClassName( const char * pszClassDir )
{
	m_classNames.clear() ;
	m_classNames.push_back( std::string() ) ;

	NNMLPShellGenFileIterator::EnumerateDirectories
		( pszClassDir, [this]( const std::filesystem::path& pathDir )
	{
		m_classNames.push_back( pathDir.filename().string() ) ;
	} ) ;

	std::sort( m_classNames.begin(), m_classNames.end() ) ;
}

// 分類名取得
//////////////////////////////////////////////////////////////////////////////
const std::string& NNMLPShellFileClassifier::GetClassNameAt( size_t iClass ) const
{
	return	m_classNames.at( iClass ) ;
}

size_t NNMLPShellFileClassifier::GetClassCount( void ) const
{
	return	m_classNames.size() ;
}

// 分類インデックス
//////////////////////////////////////////////////////////////////////////////
int NNMLPShellFileClassifier::GetClassIndexOf( const char * pszClassName ) const
{
	for ( size_t i = 0; i < m_classNames.size(); i ++ )
	{
		if ( m_classNames.at(i) == pszClassName )
		{
			return	(int) i ;
		}
	}
	return	NNMLPShell::Classifier::classInvalid ;
}

// One-Hot データ作成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellFileClassifier::MakeOneHot( size_t iClass ) const
{
	std::shared_ptr<NNBuffer>	pBuf = std::make_shared<NNBuffer>() ;
	if ( m_formatIndex )
	{
		pBuf->Create( 1, 1, 1 ) ;
		pBuf->GetBuffer()[0] = (float) iClass ;
	}
	else
	{
		pBuf->Create( 1, 1, m_classNames.size() ) ;
		pBuf->Fill( 0.0f ) ;

		assert( iClass < m_classNames.size() ) ;
		if ( iClass < m_classNames.size() )
		{
			pBuf->GetBuffer()[iClass] = 1.0f ;
		}
	}
	return	pBuf ;
}

// One-Hot インデックス形式か？
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellFileClassifier::IsOneHotIndexFormat( void ) const
{
	return	m_formatIndex ;
}

// One-Hot 表現をインデックス形式に設定
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellFileClassifier::SetOneHotToIndexFormat( bool formatIndex )
{
	m_formatIndex = formatIndex ;
}

// 分類名記述ファイル読み込み
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer>
	NNMLPShellFileClassifier::ParseFile( const std::filesystem::path& path )
{
	try
	{
		std::ifstream	ifs( path, std::ios_base::in ) ;
		if ( ifs.is_open() )
		{
			std::string	strClassName ;
			ifs >> strClassName ;

			int	iClass = GetClassIndexOf( strClassName.c_str() ) ;
			if ( iClass != NNMLPShell::Classifier::classInvalid )
			{
				return	MakeOneHot( (size_t) iClass ) ;
			}
		}
	}
	catch ( const std::exception& e )
	{
		TRACE( "exception at SaveModel: %s\r\n", e.what() ) ;
	}
	return	MakeOneHot( NNMLPShell::Classifier::classFalse ) ;
}



//////////////////////////////////////////////////////////////////////////////
// 分類器ファイル入力
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellFileClassIterator::NNMLPShellFileClassIterator
		( const char * pszSourceDir, bool flagPrediction,
			const char * pszClassDir, bool formatIndex,
			bool flagRandValidation, double rateValidation )
	: NNMLPShellGenFileIterator( !flagPrediction ),
		NNMLPShellFileClassifier( nullptr, formatIndex ),
		m_pathSourceDir( pszSourceDir ), m_flagPrediction( flagPrediction ),
		m_strClassDir( (pszClassDir != nullptr) ? pszClassDir : "" ),
		m_formatIndex( formatIndex ),
		m_flagRandValidation( flagRandValidation ),
		m_rateValidation( rateValidation )
{
	assert( rateValidation < 1.0 ) ;
	assert( rateValidation >= 0.0 ) ;
}

// 初期化処理
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellFileClassIterator::InitializeIterator( void )
{
	std::vector<std::filesystem::path>	vValidations ;
	std::vector<size_t>					vValidClass ;
	if ( m_flagPrediction )
	{
		// ファイル列挙
		EnumerateFiles
			( m_pathSourceDir.string().c_str(),
				[&]( const std::filesystem::path& pathFile )
		{
			assert( m_classIndices.size() == m_files.size() ) ;
			m_files.push_back( pathFile ) ;
			m_classIndices.push_back( classFalse ) ;
		} ) ;
		if ( !m_strClassDir.empty() )
		{
			DirectoryAsClassName( m_strClassDir.c_str() ) ;
		}
	}
	else
	{
		// 分類名（ディレクトリ）列挙
		DirectoryAsClassName( m_pathSourceDir.string().c_str() ) ;

		// 分類ごとの入力ファイルを列挙
		for ( size_t iClass = classFirstIndex; iClass < m_classNames.size(); iClass ++ )
		{
			std::filesystem::path	pathDir = m_pathSourceDir ;
			pathDir /= m_classNames.at( iClass ) ;

			// ファイル列挙
			std::vector<std::filesystem::path>	vFiles ;
			EnumerateFiles( pathDir.string().c_str(),
							[&]( const std::filesystem::path& pathFile )
			{
				vFiles.push_back( pathFile ) ;
			} ) ;

			if ( m_flagRandValidation )
			{
				Shuffle( 0, vFiles.size(),
					[&]( size_t iShuffle1, size_t iShuffle2 )
						{
							auto	temp = vFiles.at(iShuffle1) ;
							vFiles.at(iShuffle1) = vFiles.at(iShuffle2) ;
							vFiles.at(iShuffle2) = temp ;
						} ) ;
			}
			size_t	iValid = vFiles.size()
							- (size_t) floor(vFiles.size() * m_rateValidation) ;
			for ( size_t i = 0; i < iValid; i ++ )
			{
				assert( m_classIndices.size() == m_files.size() ) ;
				m_files.push_back( vFiles.at(i) ) ;
				m_classIndices.push_back( iClass ) ;
			}
			for ( size_t i = iValid; i < vFiles.size(); i ++ )
			{
				assert( vValidClass.size() == vValidations.size() ) ;
				vValidations.push_back( vFiles.at(i) ) ;
				vValidClass.push_back( iClass ) ;
			}
		}
	}

	m_iValidation = m_files.size() ;

	assert( vValidClass.size() == vValidations.size() ) ;
	for ( size_t i = 0; i < vValidations.size(); i ++ )
	{
		m_files.push_back( vValidations.at(i) ) ;
		m_classIndices.push_back( vValidClass.at(i) ) ;
	}

	ResetIterator() ;
}

// 最後に取得した入力データに対応する予測データを出力する
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellFileClassIterator::OutputPrediction( const NNBuffer& bufOutput )
{
	WriteToFile( std::cout, bufOutput ) ;
	return	true ;
}

// GetTeachingData で取得できる教師データのパスを取得する
//////////////////////////////////////////////////////////////////////////////
std::string NNMLPShellFileClassIterator::GetTeachingDataPath( void )
{
	return	std::string() ;
}

// OutputPrediction で出力するパスを取得する
//////////////////////////////////////////////////////////////////////////////
std::string NNMLPShellFileClassIterator::GetOutputPath( void )
{
	return	std::string() ;
}

// 入力ファイルをシャッフルする（指定指標のファイルを入れ替える）
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellFileClassIterator::ShuffleFile( size_t iFile1, size_t iFile2 )
{
	NNMLPShellGenFileIterator::ShuffleFile( iFile1, iFile2 ) ;

	assert( iFile1 < m_classIndices.size() ) ;
	assert( iFile2 < m_classIndices.size() ) ;
	size_t	temp = m_classIndices.at(iFile1) ;
	m_classIndices.at(iFile1) = m_classIndices.at(iFile2) ;
	m_classIndices.at(iFile2) = temp ;
}

// 次のデータを用意する
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellFileClassIterator::PrepareNextDataAt( size_t iFile )
{
	assert( iFile < m_files.size() ) ;
	const std::filesystem::path&	pathSource = m_files.at(iFile) ;

	std::shared_ptr<NNBuffer>	pSource = LoadFromFile( pathSource ) ;
	std::shared_ptr<NNBuffer>	pTeaching ;
	if ( pSource == nullptr )
	{
		return	false ;
	}
	if ( !m_flagPrediction )
	{
		assert( iFile < m_classIndices.size() ) ;
		pTeaching = MakeOneHot( m_classIndices.at(iFile) ) ;
	}
	if ( SetNextDataOnLoaded( pSource, pTeaching ) )
	{
		m_pathSourceFile = pathSource ;
		return	true ;
	}
	return	false ;
}

// バッファを形式変換してファイルに書き込む
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellFileClassIterator::SaveToFile
	( const std::filesystem::path& path, const NNBuffer& bufOutput )
{
	try
	{
		std::ofstream	ofs( path, std::ios_base::out ) ;
		if ( !ofs )
		{
			return	false ;
		}
		WriteToFile( ofs, bufOutput ) ;
	}
	catch ( const std::exception& e )
	{
		TRACE( "exception at SaveToFile: %s\r\n", e.what() ) ;
		return	false ;
	}
	return	true ;
}

void NNMLPShellFileClassIterator::WriteToFile
	( std::ostream& ofs, const NNBuffer& bufOutput )
{
	ofs << "{ " ;

	NNBufDim	dimOut = bufOutput.GetSize() ;
	if ( IsOneHotIndexFormat() )
	{
		// argmax 形式
		for ( size_t i = 0; i + (argmaxChannelCount-1) < dimOut.z; i += argmaxChannelCount )
		{
			if ( i > 0 )
			{
				ofs << ", " ;
			}
			size_t	iClass = (size_t) floor( bufOutput.GetConstBuffer()[i+argmaxIndex] ) ;
			if ( iClass < m_classNames.size() )
			{
				ofs << m_classNames.at(iClass) ;
			}
			else
			{
				ofs << iClass ;
			}
			ofs << ": " ;
			ofs << bufOutput.GetConstBuffer()[i+argmaxProbability] ;
		}
	}
	else
	{
		// softmax 形式
		if ( dimOut.n >= 1 )
		{
			for ( size_t i = 0; i < dimOut.z; i ++ )
			{
				if ( i > 0 )
				{
					ofs << ", " ;
				}
				if ( i < m_classNames.size() )
				{
					ofs << m_classNames.at(i) ;
				}
				else
				{
					ofs << i ;
				}
				ofs << ": " ;
				ofs << bufOutput.GetConstBuffer()[i] ;
			}
		}
	}

	ofs << " }" ;
}



//////////////////////////////////////////////////////////////////////////////
// GAN 学習用反復器
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellGANIterator::NNMLPShellGANIterator
		( std::shared_ptr<NNMLPShellFileClassIterator> pClassifier )
	: m_pClassifier( pClassifier ), m_iNext( 0 )
{
}

// 初期化処理
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellGANIterator::InitializeIterator( void )
{
	NNMLPShell::Iterator *	pClassifierIter =
				dynamic_cast<NNMLPShell::Iterator*>( m_pClassifier.get() ) ;
	if ( pClassifierIter != nullptr )
	{
		pClassifierIter->InitializeIterator() ;
	}
}

// 初めから
//////////////////////////////////////////////////////////////////////////////
void NNMLPShellGANIterator::ResetIterator( void )
{
	assert( m_pClassifier != nullptr ) ;

	m_classes.resize
		( (size_t) __max((int) m_pClassifier->GetClassCount()
							- NNMLPShell::Classifier::classFirstIndex, 0) ) ;
	//
	for ( size_t i = 0; i < m_classes.size(); i ++ )
	{
		m_classes.at(i) = NNMLPShell::Classifier::classFirstIndex + i ;
	}
	auto	funcShuffle = [this]( size_t iShuffle1, size_t iShuffle2 )
	{
		assert( iShuffle1 < m_classes.size() ) ;
		assert( iShuffle2 < m_classes.size() ) ;
		size_t	temp = m_classes.at(iShuffle1) ;
		m_classes.at(iShuffle1) = m_classes.at(iShuffle2) ;
		m_classes.at(iShuffle2) = temp ;
	} ;
	NNMLPShellGenFileIterator::Shuffle( 0, m_classes.size(), funcShuffle ) ;

	m_pCounterfeit = m_pClassifier->MakeOneHot( NNMLPShell::Classifier::classFalse ) ;
	m_iNext = 0 ;
}

// 次の入力データを取得する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellGANIterator::NextSource( void )
{
	assert( m_pClassifier != nullptr ) ;
	if ( m_iNext < m_classes.size() )
	{
		m_pTeaching = m_pClassifier->MakeOneHot( m_classes.at(m_iNext ++) ) ;
		return	m_pTeaching ;
	}
	return	nullptr ;
}

// 次の検証データを取得する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellGANIterator::NextValidation( void )
{
	return	nullptr ;
}

// 最後に取得した{入力|検証}データに対応する教師データを取得する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellGANIterator::GetTeachingData( void )
{
	return	m_pTeaching ;
}

// 最後に取得した入力データに対応する予測データを出力する
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellGANIterator::OutputPrediction( const NNBuffer& bufOutput )
{
	return	true ;
}

// {NextSource|NextValidation} で取得した入力データのパスを取得する
//////////////////////////////////////////////////////////////////////////////
std::string NNMLPShellGANIterator::GetSourcePath( void )
{
	return	std::string() ;
}

// GetTeachingData で取得できる教師データのパスを取得する
//////////////////////////////////////////////////////////////////////////////
std::string NNMLPShellGANIterator::GetTeachingDataPath( void )
{
	return	std::string() ;
}

// OutputPrediction で出力するパスを取得する
//////////////////////////////////////////////////////////////////////////////
std::string NNMLPShellGANIterator::GetOutputPath( void )
{
	return	std::string() ;
}

// 分類器への偽物教師データを取得する（GAN用）
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellGANIterator::GetCounterfeitData( void )
{
	return	m_pCounterfeit ;
}

// 分類器用の反復器取得
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNMLPShell::Iterator> NNMLPShellGANIterator::GetClassifierIterator( void )
{
	return	m_pClassifier ;
}



//////////////////////////////////////////////////////////////////////////////
// 生成器用反復器
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellGenerativeIterator::NNMLPShellGenerativeIterator
	( const char * pszSourceDir, const char * pszOutputDir,
			const char * pszClassDir, bool formatIndex )
	: NNMLPShellFileIterator( pszSourceDir, pszOutputDir, true ),
		NNMLPShellFileClassifier( pszClassDir, formatIndex )
{
}
