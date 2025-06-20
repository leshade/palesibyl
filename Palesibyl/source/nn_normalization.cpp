
#include "nn_normalization.h"
#include "nn_cuda_kernel.h"

#ifdef	min
	#undef	min
#endif
#ifdef	max
	#undef	max
#endif

using namespace Palesibyl ;



//////////////////////////////////////////////////////////////////////////////
// 正規化作業バッファ
//////////////////////////////////////////////////////////////////////////////

size_t NNNormalizationFilter::WorkBuf::GetBufferBytes( void ) const
{
	return	bufParameter.GetBufferBytes()
			+ bufVariance.GetBufferBytes()
			+ bufAggregation[0].GetBufferBytes()
			+ bufAggregation[1].GetBufferBytes()
			+ bufAggregation[2].GetBufferBytes()
			+ bufGradient.GetBufferBytes() ;
}

size_t NNNormalizationFilter::WorkBuf::GetCudaBufferBytes( void ) const
{
	return	bufParameter.GetCudaBufferBytes()
			+ bufVariance.GetCudaBufferBytes()
			+ bufAggregation[0].GetCudaBufferBytes()
			+ bufAggregation[1].GetCudaBufferBytes()
			+ bufAggregation[2].GetCudaBufferBytes()
			+ bufGradient.GetCudaBufferBytes() ;
}



//////////////////////////////////////////////////////////////////////////////
// 正規化
//////////////////////////////////////////////////////////////////////////////

std::map< std::string, std::function< std::shared_ptr<NNNormalizationFilter>() > >
	NNNormalizationFilter::s_mapMakeFilter ;

// 関数生成準備
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::InitMake( void )
{
	s_mapMakeFilter.clear() ;
	Register<NNLayerNormalization>() ;
	Register<NNGroupNormalization>() ;
	Register<NNInstanceNormalization>() ;
}

// 関数生成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNNormalizationFilter> NNNormalizationFilter::Make( const char * pszName )
{
	decltype(s_mapMakeFilter)::iterator iter = s_mapMakeFilter.find(pszName) ;
	assert( iter != s_mapMakeFilter.end() ) ;
	if ( iter != s_mapMakeFilter.end() )
	{
		return	(iter->second)() ;
	}
	return	nullptr ;
}

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNNormalizationFilter::NNNormalizationFilter( void )
{
}

NNNormalizationFilter::NNNormalizationFilter( const Hyperparameter& hyparam )
	: m_hyparam( hyparam )
{
}

// ハイパーパラメータ
//////////////////////////////////////////////////////////////////////////////
const NNNormalizationFilter::Hyperparameter&
		NNNormalizationFilter::GetHyperparameter( void ) const
{
	return	m_hyparam ;
}

void NNNormalizationFilter::SetHyperparameter
		( const NNNormalizationFilter::Hyperparameter& hyparam )
{
	m_hyparam = hyparam ;
}

// 生成
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::CreateFilter( size_t zChannels )
{
	const size_t	nChannel = NormalizeChannelCount( zChannels ) ;
	const size_t	zSampling = SamplingChannels( zChannels ) ;

	m_params.resize( nChannel ) ;
	m_aggregation.resize( nChannel ) ;
	m_vecIndices.resize( zChannels ) ;

	for ( size_t i = 0; i < nChannel; i ++ )
	{
		m_params.at(i).scale = 1.0 ;
		m_params.at(i).shift = 0.0f ;
	}

	for ( size_t i = 0; i < nChannel; i ++ )
	{
		m_aggregation.at(i).sum = 0.0f ;
		m_aggregation.at(i).sum2 = 0.0f ;
		m_aggregation.at(i).num = 0.0f ;
	}

	for ( size_t i = 0; i < zChannels; i ++ )
	{
		assert( (i / zSampling) < m_params.size() ) ;
		m_vecIndices.at(i) = (uint32_t) (i / zSampling) ;
	}
}

// アフィンパラメータ乗算
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::ScaleParameter( float scale )
{
	for ( size_t i = 0; i < m_params.size(); i ++ )
	{
		m_params.at(i).scale *= scale ;
		m_params.at(i).shift *= scale ;
	}
}

// 作業バッファ準備
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::PrepareWorkBuf
	( NNNormalizationFilter::WorkBuf& bufWork,
		const NNBufDim& dimSample, bool forLearning, const NNLoopStream& stream ) const
{
	bufWork.nThreads = stream.m_ploop.GetThreadCount() ;
	bufWork.nChannels = m_params.size() ;
	bufWork.transParam = false ;
	bufWork.forLearning = forLearning ;
	bufWork.dimSample = dimSample ;
	bufWork.vecVariance.resize( bufWork.nChannels ) ;
	bufWork.vecAggregation.resize( bufWork.nChannels * bufWork.nThreads ) ;
	bufWork.vecGradients.resize( bufWork.nChannels * bufWork.nThreads ) ;

	if ( stream.m_useCuda )
	{
		bufWork.bufParameter.Create
			( 1, 1, bufWork.nChannels * 2, 0, NNBuffer::cudaAllocate ) ;
		bufWork.bufVariance.Create
			( 1, 1, bufWork.nChannels * 2, 0, NNBuffer::cudaAllocate ) ;

		bufWork.iAggregate = 0 ;

		if ( forLearning )
		{
			NNBufDim	dimAggr( nncuda_CalcAggregateSize(dimSample.x),
									nncuda_CalcAggregateSize(dimSample.y),
									bufWork.nChannels * 3 ) ;
			bufWork.bufAggregation[0].Create( dimAggr, NNBuffer::cudaAllocate ) ;
			//
			NNBufDim	dimAggr2( nncuda_CalcAggregateSize(dimAggr.x),
									nncuda_CalcAggregateSize(dimAggr.y),
									dimAggr.z ) ;
			bufWork.bufAggregation[1].Create( dimAggr2, NNBuffer::cudaAllocate ) ;
			//
			bufWork.bufAggregation[2].Create
				( 1, 1, bufWork.nChannels * 3, 0, NNBuffer::cudaAllocate ) ;

			NNBufDim	dimGradient( nncuda_CalcAggregateSize(dimSample.x),
									dimSample.y, dimSample.z * 2 ) ;
			bufWork.bufGradient.Create( dimGradient, NNBuffer::cudaAllocate ) ;
		}

		// 分布計算
		float *	pVariance = bufWork.bufVariance.GetBuffer() ;
		for ( size_t i = 0; i < m_aggregation.size(); i ++ )
		{
			const Aggregation&	aggregation = m_aggregation.at(i) ;
			const float			n = std::max( aggregation.num, 1.0e-7f ) ;
			const float			mean = aggregation.sum / n ;
			pVariance[0] = mean ;
			pVariance[1] = 1.0f / sqrt( std::max( aggregation.sum2 / n - mean * mean, 1.0e-10f ) ) ;
			pVariance += 2 ;
		}
		bufWork.bufVariance.CheckOverun() ;
		bufWork.bufVariance.CudaAsyncToDevice( stream.m_cudaStream ) ;
	}

	ResetWorkBuf( bufWork ) ;
}

void NNNormalizationFilter::PrepareGradBuf
	( NNNormalizationFilter::GradientBuf& bufGrad ) const
{
	bufGrad.vecGradients.resize( m_params.size() ) ;

	bufGrad.ResetGradient() ;
}

// 正規化チャネル数
//////////////////////////////////////////////////////////////////////////////
size_t NNNormalizationFilter::NormalizeChannelCount( size_t zChannels ) const
{
	const size_t	zSampling = SamplingChannels( zChannels ) ;
	assert( zSampling >= 1 ) ;
	return	(zChannels + zSampling - 1) / zSampling ;
}

// 作業バッファ初期化
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::ResetWorkBuf( NNNormalizationFilter::WorkBuf& bufWork ) const
{
	bufWork.transParam = false ;

	assert( bufWork.vecAggregation.size() >= m_aggregation.size() ) ;
	for ( size_t i = 0; i < bufWork.vecAggregation.size(); i ++ )
	{
		Aggregation&	aggregation = bufWork.vecAggregation.at(i) ;
		aggregation.sum = 0.0f ;
		aggregation.sum2 = 0.0f ;
		aggregation.num = 0.0f ;
	}

	for ( size_t i = 0; i < bufWork.vecGradients.size(); i ++ )
	{
		Gradient&	gradient = bufWork.vecGradients.at(i) ;
		gradient.scale = 0.0f ;
		gradient.shift = 0.0f ;
		gradient.num = 0 ;
	}
}

void NNNormalizationFilter::GradientBuf::ResetGradient( void )
{
	for ( size_t i = 0; i < vecGradients.size(); i ++ )
	{
		Gradient&	gradient = vecGradients.at(i) ;
		gradient.scale = 0.0f ;
		gradient.shift = 0.0f ;
		gradient.num = 0 ;
	}
}

// 分布を計算して正規化
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::cpuNormalize
	( NNBuffer& bufSample,
		NNNormalizationFilter::WorkBuf& bufWork,
		NNLoopStream& stream, size_t xLeftBounds ) const
{
	const NNBufDim	dimSample = bufSample.GetSize() ;
	const size_t	nThreads = stream.m_ploop.GetThreadCount() ;

	assert( nThreads == bufWork.nThreads ) ;
	assert( bufWork.vecVariance.size() >= m_aggregation.size() ) ;
	assert( m_vecIndices.size() >= dimSample.z ) ;

	// 分布計算
	for ( size_t i = 0; i < m_aggregation.size(); i ++ )
	{
		const Aggregation&	aggregation = m_aggregation.at(i) ;
		MeanAndVariance&	mav = bufWork.vecVariance.at(i) ;
		const float			n = std::max( aggregation.num, 1.0e-7f ) ;
		mav.mean = aggregation.sum / n ;
		mav.rcpvar = 1.0f / sqrt( std::max( aggregation.sum2 / n
										- mav.mean * mav.mean, 1.0e-10f ) ) ;
	}

	// 正規化
	stream.m_ploop.Loop( 0, dimSample.y, [&]( size_t iThread, size_t y )
	{
		const uint32_t *		pIndices = m_vecIndices.data() ;
		const Parameter *		param = m_params.data() ;
		const MeanAndVariance *	pmav = bufWork.vecVariance.data() ;
		float *					pSample = bufSample.GetBufferAt( xLeftBounds, y ) ;
		for ( size_t x = xLeftBounds; x < dimSample.x; x ++ )
		{
			for ( size_t z = 0; z < dimSample.z; z ++ )
			{
				const size_t	i = (size_t) pIndices[z] ;
				const float		s = pSample[z] ;
				const float		x = (s - pmav[i].mean) * pmav[i].rcpvar ;
				pSample[z] = x * param[i].scale + param[i].shift ;
			}
			pSample += dimSample.z ;
		}
	} ) ;

	bufSample.CheckOverun() ;
}

void NNNormalizationFilter::cudaNormalize
	( NNBuffer& bufSample,
		NNNormalizationFilter::WorkBuf& bufWork,
		NNLoopStream& stream, size_t xLeftBounds ) const
{
	// パラメータ転送
	if ( !bufWork.transParam )
	{
		float *	pParameter = bufWork.bufParameter.GetBuffer() ;
		for ( size_t i = 0; i < m_params.size(); i ++ )
		{
			const Parameter&	param = m_params.at(i) ;
			pParameter[0] = param.scale ;
			pParameter[1] = param.shift ;
			pParameter += 2 ;
		}
		bufWork.bufParameter.CheckOverun() ;
		bufWork.bufParameter.CudaAsyncToDevice( stream.m_cudaStream ) ;
		stream.m_cudaStream.VerifySync() ;
		bufWork.transParam = true ;
	}

	// 正規化
	bufSample.CommitCuda() ;

	nncuda_Normalize
		( bufSample.GetCudaPtr(), bufSample.GetSize(), xLeftBounds,
			bufWork.bufParameter.GetCudaPtr(),
			bufWork.bufVariance.GetCudaPtr(),
			(int) SamplingChannels( m_vecIndices.size() ), stream.m_cudaStream ) ;
	stream.m_cudaStream.VerifySync() ;
}

// サンプルを集計
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::cpuAggregateSample
	( NNNormalizationFilter::WorkBuf& bufWork,
		const NNBuffer& bufSample, NNLoopStream& stream )
{
	const NNBufDim	dimSample = bufSample.GetSize() ;
	const size_t	nThreads = stream.m_ploop.GetThreadCount() ;

	assert( nThreads == bufWork.nThreads ) ;
	assert( m_vecIndices.size() >= dimSample.z ) ;
	assert( NormalizeChannelCount(dimSample.z) == bufWork.nChannels ) ;
	assert( bufWork.vecAggregation.size() >= bufWork.nChannels * nThreads ) ;

	for ( size_t i = 0; i < bufWork.vecAggregation.size(); i ++ )
	{
		Aggregation&	aggregation = bufWork.vecAggregation.at(i) ;
		aggregation.sum = 0.0f ;
		aggregation.sum2 = 0.0f ;
		aggregation.num = 0.0f ;
	}

	stream.m_ploop.Loop( 0, dimSample.y, [&]( size_t iThread, size_t y )
	{
		const uint32_t *	pIndices = m_vecIndices.data() ;
		Aggregation *		pAggregation = bufWork.vecAggregation.data()
											+ (iThread * bufWork.nChannels) ;
		const float *		pSample = bufSample.GetConstBufferAt( 0, y ) ;
		for ( size_t x = 0; x < dimSample.x; x ++ )
		{
			for ( size_t z = 0; z < dimSample.z; z ++ )
			{
				const size_t	i = (size_t) pIndices[z] ;
				const float		s = pSample[z] ;
				pAggregation[i].sum += s ;
				pAggregation[i].sum2 += s * s ;
				pAggregation[i].num += 1.0f ;
			}
			pSample += dimSample.z ;
		}
	} ) ;

	assert( m_aggregation.size() >= bufWork.nChannels ) ;

	for ( size_t i = 0; i < bufWork.nChannels; i ++ )
	{
		Aggregation&	aggregation = m_aggregation.at(i) ;
		for ( size_t j = 0; j < bufWork.nThreads; j ++ )
		{
			Aggregation&	aggr = bufWork.vecAggregation.at(j * bufWork.nChannels + i) ;
			aggregation.sum += aggr.sum ;
			aggregation.sum2 += aggr.sum2 ;
			aggregation.num += aggr.num ;
		}
	}
}

void NNNormalizationFilter::cudaAggregateSample
	( NNNormalizationFilter::WorkBuf& bufWork,
			const NNBuffer& bufSample, NNLoopStream& stream )
{
	const NNBufDim	dimSample = bufSample.GetSize() ;

	// 集計
	size_t		iAggregate = 0 ;
	NNBufDim	dimAggregate( nncuda_CalcAggregateSize(dimSample.x),
								nncuda_CalcAggregateSize(dimSample.y),
								bufWork.nChannels * 3 ) ;
	assert( dimAggregate.x <= bufWork.bufAggregation[0].GetSize().x ) ;
	assert( dimAggregate.y <= bufWork.bufAggregation[0].GetSize().y ) ;
	assert( dimAggregate.z <= bufWork.bufAggregation[0].GetSize().z ) ;
	nncuda_AggregateSample
		( bufWork.bufAggregation[0].GetCudaPtr(), dimAggregate,
			bufSample.GetCudaPtr(), dimSample,
			(int) SamplingChannels( m_vecIndices.size() ), stream.m_cudaStream ) ;
	stream.m_cudaStream.VerifySync() ;

	for ( ; ; )
	{
		size_t		iNext = (iAggregate + 1) % 2 ;
		NNBufDim	dimNext( nncuda_CalcAggregateSize(dimAggregate.x),
								nncuda_CalcAggregateSize(dimAggregate.y),
								dimAggregate.z ) ;
		if ( dimNext.x * dimNext.y <= 1 )
		{
			break ;
		}
		assert( dimNext.x * dimNext.y * dimNext.z
					<= bufWork.bufAggregation[iNext].GetSize().n
						* bufWork.bufAggregation[iNext].GetSize().z ) ;
		nncuda_AggregateSample2
			( bufWork.bufAggregation[iNext].GetCudaPtr(), dimNext,
				bufWork.bufAggregation[iAggregate].GetCudaPtr(), dimAggregate,
				stream.m_cudaStream ) ;
		stream.m_cudaStream.VerifySync() ;

		iAggregate = iNext ;
		dimAggregate = dimNext ;
	}
	bufWork.iAggregate = iAggregate ;
	bufWork.dimAggregate = dimAggregate ;
	bufWork.bufAggregation[iAggregate].CudaAsyncFromDevice( stream.m_cudaStream ) ;
	stream.m_cudaStream.VerifySync() ;

	// 分布計算
	float *	pAggregation = bufWork.bufAggregation[2].GetBuffer() ;
	for ( size_t i = 0; i < m_aggregation.size(); i ++ )
	{
		const Aggregation&	aggregation = m_aggregation.at(i) ;
		pAggregation[0] = aggregation.sum ;
		pAggregation[1] = aggregation.sum2 ;
		pAggregation[2] = aggregation.num ;
		pAggregation += 3 ;
	}
	bufWork.bufAggregation[2].CheckOverun() ;
	bufWork.bufAggregation[2].CudaAsyncToDevice( stream.m_cudaStream ) ;
	stream.m_cudaStream.VerifySync() ;

	nncuda_CalcDistribution
		( bufWork.bufVariance.GetCudaPtr(), bufWork.bufVariance.GetSize(),
			bufWork.bufAggregation[2].GetCudaPtr(), bufWork.bufAggregation[2].GetSize(),
			bufWork.bufAggregation[iAggregate].GetCudaPtr(), dimAggregate,
			stream.m_cudaStream ) ;
	stream.m_cudaStream.VerifySync() ;
}

// δ逆伝播と勾配計算
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::cpuDeltaBack
	( NNNormalizationFilter::WorkBuf& bufWork, NNBuffer& bufDelta,
		const NNBuffer& bufDstSample, NNLoopStream& stream ) const
{
	const NNBufDim	dimDelta = bufDelta.GetSize() ;
	const NNBufDim	dimSample = bufDstSample.GetSize() ;
	const size_t	nThreads = stream.m_ploop.GetThreadCount() ;

	assert( nThreads == bufWork.nThreads ) ;
	assert( m_vecIndices.size() >= dimSample.z ) ;
	assert( dimDelta == dimSample ) ;
	assert( NormalizeChannelCount(dimSample.z) == bufWork.nChannels ) ;
	assert( bufWork.vecGradients.size() >= bufWork.nChannels * nThreads ) ;

	stream.m_ploop.Loop( 0, dimDelta.y, [&]( size_t iThread, size_t y )
	{
		const Parameter *		param = m_params.data() ;
		const uint32_t *		pIndices = m_vecIndices.data() ;
		const MeanAndVariance *	pmav = bufWork.vecVariance.data() ;
		Gradient *				pGradient = bufWork.vecGradients.data()
											+ (iThread * bufWork.nChannels) ;
		const float *			pSample = bufDstSample.GetConstBufferAt( 0, y ) ;
		float *					pDelta = bufDelta.GetBufferAt( 0, y ) ;
		for ( size_t x = 0; x < dimDelta.x; x ++ )
		{
			for ( size_t z = 0; z < dimDelta.z; z ++ )
			{
				const size_t	i = (size_t) pIndices[z] ;
				const float		s = pSample[z] ;
				const float		r = (fabs(param[i].scale) > 1.0e-5f)
										? (1.0f / param[i].scale) : 1.0f ;
				const float		x = (s - param[i].shift) * r ;
				const float		d = pDelta[z] ;

				// 勾配計算
				pGradient[i].scale += d * x ;
				pGradient[i].shift += d ;
				pGradient[i].num ++ ;

				// δ逆伝播
				pDelta[z] = d * param[i].scale * pmav[i].rcpvar ;
			}
			pDelta += dimDelta.z ;
		}
	} ) ;

	bufDelta.CheckOverun() ;
}

void NNNormalizationFilter::cudaDeltaBack
	( NNNormalizationFilter::WorkBuf& bufWork, NNBuffer& bufDelta,
			const NNBuffer& bufDstSample, NNLoopStream& stream ) const
{
	const NNBufDim	dimDelta = bufDelta.GetSize() ;
	const NNBufDim	dimDstSample = bufDstSample.GetSize() ;
	const NNBufDim	dimGradient = bufWork.bufGradient.GetSize() ;

	assert( dimDelta == dimDstSample ) ;

	nncuda_NormDeltaBack
		( bufDelta.GetCudaPtr(), dimDelta,
			bufWork.bufGradient.GetCudaPtr(), dimGradient,
			bufDstSample.GetCudaPtr(), dimDstSample,
			bufWork.bufParameter.GetCudaPtr(),
			bufWork.bufVariance.GetCudaPtr(),
			(int) SamplingChannels( m_vecIndices.size() ), stream.m_cudaStream ) ;
	stream.m_cudaStream.VerifySync() ;

	bufWork.bufGradient.CudaAsyncFromDevice( stream.m_cudaStream ) ;
	stream.m_cudaStream.VerifySync() ;
}

// 更新用行列勾配を統合する
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::cpuIntegrateGradient
	( NNNormalizationFilter::GradientBuf& bufGrad,
		NNNormalizationFilter::WorkBuf& bufWork )
{
	assert( m_params.size() >= bufWork.nChannels ) ;
	assert( bufGrad.vecGradients.size() >= bufWork.nChannels ) ;
	assert( bufWork.vecGradients.size() >= bufWork.nChannels * bufWork.nThreads ) ;

	for ( size_t i = 0; i < bufWork.nChannels; i ++ )
	{
		Gradient&	gradient = bufGrad.vecGradients.at(i) ;
		for ( size_t j = 0; j < bufWork.nThreads; j ++ )
		{
			Gradient&	grad = bufWork.vecGradients.at(j * bufWork.nChannels + i) ;
			gradient.scale += grad.scale ;
			gradient.shift += grad.shift ;
			gradient.num += grad.num ;
			grad.scale = 0.0f ;
			grad.shift = 0.0f ;
			grad.num = 0 ;
		}
	}
}

void NNNormalizationFilter::cudaIntegrateGradient
	( NNNormalizationFilter::GradientBuf& bufGrad,
		NNNormalizationFilter::WorkBuf& bufWork )
{
	// 勾配加算
	const NNBufDim		dimGradient = bufWork.bufGradient.GetSize() ;
	const float *		pGradientBuf = bufWork.bufGradient.GetConstBuffer() ;
	Gradient *			pGradient = bufGrad.vecGradients.data() ;
	const uint32_t *	pIndices = m_vecIndices.data() ;

	assert( dimGradient.z == m_vecIndices.size() * 2 ) ;

	for ( size_t i = 0; i < dimGradient.n; i ++ )
	{
		for ( size_t j = 0; j < dimGradient.z; j += 2 )
		{
			const size_t	z = j / 2 ;
			const size_t	i = (size_t) pIndices[z] ;
			Gradient&		grad = pGradient[i] ;
			grad.scale += pGradientBuf[j] ;
			grad.shift += pGradientBuf[j + 1] ;
		}
		pGradientBuf += dimGradient.z ;
	}

	for ( size_t z = 0; z < m_vecIndices.size(); z ++ )
	{
		const size_t	i = (size_t) pIndices[z] ;
		Gradient&		grad = pGradient[i] ;
		grad.num += bufWork.dimSample.n ;
	}

	// サンプル集計
	const float *	pAggrBuf = bufWork.bufAggregation
									[bufWork.iAggregate].GetConstBuffer() ;
	const NNBufDim	dimAggr = bufWork.dimAggregate ;
	Aggregation *	pAggregation = m_aggregation.data() ;

	assert( dimAggr.z == m_aggregation.size() * 3 ) ;
	assert( bufWork.nChannels == m_aggregation.size() ) ;

	for ( size_t i = 0; i < dimAggr.n; i ++ )
	{
		for ( size_t z = 0; z < bufWork.nChannels; z ++ )
		{
			pAggregation[z].sum += pAggrBuf[0] ;
			pAggregation[z].sum2 += pAggrBuf[1] ;
			pAggregation[z].num += pAggrBuf[2] ;
			pAggrBuf += 3 ;
		}
	}
}

// 勾配をパラメータに更新する
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::AddGradient
	( const NNNormalizationFilter::GradientBuf& bufGrad, float deltaRate, float l2reg )
{
	assert( m_params.size() >= bufGrad.vecGradients.size() ) ;

	const float	rate = std::min( m_hyparam.delta * deltaRate + m_hyparam.deltac, 0.1f ) ;

	// パラメータ更新
	for ( size_t i = 0; i < bufGrad.vecGradients.size(); i ++ )
	{
		const Gradient&	gradient = bufGrad.vecGradients.at(i) ;
		if ( gradient.num > 0 )
		{
			Parameter&	param = m_params.at(i) ;
			const float	r = rate / (float) gradient.num ;
			param.scale -= gradient.scale * r ;
			param.shift -= gradient.shift * r
							+ param.shift * rate * l2reg ;
		}
		if ( m_hyparam.flags & flagZeroBias )
		{
			m_params.at(i).shift = 0.0f ;
		}
	}

	// 集計値をスケーリング
	ScaleAggregate( m_hyparam.alpha ) ;
}

void NNNormalizationFilter::GradientBuf::AddGradient
	( const NNNormalizationFilter::GradientBuf& bufSrc )
{
	assert( vecGradients.size() == bufSrc.vecGradients.size() ) ;
	for ( size_t i = 0; i < vecGradients.size(); i ++ )
	{
		Gradient&		gradDst = vecGradients.at(i) ;
		const Gradient&	gradSrc = bufSrc.vecGradients.at(i) ;
		gradDst.scale += gradSrc.scale ;
		gradDst.shift += gradSrc.shift ;
		gradDst.num += gradSrc.num ;
	}
}

// エポック開始時の集計データスケーリング
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::OnBeginEpoch( void )
{
	ScaleAggregate( m_hyparam.beta ) ;
}

// エポック終了時の集計データスケーリング
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::OnEndEpoch
	( NNNormalizationFilter::WorkBuf& bufWork )
{
}

// 集計値のスケーリング
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::ScaleAggregate( float scale )
{
	for ( size_t i = 0; i < m_aggregation.size(); i ++ )
	{
		Aggregation&	aggregation = m_aggregation.at(i) ;
		aggregation.sum *= scale ;
		aggregation.sum2 *= scale ;
		aggregation.num *= scale ;
	}
}

// シリアライズ
//////////////////////////////////////////////////////////////////////////////
void NNNormalizationFilter::Serialize( NNSerializer& ser )
{
	uint32_t	nParams = (uint32_t) m_params.size() ;
	ser.Write( &nParams, sizeof(nParams) ) ;
	ser.Write( m_params.data(), sizeof(Parameter) * nParams ) ;
	ser.Write( m_aggregation.data(), sizeof(Aggregation) * nParams ) ;
	//
	uint32_t	nHyparamSize = sizeof(m_hyparam) ;
	ser.Write( &nHyparamSize, sizeof(nHyparamSize) ) ;
	ser.Write( &m_hyparam, nHyparamSize ) ;
	//
	uint32_t	nIndices = (uint32_t) m_vecIndices.size() ;
	ser.Write( &nIndices, sizeof(nIndices) ) ;
	ser.Write( m_vecIndices.data(), sizeof(uint32_t) * nIndices ) ;
}

// デシリアライズ
//////////////////////////////////////////////////////////////////////////////
bool NNNormalizationFilter::Deserialize( NNDeserializer & dsr )
{
	uint32_t	nParams = 0 ;
	dsr.Read( &nParams, sizeof(nParams) ) ;
	m_params.resize( (size_t) nParams ) ;
	m_aggregation.resize( (size_t) nParams ) ;
	dsr.Read( m_params.data(), sizeof(Parameter) * nParams ) ;
	dsr.Read( m_aggregation.data(), sizeof(Aggregation) * nParams ) ;
	//
	uint32_t	nHyparamSize = sizeof(m_hyparam) ;
	dsr.Read( &nHyparamSize, sizeof(nHyparamSize) ) ;
	dsr.Read( &m_hyparam, std::min((size_t)nHyparamSize,sizeof(m_hyparam)) ) ;
	dsr.Skip( nHyparamSize - std::min((size_t)nHyparamSize,sizeof(m_hyparam)) ) ;
	//
	uint32_t	nIndices = 0 ;
	dsr.Read( &nIndices, sizeof(nIndices) ) ;
	m_vecIndices.resize( (size_t) nIndices ) ;
	dsr.Read( m_vecIndices.data(), sizeof(uint32_t) * nIndices ) ;
	return	true ;
}



//////////////////////////////////////////////////////////////////////////////
// レイヤー正規化基底
//////////////////////////////////////////////////////////////////////////////

// 標本範囲チャネル数
//////////////////////////////////////////////////////////////////////////////
size_t NNLayerNormalization::SamplingChannels( size_t zChannels ) const
{
	return	zChannels ;
}



//////////////////////////////////////////////////////////////////////////////
// グループ正規化
//////////////////////////////////////////////////////////////////////////////

// 標本範囲チャネル数
//////////////////////////////////////////////////////////////////////////////
size_t NNGroupNormalization::SamplingChannels( size_t zChannels ) const
{
	return	m_zSampling ;
}

// シリアライズ
//////////////////////////////////////////////////////////////////////////////
void NNGroupNormalization::Serialize( NNSerializer& ser )
{
	NNNormalizationFilter::Serialize( ser ) ;

	uint32_t	zSampling = (uint32_t) m_zSampling ;
	ser.Write( &zSampling, sizeof(zSampling) ) ;
}

// デシリアライズ
//////////////////////////////////////////////////////////////////////////////
bool NNGroupNormalization::Deserialize( NNDeserializer & dsr )
{
	NNNormalizationFilter::Deserialize( dsr ) ;

	uint32_t	zSampling = (uint32_t) m_zSampling ;
	dsr.Read( &zSampling, sizeof(zSampling) ) ;
	m_zSampling = (size_t) zSampling ;
	return	true ;
}

