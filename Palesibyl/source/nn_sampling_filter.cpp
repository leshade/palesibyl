
#include "nn_perceptron.h"

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// 入力サンプリング・フィルタ関数
//////////////////////////////////////////////////////////////////////////////

std::map< std::string, std::function< std::shared_ptr<NNSamplingFilter>() > >
	NNSamplingFilter::s_mapMakeFilter ;

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNSamplingFilter::NNSamplingFilter
		( int xStride, int yStride, int xOffset, int yOffset,
			int xConv, int yConv, int xPad, int yPad,
			int xPitch, int yPitch,  int xUpScale, int yUpScale )
	: NNSamplingParam
		( xStride, yStride, xOffset, yOffset,
			xConv, yConv, xPad, yPad, xPitch, yPitch, xUpScale, yUpScale )
{
}

// 出力サイズ計算
//////////////////////////////////////////////////////////////////////////////
NNBufDim NNSamplingFilter::CalcOutputDim( const NNBufDim& dimSrc ) const
{
	return	CalcMatrixPointDim( dimSrc ) ;
}

NNBufDim NNSamplingFilter::CalcMatrixPointDim( const NNBufDim& dimSrc ) const
{
	return	NNBufDim( (dimSrc.x + m_xStride + m_xPadding - m_xConv) / m_xStride,
						(dimSrc.y + m_yStride + m_yPadding - m_yConv) / m_yStride, dimSrc.z ) ;
}

size_t NNSamplingFilter::CalcOutputChannels( size_t nMatrixLines ) const
{
	return	nMatrixLines ;
}

size_t NNSamplingFilter::UpSamplingScaleX( void ) const
{
	return	m_xUpScale ;
}

size_t NNSamplingFilter::UpSamplingScaleY( void ) const
{
	return	m_yUpScale ;
}

// CPU での行列勾配計算（加算）
//////////////////////////////////////////////////////////////////////////////
void NNSamplingFilter::cpuCalcMatrixGradient
	( NNMatrix& matGradient, int xPos, int yPos,
		float * pSrcVecBuf, const float * pSrc, NNBufDim dimSrc, size_t zSrcCount,
		float * pDeltaBuf, const float * pDelta, NNBufDim dimDelta )
{
	const size_t	nLines = matGradient.GetLineCount() ;
	const size_t	nColumns = matGradient.GetColumnCount() ;

	cpuSample
		( pSrcVecBuf,
			xPos * (int) UpSamplingScaleX(),
			yPos * (int) UpSamplingScaleY(),
			nColumns, pSrc, dimSrc, zSrcCount ) ;
	cpuBackSample
		( pDeltaBuf, nLines, pDelta, dimDelta, xPos, yPos ) ;

	float *	pGradient = matGradient.GetArray() ;
	for ( size_t j = 0; j < nLines; j ++ )
	{
		for ( size_t k = 0; k < nColumns; k ++ )
		{
			pGradient[k] += pSrcVecBuf[k] * pDeltaBuf[j] ;
		}
		pGradient += nColumns ;
	}
}

void NNSamplingFilter::cpuSparseCalcMatrixGradient
	( NNMatrix& matGradient, int xPos, int yPos,
		float * pSrcVecBuf, const float * pSrc, NNBufDim dimSrc, size_t zSrcCount,
		float * pDeltaBuf, const float * pDelta, NNBufDim dimDelta )
{
	const size_t	nLines = matGradient.GetLineCount() ;
	const size_t	nColumns = matGradient.GetColumnCount() ;

	cpuSample
		( pSrcVecBuf,
			xPos * (int) UpSamplingScaleX(),
			yPos * (int) UpSamplingScaleY(),
			nColumns, pSrc, dimSrc, zSrcCount ) ;
	cpuBackSample
		( pDeltaBuf, nLines, pDelta, dimDelta, xPos, yPos ) ;

	float *	pGradient = matGradient.GetArray() ;
	for ( size_t j = 0; j < nLines; j ++ )
	{
		if ( pDeltaBuf[j] != 0.0f )
		{
			for ( size_t k = 0; k < nColumns; k ++ )
			{
				pGradient[k] += pSrcVecBuf[k] * pDeltaBuf[j] ;
			}
		}
		pGradient += nColumns ;
	}
}

// シリアライズ
//////////////////////////////////////////////////////////////////////////////
void NNSamplingFilter::Serialize( NNSerializer& ser )
{
	const NNSamplingParam&	sp = *this ;
	ser.Write( &sp, sizeof(NNSamplingParam) ) ;
}

// デシリアライズ
//////////////////////////////////////////////////////////////////////////////
bool NNSamplingFilter::Deserialize( NNDeserializer & dsr )
{
	NNSamplingParam&	sp = *this ;
	size_t	nLeftBytes = dsr.GetChunkBytes() - (size_t) dsr.LocalPosition() ;
	dsr.Read( &sp, __min(sizeof(NNSamplingParam),nLeftBytes) ) ;
	return	true ;
}

// フィルタ生成準備
//////////////////////////////////////////////////////////////////////////////
void NNSamplingFilter::InitMake( void )
{
	s_mapMakeFilter.clear() ;
	Register<NNSamplerInjection>( NNBufSampler::SamplerName ) ;
	Register<NNSamplerClamp>( NNBufClampSampler::SamplerName ) ;
	Register<NNSamplerEdge>( NNBufEdgeSampler::SamplerName ) ;
	Register<NNSamplerUpSampler>( NNBufUpSampler::SamplerName ) ;
	Register<NNSamplerUp2x2>( NNBufUpSampler2x2::SamplerName ) ;
	Register<NNConvClampFilter>( NNBufConvClampSampler::SamplerName ) ;
	Register<NNConvEdgeFilter>( NNBufConvEdgeSampler::SamplerName ) ;
	Register<NNSamplerOneHot>( NNBufOneHotSampler::SamplerName ) ;
	Register<NNSamplerSparse>( NNSamplerSparse::SamplerName ) ;
	Register<NNSparseConvFilter>( NNSparseConvFilter::SamplerName ) ;
}

// 関数生成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNSamplingFilter> NNSamplingFilter::Make( const char * pszName )
{
	decltype(s_mapMakeFilter)::iterator iter = s_mapMakeFilter.find(pszName) ;
	assert( iter != s_mapMakeFilter.end() ) ;
	if ( iter != s_mapMakeFilter.end() )
	{
		return	(iter->second)() ;
	}
	return	nullptr ;
}



//////////////////////////////////////////////////////////////////////////////
// One-Hot 入力特殊化
//////////////////////////////////////////////////////////////////////////////

// CPU での１サンプル行列計算（出力座標は出力チャネル操作のみに影響）
//////////////////////////////////////////////////////////////////////////////
void NNSamplerOneHot::cpuMatrix
	( float * pDst, int xDst, int yDst, const NNMatrix& matrix,
		float * pSrcBuf, size_t zSrcChannels,
		const float * pSrc, NNBufDim dimSrc,
		size_t iMatrixBias, size_t nDepthwise )
{
	const size_t	nLines = matrix.GetLineCount() ;
	if ( (xDst >= 0) && (xDst < (int) dimSrc.x)
		&& (yDst >= 0) && (yDst < (int) dimSrc.y) )
	{
		const size_t	one_hot = (size_t) floor( pSrc[((yDst * dimSrc.x) + xDst) * dimSrc.z] ) ;
		if ( one_hot < iMatrixBias )
		{
			const size_t	nColumns = matrix.GetColumnCount() ;
			const float *	pMatrix = matrix.GetConstArray() ;
			for ( size_t i = 0; i < nLines; i ++ )
			{
				pDst[i] = pMatrix[i * nColumns + one_hot] ;
			}
			return ;
		}
	}
	for ( size_t i = 0; i < nLines; i ++ )
	{
		pDst[i] = 0.0f ;
	}
}

// CPU での行列勾配計算（加算）
//////////////////////////////////////////////////////////////////////////////
void NNSamplerOneHot::cpuCalcMatrixGradient
	( NNMatrix& matGradient, int xPos, int yPos,
		float * pSrcVecBuf, const float * pSrc, NNBufDim dimSrc, size_t zSrcCount,
		float * pDeltaBuf, const float * pDelta, NNBufDim dimDelta )
{
	const size_t	nLines = matGradient.GetLineCount() ;
	const size_t	nColumns = matGradient.GetColumnCount() ;

	const int	xSrc = xPos * (int) UpSamplingScaleX() ;
	const int	ySrc = yPos * (int) UpSamplingScaleY() ;
	if ( (xSrc < 0) || (xSrc >= (int) dimSrc.x)
		|| (ySrc < 0) || (ySrc >= (int) dimSrc.y) )
	{
		return ;
	}
	const size_t	one_hot = (size_t) floor( pSrc[((ySrc * dimSrc.x) + xSrc) * dimSrc.z] ) ;
	if ( one_hot >= zSrcCount )
	{
		return ;
	}
	cpuBackSample
		( pDeltaBuf, nLines, pDelta, dimDelta, xPos, yPos ) ;

	float *	pGradient = matGradient.GetArray() ;
	for ( size_t j = 0; j < nLines; j ++ )
	{
		pGradient[one_hot] += pSrcVecBuf[one_hot] * pDeltaBuf[j] ;
		pGradient += nColumns ;
	}
}



