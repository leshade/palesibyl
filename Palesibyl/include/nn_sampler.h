
#ifndef	__NN_SAMPLER_H__
#define	__NN_SAMPLER_H__

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// サンプリング関数
//////////////////////////////////////////////////////////////////////////////
class	NNBufSampler
{
public:
	constexpr static const char		SamplerName[] = "injection" ;
	constexpr static const bool		MustBeInputLayer = false ;	// 入力層でなければならないか？

	// 入力チャネル数から行列へ入力する要素数計算
	static inline size_t ConvChannelCount
			( size_t zSrc, int xConv, int yConv, size_t xMatrix )
	{
		return	zSrc ;
	}
	// 行列の出力サイズから実際の出力サイズを計算
	static inline NNBufDim CalcOutputDim( const NNBufDim& dimSrc, const NNSamplingParam& sp )
	{
		return	dimSrc ;
	}
	static inline size_t CalcOutputChannels( size_t nMatrixLines, const NNSamplingParam& sp )
	{
		return	nMatrixLines ;
	}
	// 畳み込み元の相対座標とチャネルから行列の列番号を計算
	static inline __NN_CUDA_DEV__ size_t ConvChannelIndex
		( int xSubSrc, int ySubSrc, size_t zSrc, size_t zChannels, int xConv )
	{
		return	zSrc ;
	}
	// 出力先座標とチャネルから行列の行番号を計算
	static inline __NN_CUDA_DEV__ size_t SampleMatrixLine
		( int xDst, int yDst, size_t zChannel, size_t zChannelCount, const NNSamplingParam& sp )
	{
		return	zChannel ;
	}
	// 行列への入力をサンプリング
	static inline __NN_CUDA_DEV__ float Sample
		( const float * pSrc, NNBufDim dimSrc,
			int xSrc, int ySrc, size_t chSrc, const NNSamplingParam& sp )
	{
		return	(chSrc >= dimSrc.z) ? 1.0f :
					pSrc[(ySrc * dimSrc.x + xSrc) * dimSrc.z + chSrc] ;
	}
	// 行列の出力をサンプリング
	static inline __NN_CUDA_DEV__ float BackSample
		( size_t chOut, const float * pSrc, NNBufDim dimSrc,
					int xSrc, int ySrc, const NNSamplingParam& sp )
	{
		return	(chOut >= dimSrc.z) ? 0.0f :
					pSrc[(ySrc * dimSrc.x + xSrc) * dimSrc.z + chOut] ;
	}
	// CPU での１サンプル行列計算（出力座標は出力チャネル操作のみに影響）
	static inline void cpuMatrix
		( float * pDst, int xDst, int yDst, const NNMatrix& matrix,
			const float * pSrc, size_t nDepthwise,
			size_t iMatrixBias, const NNSamplingParam& sp )
	{
		if ( nDepthwise <= 1 )
		{
			matrix.ProductVector( pDst, pSrc ) ;
		}
		else
		{
			matrix.DepthwiseProductVector( pDst, pSrc, nDepthwise, iMatrixBias ) ;
		}
	}
	// CUDA での行列計算
	static inline void cudaMatrix
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			const float * pMatrix,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_Matrix_Clamp
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				nDepthwise, sp, stream ) ;
	}
	// CUDA での更新用行列勾配計算
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Clamp
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
	}
	// CUDA でのδ逆伝播
	static inline void cudaMatrixDeltaBack
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pMatrix,
			int xMatrix, int yMatrix, size_t zSrcChannels,
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_Matrix_DeltaBack_Injection
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix, zSrcChannels, nDepthwise, sp, stream ) ;
	}
} ;


// サンプリング関数（端の値で延長）
//////////////////////////////////////////////////////////////////////////////
class	NNBufClampSampler	: public NNBufSampler
{
public:
	constexpr static const char	SamplerName[] = "clamp" ;

	static inline __NN_CUDA_DEV__ float Sample
		( const float * pSrc, NNBufDim dimSrc,
			int xSrc, int ySrc, size_t chSrc, const NNSamplingParam& sp )
	{
		if ( chSrc >= dimSrc.z )
		{
			return	1.0f ;	// バイアス項
		}
		const int	x = (xSrc < 0) ? 0 : ((xSrc >= (int) dimSrc.x) ? (int) dimSrc.x - 1 : xSrc) ;
		const int	y = (ySrc < 0) ? 0 : ((ySrc >= (int) dimSrc.y) ? (int) dimSrc.y - 1 : ySrc) ;
		return	pSrc[((y * dimSrc.x) + x) * dimSrc.z + chSrc] ;
	}
} ;


// サンプリング関数（範囲外は０）
//////////////////////////////////////////////////////////////////////////////
class	NNBufEdgeSampler	: public NNBufSampler
{
public:
	constexpr static const char	SamplerName[] = "edge" ;

	static inline __NN_CUDA_DEV__ float Sample
		( const float * pSrc, NNBufDim dimSrc,
			int xSrc, int ySrc, size_t chSrc, const NNSamplingParam& sp )
	{
		if ( chSrc >= dimSrc.z )
		{
			return	1.0f ;	// バイアス項
		}
		if ( (xSrc < 0) || (xSrc >= (int) dimSrc.x)
			|| (ySrc < 0) || (ySrc >= (int) dimSrc.y) )
		{
			return	0.0f ;
		}
		return	pSrc[((ySrc * dimSrc.x) + xSrc) * dimSrc.z + chSrc] ;
	}
	static inline void cudaMatrix
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			const float * pMatrix,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias, 
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_Matrix_Edge
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				nDepthwise, sp, stream ) ;
	}
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Edge
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
	}
} ;


// 畳み込みサンプリング関数
//////////////////////////////////////////////////////////////////////////////
template <class S> class	NNBufConvSampler	: public S
{
public:
	static inline size_t ConvChannelCount
		( size_t zSrc, int xConv, int yConv, size_t xMatrix )
	{
		return	zSrc * xConv * yConv ;
	}
	static inline __NN_CUDA_DEV__ float Sample
		( const float * pSrc, NNBufDim dimSrc,
			int xSrc, int ySrc, size_t chSrc, const NNSamplingParam& sp )
	{
		const size_t	iConv = chSrc / dimSrc.z ;
		const size_t	zSub = chSrc - iConv * dimSrc.z ;
		const int		ySub = (int) iConv / sp.m_xConv ;
		const int		xSub = (int) iConv - ySub * sp.m_xConv ;
		return	S::Sample( pSrc, dimSrc, xSrc + xSub * sp.m_xPitch,
										ySrc + ySub * sp.m_yPitch, zSub, sp ) ;
	}
	static inline __NN_CUDA_DEV__ size_t ConvChannelIndex
		( int xSubSrc, int ySubSrc, size_t zSrc, size_t zChannels, int xConv )
	{
		return	(size_t) (ySubSrc * xConv + xSubSrc) * zChannels + zSrc ;
	}
	static inline void cudaMatrixDeltaBack
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pMatrix,
			int xMatrix, int yMatrix, size_t zSrcChannels,
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_Matrix_DeltaBack_Conv
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix, zSrcChannels, nDepthwise, sp, stream ) ;
	}
} ;

class	NNBufConvClampSampler	: public NNBufConvSampler<NNBufClampSampler>
{
public:
	constexpr static const char	SamplerName[] = "conv_clamp" ;

	static inline void cudaMatrix
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			const float * pMatrix,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias, 
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_Matrix_Conv_Clamp
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				nDepthwise, sp, stream ) ;
	}
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Conv_Clamp
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
	}
} ;

class	NNBufConvEdgeSampler	: public NNBufConvSampler<NNBufEdgeSampler>
{
public:
	constexpr static const char	SamplerName[] = "conv_edge" ;

	static inline void cudaMatrix
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			const float * pMatrix,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_Matrix_Conv_Edge
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				nDepthwise, sp, stream ) ;
	}
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Conv_Edge
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
	}
} ;


// アップサンプリング
//////////////////////////////////////////////////////////////////////////////
class NNBufUpSampler	: public NNBufEdgeSampler
{
public:
	constexpr static const char	SamplerName[] = "upsampler" ;

	static inline __NN_CUDA_DEV__ NNBufDim CalcOutputDim( const NNBufDim& dimSrc, const NNSamplingParam& sp )
	{
		return	NNBufDim( dimSrc.x * (size_t) sp.m_xUpScale,
							dimSrc.y * (size_t) sp.m_yUpScale,
							dimSrc.z / (size_t) (sp.m_xUpScale * sp.m_yUpScale) ) ;
	}
	static inline size_t CalcOutputChannels( size_t nMatrixLines, const NNSamplingParam& sp )
	{
		return	nMatrixLines / (size_t) (sp.m_xUpScale * sp.m_yUpScale) ;
	}
	static inline __NN_CUDA_DEV__ size_t SampleMatrixLine
		( int xDst, int yDst, size_t zChannel, size_t zChannelCount, const NNSamplingParam& sp )
	{
		return	((yDst % sp.m_yUpScale) * sp.m_xUpScale
					+ (xDst % sp.m_xUpScale)) * zChannelCount + zChannel ;
	}
	static inline __NN_CUDA_DEV__ float Sample
		( const float * pSrc, NNBufDim dimSrc,
			int xSrc, int ySrc, size_t chSrc, const NNSamplingParam& sp )
	{
		return	NNBufEdgeSampler::Sample
					( pSrc, dimSrc,
						xSrc / sp.m_xUpScale,
						ySrc / sp.m_yUpScale, chSrc, sp ) ;
	}
	static inline __NN_CUDA_DEV__ float BackSample
		( size_t chOut, const float * pSrc, NNBufDim dimSrc,
					int xSrc, int ySrc, const NNSamplingParam& sp )
	{
		size_t		iUp = chOut / dimSrc.z ;
		size_t		chSrc = chOut - iUp * dimSrc.z ;
		const int	yUpOdd = (int) (iUp / sp.m_xUpScale) ;
		const int	xUpOdd = (int) (iUp - yUpOdd * sp.m_xUpScale) ;
		return	NNBufEdgeSampler::BackSample
				( chSrc, pSrc, dimSrc,
					xSrc * sp.m_xUpScale + xUpOdd,
					ySrc * sp.m_yUpScale + yUpOdd, sp ) ;
	}
	static inline void cpuMatrix
		( float * pDst, int xDst, int yDst, const NNMatrix& matrix,
			const float * pSrc, size_t nDepthwise,
			size_t iMatrixBias, const NNSamplingParam& sp )
	{
		const size_t	nDstChannels = matrix.GetLineCount() / (sp.m_xUpScale * sp.m_yUpScale) ;
		const size_t	iDstOffset = ((yDst % sp.m_yUpScale) * sp.m_xUpScale
										+ (xDst % sp.m_xUpScale)) * nDstChannels ;
		matrix.ProductVectorLines
			( pDst, iDstOffset, nDstChannels, pSrc, nDepthwise, iMatrixBias ) ;
	}
	static inline void cudaMatrix
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			const float * pMatrix,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_Matrix_UpSampler
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				nDepthwise, sp, stream ) ;
	}
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_UpSampler
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
	}
	static inline void cudaMatrixDeltaBack
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pMatrix,
			int xMatrix, int yMatrix, size_t zSrcChannels,
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_Matrix_DeltaBack_UpSampler
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix, zSrcChannels, nDepthwise, sp, stream ) ;
	}
} ;

class	NNBufUpSampler2x2	: public NNBufUpSampler
{
public:
	constexpr static const char	SamplerName[] = "up2x2" ;
} ;


// サンプリング関数（One-Hot 入力）
//（※この層は正しくδ逆伝播出来ない為、入力層にしか使用してはならない）
//////////////////////////////////////////////////////////////////////////////
class	NNBufOneHotSampler	: public NNBufSampler
{
public:
	constexpr static const char	SamplerName[] = "onehot" ;
	constexpr static const bool	MustBeInputLayer = true ;	// 入力層でなければならないか？

	static inline size_t ConvChannelCount
		( size_t zSrc, int xConv, int yConv, size_t xMatrix )
	{
		return	xMatrix ;
	}
	static inline __NN_CUDA_DEV__ float Sample
		( const float * pSrc, NNBufDim dimSrc,
			int xSrc, int ySrc, size_t chSrc, const NNSamplingParam& sp )
	{
		if ( (xSrc < 0) || (xSrc >= (int) dimSrc.x)
			|| (ySrc < 0) || (ySrc >= (int) dimSrc.y) )
		{
			return	0.0f ;
		}
		const float	one_hot = floor( pSrc[((ySrc * dimSrc.x) + xSrc) * dimSrc.z] ) ;
		return	(one_hot == (float) chSrc) ? 1.0f : 0.0f ;
	}
	static inline void cudaMatrix
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			const float * pMatrix,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_Matrix_OneHot
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				nDepthwise, sp, stream ) ;
	}
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_OneHot
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
	}
} ;


}

#endif
