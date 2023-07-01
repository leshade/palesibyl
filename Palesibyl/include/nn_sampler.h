
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

	constexpr static const size_t	UpSamplingScaleX = 1 ;
	constexpr static const size_t	UpSamplingScaleY = 1 ;

	// 入力チャネル数から行列へ入力する要素数計算
	static inline size_t ConvChannelCount
			( size_t zSrc, int xConv, int yConv, size_t xMatrix )
	{
		return	zSrc ;
	}
	// 行列の出力サイズから実際の出力サイズを計算
	static inline NNBufDim CalcOutputDim( const NNBufDim& dimSrc )
	{
		return	dimSrc ;
	}
	static inline size_t CalcOutputChannels( size_t nMatrixLines )
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
		( int xDst, int yDst, size_t zChannel, size_t zChannelCount )
	{
		return	zChannel ;
	}
	// 行列への入力をサンプリング
	static inline __NN_CUDA_DEV__ float Sample
		( const float * pSrc, NNBufDim dimSrc,
			int xSrc, int ySrc, size_t chSrc, int xConv )
	{
		return	(chSrc >= dimSrc.z) ? 1.0f :
					pSrc[(ySrc * dimSrc.x + xSrc) * dimSrc.z + chSrc] ;
	}
	// 行列の出力をサンプリング
	static inline __NN_CUDA_DEV__ float BackSample
		( size_t chOut, const float * pSrc, NNBufDim dimSrc, int xSrc, int ySrc )
	{
		return	(chOut >= dimSrc.z) ? 0.0f :
					pSrc[(ySrc * dimSrc.x + xSrc) * dimSrc.z + chOut] ;
	}
	// CPU での１サンプル行列計算（出力座標は出力チャネル操作のみに影響）
	static inline void cpuMatrix
		( float * pDst, int xDst, int yDst, const NNMatrix& matrix,
			const float * pSrc, size_t nDepthwise, size_t iMatrixBias )
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
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_Clamp
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				xStride, yStride, xOffset, yOffset,
				nDepthwise, xConv, yConv, stream ) ;
	}
	// CUDA での更新用行列勾配計算
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Clamp
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc,
				xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
	}
	// CUDA でのδ逆伝播
	static inline void cudaMatrixDeltaBack
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pMatrix,
			int xMatrix, int yMatrix, size_t zSrcChannels,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_DeltaBack_Injection
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix, zSrcChannels,
				xStride, yStride, xOffset, yOffset, nDepthwise, xConv, yConv, stream ) ;
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
			int xSrc, int ySrc, size_t chSrc, int xConv )
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
			int xSrc, int ySrc, size_t chSrc, int xConv )
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
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_Edge
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				xStride, yStride, xOffset, yOffset,
				nDepthwise, xConv, yConv, stream ) ;
	}
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Edge
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc,
				xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
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
			int xSrc, int ySrc, size_t chSrc, int xConv )
	{
		const size_t	iConv = chSrc / dimSrc.z ;
		const size_t	zSub = chSrc - iConv * dimSrc.z ;
		const int		ySub = (int) iConv / xConv ;
		const int		xSub = (int) iConv - ySub * xConv ;
		return	S::Sample( pSrc, dimSrc, xSrc + xSub, ySrc + ySub, zSub, xConv ) ;
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
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_DeltaBack_Conv
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix, zSrcChannels,
				xStride, yStride, xOffset, yOffset, nDepthwise, xConv, yConv, stream ) ;
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
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_Conv_Clamp
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				xStride, yStride, xOffset, yOffset,
				nDepthwise, xConv, yConv, stream ) ;
	}
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Conv_Clamp
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc,
				xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
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
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_Conv_Edge
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				xStride, yStride, xOffset, yOffset,
				nDepthwise, xConv, yConv, stream ) ;
	}
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Conv_Edge
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc,
				xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
	}
} ;


// アップサンプリング
//////////////////////////////////////////////////////////////////////////////
template <int SX, int SY>  class NNBufUpSampler	: public NNBufEdgeSampler
{
public:
	constexpr static const size_t	UpSamplingScaleX = SX ;
	constexpr static const size_t	UpSamplingScaleY = SY ;

	static inline __NN_CUDA_DEV__ NNBufDim CalcOutputDim( const NNBufDim& dimSrc )
	{
		return	NNBufDim( dimSrc.x * SX, dimSrc.y * SY, dimSrc.z / (SX*SY) ) ;
	}
	static inline size_t CalcOutputChannels( size_t nMatrixLines )
	{
		return	nMatrixLines / (SX*SY) ;
	}
	static inline __NN_CUDA_DEV__ size_t SampleMatrixLine
		( int xDst, int yDst, size_t zChannel, size_t zChannelCount )
	{
		return	((yDst % SY) * SX + (xDst % SX)) * zChannelCount + zChannel ;
	}
	static inline __NN_CUDA_DEV__ float Sample
		( const float * pSrc, NNBufDim dimSrc,
			int xSrc, int ySrc, size_t chSrc, int xConv )
	{
		return	NNBufEdgeSampler::Sample
					( pSrc, dimSrc, xSrc / SX, ySrc / SY, chSrc, xConv ) ;
	}
	static inline __NN_CUDA_DEV__ float BackSample
		( size_t chOut, const float * pSrc, NNBufDim dimSrc, int xSrc, int ySrc )
	{
		size_t		iUp = chOut / dimSrc.z ;
		size_t		chSrc = chOut - iUp * dimSrc.z ;
		const int	yUpOdd = (int) (iUp / SX) ;
		const int	xUpOdd = (int) (iUp - yUpOdd * SX) ;
		return	NNBufEdgeSampler::BackSample
				( chSrc, pSrc, dimSrc, xSrc * SX + xUpOdd, ySrc * SY + yUpOdd ) ;
	}
	static inline void cpuMatrix
		( float * pDst, int xDst, int yDst, const NNMatrix& matrix,
			const float * pSrc, size_t nDepthwise, size_t iMatrixBias )
	{
		const size_t	nDstChannels = matrix.GetLineCount() / (SX*SY) ;
		const size_t	iDstOffset = ((yDst % SY) * SX + (xDst % SX)) * nDstChannels ;
		matrix.ProductVectorLines
			( pDst, iDstOffset, nDstChannels, pSrc, nDepthwise, iMatrixBias ) ;
	}
} ;

class	NNBufUpSampler2x2	: public NNBufUpSampler<2,2>
{
public:
	constexpr static const char	SamplerName[] = "up2x2" ;

	static inline void cudaMatrix
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			const float * pMatrix,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias, 
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_Up2x2
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				xStride, yStride, xOffset, yOffset,
				nDepthwise, xConv, yConv, stream ) ;
	}
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Up2x2
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc,
				xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
	}
	static inline void cudaMatrixDeltaBack
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pMatrix,
			int xMatrix, int yMatrix, size_t zSrcChannels,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_DeltaBack_Up2x2
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix, zSrcChannels,
				xStride, yStride, xOffset, yOffset, nDepthwise, xConv, yConv, stream ) ;
	}
} ;

class	NNBufUpSampler4x4	: public NNBufUpSampler<4,4>
{
public:
	constexpr static const char	SamplerName[] = "up4x4" ;

	static inline void cudaMatrix
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			const float * pMatrix,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias, 
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_Up4x4
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				xStride, yStride, xOffset, yOffset,
				nDepthwise, xConv, yConv, stream ) ;
	}
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Up4x4
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc,
				xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
	}
	static inline void cudaMatrixDeltaBack
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pMatrix,
			int xMatrix, int yMatrix, size_t zSrcChannels,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_DeltaBack_Up4x4
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix, zSrcChannels,
				xStride, yStride, xOffset, yOffset, nDepthwise, xConv, yConv, stream ) ;
	}
} ;

class	NNBufUpSampler8x8	: public NNBufUpSampler<8,8>
{
public:
	constexpr static const char	SamplerName[] = "up8x8" ;

	static inline void cudaMatrix
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			const float * pMatrix,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias, 
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_Up8x8
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				xStride, yStride, xOffset, yOffset,
				nDepthwise, xConv, yConv, stream ) ;
	}
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Up8x8
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc,
				xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
	}
	static inline void cudaMatrixDeltaBack
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pMatrix,
			int xMatrix, int yMatrix, size_t zSrcChannels,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_DeltaBack_Up8x8
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix, zSrcChannels,
				xStride, yStride, xOffset, yOffset, nDepthwise, xConv, yConv, stream ) ;
	}
} ;

class	NNBufUpSampler16x16	: public NNBufUpSampler<16,16>
{
public:
	constexpr static const char	SamplerName[] = "up16x16" ;

	static inline void cudaMatrix
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			const float * pMatrix,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias, 
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_Up16x16
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				xStride, yStride, xOffset, yOffset,
				nDepthwise, xConv, yConv, stream ) ;
	}
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Up16x16
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc,
				xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
	}
	static inline void cudaMatrixDeltaBack
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pMatrix,
			int xMatrix, int yMatrix, size_t zSrcChannels,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_DeltaBack_Up16x16
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix, zSrcChannels,
				xStride, yStride, xOffset, yOffset, nDepthwise, xConv, yConv, stream ) ;
	}
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
			int xSrc, int ySrc, size_t chSrc, int xConv )
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
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_Matrix_OneHot
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				xStride, yStride, xOffset, yOffset,
				nDepthwise, xConv, yConv, stream ) ;
	}
	static inline void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc,
			int xStride, int yStride, int xOffset, int yOffset,
			int nDepthwise, int xConv, int yConv, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_OneHot
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc,
				xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
	}
	// ※現状 CUDA 実装は特殊化していないので高速化しません（CPU のみ高速化）
} ;


}

#endif
