
#ifndef	__NN_SAMPLING_FILTER_H__
#define	__NN_SAMPLING_FILTER_H__

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// 入力サンプリング・フィルタ関数
//////////////////////////////////////////////////////////////////////////////

class	NNSamplingFilter	: public NNSamplingParam
{
protected:
	static std::map
		< std::string,
			std::function
				< std::shared_ptr
					<NNSamplingFilter>() > >	s_mapMakeFilter ;

public:
	// フィルタ生成準備
	static void InitMake( void ) ;
	// 関数生成
	static std::shared_ptr<NNSamplingFilter> Make( const char * pszName ) ;
	// 登録
	template <class T> static void Register( const char * pszName )
	{
		s_mapMakeFilter.insert
			( std::make_pair(std::string(pszName),
							[]() { return std::make_shared<T>(); } ) ) ;
	}

public:
	// 構築関数
	NNSamplingFilter
		( int xStride = 1, int yStride = 1,
			int xOffset = 0, int yOffset = 0,
			int xConv = 1, int yConv = 1,
			int xPad = 0, int yPad = 0,
			int xPitch = 1, int yPitch = 1,
			int xUpScale = 1, int yUpScale = 1 ) ;
	// フィルタ名
	virtual const char * GetSamplerName( void) const = 0 ;
	// 入力層でなければならないか？
	virtual bool MustBeInputLayer( void ) const = 0 ;
	// 出力サイズ計算
	virtual NNBufDim CalcOutputDim( const NNBufDim& dimSrc ) const ;
	virtual NNBufDim CalcMatrixPointDim( const NNBufDim& dimSrc ) const ;
	virtual size_t CalcOutputChannels( size_t nMatrixLines ) const ;
	virtual size_t UpSamplingScaleX( void ) const ;
	virtual size_t UpSamplingScaleY( void ) const ;
	// 入力チャネル数計算
	virtual size_t ConvChannelCount( size_t zSrc, size_t xMatrix ) const = 0 ;
	// チャネル指標変換
	virtual size_t ConvChannelIndex
		( int xSubSrc, int ySubSrc, size_t zSrc, size_t zChannels, int xConv ) = 0 ;
	// サンプリング
	virtual void cpuSample
		( float * pDst, int xDst, int yDst, size_t zChannels,
			const float * pSrc, NNBufDim dimSrc, size_t zSrcCount ) = 0 ;
	// 逆サンプリング
	virtual void cpuBackSample
		( float * pDst, size_t zChannels,
			const float * pSrc, NNBufDim dimSrc, int xSrc, int ySrc ) = 0 ;
	// CPU での１サンプル行列計算（出力座標は出力チャネル操作のみに影響）
	virtual void cpuMatrix
		( float * pDst, int xDst, int yDst, const NNMatrix& matrix,
			float * pSrcBuf, size_t zSrcChannels,
			const float * pSrc, NNBufDim dimSrc,
			size_t iMatrixBias, size_t nDepthwise ) = 0 ;
	// CUDA 行列計算
	virtual void cudaMatrix
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			const float * pMatrix,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			size_t xLeftBounds, int nDepthwise, cudaStream_t stream ) = 0 ;
	// CPU での行列勾配計算（加算）
	virtual void cpuCalcMatrixGradient
		( NNMatrix& matGradient, int xPos, int yPos,
			float * pSrcVecBuf, const float * pSrc, NNBufDim dimSrc, size_t zSrcCount,
			float * pDeltaBuf, const float * pDelta, NNBufDim dimDelta ) ;
	void cpuSparseCalcMatrixGradient
		( NNMatrix& matGradient, int xPos, int yPos,
			float * pSrcVecBuf, const float * pSrc, NNBufDim dimSrc, size_t zSrcCount,
			float * pDeltaBuf, const float * pDelta, NNBufDim dimDelta ) ;
	// CUDA 行列勾配計算
	virtual void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias, int nDepthwise,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc, cudaStream_t stream ) = 0 ;
	// 行列のδ逆伝播
	virtual void cpuAddMatrixDeltaBackAt
		( float * pDstDelta, size_t zDstChannels,
			const float * pSrcDelta, size_t zSrcChannels, size_t nDepthwise,
			const float * pMatrix, size_t nMatrixColumns, int xSubSrc, int ySubSrc ) = 0 ;
	virtual void cudaMatrixDeltaBack
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pMatrix, int xMatrix, int yMatrix, int nDepthwise,
			size_t zSrcChannels, cudaStream_t stream ) = 0 ;
	// シリアライズ
	virtual void Serialize( NNSerializer& ser ) ;
	// デシリアライズ
	virtual bool Deserialize( NNDeserializer & dsr ) ;
} ;

template <class S> class NNGenSamplingFilter	: public NNSamplingFilter
{
public:
	// 構築関数
	NNGenSamplingFilter<S>
		( int xStride = 1, int yStride = 1,
			int xOffset = 0, int yOffset = 0,
			int xConv = 1, int yConv = 1,
			int xPad = 0, int yPad = 0,
			int xPitch = 1, int yPitch = 1,
			int xUpScale = 1, int yUpScale = 1 )
		: NNSamplingFilter
			( xStride, yStride, xOffset, yOffset,
				xConv, yConv, xPad, yPad, xPitch, yPitch, xUpScale, yUpScale ) { }
	// フィルタ名
	virtual const char * GetSamplerName( void) const
	{
		return	S::SamplerName ;
	}
	// 入力層でなければならないか？
	virtual bool MustBeInputLayer( void ) const
	{
		return	S::MustBeInputLayer ;
	}
	// 出力サイズ計算
	virtual NNBufDim CalcOutputDim( const NNBufDim& dimSrc ) const
	{
		return	S::CalcOutputDim( NNSamplingFilter::CalcOutputDim( dimSrc ), *this ) ;
	}
	virtual size_t CalcOutputChannels( size_t nMatrixLines ) const
	{
		return	S::CalcOutputChannels( nMatrixLines, *this ) ;
	}
	// 入力チャネル数計算
	virtual size_t ConvChannelCount( size_t zSrc, size_t xMatrix ) const
	{
		return	S::ConvChannelCount( zSrc, m_xConv, m_yConv, xMatrix ) ;
	}
	// チャネル指標変換
	virtual size_t ConvChannelIndex
		( int xSubSrc, int ySubSrc, size_t zSrc, size_t zChannels, int xConv )
	{
		return	S::ConvChannelIndex( xSubSrc, ySubSrc, zSrc, zChannels, xConv ) ;
	}
	// サンプリング
	virtual void cpuSample
		( float * pDst, int xDst, int yDst, size_t zChannels,
			const float * pSrc, NNBufDim dimSrc, size_t zSrcCount )
	{
		const int	xSrc = xDst * m_xStride + m_xOffset ;
		const int	ySrc = yDst * m_yStride + m_yOffset ;
		for ( size_t z = 0; z < zSrcCount; z ++ )
		{
			pDst[z] = S::Sample( pSrc, dimSrc, xSrc, ySrc, z, *this ) ;
		}
		for ( size_t z = zSrcCount; z < zChannels; z ++ )
		{
			pDst[z] = 1.0f ;
		}
	}
	// 逆サンプリング
	virtual void cpuBackSample
		( float * pDst, size_t zChannels,
			const float * pSrc, NNBufDim dimSrc, int xSrc, int ySrc )
	{
		for ( size_t z = 0; z < zChannels; z ++ )
		{
			pDst[z] = S::BackSample( z, pSrc, dimSrc, xSrc, ySrc, *this ) ;
		}
	}
	// CPU での１サンプル行列計算（出力座標は出力チャネル操作のみに影響）
	virtual void cpuMatrix
		( float * pDst, int xDst, int yDst, const NNMatrix& matrix,
			float * pSrcBuf, size_t zSrcChannels,
			const float * pSrc, NNBufDim dimSrc,
			size_t iMatrixBias, size_t nDepthwise )
	{
		cpuSample( pSrcBuf, xDst, yDst, zSrcChannels, pSrc, dimSrc, iMatrixBias ) ;
		S::cpuMatrix( pDst, xDst, yDst, matrix, pSrcBuf, nDepthwise, iMatrixBias, *this ) ;
	}
	// CUDA 行列計算
	virtual inline void cudaMatrix
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			const float * pMatrix,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
			size_t xLeftBounds, int nDepthwise, cudaStream_t stream )
	{
		S::cudaMatrix
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, xMatrix, yMatrix, iMatrixBias,
				xLeftBounds, nDepthwise, *this, stream ) ;
	}
	// CUDA 行列勾配計算
	virtual void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias, int nDepthwise,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc, cudaStream_t stream )
	{
		S::cudaCalcMatrixGradient
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc,
				nDepthwise, *this, stream ) ;
	}
	// 行列のδ逆伝播
	virtual void cpuAddMatrixDeltaBackAt
		( float * pDstDelta, size_t zDstChannels,
			const float * pSrcDelta, size_t zSrcChannels, size_t nDepthwise,
			const float * pMatrix, size_t nMatrixColumns, int xSubSrc, int ySubSrc )
	{
		for ( size_t z = 0; z < zDstChannels; z ++ )
		{
			size_t	col = S::ConvChannelIndex
							( xSubSrc, ySubSrc, z, zDstChannels, m_xConv ) ;
			float	d = 0.0f ;
			for ( size_t i = (col % nDepthwise); i < zSrcChannels; i += nDepthwise )
			{
				d += pSrcDelta[i] * pMatrix[i * nMatrixColumns + col] ;
			}
			pDstDelta[z] += d ;
		}
	}
	virtual void cudaMatrixDeltaBack
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pMatrix, int xMatrix, int yMatrix, int nDepthwise,
			size_t zSrcChannels, cudaStream_t stream )
	{
		S::cudaMatrixDeltaBack
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix, zSrcChannels,
				nDepthwise, *this, stream ) ;
	}
} ;

template <class S> class NNSparseSamplingFilter	: public NNGenSamplingFilter<S>
{
public:
	// 構築関数
	NNSparseSamplingFilter<S>
		( int xStride = 1, int yStride = 1,
			int xOffset = 0, int yOffset = 0,
			int xConv = 1, int yConv = 1,
			int xPad = 0, int yPad = 0 )
		: NNGenSamplingFilter<S>
			( xStride, yStride, xOffset, yOffset, xConv, yConv, xPad, yPad ) {}
	// CPU での行列勾配計算（加算）
	virtual void cpuCalcMatrixGradient
		( NNMatrix& matGradient, int xPos, int yPos,
			float * pSrcVecBuf, const float * pSrc, NNBufDim dimSrc, size_t zSrcCount,
			float * pDeltaBuf, const float * pDelta, NNBufDim dimDelta )
	{
		NNGenSamplingFilter<S>::cpuSparseCalcMatrixGradient
			( matGradient, xPos, yPos,
				pSrcVecBuf, pSrc, dimSrc, zSrcCount,
				pDeltaBuf, pDelta, dimDelta ) ;
	}
	// 行列のδ逆伝播
	virtual void cpuAddMatrixDeltaBackAt
		( float * pDstDelta, size_t zDstChannels,
			const float * pSrcDelta, size_t zSrcChannels, size_t nDepthwise,
			const float * pMatrix, size_t nMatrixColumns, int xSubSrc, int ySubSrc )
	{
		for ( size_t i = 0; i < zSrcChannels; i ++ )
		{
			if ( pSrcDelta[i] != 0.0f )
			{
				for ( size_t z = 0; z < zDstChannels; z ++ )
				{
					size_t	col = S::ConvChannelIndex
									( xSubSrc, ySubSrc, z, zDstChannels,
										NNGenSamplingFilter<S>::m_xConv ) ;
					pDstDelta[z] += pSrcDelta[i] * pMatrix[i * nMatrixColumns + col] ;
				}
			}
		}
	}
} ;

class	NNSamplerInjection	: public NNGenSamplingFilter<NNBufSampler> {} ;
class	NNSamplerClamp	: public NNGenSamplingFilter<NNBufClampSampler> {} ;
class	NNSamplerEdge	: public NNGenSamplingFilter<NNBufEdgeSampler> {} ;

class	NNSamplerUpSampler	: public NNGenSamplingFilter<NNBufUpSampler>
{
public:
	NNSamplerUpSampler( int xUpScale = 2, int yUpScale = 2 )
	{
		NNSamplingParam::m_xUpScale = (int32_t) xUpScale ;
		NNSamplingParam::m_yUpScale = (int32_t) yUpScale ;
	}
} ;

class	NNSamplerUp2x2	: public NNGenSamplingFilter<NNBufUpSampler2x2>
{
public:
	NNSamplerUp2x2( void )
	{
		NNSamplingParam::m_xUpScale = 2 ;
		NNSamplingParam::m_yUpScale = 2 ;
	}
} ;

class	NNSamplerOneHot	: public NNGenSamplingFilter<NNBufOneHotSampler>
{
public:
	// CPU での１サンプル行列計算（出力座標は出力チャネル操作のみに影響）
	virtual void cpuMatrix
		( float * pDst, int xDst, int yDst, const NNMatrix& matrix,
			float * pSrcBuf, size_t zSrcChannels,
			const float * pSrc, NNBufDim dimSrc,
			size_t iMatrixBias, size_t nDepthwise ) ;
	// CPU での行列勾配計算（加算）
	virtual void cpuCalcMatrixGradient
		( NNMatrix& matGradient, int xPos, int yPos,
			float * pSrcVecBuf, const float * pSrc, NNBufDim dimSrc, size_t zSrcCount,
			float * pDeltaBuf, const float * pDelta, NNBufDim dimDelta ) ;
} ;

class	NNConvClampFilter	: public NNGenSamplingFilter<NNBufConvClampSampler>
{
public:
	NNConvClampFilter
		( int xConv = 1, int yConv = 1, bool padding = true,
			int xStride = 1, int yStride = 1,
			int xOffset = 0, int yOffset = 0 )
		: NNGenSamplingFilter<NNBufConvClampSampler>
			( xStride, yStride,
				xOffset - (padding ? xConv/2 : 0),
				yOffset - (padding ? yConv/2 : 0),
				xConv, yConv,
				(padding ? (xConv - 1) : 0), (padding ? (yConv - 1) : 0) ) { }
} ;

class	NNConvEdgeFilter	: public NNGenSamplingFilter<NNBufConvEdgeSampler>
{
public:
	NNConvEdgeFilter
		( int xConv = 1, int yConv = 1, bool padding = false,
			int xStride = 1, int yStride = 1,
			int xOffset = 0, int yOffset = 0 )
		: NNGenSamplingFilter<NNBufConvEdgeSampler>
			( xStride, yStride,
				xOffset - (padding ? xConv/2 : 0),
				yOffset - (padding ? yConv/2 : 0),
				xConv, yConv,
				(padding ? (xConv - 1) : 0), (padding ? (yConv - 1) : 0) ) { }
} ;

class	NNSamplerSparse	: public NNSparseSamplingFilter<NNBufSampler>
{
public:
	constexpr static const char	SamplerName[] = "spars" ;

	// フィルタ名
	virtual const char * GetSamplerName( void) const
	{
		return	SamplerName ;
	}
	// CUDA 行列勾配計算
	virtual void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias, int nDepthwise,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Edge_Sp
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc, *this, stream ) ;
	}
	// CUDA 行列のδ逆伝播
	virtual void cudaMatrixDeltaBack
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pMatrix, int xMatrix, int yMatrix, int nDepthwise,
			size_t zSrcChannels, cudaStream_t stream )
	{
		nncuda_Matrix_DeltaBack_Injection_Sp
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix, zSrcChannels,
				nDepthwise, *this, stream ) ;
	}
} ;

class	NNSparseConvFilter	: public NNSparseSamplingFilter<NNBufConvEdgeSampler>
{
public:
	constexpr static const char	SamplerName[] = "conv_spars" ;

	NNSparseConvFilter
		( int xConv = 1, int yConv = 1, bool padding = false,
			int xStride = 1, int yStride = 1,
			int xOffset = 0, int yOffset = 0 )
		: NNSparseSamplingFilter<NNBufConvEdgeSampler>
			( xStride, yStride,
				xOffset - (padding ? xConv/2 : 0),
				yOffset - (padding ? yConv/2 : 0),
				xConv, yConv,
				(padding ? (xConv - 1) : 0), (padding ? (yConv - 1) : 0) ) { }
	// フィルタ名
	virtual const char * GetSamplerName( void) const
	{
		return	SamplerName ;
	}
	// CUDA 行列勾配計算
	virtual void cudaCalcMatrixGradient
		( float * pGradient, NNBufDim dimGradient,
			size_t xGradientBlockSize, size_t yGradientBlockSize,
			size_t xMatrix, size_t yMatrix, size_t iMatrixBias, int nDepthwise,
			const float * pDelta, NNBufDim dimDelta,
			const float * pSrc, NNBufDim dimSrc, cudaStream_t stream )
	{
		nncuda_CalcMatrixGradient_Conv_Edge_Sp
			( pGradient, dimGradient,
				xGradientBlockSize, yGradientBlockSize,
				xMatrix, yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc, *this, stream ) ;
	}
	// CUDA 行列のδ逆伝播
	virtual void cudaMatrixDeltaBack
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pMatrix, int xMatrix, int yMatrix, int nDepthwise,
			size_t zSrcChannels, cudaStream_t stream )
	{
		nncuda_Matrix_DeltaBack_Conv_Sp
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix, zSrcChannels,
				nDepthwise, *this, stream ) ;
	}
} ;

}

#endif
