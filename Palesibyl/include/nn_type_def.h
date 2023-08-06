
#ifndef	__NN_TYPE_DEF_H__
#define	__NN_TYPE_DEF_H__

#ifndef	__NN_CUDA_DEV__
// CUDA コードを生成する場合には __NN_CUDA_DEV__ に __device__ を予め定義しておく
#define	__NN_CUDA_DEV__
#endif

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// バッファサイズ
//////////////////////////////////////////////////////////////////////////////

struct	NNBufDim
{
	size_t	x, y, z ;	// 幅・高さ・チャネル数
	size_t	n ;			// 最大サンプル数（通常は x * y）

	__NN_CUDA_DEV__ NNBufDim( size_t dx = 1, size_t dy = 1, size_t dz = 1 )
		: x( dx ), y( dy ), z( dz ), n( dx * dy ) {}
	__NN_CUDA_DEV__ NNBufDim( const NNBufDim& bd )
		: x( bd.x ), y( bd.y ), z( bd.z ), n( bd.n ) {}
	__NN_CUDA_DEV__ const NNBufDim& operator = ( const NNBufDim& bd )
	{
		x = bd.x ;
		y = bd.y ;
		z = bd.z ;
		n = bd.n ;
		return	*this ;
	}
	bool operator == ( const NNBufDim& bd ) const
	{
		return	(x == bd.x) && (y == bd.y) && (z == bd.z) && (n == bd.n) ;
	}
	bool operator != ( const NNBufDim& bd ) const
	{
		return	(x != bd.x) || (y != bd.y) || (z != bd.z) || (n != bd.n) ;
	}
} ;

struct	NNBufDimCompareLess
{
	bool operator ()(const NNBufDim& x, const NNBufDim& y) const
	{
		return	(x.n < y.n)
				|| ((x.n == y.n)
					&& ((x.z < y.z)
						|| ((x.z == y.z)
							&& ((x.y < y.y)
								|| ((x.y == y.y)
									&& (x.x < y.x) )
								)
							)
						)
					) ;
	}
} ;



//////////////////////////////////////////////////////////////////////////////
// 損失関数ハイパーパラメータ
//////////////////////////////////////////////////////////////////////////////

struct	NNLossParam
{
	// ※デフォルトの損失関数にパラメータはない
} ;

struct	NNLossParam2
{
	float	lossFactor ;
	float	deltaFactor ;

	NNLossParam2( void ) : lossFactor(1.0f), deltaFactor(1.0f) {}
	NNLossParam2( float lf, float df ) : lossFactor(lf), deltaFactor(df) {}
} ;



//////////////////////////////////////////////////////////////////////////////
// サンプリング・パラメータ
//////////////////////////////////////////////////////////////////////////////

struct	NNSamplingParam
{
	int32_t	m_xStride, m_yStride ;		// サンプリング座標 = stride * 出力座標 + offset
	int32_t	m_xOffset, m_yOffset ;
	int32_t	m_xConv, m_yConv ;			// 畳み込みサイズ
	int32_t	m_xPadding, m_yPadding ;	// 出力サイズを変えない場合: m_xConv-1, m_yConv-1
	int32_t	m_xPitch, m_yPitch ;		// 畳み込みピッチ
	int32_t	m_xUpScale, m_yUpScale ;	// アップサンプリング用

	__NN_CUDA_DEV__ NNSamplingParam
		( int xStride = 1, int yStride = 1,
			int xOffset = 0, int yOffset = 0,
			int xConv = 1, int yConv = 1,
			int xPad = 0, int yPad = 0,
			int xPitch = 1, int yPitch = 1,
			int xUpScale = 1, int yUpScale = 1 )
		: m_xStride( (int32_t)xStride ), m_yStride( (int32_t)yStride ),
			m_xOffset( (int32_t)xOffset ), m_yOffset( (int32_t)yOffset ),
			m_xConv( (int32_t)xConv ), m_yConv( (int32_t)yConv ),
			m_xPadding( (int32_t)xPad ), m_yPadding( (int32_t)yPad ),
			m_xPitch( (int32_t)xPitch ), m_yPitch( (int32_t)yPitch ),
			m_xUpScale( (int32_t)xUpScale ), m_yUpScale( (int32_t)yUpScale ) { }
	__NN_CUDA_DEV__ NNSamplingParam( const NNSamplingParam& sp )
		: m_xStride( sp.m_xStride ), m_yStride( sp.m_yStride ),
			m_xOffset( sp.m_xOffset ), m_yOffset( sp.m_yOffset ),
			m_xConv( sp.m_xConv ), m_yConv( sp.m_xConv ),
			m_xPadding( sp.m_xPadding ), m_yPadding( sp.m_yPadding ),
			m_xPitch( sp.m_xPitch ), m_yPitch( sp.m_yPitch ),
			m_xUpScale( sp.m_xUpScale ), m_yUpScale( sp.m_yUpScale ) {}
	__NN_CUDA_DEV__ const NNSamplingParam& operator = ( const NNSamplingParam& sp )
	{
		m_xStride = sp.m_xStride ;
		m_yStride = sp.m_yStride ;
		m_xOffset = sp.m_xOffset ;
		m_yOffset = sp.m_yOffset ;
		m_xConv = sp.m_xConv ;
		m_yConv = sp.m_yConv ;
		m_xPadding = sp.m_xPadding ;
		m_yPadding = sp.m_yPadding ;
		m_xPitch = sp.m_xPitch ;
		m_yPitch = sp.m_yPitch ;
		m_xUpScale = sp.m_xUpScale ;
		m_yUpScale = sp.m_yUpScale ;
		return	*this ;
	}
	// ストライド情報
	NNSamplingParam * SetStride
		( int xStride, int yStride,
			int xOffset = 0, int yOffset = 0 )
	{
		m_xStride = (int32_t) xStride ;
		m_yStride = (int32_t) yStride ;
		m_xOffset = (int32_t) xOffset ;
		m_yOffset = (int32_t) yOffset ;
		return	this ;
	}
	// 畳み込み情報
	NNSamplingParam * SetConvolution
		( int xConv, int yConv, bool padding = true )
	{
		m_xConv = (int32_t) xConv ;
		m_yConv = (int32_t) yConv ;
		m_xPadding = padding ? ((int32_t) xConv - 1) : 0 ;
		m_yPadding = padding ? ((int32_t) yConv - 1) : 0 ;
		return	this ;
	}
	// 畳み込みピッチ・スケーリング
	NNSamplingParam * ScaleConvPitch( int xScale, int yScale )
	{
		m_xOffset *= (int32_t) xScale ;
		m_yOffset *= (int32_t) yScale ;
		m_xPitch *= (int32_t) xScale ;
		m_yPitch *= (int32_t) yScale ;
		return	this ;
	}
	// アップサンプリング
	NNSamplingParam * SetUpsampling( int xUpScale, int yUpScale )
	{
		m_xUpScale = (int32_t) xUpScale ;
		m_yUpScale = (int32_t) yUpScale ;
		return	this ;
	}

} ;



}

#endif

