
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
	__NN_CUDA_DEV__ NNBufDim( const NNBufDim& nnbd )
		: x( nnbd.x ), y( nnbd.y ), z( nnbd.z ), n( nnbd.n ) {}
	__NN_CUDA_DEV__ const NNBufDim& operator = ( const NNBufDim& nnbd )
	{
		x = nnbd.x ;
		y = nnbd.y ;
		z = nnbd.z ;
		n = nnbd.n ;
		return	*this ;
	}
	bool operator == ( const NNBufDim& nnbd ) const
	{
		return	(x == nnbd.x) && (y == nnbd.y) && (z == nnbd.z) && (n == nnbd.n) ;
	}
	bool operator != ( const NNBufDim& nnbd ) const
	{
		return	(x != nnbd.x) || (y != nnbd.y) || (z != nnbd.z) || (n != nnbd.n) ;
	}
} ;


//////////////////////////////////////////////////////////////////////////////
// 損失関数ハイパーパラメータ
//////////////////////////////////////////////////////////////////////////////

struct	NNLossParam
{
	// ※デフォルトの損失関数にパラメータはない
} ;



}

#endif

