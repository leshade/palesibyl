
#define	__NN_CUDA_DEV__	__device__

#include "nn_cuda_kernel.h"

using namespace Palesibyl ;


// スレッド分割数計算
//////////////////////////////////////////////////////////////////////////////

inline dim3 CalcThreadCount( const NNBufDim& dimDst )
{
	unsigned int xThreads = (unsigned int) dimDst.z ;
	unsigned int yThreads = (unsigned int) cudaMaxThreadCount / xThreads ;
	if ( xThreads >= cudaMaxThreadCount )
	{
		xThreads = cudaMaxThreadCount ;
		yThreads = 1 ;
	}
	else if ( yThreads >= dimDst.x )
	{
		yThreads = (unsigned int) dimDst.x ;
	}
	assert( xThreads * yThreads <= cudaMaxThreadCount ) ;
	return	dim3( xThreads, yThreads ) ;
}


// 値で埋める
//////////////////////////////////////////////////////////////////////////////

__global__ void nnkernel_FillMemory
	( float * pDst, NNBufDim dimDst, float fill, int xThreads, int yThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	bx = blockIdx.x * yThreads + ty ;
	const int	by = blockIdx.y ;
	const int	bi = bx + by * dimDst.x ;

	if ( (bx < dimDst.x) && (by < dimDst.y) && (bi < dimDst.n) )
	{
		for ( int i = tx; i < dimDst.z; i += xThreads )
		{
			pDst[bi * dimDst.z + i] = fill ;
		}
	}
}

void Palesibyl::nncuda_FillMemory
	( float * pDst, NNBufDim dimDst, float fill, cudaStream_t stream )
{
	dim3			threads = CalcThreadCount( dimDst ) ;
	unsigned int	xThreads = threads.x ;
	unsigned int	yThreads = threads.y ;

	dim3	grid( ((unsigned int) dimDst.x + yThreads - 1) / yThreads,
					(unsigned int) dimDst.y ) ;

	nnkernel_FillMemory
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst, fill, xThreads, yThreads ) ;
}


// サンプルごとに pMask[dimDst.z] で乗算する
//////////////////////////////////////////////////////////////////////////////

__global__ void nnkernel_MaskPattern
	( float * pDst, NNBufDim dimDst, const float * pMask, int xThreads, int yThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	bx = blockIdx.x * yThreads + ty ;
	const int	by = blockIdx.y ;
	const int	bi = bx + by * dimDst.x ;

	if ( (bx < dimDst.x) && (by < dimDst.y) && (bi < dimDst.n) )
	{
		for ( int i = tx; i < dimDst.z; i += xThreads )
		{
			float	mask = pMask[i] ;
			float	masked = pDst[bi * dimDst.z + i] * mask ;
			pDst[bi * dimDst.z + i] = masked ;
		}
	}
}

void Palesibyl::nncuda_MaskPattern
	( float * pDst, NNBufDim dimDst, const float * pMask, cudaStream_t stream )
{
	dim3			threads = CalcThreadCount( dimDst ) ;
	unsigned int	xThreads = threads.x ;
	unsigned int	yThreads = threads.y ;

	dim3	grid( ((unsigned int) dimDst.x + yThreads - 1) / yThreads,
					(unsigned int) dimDst.y ) ;

	nnkernel_MaskPattern
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst, pMask, xThreads, yThreads ) ;
}



// 操作クラス（コピー）
//////////////////////////////////////////////////////////////////////////////

class	OperationCopy
{
public:
	static inline __device__ float LoadDst( float * pDst )
	{
		return	0.0f ;
	}
	static inline __device__ float Operate( float dst, float src )
	{
		return	src ;
	}
} ;


// 操作クラス（加算）
//////////////////////////////////////////////////////////////////////////////

class	OperationAdd
{
public:
	static inline __device__ float LoadDst( float * pDst )
	{
		return	*pDst ;
	}
	static inline __device__ float Operate( float dst, float src )
	{
		return	dst + src ;
	}
} ;


// サンプルを移動しながらチャネルをコピー
// （出力先のシフト元が範囲外の場合、ソースをシフトせずに操作）
//////////////////////////////////////////////////////////////////////////////

template <class Op> __global__ void nncuda_ShiftOperationMemory
	( float * pDst, NNBufDim dimDst, size_t iDstChannel,
		float * pSrc, NNBufDim dimSrc, int xShift, int yShift,
		size_t iSrcChannel, size_t nChannelCount,
		float scaleFactor, int xThreads, int yThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	bx = blockIdx.x * yThreads + ty ;
	const int	by = blockIdx.y ;
	const int	bi = bx + by * dimDst.x ;

	int	xSrc = bx - xShift ;
	int	ySrc = by - yShift ;
	if ( (xSrc < 0) || (xSrc >= dimSrc.x) )
	{
		xSrc = bx ;
	}
	if ( (ySrc < 0) || (ySrc >= dimSrc.y) )
	{
		ySrc = by ;
	}
	const int	iSrc = (ySrc * dimSrc.x) + xSrc ;

	if ( (bx < dimDst.x) && (by < dimDst.y) && (bi < dimDst.n) )
	{
		for ( size_t iChannel = tx;
				(iChannel < nChannelCount) && ((iDstChannel + iChannel) < dimDst.z);
				iChannel += xThreads )
		{
			// 出力先の値を準備する
			const int	iDst = bi * dimDst.z + iDstChannel + iChannel ;
			float		dst = Op::LoadDst( pDst + iDst ) ;

			// 入力元の値を準備する
			float	src = 0.0f ;
			if ( (xSrc < dimSrc.x) && (ySrc < dimSrc.y)
				&& ((iSrcChannel + iChannel) < dimSrc.z) )
			{
				src = pSrc[iSrc * dimSrc.z + iSrcChannel + iChannel] ;
			}

			// 出力する
			pDst[iDst] = Op::Operate( dst, src * scaleFactor ) ;
		}
	}
}

void Palesibyl::nncuda_ShiftMoveMemory
	( float * pDst, NNBufDim dimDst, size_t iDstChannel,
		float * pSrc, NNBufDim dimSrc, int xShift, int yShift,
		size_t iSrcChannel, size_t nChannelCount,
		float scaleFactor, cudaStream_t stream )
{
	dim3			threads = CalcThreadCount( dimDst ) ;
	unsigned int	xThreads = threads.x ;
	unsigned int	yThreads = threads.y ;

	dim3	grid( ((unsigned int) dimDst.x + yThreads - 1) / yThreads,
					(unsigned int) dimDst.y ) ;

	nncuda_ShiftOperationMemory<OperationCopy>
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst, iDstChannel,
				pSrc, dimSrc, xShift, yShift,
				iSrcChannel, nChannelCount,
				scaleFactor, xThreads, yThreads ) ;
}

void Palesibyl::nncuda_ShiftAddMemory
	( float * pDst, NNBufDim dimDst, size_t iDstChannel,
		float * pSrc, NNBufDim dimSrc, int xShift, int yShift,
		size_t iSrcChannel, size_t nChannelCount,
		float scaleFactor, cudaStream_t stream )
{
	dim3			threads = CalcThreadCount( dimDst ) ;
	unsigned int	xThreads = threads.x ;
	unsigned int	yThreads = threads.y ;

	dim3	grid( ((unsigned int) dimDst.x + yThreads - 1) / yThreads,
					(unsigned int) dimDst.y ) ;

	nncuda_ShiftOperationMemory<OperationAdd>
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst, iDstChannel,
				pSrc, dimSrc, xShift, yShift,
				iSrcChannel, nChannelCount,
				scaleFactor, xThreads, yThreads ) ;
}


