
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


// 矩形の外側を値で埋める
//////////////////////////////////////////////////////////////////////////////

__global__ void nnkernel_FillMemory
	( float * pDst, NNBufDim dimDst,
		size_t xLeft, size_t yTop,
		size_t xRight, size_t yBottom,
		float fill, int xThreads, int yThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	bx = blockIdx.x * yThreads + ty ;
	const int	by = blockIdx.y ;
	const int	bi = bx + by * dimDst.x ;

	if ( (bx < dimDst.x) && (by < dimDst.y) && (bi < dimDst.n)
		&& ((bx < xLeft) || (bx >= xRight)
			|| (by < yTop) || (by >= yBottom)) )
	{
		for ( int i = tx; i < dimDst.z; i += xThreads )
		{
			pDst[bi * dimDst.z + i] = fill ;
		}
	}
}

void Palesibyl::nncuda_FillExterior
	( float * pDst, NNBufDim dimDst,
		size_t xLeft, size_t yTop,
		size_t xRight, size_t yBottom, float fill, cudaStream_t stream )
{
	dim3			threads = CalcThreadCount( dimDst ) ;
	unsigned int	xThreads = threads.x ;
	unsigned int	yThreads = threads.y ;

	dim3	grid( ((unsigned int) dimDst.x + yThreads - 1) / yThreads,
					(unsigned int) dimDst.y ) ;

	nnkernel_FillMemory
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst, xLeft, yTop, xRight, yBottom, fill, xThreads, yThreads ) ;
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


// 操作クラス（乗算）
//////////////////////////////////////////////////////////////////////////////

class	OperationMul
{
public:
	static inline __device__ float Operate( float src1, float src2 )
	{
		return	src1 * src2 ;
	}
} ;


// サンプルを移動しながらチャネルをコピー
// （出力先のシフト元が範囲外の場合、ソースをシフトせずに操作）
//////////////////////////////////////////////////////////////////////////////

template <class Op> __global__ void nncuda_ShiftOperationMemory
	( float * pDst, NNBufDim dimDst,
		size_t xDstOffset, size_t yDstOffset,
		size_t nDstWidth, size_t nDstHeight, size_t iDstChannel,
		float * pSrc, NNBufDim dimSrc, int xShift, int yShift,
		size_t iSrcChannel, size_t nChannelCount,
		float scaleFactor, int xThreads, int yThreads )
{
	const size_t	tx = threadIdx.x ;
	const size_t	ty = threadIdx.y ;
	const size_t	bx0 = blockIdx.x * yThreads + ty ;
	const size_t	by0 = blockIdx.y ;
	const size_t	bx = bx0 + xDstOffset ;
	const size_t	by = by0 + yDstOffset ;
	const size_t	bi = bx + by * dimDst.x ;

	int	xSrc = bx0 - xShift ;
	int	ySrc = by0 - yShift ;
	if ( (xSrc < 0) || (xSrc >= dimSrc.x) )
	{
		xSrc = bx0 ;
	}
	if ( (ySrc < 0) || (ySrc >= dimSrc.y) )
	{
		ySrc = by0 ;
	}
	const int	iSrc = (ySrc * dimSrc.x) + xSrc ;

	if ( (bx0 < nDstWidth) && (bx < dimDst.x) && (by < dimDst.y) && (bi < dimDst.n) )
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
	( float * pDst, NNBufDim dimDst,
		size_t xDstOffset, size_t yDstOffset, size_t iDstChannel,
		size_t nDstWidth, size_t nDstHeight,
		float * pSrc, NNBufDim dimSrc, int xShift, int yShift,
		size_t iSrcChannel, size_t nChannelCount,
		float scaleFactor, cudaStream_t stream )
{
	dim3			threads = CalcThreadCount( dimDst ) ;
	unsigned int	xThreads = threads.x ;
	unsigned int	yThreads = threads.y ;

	assert( xDstOffset + nDstWidth <= dimDst.x ) ;
	assert( yDstOffset + nDstHeight <= dimDst.y ) ;
	dim3	grid( ((unsigned int) nDstWidth + yThreads - 1) / yThreads,
					(unsigned int) nDstHeight ) ;

	nncuda_ShiftOperationMemory<OperationCopy>
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst, xDstOffset, yDstOffset,
				nDstWidth, nDstHeight, iDstChannel,
				pSrc, dimSrc, xShift, yShift,
				iSrcChannel, nChannelCount,
				scaleFactor, xThreads, yThreads ) ;
}

void Palesibyl::nncuda_ShiftAddMemory
	( float * pDst, NNBufDim dimDst,
		size_t xDstOffset, size_t yDstOffset, size_t iDstChannel,
		size_t nDstWidth, size_t nDstHeight,
		float * pSrc, NNBufDim dimSrc, int xShift, int yShift,
		size_t iSrcChannel, size_t nChannelCount,
		float scaleFactor, cudaStream_t stream )
{
	dim3			threads = CalcThreadCount( dimDst ) ;
	unsigned int	xThreads = threads.x ;
	unsigned int	yThreads = threads.y ;

	assert( xDstOffset + nDstWidth <= dimDst.x ) ;
	assert( yDstOffset + nDstHeight <= dimDst.y ) ;
	dim3	grid( ((unsigned int) nDstWidth + yThreads - 1) / yThreads,
					(unsigned int) nDstHeight ) ;

	nncuda_ShiftOperationMemory<OperationAdd>
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst, xDstOffset, yDstOffset,
				nDstWidth, nDstHeight, iDstChannel,
				pSrc, dimSrc, xShift, yShift,
				iSrcChannel, nChannelCount,
				scaleFactor, xThreads, yThreads ) ;
}



// 単純計算
//////////////////////////////////////////////////////////////////////////////

template <class Op> __global__ void nncuda_PrimitiveOperation
	( float * pDst, NNBufDim dimDst,
		const float * pSrc0, NNBufDim dimSrc0,
		size_t iChannel0, int xShift0, int yShift0,
		const float * pSrc1, NNBufDim dimSrc1,
		size_t iChannel1, int xShift1, int yShift1,
		size_t xLeftBounds, int xThreads, int yThreads )
{
	const size_t	tx = threadIdx.x ;
	const size_t	ty = threadIdx.y ;
	const size_t	bx = blockIdx.x * yThreads + ty + xLeftBounds ;
	const size_t	by = blockIdx.y ;
	const size_t	bi = bx + by * dimDst.x ;

	int	xSrc0 = bx - xShift0 ;
	int	ySrc0 = by - yShift0 ;
	if ( (xSrc0 < 0) || (xSrc0 >= dimSrc0.x) )
	{
		xSrc0 = bx ;
	}
	if ( (ySrc0 < 0) || (ySrc0 >= dimSrc0.y) )
	{
		ySrc0 = by ;
	}
	const int	iSrc0 = (ySrc0 * dimSrc0.x) + xSrc0 ;

	int	xSrc1 = bx - xShift1 ;
	int	ySrc1 = by - yShift1 ;
	if ( (xSrc1 < 0) || (xSrc1 >= dimSrc1.x) )
	{
		xSrc1 = bx ;
	}
	if ( (ySrc1 < 0) || (ySrc1 >= dimSrc1.y) )
	{
		ySrc1 = by ;
	}
	const int	iSrc1 = (ySrc1 * dimSrc1.x) + xSrc1 ;

	if ( (bx < dimDst.x) && (by < dimDst.y) && (bi < dimDst.n) )
	{
		for ( size_t iChannel = tx; iChannel < dimDst.z; iChannel += xThreads )
		{
			const int	iDst = bi * dimDst.z + iChannel ;

			float	src0 = 0.0f ;
			if ( (iChannel0 + iChannel) < dimSrc0.z )
			{
				src0 = pSrc0[iSrc0 * dimSrc0.z + iChannel0 + iChannel] ;
			}
			float	src1 = 0.0f ;
			if ( (iChannel1 + iChannel) < dimSrc1.z )
			{
				src1 = pSrc1[iSrc1 * dimSrc1.z + iChannel1 + iChannel] ;
			}

			pDst[iDst] = Op::Operate( src0, src1 ) ;
		}
	}
}

void Palesibyl::nncuda_Primitive_Add
	( float * pDst, NNBufDim dimDst,
		const float * pSrc0, NNBufDim dimSrc0,
		size_t iChannel0, int xShift0, int yShift0,
		const float * pSrc1, NNBufDim dimSrc1,
		size_t iChannel1, int xShift1, int yShift1,
		size_t xLeftBounds, cudaStream_t stream )
{
	dim3			threads = CalcThreadCount( dimDst ) ;
	unsigned int	xThreads = threads.x ;
	unsigned int	yThreads = threads.y ;

	dim3	grid( ((unsigned int) (dimDst.x - xLeftBounds) + yThreads - 1) / yThreads,
					(unsigned int) dimDst.y ) ;

	nncuda_PrimitiveOperation<OperationAdd>
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst,
				pSrc0, dimSrc0, iChannel0, xShift0, yShift0,
				pSrc1, dimSrc1, iChannel1, xShift1, yShift1,
				xLeftBounds, xThreads, yThreads ) ;
}

void Palesibyl::nncuda_Primitive_Multiply
	( float * pDst, NNBufDim dimDst,
		const float * pSrc0, NNBufDim dimSrc0,
		size_t iChannel0, int xShift0, int yShift0,
		const float * pSrc1, NNBufDim dimSrc1,
		size_t iChannel1, int xShift1, int yShift1,
		size_t xLeftBounds, cudaStream_t stream )
{
	dim3			threads = CalcThreadCount( dimDst ) ;
	unsigned int	xThreads = threads.x ;
	unsigned int	yThreads = threads.y ;

	dim3	grid( ((unsigned int) (dimDst.x - xLeftBounds) + yThreads - 1) / yThreads,
					(unsigned int) dimDst.y ) ;

	nncuda_PrimitiveOperation<OperationMul>
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst,
				pSrc0, dimSrc0, iChannel0, xShift0, yShift0,
				pSrc1, dimSrc1, iChannel1, xShift1, yShift1,
				xLeftBounds, xThreads, yThreads ) ;
}

