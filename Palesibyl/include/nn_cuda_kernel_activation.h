
#define	__NN_CUDA_DEV__	__device__

#include "nn_cuda_kernel.h"
#include "nn_function2.h"

using namespace Palesibyl ;

constexpr const unsigned int	maxBatchSamples = 64 ;


// バッチサンプル数計算
//////////////////////////////////////////////////////////////////////////////

inline unsigned int CalcBatchSamples( size_t nBufCaps, size_t nDstWidth )
{
	unsigned int	nBatchSamples = (unsigned int) nBufCaps ;
	assert( nBatchSamples > 0 ) ;
	if ( (nBatchSamples > nDstWidth / 2) && (nDstWidth >= 2) )
	{
		nBatchSamples = (unsigned int) nDstWidth / 2 ;
	}
	if ( nBatchSamples > maxBatchSamples )
	{
		nBatchSamples = maxBatchSamples ;
	}
	return	nBatchSamples ;
}



//////////////////////////////////////////////////////////////////////////////
// 活性化関数
//////////////////////////////////////////////////////////////////////////////

template <class A> __global__ void nnkernel_Activation
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		int xLeftBounds, int nDepthwise, int xThreads, int yThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	bx = blockIdx.x * yThreads + ty + xLeftBounds ;
	const int	by = blockIdx.y ;
	const int	bi = bx + by * dimDst.x ;
	if ( (bx > dimDst.x) || (bi >= dimDst.n) )
	{
		return ;
	}
	const int	tyLine = ty * dimSrc.z ;

	__shared__ float	vSrc[cudaSharedMemorySize/2/sizeof(float)] ;
	__shared__ float	vDst[cudaSharedMemorySize/2/sizeof(float)] ;

	// 入力ベクトルを読み込む
	const int	iSrcBase = bi * dimSrc.z ;
	for ( int i = tx; i < dimSrc.z; i += xThreads )
	{
		vSrc[tyLine + i] =
			A::kernelPreActivation( pSrc + iSrcBase, i, dimSrc.z, nDepthwise ) ;
	}
	__syncthreads() ;

	// 活性化関数
	for ( int i = tx; i < dimDst.z; i += xThreads )
	{
		vDst[tyLine + i] =
			A::kernelActivation( i, &vSrc[tyLine], dimSrc.z, nDepthwise ) ;
	}
	__syncthreads() ;

	// 計算結果を書き出す
	const int	iDstBase = bi * dimDst.z ;
	for ( int i = tx; i < dimDst.z; i += xThreads )
	{
		pDst[iDstBase + i] = vDst[tyLine + i] ;
	}
}

template <class S> void nncuda_Activation
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream )
{
	assert( xLeftBounds < dimDst.x ) ;
	unsigned int	nBatchSamples =
		CalcBatchSamples
			( (cudaSharedMemorySize/2/sizeof(float)) / dimSrc.z, dimDst.x - xLeftBounds ) ;

	unsigned int	xThreads = (unsigned int) dimSrc.z ;
	unsigned int	yThreads = 1 ;
	if ( xThreads >= cudaMaxThreadCount )
	{
		xThreads = (unsigned int) cudaMaxThreadCount ;
		nBatchSamples = 1 ;
	}
	else
	{
		yThreads = cudaMaxThreadCount / xThreads ;
		if ( yThreads > nBatchSamples )
		{
			yThreads = nBatchSamples ;
		}
	}

	dim3	threads( xThreads, yThreads ) ;
	dim3	grid( ((unsigned int) (dimDst.x - xLeftBounds) + yThreads - 1) / yThreads,
					(unsigned int) dimDst.y ) ;

	assert( dimSrc.z <= cudaSharedMemorySize/2/sizeof(float) ) ;
	assert( dimDst.z <= cudaSharedMemorySize/2/sizeof(float) ) ;

	nnkernel_Activation<S>
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst, pSrc, dimSrc,
				(int) xLeftBounds,  nDepthwise, xThreads, yThreads ) ;
}


//////////////////////////////////////////////////////////////////////////////
// 活性化関数のδ逆伝播
//////////////////////////////////////////////////////////////////////////////

template <class A> __global__ void nnkernel_Activation_DeltaBack
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct,
		int nDepthwise, int xThreads, int yThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	bx = blockIdx.x * yThreads + ty ;
	const int	by = blockIdx.y ;
	const int	bi = bx + by * dimDstDelta.x ;

	__shared__ float	vSrcAct[cudaSharedMemorySize/4/sizeof(float)] ;
	__shared__ float	vDiffSrcAct[cudaSharedMemorySize/4/sizeof(float)] ;
	__shared__ float	vSrcDelta[cudaSharedMemorySize/4/sizeof(float)] ;
	__shared__ float	vDstDelta[cudaSharedMemorySize/4/sizeof(float)] ;

	// 入力ベクトルを読み込む
	const int	iSrcActBase = bi * dimSrcAct.z ;
	const int	iOutActBase = bi * dimOutAct.z ;
	if ( (bx < dimDstDelta.x) && (bi < dimDstDelta.n) )
	{
		for ( int i = tx; i < dimSrcAct.z; i += xThreads )
		{
			vSrcAct[ty * dimSrcAct.z + i] =
				A::kernelPreDifferential
					( pSrcAct + iSrcActBase,
						pOutAct + iOutActBase, i, dimSrcAct.z, nDepthwise ) ;
		}
	}
	__syncthreads() ;

	// 入力ベクトルの微分
	if ( (bx < dimDstDelta.x) && (bi < dimDstDelta.n) )
	{
		for ( int i = tx; i < dimSrcAct.z; i += xThreads )
		{
			vDiffSrcAct[ty * dimSrcAct.z + i] =
				A::kernelDifferential
					( i, &(vSrcAct[ty * dimSrcAct.z]), dimSrcAct.z, nDepthwise ) ;
		}
	}
	__syncthreads() ;

	// 前レイヤーのδを読み込む
	const int	iSrcDeltaBase = bi * dimSrcDelta.z ;
	if ( (bx < dimDstDelta.x) && (bi < dimDstDelta.n) )
	{
		for ( int i = tx; i < dimSrcDelta.z; i += xThreads )
		{
			vSrcDelta[ty * dimSrcDelta.z + i] = pSrcDelta[iSrcDeltaBase + i] ;
		}
	}
	__syncthreads() ;

	// 活性化関数のδ逆伝播
	if ( (bx < dimDstDelta.x) && (bi < dimDstDelta.n) )
	{
		for ( int i = tx; i < dimDstDelta.z; i += xThreads )
		{
			vDstDelta[ty * dimDstDelta.z + i] =
				A::BackDelta( i, &(vSrcDelta[ty * dimSrcDelta.z]),
								&(vDiffSrcAct[ty * dimSrcAct.z]),
								dimSrcAct.z, nDepthwise ) ;
		}
	}
	__syncthreads() ;

	// 計算結果を書き出す
	const int	iDstDeltaBase = bi * dimDstDelta.z ;
	if ( (bx < dimDstDelta.x) && (bi < dimDstDelta.n) )
	{
		for ( int i = tx; i < dimDstDelta.z; i += xThreads )
		{
			pDstDelta[iDstDeltaBase + i] = vDstDelta[ty * dimDstDelta.z + i] ;
		}
	}
}

template <class A> void nncuda_Activation_DeltaBack
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct, int nDepthwise, cudaStream_t stream )
{
	unsigned int	nBatchSamples =
		CalcBatchSamples
			( (cudaSharedMemorySize/4/sizeof(float)) / dimDstDelta.z, dimDstDelta.x ) ;

	unsigned int	xThreads = (unsigned int) cudaMaxThreadCount / nBatchSamples ;
	unsigned int	yThreads = nBatchSamples ;
	if ( xThreads >= dimSrcDelta.z )
	{
		xThreads = (unsigned int) dimSrcDelta.z ;
	}

	dim3	threads( xThreads, yThreads ) ;
	dim3	grid( ((unsigned int) dimDstDelta.x + yThreads - 1) / yThreads,
					(unsigned int) dimDstDelta.y ) ;

	assert( dimSrcDelta.z * yThreads <= cudaSharedMemorySize/4/sizeof(float) ) ;
	assert( dimSrcDelta.z * yThreads <= cudaSharedMemorySize/4/sizeof(float) ) ;
	assert( dimSrcAct.z * yThreads <= cudaSharedMemorySize/4/sizeof(float) ) ;
	assert( dimSrcAct == dimDstDelta ) ;
	assert( dimOutAct.x == dimDstDelta.x ) ;
	assert( dimOutAct.y == dimDstDelta.y ) ;
	assert( dimOutAct.z == A::CalcOutChannels(dimSrcAct.z,nDepthwise) ) ;

	nnkernel_Activation_DeltaBack<A>
		<<<grid, threads, 0, stream>>>
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct,
				nDepthwise, xThreads, yThreads ) ;
}



//////////////////////////////////////////////////////////////////////////////
// 損失関数
//////////////////////////////////////////////////////////////////////////////

template <class L, typename P> __global__ void nnkernel_LossDelta
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, P lp, int xThreads, int yThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	tn = xThreads ;
	const int	bx = blockIdx.x * yThreads + ty ;
	const int	by = blockIdx.y ;
	const int	bi = bx + by * dimLossDelta.x ;

	__shared__ float	vOutput[cudaSharedMemorySize/4/sizeof(float)] ;
	__shared__ float	vTeaching[cudaSharedMemorySize/4/sizeof(float)] ;

	// 予測値読み込み
	const int	iOutBase = bi * dimOutput.z ;
	if ( (bx < dimLossDelta.x) && (bi < dimLossDelta.n) )
	{
		for ( int i = tx; i < dimOutput.z; i += tn )
		{
			vOutput[ty * dimOutput.z + i] = pOutput[iOutBase + i] ;
		}
	}
	__syncthreads() ;

	// 教師データ読み込み
	const int	iTeachingBase = bi * dimTeaching.z ;
	if ( (bx < dimLossDelta.x) && (bi < dimLossDelta.n) )
	{
		for ( int i = tx; i < dimTeaching.z; i += tn )
		{
			vTeaching[ty * dimTeaching.z + i] = pTeaching[iTeachingBase + i] ;
		}
	}
	__syncthreads() ;

	// 損失出力
	const int	iDeltaBase = bi * dimLossDelta.z ;
	const int	iInActBase = bi * dimInAct.z ;
	if ( (bx < dimLossDelta.x) && (bi < dimLossDelta.n) )
	{
		for ( int i = tx; i < dimLossDelta.z; i += tn )
		{
			pLossDelta[iDeltaBase + i] =
				L::kernelLossDelta
					( i, pInAct + iInActBase,
						&(vOutput[ty * dimOutput.z]),
						&(vTeaching[ty * dimTeaching.z]),
						dimLossDelta.z, nDepthwise, lp ) ;
		}
	}
}

template <class L, typename P> __global__ void nnkernel_LossDelta_huge
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, P lp, int xThreads, int yThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	bx = blockIdx.x * yThreads + ty ;
	const int	by = blockIdx.y ;
	const int	bi = bx + by * dimLossDelta.x ;

	__shared__ float	vLoss[cudaSharedMemorySize/4/sizeof(float)] ;

	const int	iOutBase = bi * dimOutput.z ;
	const int	iTeachingBase = bi * dimTeaching.z ;
	const int	iDeltaBase = bi * dimLossDelta.z ;
	const int	iInActBase = bi * dimInAct.z ;
	if ( (bx < dimLossDelta.x) && (bi < dimLossDelta.n) )
	{
		for ( int zBase = 0; zBase < dimLossDelta.z; zBase += xThreads )
		{
			// 損失計算
			if ( zBase + tx < dimLossDelta.z )
			{
				vLoss[ty * xThreads + tx] =
					L::kernelLossDelta
						( zBase + tx, pInAct + iInActBase,
							pOutput + iOutBase,
							pTeaching + iTeachingBase,
							dimLossDelta.z, nDepthwise, lp ) ;
			}
			__syncthreads() ;

			// 損失出力
			if ( zBase + tx < dimLossDelta.z )
			{
				pLossDelta[iDeltaBase + zBase + tx] = vLoss[ty * xThreads + tx] ;
			}
			__syncthreads() ;
		}
	}
}

template <class L, typename P> void nncuda_LossDelta
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, const P& lp, cudaStream_t stream )
{
	const size_t	zChannles = __max( dimLossDelta.z, dimTeaching.z ) ;
	if ( zChannles <= cudaSharedMemorySize/4/sizeof(float) )
	{
		unsigned int	nBatchSamples =
			CalcBatchSamples
				( (cudaSharedMemorySize/4/sizeof(float)) / zChannles, dimLossDelta.x ) ;

		unsigned int	xThreads = (unsigned int) cudaMaxThreadCount / nBatchSamples ;
		unsigned int	yThreads = nBatchSamples ;
		if ( xThreads >= zChannles )
		{
			xThreads = (unsigned int) zChannles ;
		}

		dim3	threads( xThreads, yThreads ) ;
		dim3	grid( ((unsigned int) dimLossDelta.x + yThreads - 1) / yThreads,
						(unsigned int) dimLossDelta.y ) ;

		assert( dimLossDelta.z * yThreads <= cudaSharedMemorySize/4/sizeof(float) ) ;
		assert( dimTeaching.z * yThreads <= cudaSharedMemorySize/4/sizeof(float) ) ;

		nnkernel_LossDelta<L,P>
			<<<grid, threads, 0, stream>>>
				( pLossDelta, dimLossDelta,
					pInAct, dimInAct,
					pOutput, dimOutput,
					pTeaching, dimTeaching,
					nDepthwise, lp, xThreads, yThreads ) ;
	}
	else
	{
		unsigned int	xThreads = 64 ;
		unsigned int	yThreads = cudaMaxThreadCount / xThreads ;

		dim3	threads( xThreads, yThreads ) ;
		dim3	grid( ((unsigned int) dimLossDelta.x + yThreads - 1) / yThreads,
						(unsigned int) dimLossDelta.y ) ;

		nnkernel_LossDelta_huge<L,P>
			<<<grid, threads, 0, stream>>>
				( pLossDelta, dimLossDelta,
					pInAct, dimInAct,
					pOutput, dimOutput,
					pTeaching, dimTeaching,
					nDepthwise, lp, xThreads, yThreads ) ;
	}
}