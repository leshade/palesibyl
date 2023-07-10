
#define	__NN_CUDA_DEV__	__device__

#include "nn_cuda_kernel.h"
#include "nn_function.h"

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// argmax 活性化関数
//////////////////////////////////////////////////////////////////////////////

__global__ void nnkernel_Activation_Argmax
	( float * pDst, NNBufDim dimDst, int xLeftBounds,
		const float * pSrc, NNBufDim dimSrc, int xThreads, int yThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	ti = ty * xThreads + tx ;
	const int	bx = blockIdx.x * yThreads + ty + xLeftBounds ;
	const int	by = blockIdx.y ;
	const int	bi = by * dimDst.x + bx ;
	if ( (bx > dimDst.x) || (bi >= dimDst.n) )
	{
		return ;
	}

	__shared__ float	eSum[cudaMaxThreadCount] ;
	__shared__ float	eMax[cudaMaxThreadCount] ;
	__shared__ int		zMax[cudaMaxThreadCount] ;
	__shared__ float	vDst[cudaMaxThreadCount*argmaxChannelCount] ;

	// Σexp(x(i)) と最大値を走査する
	const int	iSrcBase = bi * dimSrc.z ;
	eSum[ti] = 0.0f ;
	eMax[ti] = 0.0f ;
	zMax[ti] = 0 ;
	for ( int zBase = 0; zBase < dimSrc.z; zBase += xThreads )
	{
		const int	z = zBase + tx ;
		if ( z < dimSrc.z )
		{
			const float	e = exp_sd( pSrc[iSrcBase + z] ) ;
			if ( e > eMax[ti] )
			{
				eMax[ti] = e ;
				zMax[ti] = z ;
			}
			eSum[ti] += e ;
		}
	}
	__syncthreads() ;

	if ( tx == 0 )
	{
		for ( int i = 0; i < xThreads; i ++ )
		{
			if ( eMax[ti] < eMax[ti + i] )
			{
				eMax[ti] = eMax[ti + i] ;
				zMax[ti] = zMax[ti + i] ;
			}
			eSum[ti] += eSum[ti + i] ;
		}
		vDst[ty * argmaxChannelCount + argmaxIndex] = (float) zMax[ti] ;
		vDst[ty * argmaxChannelCount + argmaxProbability] = eMax[ti] / max( eSum[ti], 0.00001f ) ;
		vDst[ty * argmaxChannelCount + argmaxSumExp] = eSum[ti] ;
	}
	__syncthreads() ;

	// 出力
	const int	iDstBase = bi * dimDst.z ;
	if ( tx < dimDst.z )
	{
		pDst[iDstBase + tx] = vDst[ty * argmaxChannelCount + tx] ;
	}
}

void Palesibyl::nncuda_Activation_Argmax
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream )
{
	assert( dimDst.z == argmaxChannelCount ) ;
	assert( nDepthwise == 1 ) ;
	assert( dimDst.x > xLeftBounds ) ;

	unsigned int	xThreads = (unsigned int) (cudaMaxThreadCount
								/ min(dimDst.x - xLeftBounds, cudaMaxThreadCount/64)) ;
	unsigned int	yThreads = (unsigned int) cudaMaxThreadCount / xThreads ;

	dim3	threads( xThreads, yThreads ) ;
	dim3	grid( ((unsigned int) (dimDst.x - xLeftBounds) + yThreads - 1) / yThreads,
					(unsigned int) dimDst.y ) ;

	nnkernel_Activation_Argmax
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst, (int) xLeftBounds, pSrc, dimSrc, xThreads, yThreads ) ;
}


