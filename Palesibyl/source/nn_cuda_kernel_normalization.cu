
#define	__NN_CUDA_DEV__	__device__

#include "nn_cuda_kernel.h"
#include "nn_function.h"


using namespace Palesibyl ;

constexpr const int	N_BLOCK_SIZE = 16 ;

constexpr const int	I_PARAM_SCALE = 0 ;
constexpr const int	I_PARAM_SHIFT = 1 ;
constexpr const int	N_PARAM_COUNT = 2 ;

constexpr const int	I_AGGR_SUM = 0 ;
constexpr const int	I_AGGR_SUM2 = 1 ;
constexpr const int	I_AGGR_NUM = 2 ;
constexpr const int	N_AGGR_COUNT = 3 ;

constexpr const int	I_MEAN = 0 ;
constexpr const int	I_RCPVAR = 1 ;
constexpr const int	N_MEAN_VAR = 2 ;



// ブロックサイズ計算
//////////////////////////////////////////////////////////////////////////////

size_t Palesibyl::nncuda_CalcAggregateSize( size_t n )
{
	return	(n + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE ;
}



// サンプルを集計
//////////////////////////////////////////////////////////////////////////////

template <int BLOCK_SIZE>
__global__ void nnkernel_AggregateSample
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		int zSampling, int xThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	ti = ty * xThreads + tx ;
	const int	bx = blockIdx.x ;
	const int	by = blockIdx.y ;

	__shared__ float	vDstSum[cudaMaxThreadCount/BLOCK_SIZE] ;
	__shared__ float	vDstSum2[cudaMaxThreadCount/BLOCK_SIZE] ;
	__shared__ float	nDstCount[cudaMaxThreadCount/BLOCK_SIZE] ;

	__shared__ float	vSum[cudaMaxThreadCount] ;
	__shared__ float	vSum2[cudaMaxThreadCount] ;
	__shared__ float	nCount[cudaMaxThreadCount] ;

	if ( ti < cudaMaxThreadCount/BLOCK_SIZE )
	{
		vDstSum[ti] = 0.0f ;
		vDstSum2[ti] = 0.0f ;
		nDstCount[ti] = 0 ;
	}
	__syncthreads() ;

	int	zDstOffset = 0 ;
	for ( int zBase = 0; zBase < dimSrc.z; zBase += xThreads )
	{
		const int	zDst = (zBase + tx) / zSampling ;
		const int	zDstLast = (zBase + xThreads - 1) / zSampling ;
		const int	zFixedDst = (zBase + xThreads < dimSrc.z)
								? (zBase + xThreads) / zSampling : dimDst.z / 3 ;

		// 垂直加算
		vSum[ti] = 0.0f ;
		vSum2[ti] = 0.0f ;
		nCount[ti] = 0.0f ;
		for ( int ly = 0; ly < BLOCK_SIZE; ly ++ )
		{
			const int	x = bx * BLOCK_SIZE + ty ;
			const int	y = by * BLOCK_SIZE + ly ;
			const int	z = zBase + tx ;
			const int	iSrc = ((y * dimSrc.x) + x) * dimSrc.z ;
			if ( (x < dimSrc.x) && (y < dimSrc.y) && (z < dimSrc.z) )
			{
				const float	s = pSrc[iSrc + z] ;
				vSum[ti] += s ;
				vSum2[ti] += s * s ;
				nCount[ti] += 1.0f ;
			}
		}
		__syncthreads() ;

		// 水平加算
		if ( ty == 0 )
		{
			for ( int i = 1; i < BLOCK_SIZE; i ++ )
			{
				vSum[tx] += vSum[i * xThreads + tx] ;
				vSum2[tx] += vSum2[i * xThreads + tx] ;
				nCount[tx] += nCount[i * xThreads + tx] ;
			}
		}
		__syncthreads() ;

		// 深度方向加算
		if ( ((tx == 0) || (zDst * zSampling == zBase + tx))
			&& (ty == 0) && (zDst * 3 < dimDst.z) )
		{
			for ( int i = 0; (i < zSampling)
							&& (tx + i < xThreads)
							&& ((zBase + tx + i) / zSampling == zDst); i ++ )
			{
				vDstSum[zDst - zDstOffset] += vSum[tx + i] ;
				vDstSum2[zDst - zDstOffset] += vSum2[tx + i] ;
				nDstCount[zDst - zDstOffset] += nCount[tx + i] ;
			}
		}
		__syncthreads() ;

		// 出力
		if ( ((tx == 0) || (zDst * zSampling == zBase + tx))
			&& (zDst * N_AGGR_COUNT < dimDst.z)
			&& (zDst < zFixedDst) && (ty < N_AGGR_COUNT) )
		{
			const int	iDst = ((by * dimDst.x) + bx) * dimDst.z
											+ zDst * N_AGGR_COUNT ;
			if ( ty == I_AGGR_SUM )
			{
				pDst[iDst + ty] = vDstSum[zDst - zDstOffset] ;
			}
			else if ( ty == I_AGGR_SUM2 )
			{
				pDst[iDst + ty] = vDstSum2[zDst - zDstOffset] ;
			}
			else if ( ty == I_AGGR_NUM )
			{
				pDst[iDst + ty] = nDstCount[zDst - zDstOffset] ;
			}
		}
		if ( (ti == 0) && (zDstLast == zFixedDst) )
		{
			vDstSum[0] = vDstSum[zDstLast - zDstOffset] ;
			vDstSum2[0] = vDstSum2[zDstLast - zDstOffset] ;
			nDstCount[0] = nDstCount[zDstLast - zDstOffset] ;
		}
		__syncthreads() ;

		if ( (ti < cudaMaxThreadCount/BLOCK_SIZE)
			&& ((ti != 0) || (zDstLast != zFixedDst)) )
		{
			vDstSum[ti] = 0.0f ;
			vDstSum2[ti] = 0.0f ;
			nDstCount[ti] = 0 ;
		}
		zDstOffset = zFixedDst ;
	}
}

void Palesibyl::nncuda_AggregateSample
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc, int zSampling, cudaStream_t stream )
{
	const int	xThreads = (int) min( dimSrc.z, cudaMaxThreadCount/N_BLOCK_SIZE ) ;

	dim3	threads( xThreads, N_BLOCK_SIZE ) ;
	dim3	grid( (unsigned int) (dimSrc.x + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE,
					(unsigned int)  (dimSrc.y + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE ) ;

	assert( dimDst.x >= (dimSrc.x + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE ) ;
	assert( dimDst.y >= (dimSrc.y + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE ) ;

	nnkernel_AggregateSample<N_BLOCK_SIZE>
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst, pSrc, dimSrc, zSampling, xThreads ) ;
}



// サンプルを集計２
//////////////////////////////////////////////////////////////////////////////

template <int BLOCK_SIZE>
__global__ void nnkernel_AggregateSample2
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc, int xThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	ti = ty * xThreads + tx ;
	const int	bx = blockIdx.x ;
	const int	by = blockIdx.y ;

	__shared__ float	vSum[cudaMaxThreadCount] ;

	for ( int zBase = 0; zBase < dimSrc.z; zBase += xThreads )
	{
		// 垂直加算
		float	sum = 0.0f ;
		for ( int ly = 0; ly < BLOCK_SIZE; ly ++ )
		{
			const int	x = bx * BLOCK_SIZE + ty ;
			const int	y = by * BLOCK_SIZE + ly ;
			const int	z = zBase + tx ;
			const int	iSrc = ((y * dimSrc.x) + x) * dimSrc.z ;
			if ( (x < dimSrc.x) && (y < dimSrc.y) && (z < dimSrc.z) )
			{
				sum += pSrc[iSrc + z] ;
			}
		}
		vSum[ti] = sum ;
		__syncthreads() ;

		// 水平加算と出力
		if ( ty == 0 )
		{
			for ( int i = 1; i < BLOCK_SIZE; i ++ )
			{
				sum += vSum[i * xThreads + tx] ;
			}
			const int	iDst = ((by * dimDst.x) + bx) * dimDst.z ;
			pDst[iDst + zBase + tx] = sum ;
		}
		__syncthreads() ;
	}
}

void Palesibyl::nncuda_AggregateSample2
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc, cudaStream_t stream )
{
	const int	xThreads = (int) min( dimSrc.z, cudaMaxThreadCount/N_BLOCK_SIZE ) ;

	dim3	threads( xThreads, N_BLOCK_SIZE ) ;
	dim3	grid( (unsigned int) (dimSrc.x + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE,
					(unsigned int)  (dimSrc.y + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE ) ;

	assert( dimDst.x >= (dimSrc.x + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE ) ;
	assert( dimDst.y >= (dimSrc.y + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE ) ;
	assert( dimDst.z == dimSrc.z ) ;

	nnkernel_AggregateSample2<N_BLOCK_SIZE>
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst, pSrc, dimSrc, xThreads ) ;
}


// 集計して分布を計算
//////////////////////////////////////////////////////////////////////////////

template <int BLOCK_SIZE>
__global__ void nnkernel_CalcDistribution
	( float * pMeanVar, NNBufDim dimMeanVar,
		const float * pSrc1, NNBufDim dimSrc1,
		const float * pSrc2, NNBufDim dimSrc2, int xThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	ti = ty * xThreads + tx ;
	const int	bx = blockIdx.x ;
	const int	by = blockIdx.y ;

	__shared__ float	vSum[cudaMaxThreadCount] ;
	__shared__ float	vDst[cudaMaxThreadCount] ;

	const int	zDstPitch = xThreads/N_AGGR_COUNT*N_MEAN_VAR ;

	for ( int zBase = 0, zDstBase = 0;
				zBase < dimSrc2.z; zBase += xThreads, zDstBase += zDstPitch )
	{
		// 垂直加算
		float	sum = 0.0f ;
		for ( int ly = 0; ly < BLOCK_SIZE; ly ++ )
		{
			const int	x = bx * BLOCK_SIZE + ty ;
			const int	y = by * BLOCK_SIZE + ly ;
			const int	z = zBase + tx ;
			const int	iSrc = ((y * dimSrc2.x) + x) * dimSrc2.z ;
			if ( (x < dimSrc2.x) && (y < dimSrc2.y) && (z < dimSrc2.z) )
			{
				sum += pSrc2[iSrc + z] ;
			}
		}
		vSum[ti] = sum ;
		__syncthreads() ;

		// 水平加算
		if ( ty == 0 )
		{
			for ( int i = 1; i < BLOCK_SIZE; i ++ )
			{
				sum += vSum[i * xThreads + tx] ;
			}
			if ( zBase + tx < dimSrc1.z )
			{
				sum += pSrc1[((by * dimSrc1.x) + bx) * dimSrc1.z + zBase + tx] ;
			}
			vSum[tx] = sum ;
		}
		__syncthreads() ;

		// 分布計算
		if ( (ty == 0) && ((tx % N_AGGR_COUNT) == 0) )
		{
			const int	zDst = tx / N_AGGR_COUNT * N_MEAN_VAR ;
			const float	sum = vSum[tx + I_AGGR_SUM] ;
			const float	sum2 = vSum[tx + I_AGGR_SUM2] ;
			const float	num = vSum[tx + I_AGGR_NUM] ;
			const float	mean = sum / num ;
			vDst[zDst + I_MEAN] = mean ;
			vDst[zDst + I_RCPVAR] = 1.0f / sqrt( max( sum2 / num - mean * mean, 1.0e-10f ) ) ;
		}
		__syncthreads() ;

		// 出力
		if ( (ti < zDstPitch) && (zDstBase + ti < dimMeanVar.z) )
		{
			const int	iDst = ((by * dimMeanVar.x) + bx) * dimMeanVar.z ;
			pMeanVar[iDst + zDstBase + ti] = vDst[ti] ;
		}
		__syncthreads() ;
	}
}

void Palesibyl::nncuda_CalcDistribution
	( float * pMeanVar, NNBufDim dimMeanVar,
		const float * pSrc1, NNBufDim dimSrc1,
		const float * pSrc2, NNBufDim dimSrc2, cudaStream_t stream )
{
	assert( dimMeanVar.x == dimSrc1.x ) ;
	assert( dimMeanVar.y == dimSrc1.y ) ;
	assert( (dimMeanVar.z % N_MEAN_VAR) == 0 ) ;
	assert( (dimSrc1.z % N_AGGR_COUNT) == 0 ) ;
	assert( (dimSrc2.z % N_AGGR_COUNT) == 0 ) ;
	assert( dimMeanVar.z == (dimSrc1.z / N_AGGR_COUNT) * N_MEAN_VAR ) ;
	assert( dimMeanVar.z == (dimSrc2.z / N_AGGR_COUNT) * N_MEAN_VAR ) ;

	const int	xThreads = min( (int) dimSrc2.z,
							((int) cudaMaxThreadCount/N_BLOCK_SIZE/N_AGGR_COUNT) * N_AGGR_COUNT ) ;
	const int	yThreads = N_BLOCK_SIZE ;
	assert( xThreads * yThreads <= cudaMaxThreadCount ) ;
	assert( xThreads % N_AGGR_COUNT == 0 ) ;

	dim3	threads( xThreads, yThreads ) ;
	dim3	grid( (unsigned int) dimMeanVar.x, (unsigned int) dimMeanVar.y ) ;

	nnkernel_CalcDistribution<N_BLOCK_SIZE>
		<<<grid, threads, 0, stream>>>
			( pMeanVar, dimMeanVar,
				pSrc1, dimSrc1, pSrc2, dimSrc2, xThreads ) ;
}



// 正規化
//////////////////////////////////////////////////////////////////////////////

__global__ void nnkernel_Normalize
	( float * pSample, NNBufDim dimSample, int xSampleBounds,
		const float * pParams,
		const float * pMeanVar,
		int zSampling, int xThreads, int yThreads )
{
	__shared__ float	scale[cudaMaxThreadCount/4] ;
	__shared__ float	shift[cudaMaxThreadCount/4] ;
	__shared__ float	mean[cudaMaxThreadCount/4] ;
	__shared__ float	rcpvar[cudaMaxThreadCount/4] ;

	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	bx = blockIdx.x * yThreads + ty + xSampleBounds ;
	const int	by = blockIdx.y ;

	for ( int zBase = 0; zBase < dimSample.z; zBase += xThreads )
	{
		// パラメーター読み込み
		if ( zBase + tx < dimSample.z )
		{
			if ( ty < N_PARAM_COUNT )
			{
				const float	p = pParams[(zBase + tx)/zSampling * N_PARAM_COUNT + ty] ;
				if ( ty == I_PARAM_SCALE )
				{
					scale[tx] = p ;
				}
				else if ( ty == I_PARAM_SHIFT )
				{
					shift[tx] = p ;
				}
			}
			else if ( ty < (N_PARAM_COUNT + N_MEAN_VAR) )
			{
				const float	p = pMeanVar[(zBase + tx)/zSampling * N_MEAN_VAR + (ty - N_PARAM_COUNT)] ;
				if ( ty - N_PARAM_COUNT == I_MEAN )
				{
					mean[tx] = p ;
				}
				else if ( ty - N_PARAM_COUNT == I_RCPVAR )
				{
					rcpvar[tx] = p ;
				}
			}
		}
		__syncthreads() ;

		// 正規化
		const int	zSrc = zBase + tx ;
		if ( (bx < dimSample.x) && (zSrc < dimSample.z) )
		{
			const int	iSample = ((by * dimSample.x) + bx) * dimSample.z ;
			float		s = pSample[iSample + zSrc] ;
			pSample[iSample + zSrc] =
				(s - mean[tx]) * rcpvar[tx] * scale[tx] + shift[tx] ;
		}
		__syncthreads() ;
	}
}

void Palesibyl::nncuda_Normalize
	( float * pSample, NNBufDim dimSample, size_t xSampleBounds,
		const float * pParams,
		const float * pMeanVar, int zSampling, cudaStream_t stream )
{
	const int	xThreads = min( (int) dimSample.z, 32 ) ;
	const int	yThreads = max( min( (int) dimSample.x/2 + 1, 16 ), N_PARAM_COUNT+N_MEAN_VAR ) ;
	assert( xThreads * yThreads <= cudaMaxThreadCount ) ;

	assert( xSampleBounds < dimSample.x ) ;
	dim3	threads( xThreads, yThreads ) ;
	dim3	grid( (unsigned int) (dimSample.x - xSampleBounds + yThreads - 1) / yThreads,
					(unsigned int) dimSample.y ) ;

	nnkernel_Normalize
		<<<grid, threads, 0, stream>>>
			( pSample, dimSample, (int) xSampleBounds,
				pParams, pMeanVar, zSampling, xThreads, yThreads ) ;
}



// δ逆伝播と勾配計算
//////////////////////////////////////////////////////////////////////////////

__global__ void nnkernel_NormDeltaBack
	( float * pDelta, NNBufDim dimDelta,
		float * pGradient, NNBufDim dimGradient,
		const float * pDstSample, NNBufDim dimDstSample,
		const float * pParams,
		const float * pMeanVar,
		int zSampling, int xThreads, int yThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	ti = ty * xThreads + tx ;
	const int	bx = blockIdx.x * yThreads + ty ;
	const int	by = blockIdx.y ;

	const int	iDelta = ((by * dimDelta.x) + bx) * dimDelta.z ;
	const int	iDstSample = ((by * dimDstSample.x) + bx) * dimDstSample.z ;
	const int	iGradient = ((by * dimGradient.x) + blockIdx.x) * dimGradient.z ;

	__shared__ float	scale[cudaMaxThreadCount/4] ;
	__shared__ float	shift[cudaMaxThreadCount/4] ;
//	__shared__ float	mean[cudaMaxThreadCount/4] ;
	__shared__ float	rcpvar[cudaMaxThreadCount/4] ;

	__shared__ float	gradScale[cudaMaxThreadCount] ;
	__shared__ float	gradShift[cudaMaxThreadCount] ;

	for ( int zBase = 0; zBase < dimDelta.z; zBase += xThreads )
	{
		const int	zSrc = zBase + tx ;
		const int	zIndex = zSrc / zSampling ;

		// パラメーター読み込み
		if ( zSrc < dimDelta.z )
		{
			if ( ty < N_PARAM_COUNT )
			{
				const float	p = pParams[zIndex * N_PARAM_COUNT + ty] ;
				if ( ty == I_PARAM_SCALE )
				{
					scale[tx] = p ;
				}
				else if ( ty == I_PARAM_SHIFT )
				{
					shift[tx] = p ;
				}
			}
			else if ( ty < (N_PARAM_COUNT + N_MEAN_VAR) )
			{
				const float	p = pMeanVar[zIndex * N_MEAN_VAR + (ty - N_PARAM_COUNT)] ;
				if ( ty - N_PARAM_COUNT == I_MEAN )
				{
//					mean[tx] = p ;
				}
				else if ( ty - N_PARAM_COUNT == I_RCPVAR )
				{
					rcpvar[tx] = p ;
				}
			}
		}
		__syncthreads() ;

		gradScale[ti] = 0.0f ;
		gradShift[ti] = 0.0f ;

		if ( (bx < dimDstSample.x) && (zSrc < dimDstSample.z) )
		{
			// サンプル読み込み
			const float	s = pDstSample[iDstSample + zSrc] ;
			const float	r = (abs(scale[tx]) > 1.0e-5f) ? (1.0f / scale[tx]) : 1.0f ;
			const float	x = (s - shift[tx]) * r ;

			// δ読み込み
			const float	d = pDelta[iDelta + zSrc] ;

			// 勾配
			gradScale[ti] = d * x ;
			gradShift[ti] = d ;

			// δ逆伝播
			pDelta[iDelta + zSrc] = d * scale[tx] * rcpvar[tx] ;
		}
		__syncthreads() ;

		if ( (ty == 0) && (zSrc * 2 < dimGradient.z) )
		{
			// 勾配水平加算
			for ( int i = 1; i < yThreads; i ++ )
			{
				const int	j = i * xThreads + tx ;
				gradScale[tx] += gradScale[j] ;
				gradShift[tx] += gradShift[j] ;
			}
		}
		__syncthreads() ;

		if ( zSrc * 2 < dimGradient.z )
		{
			// 勾配出力
			if ( ty == 0 )
			{
				pGradient[iGradient + zSrc * 2] = gradScale[tx] ;
			}
			else if ( ty == 1 )
			{
				pGradient[iGradient + zSrc * 2 + 1] = gradShift[tx] ;
			}
		}
		__syncthreads() ;
	}
}

void Palesibyl::nncuda_NormDeltaBack
	( float * pDelta, NNBufDim dimDelta,
		float * pGradient, NNBufDim dimGradient,
		const float * pDstSample, NNBufDim dimDstSample,
		const float * pParams,
		const float * pMeanVar, int zSampling, cudaStream_t stream )
{
	assert( dimDelta == dimDstSample ) ;
	assert( dimGradient.x == nncuda_CalcAggregateSize(dimDelta.x) ) ;
	assert( dimGradient.y == dimDelta.y ) ;
	assert( dimGradient.z == dimDelta.z * 2 ) ;

	const int	xThreads = (int) min( dimDelta.z, cudaMaxThreadCount/N_BLOCK_SIZE ) ;
	const int	yThreads = N_BLOCK_SIZE ;

	dim3	threads( xThreads, yThreads ) ;
	dim3	grid( (unsigned int) (dimDelta.x + yThreads - 1) / yThreads,
					(unsigned int) dimDelta.y ) ;

	nnkernel_NormDeltaBack
		<<<grid, threads, 0, stream>>>
			( pDelta, dimDelta,
				pGradient, dimGradient,
				pDstSample, dimDstSample,
				pParams, pMeanVar, zSampling, xThreads, yThreads ) ;
}

