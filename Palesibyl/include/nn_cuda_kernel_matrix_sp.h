
#ifndef	__NN_CUDA_KERNEL_MATRIX_SP_H__
#define	__NN_CUDA_KERNEL_MATRIX_SP_H__

#include "nn_cuda_kernel_matrix.h"


//////////////////////////////////////////////////////////////////////////////
// 行列δ逆伝播（δに 0.0 要素が多い最適化）
//////////////////////////////////////////////////////////////////////////////

template <class S, int MX, int MY> __global__ void nnkernel_Matrix_DeltaBack_Sp
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels, size_t zSrcBlock,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, int xThreads, int yThreads, int zThread )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	tz = threadIdx.z ;
	const int	txyi = tx + ty * xThreads ;
	const int	txyn = xThreads * yThreads ;
	const int	bx = blockIdx.x * zThread + tz ;
	const int	by = blockIdx.y ;
	const int	bi = bx + by * dimDstDelta.x ;

	__shared__ float	vMatrix[cudaSharedMemorySize/2/sizeof(float)] ;
	__shared__ float	vSrcDelta[cudaSharedMemorySize/4/sizeof(float)] ;
	__shared__ float	vDstDelta[cudaSharedMemorySize/4/sizeof(float)] ;

	const int	tzMatrix = tz * zSrcBlock * min(yMatrix,MY) ;
	const int	tzSrcDelta = tz * min(yMatrix,MY) ;
	const int	tzDstDelta = tz * min((int)dimDstDelta.z,MX) ;

	for ( int xmBase = 0; xmBase < dimDstDelta.z; xmBase += MX )
	{

	// 出力δ初期化
	for ( int i = txyi; (i < MX) && (xmBase + i < dimDstDelta.z); i += txyn )
	{
		vDstDelta[tzDstDelta + i] = 0.0f ;
	}
	__syncthreads() ;

	for ( int yc = 0; yc < yConv; yc ++ )
	{
		const int	ySrc = (by - yc - yOffset) / yStride ;
		if ( (ySrc < 0)
			|| ((size_t) ySrc * S::UpSamplingScaleY >= dimSrcDelta.y)
			|| (ySrc * yStride + yOffset != by - yc) )
		{
			continue ;
		}
		for ( int xc = 0; xc < xConv; xc ++ )
		{
			const int	xSrc = (bx - xc - xOffset) / xStride ;
			bool		flagOutOfDimension = false ;
			if ( (xSrc < 0)
				|| ((size_t) xSrc * S::UpSamplingScaleX >= dimSrcDelta.x)
				|| (xSrc * xStride + xOffset != bx - xc) )
			{
				// ※ここで continue すると __syncthreads で全スレッドが揃わない
				flagOutOfDimension = true ;
				// continue ;
			}

			for ( int ymBase = 0; ymBase < yMatrix; ymBase += MY )
			{
				// 入力δを読み込む
				if ( !flagOutOfDimension )
				for ( int i = txyi; (i < MY) && (ymBase + i < yMatrix); i += txyn )
				{
					vSrcDelta[tzSrcDelta + i] =
						S::BackSample( ymBase + i, pSrcDelta, dimSrcDelta, xSrc, ySrc ) ;
				}
				__syncthreads() ;

				bool	flagNonZeroDelta = false ;
				for ( int i = 0; (i < MY) && (ymBase + i < yMatrix); i ++ )
				{
					if ( vSrcDelta[tzSrcDelta + i] != 0.0f )
					{
						flagNonZeroDelta = true ;
						break ;
					}
				}

				const int	ymBaseMod = nDepthwise - (ymBase % nDepthwise) ;
				for ( int zb = 0; (zb < MX) && (xmBase + zb < zSrcChannels); zb += zSrcBlock )
				{
					// 行列の関係する列だけ読み込む
					if ( !flagOutOfDimension && flagNonZeroDelta )
					for ( int i = ty; (i < MY) && (ymBase + i < yMatrix); i += yThreads )
					{
						if ( vSrcDelta[tzSrcDelta + i] != 0.0f )
						{
							const int	line = ymBase + i ;
							for ( size_t z = tx; (z < zSrcBlock) && (zb + z < zSrcChannels); z += xThreads )
							{
								size_t	col = S::ConvChannelIndex( xc, yc, zb + z, zSrcChannels, xConv ) ;
								vMatrix[tzMatrix + i * zSrcBlock + z]
											= pMatrix[line * xMatrix + col] ;
							}
						}
					}
					__syncthreads() ;

					// 各列と入力δの内積を出力δに加算する
					if ( !flagOutOfDimension && flagNonZeroDelta )
					for ( size_t z = txyi;
							(z < zSrcBlock) && ((xmBase + zb) + z < zSrcChannels); z += txyn )
					{
						size_t	col = S::ConvChannelIndex
											( xc, yc, (xmBase + zb) + z, zSrcChannels, xConv ) ;
						float	d = 0.0f ;
						for ( int i = ((col + ymBaseMod) % nDepthwise);
									(i < MY) && (ymBase + i < yMatrix); i += nDepthwise )
						{
							d += vSrcDelta[tzSrcDelta + i]
								* vMatrix[tzMatrix + i * zSrcBlock + z] ;
						}
						vDstDelta[tzDstDelta + zb + z] += d ;
					}
					__syncthreads() ;
				}
			}
		}
	}

	// δ出力
	const int	iDstBase = bi * dimDstDelta.z ;
	if ( (bi < dimDstDelta.n) && (bx < dimDstDelta.x) )
	{
		for ( int i = txyi; (i < MX) && (xmBase + i < dimDstDelta.z); i += txyn )
		{
			pDstDelta[iDstBase + xmBase + i] = vDstDelta[tzDstDelta + i] ;
		}
	}

	}	// next (xmBase)
}

template <class S> void nncuda_Matrix_DeltaBack_Sp
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	const int		maxBufSize = 16 ;
	unsigned int	nBatchSamples =
		CalcBatchSamples
			( (cudaSharedMemorySize/4/sizeof(float))
				/ min(max(yMatrix,(int)dimDstDelta.z),maxBufSize), dimDstDelta.x ) ;
	unsigned int	xThreads = (unsigned int) zSrcChannels ;
	unsigned int	yThreads = (unsigned int) yMatrix ;
	unsigned int	zThreads = nBatchSamples ;
	if ( xThreads * yThreads * zThreads > cudaMaxThreadCount )
	{
		xThreads = cudaMaxThreadCount / zThreads ;
		if ( xThreads > (unsigned int) zSrcChannels )
		{
			yThreads = xThreads / (unsigned int) zSrcChannels ;
			xThreads = (unsigned int) zSrcChannels ;
		}
		else
		{
			yThreads = 1 ;
		}
	}
	size_t	zSrcBlock = zSrcChannels ;
	if ( zSrcBlock * min(yMatrix,maxBufSize) * zThreads > cudaSharedMemorySize/2/sizeof(float) )
	{
		zSrcBlock = cudaSharedMemorySize/2/sizeof(float) / (min(yMatrix,maxBufSize) * zThreads) ;
	}

	dim3	threads( xThreads, yThreads, zThreads ) ;
	dim3	grid( ((unsigned int) dimDstDelta.x + zThreads - 1) / zThreads,
					(unsigned int) dimDstDelta.y ) ;

	assert( zSrcBlock != 0 ) ;
	assert( zSrcBlock * min(yMatrix,maxBufSize) * zThreads <= cudaSharedMemorySize/2/sizeof(float) ) ;
	assert( min((int)dimDstDelta.z,maxBufSize) * zThreads <= cudaSharedMemorySize/4/sizeof(float) ) ;

	nnkernel_Matrix_DeltaBack_Sp<S,maxBufSize,maxBufSize>
		<<<grid, threads, 0, stream>>>
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix,
				zSrcChannels, zSrcBlock,
				xStride, yStride, xOffset, yOffset,
				nDepthwise, xConv, yConv, xThreads, yThreads, zThreads ) ;
}



//////////////////////////////////////////////////////////////////////////////
// 行列勾配計算（δに 0.0 要素が多い最適化）
//////////////////////////////////////////////////////////////////////////////

template <class S, int BX, int BY, int MY>
__global__ void nnkernel_CalcMatrixGradient_Sp
	( float * pGradient, NNBufDim dimGradient,
		int xMatrix, int yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		int xStride, int yStride, int xOffset, int yOffset,
		int xConv, int yConv, int xThreads, int yThreads, int zThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	tz = threadIdx.z ;
	const int	txyi = ty * xThreads + tx ;
	const int	txyn = xThreads * yThreads ;
	const int	bx = blockIdx.x * zThreads + tz ;
	const int	by = blockIdx.y ;
	const int	bi = bx + by * dimGradient.x ;

	__shared__ float	vMatrix[cudaSharedMemorySize/2/sizeof(float)] ;
	__shared__ float	vSrcDelta[cudaMaxThreadCount] ;
	__shared__ float	vDstDelta[cudaSharedMemorySize/4/sizeof(float)] ;

	const size_t	zSrcCount = iMatrixBias ;
	const int		iGradient = bi * dimGradient.z ;
	const int		yMatrixBufHeight = min(yMatrix,MY) ;
	const int		tziMatrix = tz * xThreads * yMatrixBufHeight ;
	const int		tziSrcDelta = tz * xThreads ;
	const int		tziDstDelta = tz * yMatrixBufHeight ;

	for ( int ymBase = 0; ymBase < yMatrix; ymBase += MY )
	{

	for ( int zSrcBase = 0; zSrcBase < xMatrix; zSrcBase += xThreads )
	{
		// 部分行列初期化
		for ( int i = ty; i < yMatrixBufHeight; i += yThreads )
		{
			vMatrix[tziMatrix + i * xThreads + tx] = 0.0f ;
		}
		__syncthreads() ;

		for ( int ySub = 0; ySub < BY; ySub ++ )
		{
			const int	yDelta = by * BY + ySub ;
			if ( yDelta * S::UpSamplingScaleY >= dimDelta.y )
			{
				continue ;
			}
			for ( int xSub = 0; xSub < BX; xSub ++ )
			{
				const int	xDelta = bx * BX + xSub ;
				bool		flagOutOfDimension = false ;
				if ( xDelta * S::UpSamplingScaleX >= dimDelta.x )
				{
					// ※ここで continue すると __syncthreads で全スレッドが揃わない
					flagOutOfDimension = true ;
					//continue ;
				}

				// δ読み込み
				if ( !flagOutOfDimension )
				for ( int i = txyi; i < yMatrixBufHeight; i += txyn )
				{
					vDstDelta[tziDstDelta + i] =
						S::BackSample( ymBase + i, pDelta, dimDelta, xDelta, yDelta ) ;
				}
				__syncthreads() ;

				bool	flagNonZeroDelta = false ;
				for ( int i = 0; i < yMatrixBufHeight; i ++ )
				{
					if ( vDstDelta[tziDstDelta + i] != 0.0f )
					{
						flagNonZeroDelta = true ;
						break ;
					}
				}

				// 入力サンプリング
				const int	xSrc = xDelta * S::UpSamplingScaleX * xStride + xOffset ;
				const int	ySrc = yDelta * S::UpSamplingScaleY * yStride + yOffset ;
				if ( !flagOutOfDimension && flagNonZeroDelta )
				if ( ty == 0 )
				{
					const int	z = zSrcBase + tx ;
					if ( z < zSrcCount )
					{
						vSrcDelta[tziSrcDelta + tx] =
							S::Sample( pSrc, dimSrc, xSrc, ySrc, z, (size_t) xConv ) ;
					}
					else
					{
						vSrcDelta[tziSrcDelta + tx] = 1.0f ;
					}
				}
				__syncthreads() ;

				// 勾配を計算し加算
				if ( !flagOutOfDimension && flagNonZeroDelta )
				for ( int i = ty; i < yMatrixBufHeight; i += yThreads )
				{
					vMatrix[tziMatrix + i * xThreads + tx] +=
						vDstDelta[tziDstDelta + i] * vSrcDelta[tziSrcDelta + tx] ;
				}
			}
		}

		// 部分行列出力
		if ( (zSrcBase + tx < xMatrix) && (bx < dimGradient.x) )
		{
			for ( int i = ty; i < yMatrixBufHeight; i += yThreads )
			{
				pGradient[iGradient + (ymBase + i) * xMatrix + (zSrcBase + tx)] =
											vMatrix[tziMatrix + i * xThreads + tx] ;
			}
		}
		__syncthreads() ;
	}

	}	// next (ymBase)
}

template <class S> void cuda_CalcMatrixGradient_Sp
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		int xStride, int yStride, int xOffset, int yOffset,
		int xConv, int yConv, cudaStream_t stream )
{
	const int		maxBufSize = 16 ;
	unsigned int	xThreads = (unsigned int) min( min(xMatrix,cudaMaxThreadCount),
													((cudaSharedMemorySize/2/sizeof(float))
															/ min((int)yMatrix,maxBufSize)) ) ;
	unsigned int	yThreads = (unsigned int) min( yMatrix, cudaMaxThreadCount / xThreads ) ;
	assert( xThreads <= cudaMaxThreadCount ) ;
	assert( yThreads != 0 ) ;
	assert( xThreads * yThreads <= cudaMaxThreadCount ) ;

	unsigned int	zThreads = (unsigned int) cudaMaxThreadCount / (xThreads * yThreads) ;
	assert( zThreads >= 1 ) ;
	if ( zThreads > dimGradient.x )
	{
		zThreads = (unsigned int) dimGradient.x ;
	}

	dim3	threads( xThreads, yThreads, zThreads ) ;
	dim3	grid( ((unsigned int) dimGradient.x + zThreads - 1) / zThreads,
					(unsigned int) dimGradient.y ) ;

	assert( cudaSharedMemorySize/4/sizeof(float) >= min((int)yMatrix,maxBufSize) * zThreads ) ;
	assert( cudaSharedMemorySize/2/sizeof(float) >= min((int)yMatrix,maxBufSize) * xThreads * zThreads ) ;
	assert( xGradientBlockSize == GradientBlockX ) ;
	assert( yGradientBlockSize == GradientBlockY ) ;

	nnkernel_CalcMatrixGradient_Sp<S,GradientBlockX,GradientBlockY,maxBufSize>
		<<<grid, threads, 0, stream>>>
			( pGradient, dimGradient,
				(int) xMatrix, (int) yMatrix, iMatrixBias,
				pDelta, dimDelta, pSrc, dimSrc,
				xStride, yStride, xOffset, yOffset,
				xConv, yConv, xThreads, yThreads, zThreads ) ;
}


#endif

