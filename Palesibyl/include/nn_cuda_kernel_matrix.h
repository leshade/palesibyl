
#ifndef	__NN_CUDA_KERNEL_MATRIX_H__
#define	__NN_CUDA_KERNEL_MATRIX_H__

#define	__NN_CUDA_DEV__	__device__

#include "nn_cuda_kernel.h"
#include "nn_function.h"

using namespace Palesibyl ;

constexpr const unsigned int	maxBatchSamples = 64 ;
constexpr const size_t			maxMatrixStrideX = 64 ;
constexpr const size_t			GradientBlockX = 32 ;
constexpr const size_t			GradientBlockY = 8 ;


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
// 行列計算
//////////////////////////////////////////////////////////////////////////////

template <class S, int MX> __global__ void nnkernel_Matrix
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix, int xMatrix, int yMatrix, size_t iMatrixBias,
		int nDepthwise, NNSamplingParam sp, int xThreads, int yThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	bx0 = blockIdx.x * yThreads ;
	const int	bx = bx0 + ty ;
	const int	by = blockIdx.y ;
	const int	bi = bx + by * dimDst.x ;

	__shared__ float	vMatrixLine[cudaSharedMemorySize/3/sizeof(float)] ;
	__shared__ float	vSrc[cudaSharedMemorySize/3/sizeof(float)] ;
	__shared__ float	vDst[maxBatchSamples*maxBatchSamples] ;
	
	// 入力ベクトルを読み込む（一括）
	const int		xUpScale = sp.m_xUpScale ;
	const int		xMatrixBufWidth = min(xMatrix,MX) ;
	const int		nMatrixBufSize = xMatrixBufWidth * yThreads ;
	const int		tyLine = ty * xMatrixBufWidth ;
	const int		xSrc = bx * sp.m_xStride + sp.m_xOffset ;
	const int		ySrc = by * sp.m_yStride + sp.m_yOffset ;
	const size_t	zSrcCount = iMatrixBias ;
	if ( xMatrix <= MX )
	{
		for ( int i = tx; i < xMatrix; i += xThreads )
		{
			if ( bx < dimDst.x )
			{
				if ( i < zSrcCount )
				{
					vSrc[tyLine + i] = S::Sample( pSrc, dimSrc, xSrc, ySrc, i, sp ) ;
				}
				else
				{
					vSrc[tyLine + i] = 1.0f ;
				}
			}
			else
			{
				vSrc[tyLine + i] = 0.0f ;
			}
		}
		__syncthreads() ;
	}

	const int	iDstBase = bi * dimDst.z ;
	const int	tyDst = ty * yThreads ;
	const int	iMatrixBlock = (ty % xUpScale) * nMatrixBufSize ;
	for ( int lineBase = 0; lineBase < dimDst.z; lineBase += yThreads )
	{
		// 計算結果をゼロクリア
		for ( int i = tx; i < yThreads; i += xThreads )
		{
			vDst[tyDst + i] = 0.0f ;
		}
		__syncthreads() ;

		for ( int colBase = 0; colBase < xMatrix; colBase += MX )
		{
			// 入力ベクトルを読み込む（分割）
			if ( xMatrix > MX )
			{
				for ( int i = tx; (i < MX) && (colBase + i < xMatrix); i += xThreads )
				{
					if ( bx < dimDst.x )
					{
						if ( colBase + i < zSrcCount )
						{
							vSrc[tyLine + i] =
								S::Sample( pSrc, dimSrc, xSrc, ySrc, colBase + i, sp ) ;
						}
						else
						{
							vSrc[tyLine + i] = 1.0f ;
						}
					}
					else
					{
						vSrc[tyLine + i] = 0.0f ;
					}
				}
				__syncthreads() ;
			}

			// 行列を yThreads 行だけ読み込む
			for ( int i = 0, iBuf = 0; i < xUpScale; i ++, iBuf += nMatrixBufSize )
			{
				// bx が ty に依存し、S::SampleMatrixLine が変化するので bx0 を使う
				const int	line = S::SampleMatrixLine(bx0 + i, by, lineBase + ty, dimDst.z, sp) ;
				if ( line < yMatrix )
				{
					const int	iLine = line * xMatrix ;
					for ( int j = tx; (j < MX) && (colBase + j < xMatrix); j += xThreads )
					{
						vMatrixLine[iBuf + tyLine + j] = pMatrix[iLine + (colBase + j)] ;
					}
				}
			}
			__syncthreads() ;

			// 各行の合計を計算
			const int	colBaseMod = nDepthwise - (colBase % nDepthwise) ;
			for ( int i = tx; (i < yThreads) && (lineBase + i < dimDst.z); i += xThreads )
			{
				const int	zOffset = (S::SampleMatrixLine
											(bx, by, lineBase + i, dimDst.z, sp)
										+ colBaseMod) % nDepthwise ;
				const int	iLine = iMatrixBlock + i * xMatrixBufWidth ;
				float	x = 0.0 ;
				for ( int j = zOffset; (j < MX) && (colBase + j < zSrcCount); j += nDepthwise )
				{
					x += vMatrixLine[iLine + j] * vSrc[tyLine + j] ;
				}
				for ( int j = zSrcCount - colBase; (j < MX) && (colBase + j < xMatrix); j ++ )
				{
					x += vMatrixLine[iLine + j] * vSrc[tyLine + j] ;
				}
				vDst[tyDst + i] += x ;
			}
			__syncthreads() ;
		}

		// 計算結果を書き出す
		if ( bx < dimDst.x )
		{
			for ( int i = tx; (i < yThreads) && (lineBase + i < dimDst.z); i += xThreads )
			{
				pDst[iDstBase + lineBase + i] = vDst[tyDst + i] ;
			}
		}
		__syncthreads() ;
	}
}

template <class S> void nncuda_Matrix
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
{
	unsigned int	nBatchSamples =
		CalcBatchSamples
			( (cudaSharedMemorySize/3/sizeof(float))
				/ (min(xMatrix,maxMatrixStrideX) * sp.m_xUpScale), dimDst.x ) ;

	unsigned int	xThreads = (unsigned int) cudaMaxThreadCount / nBatchSamples ;
	unsigned int	yThreads = nBatchSamples ;
	if ( xThreads >= xMatrix )
	{
		xThreads = (unsigned int) xMatrix ;
	}

	dim3	threads( xThreads, yThreads ) ;
	dim3	grid( ((unsigned int) dimDst.x + yThreads - 1) / yThreads,
					(unsigned int) dimDst.y ) ;

	assert( min(xMatrix,maxMatrixStrideX) * yThreads * sp.m_xUpScale <= cudaSharedMemorySize/3/sizeof(float) ) ;
	assert( yMatrix == dimDst.z * sp.m_xUpScale * sp.m_yUpScale ) ;

	nnkernel_Matrix<S,maxMatrixStrideX>
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst, pSrc, dimSrc,
				pMatrix, (int) xMatrix, (int) yMatrix, iMatrixBias,
				nDepthwise, sp, xThreads, yThreads ) ;
}




//////////////////////////////////////////////////////////////////////////////
// 行列δ逆伝播
//////////////////////////////////////////////////////////////////////////////

template <class S, int MX, int MY> __global__ void nnkernel_Matrix_DeltaBack
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels, size_t zSrcBlock,
		int nDepthwise, NNSamplingParam sp, int xThreads, int yThreads, int zThread )
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

	for ( int yc = 0; yc < sp.m_yConv; yc ++ )
	{
		const int	ySrc = (by - yc * sp.m_yPitch - sp.m_yOffset) / sp.m_yStride ;
		if ( (ySrc < 0)
			|| ((size_t) ySrc * sp.m_yUpScale >= dimSrcDelta.y)
			|| (ySrc * sp.m_yStride + sp.m_yOffset != by - yc * sp.m_yPitch) )
		{
			continue ;
		}
		for ( int xc = 0; xc < sp.m_xConv; xc ++ )
		{
			const int	xSrc = (bx - xc * sp.m_xPitch - sp.m_xOffset) / sp.m_xStride ;
			bool		flagOutOfDimension = false ;
			if ( (xSrc < 0)
				|| ((size_t) xSrc * sp.m_xUpScale >= dimSrcDelta.x)
				|| (xSrc * sp.m_xStride + sp.m_xOffset != bx - xc * sp.m_xPitch) )
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
						S::BackSample( ymBase + i, pSrcDelta, dimSrcDelta, xSrc, ySrc, sp ) ;
				}
				__syncthreads() ;

				const int	ymBaseMod = nDepthwise - (ymBase % nDepthwise) ;
				for ( int zb = 0; (zb < MX) && (xmBase + zb < zSrcChannels); zb += zSrcBlock )
				{
					// 行列の関係する列だけ読み込む
					if ( !flagOutOfDimension )
					for ( size_t z = tx; (z < zSrcBlock) && (zb + z < zSrcChannels); z += xThreads )
					{
						size_t	col = S::ConvChannelIndex( xc, yc, zb + z, zSrcChannels, sp.m_xConv ) ;
						size_t	cod = (col + ymBaseMod) % nDepthwise ;
						for ( int line = ymBase + cod + ty * nDepthwise;
									((line - ymBase) < MY) && (line < yMatrix);
									line += yThreads * nDepthwise )
						{
							vMatrix[tzMatrix + (line - ymBase) * zSrcBlock + z]
													= pMatrix[line * xMatrix + col] ;
						}
					}
					__syncthreads() ;

					// 各列と入力δの内積を出力δに加算する
					if ( !flagOutOfDimension )
					for ( size_t z = txyi;
							(z < zSrcBlock) && ((xmBase + zb) + z < zSrcChannels); z += txyn )
					{
						size_t	col = S::ConvChannelIndex
											( xc, yc, (xmBase + zb) + z, zSrcChannels, sp.m_xConv ) ;
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

template <class S> void nncuda_Matrix_DeltaBack
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
{
	const int		maxBufSize = 64 ;
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

	nnkernel_Matrix_DeltaBack<S,maxBufSize,maxBufSize>
		<<<grid, threads, 0, stream>>>
			( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
				pMatrix, xMatrix, yMatrix,
				zSrcChannels, zSrcBlock,
				nDepthwise, sp, xThreads, yThreads, zThreads ) ;
}



//////////////////////////////////////////////////////////////////////////////
// 行列勾配計算
//////////////////////////////////////////////////////////////////////////////

template <class S, int BX, int BY, int MY>
__global__ void nnkernel_CalcMatrixGradient
	( float * pGradient, NNBufDim dimGradient,
		int xMatrix, int yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		NNSamplingParam sp, int xThreads, int yThreads, int zThreads )
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
			if ( yDelta * sp.m_yUpScale >= dimDelta.y )
			{
				continue ;
			}
			for ( int xSub = 0; xSub < BX; xSub ++ )
			{
				const int	xDelta = bx * BX + xSub ;
				bool		flagOutOfDimension = false ;
				if ( xDelta * sp.m_xUpScale >= dimDelta.x )
				{
					// ※ここで continue すると __syncthreads で全スレッドが揃わない
					flagOutOfDimension = true ;
					//continue ;
				}

				// 入力サンプリング
				const int	xSrc = xDelta * sp.m_xUpScale * sp.m_xStride + sp.m_xOffset ;
				const int	ySrc = yDelta * sp.m_yUpScale * sp.m_yStride + sp.m_yOffset ;
				if ( !flagOutOfDimension )
				if ( ty == 0 )
				{
					const int	z = zSrcBase + tx ;
					if ( z < zSrcCount )
					{
						vSrcDelta[tziSrcDelta + tx] =
							S::Sample( pSrc, dimSrc, xSrc, ySrc, z, sp ) ;
					}
					else
					{
						vSrcDelta[tziSrcDelta + tx] = 1.0f ;
					}
				}
				__syncthreads() ;

				// δ読み込み
				if ( !flagOutOfDimension )
				for ( int i = txyi; i < yMatrixBufHeight; i += txyn )
				{
					vDstDelta[tziDstDelta + i] =
						S::BackSample( ymBase + i, pDelta, dimDelta, xDelta, yDelta, sp ) ;
				}
				__syncthreads() ;

				// 勾配を計算し加算
				if ( !flagOutOfDimension )
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

template <class S> void cuda_CalcMatrixGradient
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	const int		maxBufSize = 128 ;
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

	if ( (xGradientBlockSize == GradientBlockX)
		&& (yGradientBlockSize == GradientBlockY) )
	{
		nnkernel_CalcMatrixGradient<S,GradientBlockX,GradientBlockY,maxBufSize>
			<<<grid, threads, 0, stream>>>
				( pGradient, dimGradient,
					(int) xMatrix, (int) yMatrix, iMatrixBias,
					pDelta, dimDelta, pSrc, dimSrc, sp, xThreads, yThreads, zThreads ) ;
	}
	else
	{
		assert( xGradientBlockSize == GradientBlockX*GradientBlockY ) ;
		assert( yGradientBlockSize == 1 ) ;
		nnkernel_CalcMatrixGradient<S,GradientBlockX*GradientBlockY,1,maxBufSize>
			<<<grid, threads, 0, stream>>>
				( pGradient, dimGradient,
					(int) xMatrix, (int) yMatrix, iMatrixBias,
					pDelta, dimDelta, pSrc, dimSrc, sp, xThreads, yThreads, zThreads ) ;
	}
}

#endif

