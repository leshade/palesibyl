
#include "nn_cuda_kernel_matrix_sp.h"



//////////////////////////////////////////////////////////////////////////////
// 行列計算
//////////////////////////////////////////////////////////////////////////////

template <int DZ> __global__ void nnkernel_Matrix_OneHot
	( float * pDst, NNBufDim dimDst, int xLeftBounds,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix, int xMatrix, int yMatrix, size_t iMatrixBias,
		int xStride, int yStride, int xOffset, int yOffset, int xThreads, int yThreads )
{
	const int	tx = threadIdx.x ;
	const int	ty = threadIdx.y ;
	const int	txyi = ty * xThreads + tx ;
	const int	bx0 = blockIdx.x * xThreads + xLeftBounds ;
	const int	by0 = blockIdx.y * yThreads ;
	const int	bx = bx0 + tx ;
	const int	by = by0 + ty ;
	const int	bi = bx + by * dimDst.x ;

	__shared__ float	vDst[256*DZ] ;

	// 入力ベクトルを読み込む
	int	iOneHot = 0 ;
	if ( (bx < dimSrc.x) && (by < dimSrc.y) && (bi < dimSrc.n) )
	{
		iOneHot = (int) floor( pSrc[bi * dimSrc.z] ) ;
		if ( iOneHot > iMatrixBias )
		{
			iOneHot = 0 ;
		}
	}

	// DS 要素づつ出力する
	for ( int zBase = 0; zBase < dimDst.z; zBase += DZ )
	{
		// 行列要素読み込み
		for ( int z = 0; (z < DZ) && (zBase + z < yMatrix); z ++ )
		{
			vDst[txyi * DZ + z] = pMatrix[(zBase + z) * xMatrix + iOneHot] ;
		}
		__syncthreads() ;

		// バイアス項（ある場合）
		if ( iMatrixBias < xMatrix )
		{
			for ( int z = 0; (z < DZ) && (zBase + z < yMatrix); z ++ )
			{
				vDst[txyi * DZ + z] += pMatrix[(zBase + z) * xMatrix + iMatrixBias] ;
			}
			__syncthreads() ;
		}

		// 計算結果を書き出す
		for ( int xoff = 0; xoff < xThreads; xoff ++ )
		{
			const int	x = bx0 + xoff ;
			if ( x < dimDst.x )
			{
				const int	iDst = by * dimDst.x + x ;
				for ( int i = tx; (i < DZ) && (zBase + i < dimDst.z); i += xThreads )
				{
					if ( (by < dimDst.y) && (iDst < dimDst.n) )
					{
						pDst[iDst * dimDst.z + zBase + i]
							= vDst[(ty * xThreads + xoff) * DZ + i] ;
					}
				}
			}
		}
		__syncthreads() ;
	}
}

void Palesibyl::nncuda_Matrix_OneHot
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		size_t xLeftBounds, int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
{
	assert( dimSrc.z == 1 ) ;

	const unsigned int	xThreads = 32 ;
	const unsigned int	yThreads = 256/xThreads ;

	dim3	threads( xThreads, yThreads ) ;
	dim3	grid( ((unsigned int) (dimDst.x - xLeftBounds) + xThreads - 1) / xThreads,
					(unsigned int) (dimDst.y + yThreads - 1) / yThreads ) ;

	nnkernel_Matrix_OneHot<32>
		<<<grid, threads, 0, stream>>>
			( pDst, dimDst, (int) xLeftBounds, pSrc, dimSrc,
				pMatrix, (int) xMatrix, (int) yMatrix, iMatrixBias,
				sp.m_xStride, sp.m_yStride, sp.m_xOffset, sp.m_yOffset, xThreads, yThreads ) ;
}


//////////////////////////////////////////////////////////////////////////////
// 行列δ逆伝播
//////////////////////////////////////////////////////////////////////////////

void Palesibyl::nncuda_Matrix_DeltaBack_Injection_Sp
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
{
	nncuda_Matrix_DeltaBack_Sp<NNBufSampler>
		( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
			pMatrix, xMatrix, yMatrix, zSrcChannels,
			nDepthwise, sp, stream ) ;
}

void Palesibyl::nncuda_Matrix_DeltaBack_Conv_Sp
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
{
	nncuda_Matrix_DeltaBack_Sp<NNBufConvEdgeSampler>
		( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
			pMatrix, xMatrix, yMatrix, zSrcChannels,
			nDepthwise, sp, stream ) ;
}



//////////////////////////////////////////////////////////////////////////////
// 行列勾配計算
//////////////////////////////////////////////////////////////////////////////

void Palesibyl::nncuda_CalcMatrixGradient_Edge_Sp
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	cuda_CalcMatrixGradient_Sp<NNBufEdgeSampler>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_Conv_Edge_Sp
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	cuda_CalcMatrixGradient_Sp<NNBufConvEdgeSampler>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
}


