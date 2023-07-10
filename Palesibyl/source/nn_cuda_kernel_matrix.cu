
#include "nn_cuda_kernel_matrix.h"



//////////////////////////////////////////////////////////////////////////////
// 行列計算
//////////////////////////////////////////////////////////////////////////////

void Palesibyl::nncuda_Matrix_Clamp
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		size_t xLeftBounds, int nDepthwise,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	nncuda_Matrix<NNBufClampSampler>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xLeftBounds, nDepthwise, sp, stream ) ;
}

void Palesibyl::nncuda_Matrix_Edge
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		size_t xLeftBounds, int nDepthwise,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	nncuda_Matrix<NNBufEdgeSampler>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xLeftBounds, nDepthwise, sp, stream ) ;
}

void Palesibyl::nncuda_Matrix_Conv_Clamp
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		size_t xLeftBounds, int nDepthwise,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	nncuda_Matrix<NNBufConvClampSampler>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xLeftBounds, nDepthwise, sp, stream ) ;
}

void Palesibyl::nncuda_Matrix_Conv_Edge
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		size_t xLeftBounds, int nDepthwise,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	nncuda_Matrix<NNBufConvEdgeSampler>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xLeftBounds, nDepthwise, sp, stream ) ;
}

void Palesibyl::nncuda_Matrix_UpSampler
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		size_t xLeftBounds, int nDepthwise,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	assert( (sp.m_xStride == 1) && (sp.m_yStride == 1) ) ;
	assert( (sp.m_xOffset == 0) && (sp.m_yOffset == 0) ) ;
	nncuda_Matrix<NNBufUpSampler>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xLeftBounds, nDepthwise, sp, stream ) ;
}

/*
void Palesibyl::nncuda_Matrix_OneHot
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		size_t xLeftBounds, int nDepthwise,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	assert( dimSrc.z == 1 ) ;
	nncuda_Matrix<NNBufOneHotSampler>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xLeftBounds, nDepthwise, sp, stream ) ;
}
*/



//////////////////////////////////////////////////////////////////////////////
// 行列δ逆伝播
//////////////////////////////////////////////////////////////////////////////

void Palesibyl::nncuda_Matrix_DeltaBack_Injection
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
{
	nncuda_Matrix_DeltaBack<NNBufSampler>
		( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
			pMatrix, xMatrix, yMatrix, zSrcChannels,
			nDepthwise, sp, stream ) ;
}

void Palesibyl::nncuda_Matrix_DeltaBack_Conv
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
{
	nncuda_Matrix_DeltaBack<NNBufConvClampSampler>
		( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
			pMatrix, xMatrix, yMatrix, zSrcChannels,
			nDepthwise, sp, stream ) ;
}

void Palesibyl::nncuda_Matrix_DeltaBack_UpSampler
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream )
{
	assert( (sp.m_xStride == 1) && (sp.m_yStride == 1) ) ;
	assert( (sp.m_xOffset == 0) && (sp.m_yOffset == 0) ) ;
	nncuda_Matrix_DeltaBack<NNBufUpSampler>
		( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
			pMatrix, xMatrix, yMatrix, zSrcChannels,
			nDepthwise, sp, stream ) ;
}

size_t Palesibyl::nncuda_IsAcceptableMatrixSize
	( size_t xMatrix, size_t yMatrix, size_t zSrcChannels )
{
	return	true ;
//	return	(zSrcChannels <= cudaSharedMemorySize/4/sizeof(float))
//			&& (xMatrix <= cudaSharedMemorySize/4/sizeof(float))
//			&& (yMatrix <= cudaSharedMemorySize/4/sizeof(float)) ;
}



//////////////////////////////////////////////////////////////////////////////
// 行列勾配計算
//////////////////////////////////////////////////////////////////////////////


size_t Palesibyl::nncuda_CalcMatrixGradientBlockSizeX( size_t x, size_t y )
{
	if ( (y >= GradientBlockY) || (x < GradientBlockX*GradientBlockY) )
	{
		return	GradientBlockX ;
	}
	else
	{
		assert( (y < GradientBlockY) && (x >= GradientBlockX*GradientBlockY) ) ;
		return	GradientBlockX*GradientBlockY ;
	}
}

size_t Palesibyl::nncuda_CalcMatrixGradientBlockSizeY( size_t x, size_t y )
{
	if ( (y >= GradientBlockY) || (x < GradientBlockX*GradientBlockY) )
	{
		return	GradientBlockY ;
	}
	else
	{
		assert( (y < GradientBlockY) && (x >= GradientBlockX*GradientBlockY) ) ;
		return	1 ;
	}
}

size_t Palesibyl::nncuda_CalcMatrixGradientBlockX( size_t x, size_t y )
{
	const size_t	nBlockSize = nncuda_CalcMatrixGradientBlockSizeX( x, y ) ;
	return	(x + nBlockSize - 1) / nBlockSize ;
}

size_t Palesibyl::nncuda_CalcMatrixGradientBlockY( size_t x, size_t y )
{
	const size_t	nBlockSize = nncuda_CalcMatrixGradientBlockSizeY( x, y ) ;
	return	(y + nBlockSize - 1) / nBlockSize ;
}

void Palesibyl::nncuda_CalcMatrixGradient_Clamp
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	cuda_CalcMatrixGradient<NNBufClampSampler>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_Edge
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	cuda_CalcMatrixGradient<NNBufEdgeSampler>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_Conv_Clamp
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	cuda_CalcMatrixGradient<NNBufConvClampSampler>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_Conv_Edge
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	cuda_CalcMatrixGradient<NNBufConvEdgeSampler>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_UpSampler
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	assert( (sp.m_xStride == 1) && (sp.m_yStride == 1) ) ;
	assert( (sp.m_xOffset == 0) && (sp.m_yOffset == 0) ) ;
	cuda_CalcMatrixGradient<NNBufUpSampler>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_OneHot
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream )
{
	assert( dimSrc.z == 1 ) ;
	cuda_CalcMatrixGradient<NNBufOneHotSampler>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc, sp, stream ) ;
}



