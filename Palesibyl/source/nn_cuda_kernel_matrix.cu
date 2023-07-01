
#include "nn_cuda_kernel_matrix.h"



//////////////////////////////////////////////////////////////////////////////
// 行列計算
//////////////////////////////////////////////////////////////////////////////

void Palesibyl::nncuda_Matrix_Clamp
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	nncuda_Matrix<NNBufClampSampler>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xStride, yStride, xOffset, yOffset,
			nDepthwise, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_Matrix_Edge
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	nncuda_Matrix<NNBufEdgeSampler>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xStride, yStride, xOffset, yOffset,
			nDepthwise, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_Matrix_Conv_Clamp
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	nncuda_Matrix<NNBufConvClampSampler>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xStride, yStride, xOffset, yOffset,
			nDepthwise, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_Matrix_Conv_Edge
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	nncuda_Matrix<NNBufConvEdgeSampler>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xStride, yStride, xOffset, yOffset,
			nDepthwise, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_Matrix_Up2x2
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	assert( (xStride == 1) && (yStride == 1) ) ;
	assert( (xOffset == 0) && (yOffset == 0) ) ;
	nncuda_Matrix<NNBufUpSampler2x2>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xStride, yStride, xOffset, yOffset,
			nDepthwise, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_Matrix_Up4x4
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	assert( (xStride == 1) && (yStride == 1) ) ;
	assert( (xOffset == 0) && (yOffset == 0) ) ;
	nncuda_Matrix<NNBufUpSampler4x4>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xStride, yStride, xOffset, yOffset,
			nDepthwise, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_Matrix_Up8x8
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	assert( (xStride == 1) && (yStride == 1) ) ;
	assert( (xOffset == 0) && (yOffset == 0) ) ;
	nncuda_Matrix<NNBufUpSampler8x8>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xStride, yStride, xOffset, yOffset,
			nDepthwise, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_Matrix_Up16x16
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	assert( (xStride == 1) && (yStride == 1) ) ;
	assert( (xOffset == 0) && (yOffset == 0) ) ;
	nncuda_Matrix<NNBufUpSampler16x16>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xStride, yStride, xOffset, yOffset,
			nDepthwise, xConv, yConv, stream ) ;
}

/*
void Palesibyl::nncuda_Matrix_OneHot
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	assert( dimSrc.z == 1 ) ;
	nncuda_Matrix<NNBufOneHotSampler>
		( pDst, dimDst, pSrc, dimSrc,
			pMatrix, xMatrix, yMatrix, iMatrixBias,
			xStride, yStride, xOffset, yOffset,
			nDepthwise, xConv, yConv, stream ) ;
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
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	nncuda_Matrix_DeltaBack<NNBufSampler>
		( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
			pMatrix, xMatrix, yMatrix, zSrcChannels,
			xStride, yStride, xOffset, yOffset, nDepthwise, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_Matrix_DeltaBack_Conv
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	nncuda_Matrix_DeltaBack<NNBufConvClampSampler>
		( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
			pMatrix, xMatrix, yMatrix, zSrcChannels,
			xStride, yStride, xOffset, yOffset, nDepthwise, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_Matrix_DeltaBack_Up2x2
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	assert( (xStride == 1) && (yStride == 1) ) ;
	assert( (xOffset == 0) && (yOffset == 0) ) ;
	nncuda_Matrix_DeltaBack<NNBufUpSampler2x2>
		( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
			pMatrix, xMatrix, yMatrix, zSrcChannels,
			xStride, yStride, xOffset, yOffset, nDepthwise, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_Matrix_DeltaBack_Up4x4
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	assert( (xStride == 1) && (yStride == 1) ) ;
	assert( (xOffset == 0) && (yOffset == 0) ) ;
	nncuda_Matrix_DeltaBack<NNBufUpSampler4x4>
		( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
			pMatrix, xMatrix, yMatrix, zSrcChannels,
			xStride, yStride, xOffset, yOffset, nDepthwise, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_Matrix_DeltaBack_Up8x8
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	assert( (xStride == 1) && (yStride == 1) ) ;
	assert( (xOffset == 0) && (yOffset == 0) ) ;
	nncuda_Matrix_DeltaBack<NNBufUpSampler8x8>
		( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
			pMatrix, xMatrix, yMatrix, zSrcChannels,
			xStride, yStride, xOffset, yOffset, nDepthwise, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_Matrix_DeltaBack_Up16x16
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int xStride, int yStride, int xOffset, int yOffset,
		int nDepthwise, int xConv, int yConv, cudaStream_t stream )
{
	assert( (xStride == 1) && (yStride == 1) ) ;
	assert( (xOffset == 0) && (yOffset == 0) ) ;
	nncuda_Matrix_DeltaBack<NNBufUpSampler16x16>
		( pDstDelta, dimDstDelta, pSrcDelta, dimSrcDelta,
			pMatrix, xMatrix, yMatrix, zSrcChannels,
			xStride, yStride, xOffset, yOffset, nDepthwise, xConv, yConv, stream ) ;
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
	return	GradientBlockX ;
}

size_t Palesibyl::nncuda_CalcMatrixGradientBlockSizeY( size_t x, size_t y )
{
	return	GradientBlockY ;
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
		int xStride, int yStride, int xOffset, int yOffset,
		int xConv, int yConv, cudaStream_t stream )
{
	cuda_CalcMatrixGradient<NNBufClampSampler>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc,
			xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_Edge
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		int xStride, int yStride, int xOffset, int yOffset,
		int xConv, int yConv, cudaStream_t stream )
{
	cuda_CalcMatrixGradient<NNBufEdgeSampler>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc,
			xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_Conv_Clamp
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		int xStride, int yStride, int xOffset, int yOffset,
		int xConv, int yConv, cudaStream_t stream )
{
	cuda_CalcMatrixGradient<NNBufConvClampSampler>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc,
			xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_Conv_Edge
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		int xStride, int yStride, int xOffset, int yOffset,
		int xConv, int yConv, cudaStream_t stream )
{
	cuda_CalcMatrixGradient<NNBufConvEdgeSampler>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc,
			xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_Up2x2
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		int xStride, int yStride, int xOffset, int yOffset,
		int xConv, int yConv, cudaStream_t stream )
{
	assert( (xStride == 1) && (yStride == 1) ) ;
	assert( (xOffset == 0) && (yOffset == 0) ) ;
	cuda_CalcMatrixGradient<NNBufUpSampler2x2>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc,
			xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_Up4x4
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		int xStride, int yStride, int xOffset, int yOffset,
		int xConv, int yConv, cudaStream_t stream )
{
	assert( (xStride == 1) && (yStride == 1) ) ;
	assert( (xOffset == 0) && (yOffset == 0) ) ;
	cuda_CalcMatrixGradient<NNBufUpSampler4x4>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc,
			xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_Up8x8
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		int xStride, int yStride, int xOffset, int yOffset,
		int xConv, int yConv, cudaStream_t stream )
{
	assert( (xStride == 1) && (yStride == 1) ) ;
	assert( (xOffset == 0) && (yOffset == 0) ) ;
	cuda_CalcMatrixGradient<NNBufUpSampler8x8>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc,
			xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_Up16x16
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		int xStride, int yStride, int xOffset, int yOffset,
		int xConv, int yConv, cudaStream_t stream )
{
	assert( (xStride == 1) && (yStride == 1) ) ;
	assert( (xOffset == 0) && (yOffset == 0) ) ;
	cuda_CalcMatrixGradient<NNBufUpSampler16x16>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc,
			xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
}

void Palesibyl::nncuda_CalcMatrixGradient_OneHot
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		int xStride, int yStride, int xOffset, int yOffset,
		int xConv, int yConv, cudaStream_t stream )
{
	assert( dimSrc.z == 1 ) ;
	cuda_CalcMatrixGradient<NNBufOneHotSampler>
		( pGradient, dimGradient,
			xGradientBlockSize, yGradientBlockSize,
			xMatrix, yMatrix, iMatrixBias,
			pDelta, dimDelta, pSrc, dimSrc,
			xStride, yStride, xOffset, yOffset, xConv, yConv, stream ) ;
}



