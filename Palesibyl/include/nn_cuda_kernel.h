
#ifndef	__NN_CUDA_KERNEL_H__
#define	__NN_CUDA_KERNEL_H__

#include "nn_cuda_def.h"
#include "nn_type_def.h"

namespace	Palesibyl
{

// メモリ操作
//////////////////////////////////////////////////////////////////////////////

// 値で埋める
void nncuda_FillMemory
	( float * pDst, NNBufDim dimDst, float fill, cudaStream_t stream ) ;

// サンプルごとに pMask[dimDst.z] で乗算する
void nncuda_MaskPattern
	( float * pDst, NNBufDim dimDst, const float * pMask, cudaStream_t stream ) ;

// サンプルを移動しながらチャネルをコピー
// （出力先のシフト元が範囲外の場合、ソースをシフトせずにコピー）
void nncuda_ShiftMoveMemory
	( float * pDst, NNBufDim dimDst,
		size_t xDstOffset, size_t yDstOffset, size_t iDstChannel,
		size_t nDstWidth, size_t nDstHeight,
		float * pSrc, NNBufDim dimSrc, int xShift, int yShift,
		size_t iSrcChannel, size_t nChannelCount,
		float scaleFactor, cudaStream_t stream ) ;

// サンプルを移動しながらチャネルを加算
// （出力先のシフト元が範囲外の場合、ソースをシフトせずに加算）
void nncuda_ShiftAddMemory
	( float * pDst, NNBufDim dimDst,
		size_t xDstOffset, size_t yDstOffset, size_t iDstChannel,
		size_t nDstWidth, size_t nDstHeight,
		float * pSrc, NNBufDim dimSrc, int xShift, int yShift,
		size_t iSrcChannel, size_t nChannelCount,
		float scaleFactor, cudaStream_t stream ) ;


// 行列計算
//////////////////////////////////////////////////////////////////////////////

void nncuda_Matrix_Clamp
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		size_t xLeftBounds, int nDepthwise,
		const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_Matrix_Edge
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		size_t xLeftBounds, int nDepthwise,
		const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_Matrix_Conv_Clamp
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		size_t xLeftBounds, int nDepthwise,
		const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_Matrix_Conv_Edge
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		size_t xLeftBounds, int nDepthwise,
		const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_Matrix_UpSampler
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		size_t xLeftBounds, int nDepthwise,
		const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_Matrix_OneHot
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		const float * pMatrix,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		size_t xLeftBounds, int nDepthwise,
		const NNSamplingParam& sp, cudaStream_t stream ) ;



// 行列のδ逆伝播
//////////////////////////////////////////////////////////////////////////////

void nncuda_Matrix_DeltaBack_Injection
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_Matrix_DeltaBack_Injection_Sp
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_Matrix_DeltaBack_Conv
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_Matrix_DeltaBack_Conv_Sp
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_Matrix_DeltaBack_UpSampler
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pMatrix,
		int xMatrix, int yMatrix, size_t zSrcChannels,
		int nDepthwise, const NNSamplingParam& sp, cudaStream_t stream ) ;

size_t nncuda_IsAcceptableMatrixSize
	( size_t xMatrix, size_t yMatrix, size_t zSrcChannels ) ;



// 行列勾配計算
//////////////////////////////////////////////////////////////////////////////

void nncuda_CalcMatrixGradient_Clamp
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_CalcMatrixGradient_Edge
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_CalcMatrixGradient_Edge_Sp
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_CalcMatrixGradient_Conv_Clamp
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_CalcMatrixGradient_Conv_Edge
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_CalcMatrixGradient_Conv_Edge_Sp
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_CalcMatrixGradient_UpSampler
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream ) ;

void nncuda_CalcMatrixGradient_OneHot
	( float * pGradient, NNBufDim dimGradient,
		size_t xGradientBlockSize, size_t yGradientBlockSize,
		size_t xMatrix, size_t yMatrix, size_t iMatrixBias,
		const float * pDelta, NNBufDim dimDelta,
		const float * pSrc, NNBufDim dimSrc,
		const NNSamplingParam& sp, cudaStream_t stream ) ;

size_t nncuda_CalcMatrixGradientBlockSizeX( size_t x, size_t y ) ;
size_t nncuda_CalcMatrixGradientBlockSizeY( size_t x, size_t y ) ;
size_t nncuda_CalcMatrixGradientBlockX( size_t x, size_t y ) ;
size_t nncuda_CalcMatrixGradientBlockY( size_t x, size_t y ) ;


// 活性化関数
//////////////////////////////////////////////////////////////////////////////

void nncuda_Activation_Linear
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_ReLU
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_Sigmoid
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_Tanh
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_Softmax
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_Argmax
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_MaxPool
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_Multiply
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream ) ;

size_t nncuda_IsAcceptableActivationChannels( size_t zDstChannels, size_t zSrcChannels ) ;



// 活性化関数のδ逆伝播
//////////////////////////////////////////////////////////////////////////////

void nncuda_Activation_DeltaBack_Linear
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct,
		int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_DeltaBack_ReLU
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct,
		int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_DeltaBack_Sigmoid
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct,
		int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_DeltaBack_Tanh
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct,
		int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_DeltaBack_Softmax
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct,
		int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_DeltaBack_Argmax
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct,
		int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_DeltaBack_MaxPool
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct,
		int nDepthwise, cudaStream_t stream ) ;

void nncuda_Activation_DeltaBack_Multiply
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct,
		int nDepthwise, cudaStream_t stream ) ;



// 損失関数δ
//////////////////////////////////////////////////////////////////////////////

void nncuda_LossDelta_MSE
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, const NNLossParam& lp, cudaStream_t stream ) ;

void nncuda_LossDelta_MAE
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, const NNLossParam& lp, cudaStream_t stream ) ;

void nncuda_LossDelta_Argmax
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, const NNLossParam& lp, cudaStream_t stream ) ;

void nncuda_LossDelta_FastArgmax
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, const NNLossParam& lp, cudaStream_t stream ) ;



// 正規化
//////////////////////////////////////////////////////////////////////////////

size_t nncuda_CalcAggregateSize( size_t n ) ;

void nncuda_AggregateSample
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		int zSampling, cudaStream_t stream ) ;

void nncuda_AggregateSample2
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc, cudaStream_t stream ) ;

void nncuda_CalcDistribution
	( float * pMeanVar, NNBufDim dimMeanVar,
		const float * pSrc1, NNBufDim dimSrc1,
		const float * pSrc2, NNBufDim dimSrc2, cudaStream_t stream ) ;

void nncuda_Normalize
	( float * pSample, NNBufDim dimSample, size_t xSampleBounds,
		const float * pParams,
		const float * pMeanVar, int zSampling, cudaStream_t stream ) ;

void nncuda_NormDeltaBack
	( float * pDelta, NNBufDim dimDelta,
		float * pGradient, NNBufDim dimGradient,
		const float * pDstSample, NNBufDim dimDstSample,
		const float * pParams,
		const float * pMeanVar, int zSampling, cudaStream_t stream ) ;

}

#endif

