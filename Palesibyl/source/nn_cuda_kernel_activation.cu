
#include "nn_cuda_kernel_activation.h"



//////////////////////////////////////////////////////////////////////////////
// 活性化関数
//////////////////////////////////////////////////////////////////////////////

void Palesibyl::nncuda_Activation_Linear
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation<NNAFunctionLinear>
		( pDst, dimDst, pSrc, dimSrc, xLeftBounds, nDepthwise, stream ) ;
}

void Palesibyl::nncuda_Activation_ReLU
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation<NNAFunctionReLU>
		( pDst, dimDst, pSrc, dimSrc, xLeftBounds, nDepthwise, stream ) ;
}

void Palesibyl::nncuda_Activation_Sigmoid
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation<NNAFunctionSigmoid>
		( pDst, dimDst, pSrc, dimSrc, xLeftBounds, nDepthwise, stream ) ;
}

void Palesibyl::nncuda_Activation_Tanh
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation<NNAFunctionTanh>
		( pDst, dimDst, pSrc, dimSrc, xLeftBounds, nDepthwise, stream ) ;
}

void Palesibyl::nncuda_Activation_Softmax
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation<NNAFunctionSoftmax>
		( pDst, dimDst, pSrc, dimSrc, xLeftBounds, nDepthwise, stream ) ;
}

/*
void Palesibyl::nncuda_Activation_Argmax
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation<NNAFunctionArgmax>
		( pDst, dimDst, pSrc, dimSrc, xLeftBounds, nDepthwise, stream ) ;
}
*/

void Palesibyl::nncuda_Activation_MaxPool
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation<NNAFunctionMaxPool>
		( pDst, dimDst, pSrc, dimSrc, xLeftBounds, nDepthwise, stream ) ;
}

void Palesibyl::nncuda_Activation_Multiply
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation<NNAFunctionMultiply>
		( pDst, dimDst, pSrc, dimSrc, xLeftBounds, nDepthwise, stream ) ;
}


void Palesibyl::nncuda_Activation_Exp
	( float * pDst, NNBufDim dimDst,
		const float * pSrc, NNBufDim dimSrc,
		size_t xLeftBounds, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation<NNAFunctionExp>
		( pDst, dimDst, pSrc, dimSrc, xLeftBounds, nDepthwise, stream ) ;
}


size_t Palesibyl::nncuda_IsAcceptableActivationChannels
	( size_t zDstChannels, size_t zSrcChannels )
{
	return	(zDstChannels <= cudaSharedMemorySize/4/sizeof(float))
			&& (zSrcChannels <= cudaSharedMemorySize/4/sizeof(float)) ;
}



//////////////////////////////////////////////////////////////////////////////
// 活性化関数のδ逆伝播
//////////////////////////////////////////////////////////////////////////////

void Palesibyl::nncuda_Activation_DeltaBack_Linear
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation_DeltaBack<NNAFunctionLinear>
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, nDepthwise, stream ) ;
}

void Palesibyl::nncuda_Activation_DeltaBack_ReLU
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation_DeltaBack<NNAFunctionReLU>
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, nDepthwise, stream ) ;
}

void Palesibyl::nncuda_Activation_DeltaBack_Sigmoid
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation_DeltaBack<NNAFunctionSigmoid>
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, nDepthwise, stream ) ;
}

void Palesibyl::nncuda_Activation_DeltaBack_Tanh
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation_DeltaBack<NNAFunctionTanh>
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, nDepthwise, stream ) ;
}

void Palesibyl::nncuda_Activation_DeltaBack_Softmax
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation_DeltaBack<NNAFunctionSoftmax>
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, nDepthwise, stream ) ;
}

void Palesibyl::nncuda_Activation_DeltaBack_Argmax
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation_DeltaBack<NNAFunctionArgmax>
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, nDepthwise, stream ) ;
}

void Palesibyl::nncuda_Activation_DeltaBack_MaxPool
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation_DeltaBack<NNAFunctionMaxPool>
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, nDepthwise, stream ) ;
}

void Palesibyl::nncuda_Activation_DeltaBack_Multiply
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation_DeltaBack<NNAFunctionMultiply>
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, nDepthwise, stream ) ;
}


void Palesibyl::nncuda_Activation_DeltaBack_Exp
	( float * pDstDelta, NNBufDim dimDstDelta,
		const float * pSrcDelta, NNBufDim dimSrcDelta,
		const float * pSrcAct, NNBufDim dimSrcAct,
		const float * pOutAct, NNBufDim dimOutAct, int nDepthwise, cudaStream_t stream )
{
	nncuda_Activation_DeltaBack<NNAFunctionExp>
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, nDepthwise, stream ) ;
}



//////////////////////////////////////////////////////////////////////////////
// 損失関数δ
//////////////////////////////////////////////////////////////////////////////

void Palesibyl::nncuda_LossDelta_MSE
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, const NNLossParam& lp, cudaStream_t stream )
{
	nncuda_LossDelta<NNFunctionLossMSE,NNLossParam>
		( pLossDelta, dimLossDelta,
			pInAct, dimInAct, pOutput, dimOutput,
			pTeaching, dimTeaching, nDepthwise, lp, stream ) ;
}

void Palesibyl::nncuda_LossDelta_MAE
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, const NNLossParam& lp, cudaStream_t stream )
{
	nncuda_LossDelta<NNFunctionLossMAE,NNLossParam>
		( pLossDelta, dimLossDelta,
			pInAct, dimInAct, pOutput, dimOutput,
			pTeaching, dimTeaching, nDepthwise, lp, stream ) ;
}

void Palesibyl::nncuda_LossDelta_Argmax
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, const NNLossParam& lp, cudaStream_t stream )
{
	nncuda_LossDelta<NNFunctionLossArgmax,NNLossParam>
		( pLossDelta, dimLossDelta,
			pInAct, dimInAct, pOutput, dimOutput,
			pTeaching, dimTeaching, nDepthwise, lp, stream ) ;
}

void Palesibyl::nncuda_LossDelta_FastArgmax
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, const NNLossParam& lp, cudaStream_t stream )
{
	nncuda_LossDelta<NNFunctionLossFastArgmax,NNLossParam>
		( pLossDelta, dimLossDelta,
			pInAct, dimInAct, pOutput, dimOutput,
			pTeaching, dimTeaching, nDepthwise, lp, stream ) ;
}

void Palesibyl::nncuda_LossDelta_BernoulliNLL
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, const NNLossParam& lp, cudaStream_t stream )
{
	nncuda_LossDelta<NNFunctionLossBernoulliNLL,NNLossParam>
		( pLossDelta, dimLossDelta,
			pInAct, dimInAct, pOutput, dimOutput,
			pTeaching, dimTeaching, nDepthwise, lp, stream ) ;
}

void Palesibyl::nncuda_LossDelta_MeanForKLDivergence
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, const NNLossParam2& lp, cudaStream_t stream )
{
	nncuda_LossDelta<NNFunctionLossMeanForKLDivergence,NNLossParam2>
		( pLossDelta, dimLossDelta,
			pInAct, dimInAct, pOutput, dimOutput,
			pTeaching, dimTeaching, nDepthwise, lp, stream ) ;
}

void Palesibyl::nncuda_LossDelta_VarianceForKLDivergence
	( float * pLossDelta, NNBufDim dimLossDelta,
		const float * pInAct, NNBufDim dimInAct,
		const float * pOutput, NNBufDim dimOutput,
		const float * pTeaching, NNBufDim dimTeaching,
		int nDepthwise, const NNLossParam2& lp, cudaStream_t stream )
{
	nncuda_LossDelta<NNFunctionLossVarianceForKLDivergence,NNLossParam2>
		( pLossDelta, dimLossDelta,
			pInAct, dimInAct, pOutput, dimOutput,
			pTeaching, dimTeaching, nDepthwise, lp, stream ) ;
}

