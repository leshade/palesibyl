
#ifndef	__NN_FUNCTION_H__
#define	__NN_FUNCTION_H__

#include <math.h>
#include "nn_cuda_util.h"
#include "nn_type_def.h"
#include "nn_cuda_kernel.h"
#include "nn_matrix.h"
#include "nn_sampler.h"

#ifndef __max
	#define __max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef __min
	#define __min(a, b) (((a) < (b)) ? (a) : (b))
#endif


namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// 定数値
//////////////////////////////////////////////////////////////////////////////

// exp(x) の引数に入れられる凡その最大値
//（大きな値の時に inf -> nan 大量発生をさけるため）
constexpr static const float	EXP_MAX_CAP = 80.0f ; // 88.5f ; ※少し余裕を持たせる

inline __NN_CUDA_DEV__ float exp_sd( float x )
{
	return	(float) exp( __min( x, EXP_MAX_CAP ) ) ;
}

inline float exp_s( float x )
{
	assert( x < EXP_MAX_CAP ) ;
	return	(float) exp( __min( x, EXP_MAX_CAP ) ) ;
}

// argmax の出力形式
enum	ArgmaxOutputChannels
{
	argmaxIndex	= 0,	// 最大指標
	argmaxProbability,	// 最大確率
	argmaxSumExp,		// 合計 exp(x(i) - bias)
	argmaxExpBias,		// exp(x(i)-bias) が exp で計算可能な範囲に収まるための bias
	argmaxChannelCount,
} ;



//////////////////////////////////////////////////////////////////////////////
// 損失関数
//////////////////////////////////////////////////////////////////////////////

// 損失関数 : 平均二乗誤差 (L2 loss)
//////////////////////////////////////////////////////////////////////////////
class	NNFunctionLossMSE
{
public:
	constexpr static const char	FunctionName[] = "loss_mse" ;
	typedef	NNLossParam	LossParam ;

	static inline bool IsValidTeachingChannels
		( size_t nSrcActChannels, size_t nDepthwise, size_t nTeachingChannels )
	{
		return	(nSrcActChannels == nTeachingChannels) ;
	}
	static inline float Loss
		( const float * pInAct, const float * pOutAct,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		float	loss = 0.0f ;
		for ( size_t i = 0; i < nCount; i ++ )
		{
			float	d = pInAct[i] - pTeaching[i] ;
			loss += d * d ;
		}
		return	loss ;
	}
	static inline void LossDelta
		( float * pLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		for ( size_t i = 0; i < nCount; i ++ )
		{
			// ※正確には 2 * (pOutput[i] - pTeaching[i]) だが係数は省略
			pLossDelta[i] = pOutput[i] - pTeaching[i] ;
		}
	}
	static inline __NN_CUDA_DEV__ float kernelLossDelta
		( size_t iDstLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		return	pOutput[iDstLossDelta] - pTeaching[iDstLossDelta] ;
	}
	static inline void cudaLossDelta
		( float * pLossDelta, NNBufDim dimLossDelta,
			const float * pInAct, NNBufDim dimInAct,
			const float * pOutput, NNBufDim dimOutput,
			const float * pTeaching, NNBufDim dimTeaching,
			size_t nDepthwise, const LossParam& lp, cudaStream_t stream )
	{
		nncuda_LossDelta_MSE
			( pLossDelta, dimLossDelta,
				pInAct, dimInAct, pOutput, dimOutput,
				pTeaching, dimTeaching, (int) nDepthwise, lp, stream ) ;
	}
} ;

// 損失関数 : 平均絶対誤差 (L1 loss)
//////////////////////////////////////////////////////////////////////////////
class	NNFunctionLossMAE
{
public:
	constexpr static const char	FunctionName[] = "loss_mae" ;
	typedef	NNLossParam	LossParam ;

	static inline bool IsValidTeachingChannels
		( size_t nSrcActChannels, size_t nDepthwise, size_t nTeachingChannels )
	{
		return	(nSrcActChannels == nTeachingChannels) ;
	}
	static inline float Loss
		( const float * pInAct, const float * pOutAct,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		float	loss = 0.0f ;
		for ( size_t i = 0; i < nCount; i ++ )
		{
			float	d = pInAct[i] - pTeaching[i] ;
			loss += (float) fabs(d) ;
		}
		return	loss ;
	}
	static inline void LossDelta
		( float * pLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		for ( size_t i = 0; i < nCount; i ++ )
		{
			float	d = pOutput[i] - pTeaching[i] ;
			pLossDelta[i] = (d < 0.0f) ? -1.0f : ((d > 0.0f) ? 1.0f : 0.0f) ;
		}
	}
	static inline __NN_CUDA_DEV__ float kernelLossDelta
		( size_t iDstLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		float	d = pOutput[iDstLossDelta] - pTeaching[iDstLossDelta] ;
		return	(d < 0.0f) ? -1.0f : ((d > 0.0f) ? 1.0f : 0.0f) ; ;
	}
	static inline void cudaLossDelta
		( float * pLossDelta, NNBufDim dimLossDelta,
			const float * pInAct, NNBufDim dimInAct,
			const float * pOutput, NNBufDim dimOutput,
			const float * pTeaching, NNBufDim dimTeaching,
			size_t nDepthwise, const LossParam& lp, cudaStream_t stream )
	{
		nncuda_LossDelta_MAE
			( pLossDelta, dimLossDelta,
				pInAct, dimInAct, pOutput, dimOutput,
				pTeaching, dimTeaching, (int) nDepthwise, lp, stream ) ;
	}
} ;

// 損失関数 : クロスエントロピー（σ(x)）
//////////////////////////////////////////////////////////////////////////////
class	NNFunctionLossSigmoid	: public NNFunctionLossMSE
{
public:
	constexpr static const char	FunctionName[] = "loss_sigmoid" ;

	static inline float Loss
		( const float * pInAct, const float * pOutAct,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		float	loss = 0.0f ;
		for ( size_t i = 0; i < nCount; i ++ )
		{
			loss += (float) log(1.0f + exp_s(pInAct[i]))
								- pTeaching[i] * pInAct[i] ;
		}
		return	loss ;
	}
	// ※ LossDelta は σ(pInAct[i]) - pTeaching[i] となり
	//    NNLossFunctionMSE::LossDelta と一致する
} ;

// 損失関数 : クロスエントロピー（ソフトマックス関数）
//////////////////////////////////////////////////////////////////////////////
class	NNFunctionLossSoftmax	: public NNFunctionLossMSE
{
public:
	constexpr static const char	FunctionName[] = "loss_softmax" ;

	static inline float Loss
		( const float * pInAct, const float * pOutAct,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		double	loss = 0.0 ;
		for ( size_t z = 0; z < nDepthwise; z ++ )
		{
			float	ySum = 0.0f ;
			float	eSum = 0.0f ;
			for ( size_t i = z; i < nCount; i += nDepthwise )
			{
				ySum += pTeaching[i] * pInAct[i] ;
				eSum += exp_s( pInAct[i] ) ;
			}
			loss += log( eSum ) - ySum ;
		}
		return	(float) loss ;
	}
	// ※ LossDelta は Softmax(pInAct)[i] - pTeaching[i] となり
	//    NNLossFunctionMSE::LossDelta と一致する
} ;

// 損失関数 : クロスエントロピー（アーギュマックス関数・最大値インデックス出力）
//////////////////////////////////////////////////////////////////////////////
class	NNFunctionLossArgmax	: public NNFunctionLossSoftmax
{
public:
	constexpr static const char	FunctionName[] = "loss_argmax" ;

	static inline bool IsValidTeachingChannels
		( size_t nSrcActChannels, size_t nDepthwise, size_t nTeachingChannels )
	{
		return	(nDepthwise == nTeachingChannels) ;
	}
	static inline float Loss
		( const float * pInAct, const float * pOutAct,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		double	loss = 0.0 ;
		for ( size_t z = 0; z < nDepthwise; z ++ )
		{
			const size_t	iOneHot = ((size_t) floor( pTeaching[z] )) * nDepthwise + z ;
			assert( (iOneHot % nDepthwise) == z ) ;
			assert( iOneHot < nCount ) ;
			float	ySum = 0.0f ;
			float	eSum = 0.0f ;
			if ( iOneHot < nCount )
			{
				ySum += pInAct[iOneHot] ;
			}
			for ( size_t i = z; i < nCount; i += nDepthwise )
			{
				eSum += exp_s( pInAct[i] ) ;
			}
			loss += log( eSum ) - ySum ;
		}
		return	(float) loss ;
	}
	static inline void LossDelta
		( float * pLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		for ( size_t z = 0; z < nDepthwise; z ++ )
		{
			const size_t	iOneHot = ((size_t) floor( pTeaching[z] )) * nDepthwise + z ;
			//const size_t	iMax = (size_t) floor( pOutput[z * argmaxChannelCount + argmaxProbability] ) ;
			const float		eSum = __max( pOutput[z * argmaxChannelCount + argmaxSumExp], 0.0000001f ) ;
			const float		bias = pOutput[z * argmaxChannelCount + argmaxExpBias] ;
			assert( iOneHot < nCount ) ;
			assert( (iOneHot % nDepthwise) == z ) ;
			for ( size_t i = z; i < nCount; i += nDepthwise )
			{
				float	delta = exp_s( pInAct[i] - bias ) / eSum ;
				if ( i == iOneHot )
				{
					delta -= 1.0f ;
				}
				pLossDelta[i] = delta ;
			}
		}
	}
	static inline __NN_CUDA_DEV__ float kernelLossDelta
		( size_t iDstLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		const size_t	zOut = iDstLossDelta % nDepthwise ;
		const size_t	iOneHot = ((size_t) floor( pTeaching[zOut] )) * nDepthwise + zOut ;
		const float		eSum = __max( pOutput[zOut * argmaxChannelCount + argmaxSumExp], 0.0000001f ) ;
		const float		bias = pOutput[zOut * argmaxChannelCount + argmaxExpBias] ;
		float	delta = exp_sd( pInAct[iDstLossDelta] - bias ) / eSum ;	// = Softmax(x)[iDstLossDelta]
		if ( iDstLossDelta == iOneHot )
		{
			delta -= 1.0f ;
		}
		return	delta ;
	}
	static inline void cudaLossDelta
		( float * pLossDelta, NNBufDim dimLossDelta,
			const float * pInAct, NNBufDim dimInAct,
			const float * pOutput, NNBufDim dimOutput,
			const float * pTeaching, NNBufDim dimTeaching,
			size_t nDepthwise, const LossParam& lp, cudaStream_t stream )
	{
		nncuda_LossDelta_Argmax
			( pLossDelta, dimLossDelta,
				pInAct, dimInAct, pOutput, dimOutput,
				pTeaching, dimTeaching, (int) nDepthwise, lp, stream ) ;
	}
} ;

// 損失関数 : クロスエントロピー（ソフトマックス関数・逆伝播高速化）
//////////////////////////////////////////////////////////////////////////////
class	NNFunctionLossFastSoftmax	: public NNFunctionLossSoftmax
{
public:
	constexpr static const char	FunctionName[] = "loss_fast_softmax" ;

	static inline void LossDelta
		( float * pLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		for ( size_t z = 0; z < nDepthwise; z ++ )
		{
			size_t	iOneHot = z ;
			float	pHotMax = pTeaching[z] ;
			size_t	iMax = z ;
			float	pMax = pOutput[z] ;
			for ( size_t i = z; i < nCount; i += nDepthwise )
			{
				if ( pHotMax < pTeaching[i] )
				{
					iOneHot = i ;
					pHotMax = pTeaching[i] ;
				}
				if ( pMax < pOutput[i] )
				{
					iMax = i ;
					pMax = pOutput[i] ;
				}
				pLossDelta[i] = 0.0f ;
			}
			pLossDelta[iOneHot] = pOutput[iOneHot] - pTeaching[iOneHot] ;
			if ( (iMax != iOneHot) && (iMax < nCount) )
			{
				pLossDelta[iMax] = pOutput[iMax] ;
			}
		}
	}
	// ※現状高速化はCPU実装のみ
} ;

// 損失関数 : クロスエントロピー
//（アーギュマックス関数・最大値インデックス出力・逆伝播高速化）
//////////////////////////////////////////////////////////////////////////////
class	NNFunctionLossFastArgmax	: public NNFunctionLossArgmax
{
public:
	constexpr static const char	FunctionName[] = "loss_fast_argmax" ;

	static inline void LossDelta
		( float * pLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		for ( size_t z = 0; z < nDepthwise; z ++ )
		{
			for ( size_t i = z; i < nCount; i += nDepthwise )
			{
				pLossDelta[i] = 0.0f ;
			}
			const size_t	iOneHot = ((size_t) floor( pTeaching[z] )) * nDepthwise + z ;
			const float		eSum = __max( pOutput[z * argmaxChannelCount + argmaxSumExp], 0.0000001f ) ;
			assert( iOneHot < nCount ) ;
			assert( (iOneHot % nDepthwise) == z ) ;
			if ( iOneHot < nCount )
			{
				pLossDelta[iOneHot] = exp_s( pInAct[iOneHot] ) / eSum - 1.0f ;
			}
			const size_t	iMax = ((size_t) floor( pOutput[z * argmaxChannelCount + argmaxIndex] ))
										* nDepthwise + z ;
			if ( (iMax != iOneHot) && (iMax < nCount) )
			{
				pLossDelta[iMax] = exp_s( pInAct[iMax] ) / eSum ;
			}
		}
	}
	static inline __NN_CUDA_DEV__ float kernelLossDelta
		( size_t iDstLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		const size_t	zOut = iDstLossDelta % nDepthwise ;
		const size_t	iOneHot = ((size_t) floor( pTeaching[zOut] )) * nDepthwise + zOut ;
		const size_t	iMax = ((size_t) floor( pOutput[zOut * argmaxChannelCount + argmaxIndex] ))
									* nDepthwise + zOut ;
		const float		eSum = __max( pOutput[zOut * argmaxChannelCount + argmaxSumExp], 0.0000001f ) ;
		if ( iDstLossDelta == iOneHot )
		{
			return	exp_sd( pInAct[iDstLossDelta] ) / eSum - 1.0f ;
		}
		else if ( iDstLossDelta == iMax )
		{
			return	exp_sd( pInAct[iDstLossDelta] ) / eSum ;
		}
		return	0.0f ;
	}
	static inline void cudaLossDelta
		( float * pLossDelta, NNBufDim dimLossDelta,
			const float * pInAct, NNBufDim dimInAct,
			const float * pOutput, NNBufDim dimOutput,
			const float * pTeaching, NNBufDim dimTeaching,
			size_t nDepthwise, const LossParam& lp, cudaStream_t stream )
	{
		nncuda_LossDelta_FastArgmax
			( pLossDelta, dimLossDelta,
				pInAct, dimInAct, pOutput, dimOutput,
				pTeaching, dimTeaching, (int) nDepthwise, lp, stream ) ;
	}
} ;



//////////////////////////////////////////////////////////////////////////////
// 活性化関数
//////////////////////////////////////////////////////////////////////////////

// 活性化関数 : 線形関数
//////////////////////////////////////////////////////////////////////////////
class	NNAFunctionLinear
{
public:
	constexpr static const char	FunctionName[] = "linear" ;
	constexpr static const bool	MustBeLastLayer = false ;		// 最終層でしか使用できない
	constexpr static const bool	MustNotBeLastLayer = false ;	// 最終層では使用できない

	// 出力チャネル数（なんらかの結合する場合は出力数は減じ得る）
	static inline size_t CalcOutChannels( size_t chInput, size_t nDepthwise )
	{
		return	chInput ;
	}
	// CUDA で利用可能なチャネル数か？
	static bool IsAcceptableChannelsForCuda( size_t chOutput, size_t chInput )
	{
		return	nncuda_IsAcceptableActivationChannels( chOutput, chInput ) ;
	}
	// 活性化関数（ pDst 要素数は CalcOutChannels() ）
	static inline __NN_CUDA_DEV__ float kernelPreActivation
		( const float * pSrc, size_t iSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	pSrc[iSrc] ;
	}
	static inline __NN_CUDA_DEV__  float kernelActivation
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	pSrc[iDst] ;
	}
	static inline void Activation
		( float * pDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		for ( int i = 0; i < nSrcCount; i ++ )
		{
			pDst[i] = pSrc[i] ;
		}
	}
	static inline void cudaActivation
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			size_t xLeftBounds, size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_Linear
			( pDst, dimDst, pSrc, dimSrc, xLeftBounds, (int) nDepthwise, stream ) ;
	}
	// 微分関数（ pDst 要素数は pSrc 要素数と同じ nSrcCount ）
	static inline __NN_CUDA_DEV__ float kernelPreDifferential
		( const float * pSrc, const float * pActOut,
				size_t iSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	pSrc[iSrc] ;
	}
	static inline __NN_CUDA_DEV__ float kernelDifferential
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	1.0f ;
	}
	static inline void Differential
		( float * pDst, const float * pSrc,
			const float * pActOut, size_t nSrcCount, size_t nDepthwise )
	{
		for ( int i = 0; i < nSrcCount; i ++ )
		{
			pDst[i] = 1.0f ;
		}
	}
	// δ逆伝播（ pSrcDelta 要素数は CalcOutChannels(), それ以外は nSrcCount ）
	static inline __NN_CUDA_DEV__ float BackDelta
			( size_t iDstDelta, const float * pSrcDelta,
				const float * pSrcDiff, size_t nSrcCount, size_t nDepthwise )
	{
		return	pSrcDelta[iDstDelta] * pSrcDiff[iDstDelta] ;
	}
	static inline void cudaBackDelta
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pSrcAct, NNBufDim dimSrcAct,
			const float * pOutAct, NNBufDim dimOutAct,
			size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_DeltaBack_Linear
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, (int) nDepthwise, stream ) ;
	}
} ;

class	NNAFunctionLinearMAE	: public NNAFunctionLinear
{
public:
	constexpr static const char	FunctionName[] = "l1_loss" ;
} ;

// 活性化関数 : 整流関数
//////////////////////////////////////////////////////////////////////////////
class	NNAFunctionReLU	: public NNAFunctionLinear
{
public:
	constexpr static const char	FunctionName[] = "relu" ;

	static inline __NN_CUDA_DEV__ float kernelActivation
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		float	x = pSrc[iDst] ;
		return	(x >= 0.0f) ? x : 0.0f ;
	}
	static inline void Activation
		( float * pDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t i = 0; i < nSrcCount; i ++ )
		{
			float	x = pSrc[i] ;
			pDst[i] = (x >= 0.0f) ? x : 0.0f ;
		}
	}
	static inline void cudaActivation
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			size_t xLeftBounds, size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_ReLU
			( pDst, dimDst, pSrc, dimSrc, xLeftBounds, (int) nDepthwise, stream ) ;
	}
	static inline __NN_CUDA_DEV__ float kernelDifferential
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		float	x = pSrc[iDst] ;
		return	(x >= 0.0f) ? 1.0f : 0.0f ;
	}
	static inline void Differential
		( float * pDst, const float * pSrc,
			const float * pActOut, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t i = 0; i < nSrcCount; i ++ )
		{
			float	x = pSrc[i] ;
			pDst[i] = (x >= 0.0f) ? 1.0f : 0.0f ;
		}
	}
	static inline void cudaBackDelta
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pSrcAct, NNBufDim dimSrcAct,
			const float * pOutAct, NNBufDim dimOutAct,
			size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_DeltaBack_ReLU
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, (int) nDepthwise, stream ) ;
	}
} ;


// 活性化関数 : σ(x)
//////////////////////////////////////////////////////////////////////////////
class	NNAFunctionSigmoid	: public NNAFunctionLinear
{
public:
	constexpr static const char	FunctionName[] = "sigmoid" ;

	static inline __NN_CUDA_DEV__ float Sigmoid( float x )
	{
		return	1.0f / (1.0f + exp_sd(-x)) ;
	}
	static inline __NN_CUDA_DEV__ float kernelActivation
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	Sigmoid( pSrc[iDst] ) ;
	}
	static inline __NN_CUDA_DEV__ void Activation
		( float * pDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t i = 0; i < nSrcCount; i ++ )
		{
			pDst[i] = Sigmoid( pSrc[i] ) ;
		}
	}
	static inline void cudaActivation
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			size_t xLeftBounds, size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_Sigmoid
			( pDst, dimDst, pSrc, dimSrc, xLeftBounds, (int) nDepthwise, stream ) ;
	}
	static inline __NN_CUDA_DEV__ float kernelPreDifferential
		( const float * pSrc, const float * pActOut,
				size_t iSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	pActOut[iSrc] ; // == Sigmoid(x)
	}
	static inline __NN_CUDA_DEV__ float kernelDifferential
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		float	s = pSrc[iDst] ; // == Sigmoid(x)
		return	s * (1.0f - s) ;
	}
	static inline __NN_CUDA_DEV__ void Differential
		( float * pDst, const float * pSrc,
			const float * pActOut, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t i = 0; i < nSrcCount; i ++ )
		{
			float	s = pActOut[i] ; // == Sigmoid(x) ;
			pDst[i] = s * (1.0f - s) ;
		}
	}
	static inline void cudaBackDelta
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pSrcAct, NNBufDim dimSrcAct,
			const float * pOutAct, NNBufDim dimOutAct,
			size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_DeltaBack_Sigmoid
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, (int) nDepthwise, stream ) ;
	}
} ;


// 活性化関数 : tanh(x)
//////////////////////////////////////////////////////////////////////////////
class	NNAFunctionTanh	: public NNAFunctionLinear
{
public:
	constexpr static const char	FunctionName[] = "tanh" ;

	static inline __NN_CUDA_DEV__ float kernelActivation
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	tanh( pSrc[iDst] ) ;
	}
	static inline __NN_CUDA_DEV__ void Activation
		( float * pDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t i = 0; i < nSrcCount; i ++ )
		{
			pDst[i] = tanh( pSrc[i] ) ;
		}
	}
	static inline void cudaActivation
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			size_t xLeftBounds, size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_Tanh
			( pDst, dimDst, pSrc, dimSrc, xLeftBounds, (int) nDepthwise, stream ) ;
	}
	static inline __NN_CUDA_DEV__ float kernelDifferential
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		float	x = pSrc[iDst] ;
		float	y = exp_sd(x) + exp_sd(-x) ;
		return	4.0f / (y * y) ;
	}
	static inline __NN_CUDA_DEV__ void Differential
		( float * pDst, const float * pSrc,
			const float * pActOut, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t i = 0; i < nSrcCount; i ++ )
		{
			float	x = pSrc[i] ;
			float	y = exp_sd(x) + exp_sd(-x) ;
			pDst[i] = 4.0f / (y * y) ;
		}
	}
	static inline void cudaBackDelta
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pSrcAct, NNBufDim dimSrcAct,
			const float * pOutAct, NNBufDim dimOutAct,
			size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_DeltaBack_Tanh
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, (int) nDepthwise, stream ) ;
	}
} ;


// 活性化関数 : ソフトマックス関数
//////////////////////////////////////////////////////////////////////////////
class	NNAFunctionSoftmax	: public NNAFunctionSigmoid
{
public:
	constexpr static const char	FunctionName[] = "softmax" ;

	static inline __NN_CUDA_DEV__ float kernelPreActivation
		( const float * pSrc, size_t iSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	exp_sd( pSrc[iSrc] ) ;
	}
	static inline __NN_CUDA_DEV__ float kernelActivation
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		float	d = 0.0f ;
		for ( size_t i = (iDst % nDepthwise); i < nSrcCount; i += nDepthwise )
		{
			d += pSrc[i] ;
		}
		return	pSrc[iDst] / __max( d, 0.0000001f ) ;
	}
	static inline void Activation
		( float * pDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t z = 0; z < nDepthwise; z ++ )
		{
			float	d = 0.0f ;
			for ( size_t i = z; i < nSrcCount; i += nDepthwise )
			{
				float	e = exp_s( pSrc[i] ) ;
				d += e ;
				pDst[i] = e ;
			}
			float	r = 1.0f / __max( d, 0.0000001f ) ;
			for ( size_t i = z; i < nSrcCount; i += nDepthwise )
			{
				pDst[i] *= r ;
			}
		}
	}
	static inline void cudaActivation
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			size_t xLeftBounds, size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_Softmax
			( pDst, dimDst, pSrc, dimSrc, xLeftBounds, (int) nDepthwise, stream ) ;
	}
	static inline __NN_CUDA_DEV__ float kernelPreDifferential
		( const float * pSrc, const float * pActOut,
				size_t iSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	pActOut[iSrc] ;	// == Softmax(x)[iSrc]
	}
	static inline __NN_CUDA_DEV__ float kernelDifferential
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		float	s = pSrc[iDst] ; // == Softmax(x)[iSrc]
		return	s * (1.0f - s) ;
	}
	static inline void Differential
		( float * pDst, const float * pSrc,
			const float * pActOut, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t i = 0; i < nSrcCount; i ++ )
		{
			float	s = pActOut[i] ;	// == Softmax(x)[i]
			pDst[i] = s * (1.0f - s) ;	
		}
	}
	static inline void cudaBackDelta
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pSrcAct, NNBufDim dimSrcAct,
			const float * pOutAct, NNBufDim dimOutAct,
			size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_DeltaBack_Softmax
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, (int) nDepthwise, stream ) ;
	}
} ;


// 活性化関数 : アーギュマックス関数（インデックス・確率・合計分母出力）
//////////////////////////////////////////////////////////////////////////////
class	NNAFunctionArgmax	: public NNAFunctionSoftmax
{
public:
	constexpr static const char	FunctionName[] = "argmax" ;
	constexpr static const bool	MustBeLastLayer = true ;	// 最終層でしか使用できない

	static inline size_t CalcOutChannels( size_t chInput, size_t nDepthwise )
	{
		return	nDepthwise * argmaxChannelCount ;
	}
	static bool IsAcceptableChannelsForCuda( size_t chOutput, size_t chInput )
	{
		return	true ;
	}
	static inline __NN_CUDA_DEV__ float kernelPreActivation
		( const float * pSrc, size_t iSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	pSrc[iSrc] ;
	}
	static inline __NN_CUDA_DEV__ float kernelActivation
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		const size_t	iSrc = iDst / argmaxChannelCount ;
		const size_t	type = iDst - iSrc * argmaxChannelCount ;
		size_t			iMax = 0 ;
		float			eMax = 0.0f ;
		float			eSum = 0.0f ;
		float			bias = 0.0f ;
		for ( size_t i = iSrc; i < nSrcCount; i += nDepthwise )
		{
			float	src = pSrc[i] ;
			if ( src - 10.0f > bias )
			{
				float	d = exp_sd( bias - (src - 10.0f) ) ;
				eMax *= d ;
				eSum *= d ;
				bias = src - 10.0f ;
			}
			float	e = exp_sd( src - bias ) ;
			if ( e > eMax )
			{
				iMax = i ;
				eMax = e ;
			}
			eSum += e ;
		}
		switch ( type )
		{
		case	argmaxIndex:
			return	(float) ((iMax - iSrc) / nDepthwise) ;
		case	argmaxProbability:
			return	eMax / __max( eSum, 0.00001f ) ;
		case	argmaxSumExp:
			return	eSum ;
		case	argmaxExpBias:
			return	bias ;
		default:
			break ;
		}
		return	eSum ;
	}
	static inline void Activation
		( float * pDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t z = 0, zx3 = 0; z < nDepthwise; z ++, zx3 += argmaxChannelCount )
		{
			size_t	iMax = z ;
			float	eMax = 0.0f ;
			float	eSum = 0.0f ;
			float	bias = 0.0f ;
			for ( size_t i = z; i < nSrcCount; i += nDepthwise )
			{
				float	src = pSrc[i] ;
				if ( src - 10.0f > bias )
				{
					float	d = exp_s( bias - (src - 10.0f) ) ;
					eMax *= d ;
					eSum *= d ;
					bias = src - 10.0f ;
				}
				float	e = (float) exp_s( src - bias ) ;
				if ( e > eMax )
				{
					iMax = i ;
					eMax = e ;
				}
				eSum += e ;
			}
			pDst[zx3 + argmaxIndex]       = (float) ((iMax - z) / nDepthwise) ;
			pDst[zx3 + argmaxProbability] = eMax / __max( eSum, 0.0000001f ) ;
			pDst[zx3 + argmaxSumExp]      = eSum ;
			pDst[zx3 + argmaxExpBias]     = bias ;
		}
	}
	static inline void cudaActivation
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			size_t xLeftBounds, size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_Argmax
			( pDst, dimDst, pSrc, dimSrc, xLeftBounds, (int) nDepthwise, stream ) ;
	}
	static inline __NN_CUDA_DEV__ float kernelPreDifferential
		( const float * pSrc, const float * pActOut,
				size_t iSrc, size_t nSrcCount, size_t nDepthwise )
	{
		const size_t	iDst = iSrc % nDepthwise ;
		const size_t	iMax = ((size_t) floor(pActOut[iDst * argmaxChannelCount + argmaxIndex]))
								* nDepthwise + iDst ;
		if ( iMax != iSrc )
		{
			return	0.0f ;	// Positive Sampling (逆伝播出来ないので)
		}
		return	exp_sd( pSrc[iSrc] )
					/ __max(pActOut[iDst*argmaxChannelCount+argmaxSumExp],0.0000001f) ;	// == Softmax(x)[iSrc]
	}
	static inline __NN_CUDA_DEV__ float kernelDifferential
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		float	s = pSrc[iDst] ; // == Softmax(x)[iSrc]
		return	s * (1.0f - s) ;
	}
	static inline void Differential
		( float * pDst, const float * pSrc,
			const float * pActOut, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t z = 0; z < nDepthwise; z ++ )
		{
			const size_t	iMax = ((size_t) floor(pActOut[z * argmaxChannelCount + argmaxIndex]))
									* nDepthwise + z ;
			const float		eSum = __max( pActOut[z * argmaxChannelCount + argmaxSumExp], 0.0000001f ) ;
			for ( size_t i = z; i < nSrcCount; i += nDepthwise )
			{
				if ( iMax == i )
				{
					float	s = exp_s(pSrc[i]) / eSum ;	// == Softmax(x)[i]
					pDst[i] = s * (1.0f - s) ;	
				}
				else
				{
					pDst[i] = 0.0f ;	// Positive Sampling (逆伝播出来ないので)
				}
			}
		}
	}
	static inline __NN_CUDA_DEV__ float BackDelta
			( size_t iDstDelta, const float * pSrcDelta,
				const float * pSrcDiff, size_t nSrcCount, size_t nDepthwise )
	{
		return	pSrcDelta[iDstDelta % nDepthwise] * pSrcDiff[iDstDelta] ;
	}
	static inline void cudaBackDelta
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pSrcAct, NNBufDim dimSrcAct,
			const float * pOutAct, NNBufDim dimOutAct,
			size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_DeltaBack_Argmax
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, (int) nDepthwise, stream ) ;
	}

} ;


// 活性化関数 : ソフトマックス関数
//////////////////////////////////////////////////////////////////////////////
class	NNAFunctionFastSoftmax	: public NNAFunctionSoftmax
{
public:
	constexpr static const char	FunctionName[] = "fast_softmax" ;
} ;


// 活性化関数 : アーギュマックス関数（インデックス・確率・合計分母出力）
//////////////////////////////////////////////////////////////////////////////
class	NNAFunctionFastArgmax	: public NNAFunctionArgmax
{
public:
	constexpr static const char	FunctionName[] = "fast_argmax" ;
} ;


// 活性化関数 : max(x)
//////////////////////////////////////////////////////////////////////////////
class	NNAFunctionMaxPool	: public NNAFunctionLinear
{
public:
	constexpr static const char	FunctionName[] = "maxpool" ;
	constexpr static const bool	MustNotBeLastLayer = true ;	// 最終層では使用できない

	static inline size_t CalcOutChannels( size_t chInput, size_t nDepthwise )
	{
		return	nDepthwise ;
	}
	static inline __NN_CUDA_DEV__ float kernelActivation
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		float	d = 0.0f ;
		if ( iDst < nSrcCount )
		{
			d = pSrc[iDst] ;
			for ( size_t i = iDst + nDepthwise; i < nSrcCount; i += nDepthwise )
			{
				d = __max( d, pSrc[i] ) ;
			}
		}
		return	d ;
	}
	static inline void Activation
		( float * pDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t z = 0; z < nDepthwise; z ++ )
		{
			float	d = 0.0f ;
			if ( z < nSrcCount )
			{
				d = pSrc[z] ;
				for ( size_t i = z + nDepthwise; i < nSrcCount; i += nDepthwise )
				{
					float	s = pSrc[i] ;
					d = (s > d) ? s : d ;
				}
			}
			pDst[z] = d ;
		}
	}
	static inline void cudaActivation
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			size_t xLeftBounds, size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_MaxPool
			( pDst, dimDst, pSrc, dimSrc, xLeftBounds, (int) nDepthwise, stream ) ;
	}
	static inline __NN_CUDA_DEV__ float kernelPreDifferential
		( const float * pSrc, const float * pActOut,
				size_t iSrc, size_t nSrcCount, size_t nDepthwise )
	{
		const size_t	zOut = (iSrc % nDepthwise) ;
		return	(pActOut[zOut] <= pSrc[iSrc]) ? 1.0f : 0.0f ;
	}
	static inline __NN_CUDA_DEV__ float kernelDifferential
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	pSrc[iDst] ;
	}
	static inline __NN_CUDA_DEV__ void Differential
		( float * pDst, const float * pSrc,
			const float * pActOut, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t z = 0; z < nDepthwise; z ++ )
		{
			const float	maxOut = pActOut[z] ;
			for ( size_t i = z; i < nSrcCount; i += nDepthwise )
			{
				pDst[i] = (maxOut <= pSrc[i]) ? 1.0f : 0.0f ;
			}
		}
	}
	static inline __NN_CUDA_DEV__ float BackDelta
			( size_t iDstDelta, const float * pSrcDelta,
				const float * pSrcDiff, size_t nSrcCount, size_t nDepthwise )
	{
		return	pSrcDelta[iDstDelta % nDepthwise] * pSrcDiff[iDstDelta] ;
	}
	static inline void cudaBackDelta
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pSrcAct, NNBufDim dimSrcAct,
			const float * pOutAct, NNBufDim dimOutAct,
			size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_DeltaBack_MaxPool
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, (int) nDepthwise, stream ) ;
	}
} ;


// 活性化関数 : mul(x,y)
//////////////////////////////////////////////////////////////////////////////
class	NNAFunctionMultiply	: public NNAFunctionLinear
{
public:
	constexpr static const char	FunctionName[] = "multiply" ;
	constexpr static const bool	MustNotBeLastLayer = true ;	// 最終層では使用きない

	static inline size_t CalcOutChannels( size_t chInput, size_t nDepthwise )
	{
		return	nDepthwise ;
	}
	static inline __NN_CUDA_DEV__ float kernelActivation
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		float	d = 1.0f ;
		if ( iDst < nSrcCount )
		{
			d = pSrc[iDst] ;
			for ( size_t i = iDst + nDepthwise; i < nSrcCount; i += nDepthwise )
			{
				d *= pSrc[i] ;
			}
		}
		return	d ;
	}
	static inline void Activation
		( float * pDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t z = 0; z < nDepthwise; z ++ )
		{
			float	d = 1.0f ;
			if ( z < nSrcCount )
			{
				d = pSrc[z] ;
				for ( size_t i = z + nDepthwise; i < nSrcCount; i += nDepthwise )
				{
					d *= pSrc[i] ;
				}
			}
			pDst[z] = d ;
		}
	}
	static inline void cudaActivation
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			size_t xLeftBounds, size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_Multiply
			( pDst, dimDst, pSrc, dimSrc, xLeftBounds, (int) nDepthwise, stream ) ;
	}
	static inline __NN_CUDA_DEV__ float kernelDifferential
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		float	d = 1.0f ;
		for ( size_t z = (iDst % nDepthwise); z < nSrcCount; z += nDepthwise )
		{
			if ( z != iDst )
			{
				d *= pSrc[z] ;
			}
		}
		return	d ;
	}
	static inline __NN_CUDA_DEV__ void Differential
		( float * pDst, const float * pSrc,
			const float * pActOut, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t z = 0; z < nDepthwise; z ++ )
		{
			for ( size_t iDst = z; iDst < nSrcCount; iDst += nDepthwise )
			{
				float	d = 1.0f ;
				for ( size_t i = z; i < nSrcCount; i += nDepthwise )
				{
					if ( i != iDst )
					{
						d *= pSrc[i] ;
					}
				}
				pDst[iDst] = d ;
			}
		}
	}
	static inline __NN_CUDA_DEV__ float BackDelta
			( size_t iDstDelta, const float * pSrcDelta,
				const float * pSrcDiff, size_t nSrcCount, size_t nDepthwise )
	{
		return	pSrcDelta[iDstDelta % nDepthwise] * pSrcDiff[iDstDelta] ;
	}
	static inline void cudaBackDelta
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pSrcAct, NNBufDim dimSrcAct,
			const float * pOutAct, NNBufDim dimOutAct,
			size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_DeltaBack_Multiply
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, (int) nDepthwise, stream ) ;
	}
} ;


}

#endif

