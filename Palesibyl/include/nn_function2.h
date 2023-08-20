
#ifndef	__NN_FUNCTION2_H__
#define	__NN_FUNCTION2_H__

#include "nn_function.h"

namespace	Palesibyl
{

// 損失関数 : Reconstruction Error (多変量ベルヌーイ分布の -log(B(x;p)) )
//////////////////////////////////////////////////////////////////////////////
class	NNFunctionLossBernoulliNLL
{
public:
	constexpr static const char	FunctionName[] = "loss_bernoulli_nll" ;
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
			const float	y = pTeaching[i] ;
			const float	x = pOutAct[i] ;
			const float	xs = __min( __max( x, 0.00001f ), 0.99999f ) ;
			loss -= y * (float) log(xs)
					+ (1.0f - y) * (float) log(1.0f - xs) ;
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
			const float	y = pTeaching[i] ;
			const float	x = pOutput[i] ;
			const float	xs = __min( __max( x, 0.00001f ), 0.99999f ) ;
			pLossDelta[i] = (x - y) / (xs * (1.0f - xs)) ;
		}
	}
	static inline __NN_CUDA_DEV__ float kernelLossDelta
		( size_t iDstLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		const float	y = pTeaching[iDstLossDelta] ;
		const float	x = pOutput[iDstLossDelta] ;
		const float	xs = __min( __max( x, 0.00001f ), 0.99999f ) ;
		return	(x - y) / (xs * (1.0f - xs)) ;
	}
	static inline void cudaLossDelta
		( float * pLossDelta, NNBufDim dimLossDelta,
			const float * pInAct, NNBufDim dimInAct,
			const float * pOutput, NNBufDim dimOutput,
			const float * pTeaching, NNBufDim dimTeaching,
			size_t nDepthwise, const LossParam& lp, cudaStream_t stream )
	{
		nncuda_LossDelta_BernoulliNLL
			( pLossDelta, dimLossDelta,
				pInAct, dimInAct, pOutput, dimOutput,
				pTeaching, dimTeaching, (int) nDepthwise, lp, stream ) ;
	}
} ;


// 損失関数 : Gaussian KL Divergence のμに関する項 : 1/2 * μ^2
//////////////////////////////////////////////////////////////////////////////
class	NNFunctionLossMeanForKLDivergence
{
public:
	constexpr static const char	FunctionName[] = "loss_mean_kl_divergence" ;
	typedef	NNLossParam2	LossParam ;

	static inline bool IsValidTeachingChannels
		( size_t nSrcActChannels, size_t nDepthwise, size_t nTeachingChannels )
	{
		return	true ;
	}
	static inline float Loss
		( const float * pInAct, const float * pOutAct,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		float	loss = 0.0f ;
		for ( size_t i = 0; i < nCount; i ++ )
		{
			const float	x = pOutAct[i] ;	// == μ
			loss += x * x * 0.5f ;
		}
		return	loss * lp.lossFactor ;
	}
	static inline void LossDelta
		( float * pLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		for ( size_t i = 0; i < nCount; i ++ )
		{
			pLossDelta[i] = pOutput[i] * lp.deltaFactor ;
		}
	}
	static inline __NN_CUDA_DEV__ float kernelLossDelta
		( size_t iDstLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		return	pOutput[iDstLossDelta] * lp.deltaFactor ;
	}
	static inline void cudaLossDelta
		( float * pLossDelta, NNBufDim dimLossDelta,
			const float * pInAct, NNBufDim dimInAct,
			const float * pOutput, NNBufDim dimOutput,
			const float * pTeaching, NNBufDim dimTeaching,
			size_t nDepthwise, const LossParam& lp, cudaStream_t stream )
	{
		nncuda_LossDelta_MeanForKLDivergence
			( pLossDelta, dimLossDelta,
				pInAct, dimInAct, pOutput, dimOutput,
				pTeaching, dimTeaching, (int) nDepthwise, lp, stream ) ;
	}
} ;


// 損失関数 : Gaussian KL Divergence のσ^2 に関する項 : (σ^2 - log(σ^2) - 1)/2
//////////////////////////////////////////////////////////////////////////////
class	NNFunctionLossVarianceForKLDivergence
{
public:
	constexpr static const char	FunctionName[] = "loss_variance_kl_divergence" ;
	typedef	NNLossParam2	LossParam ;

	static inline bool IsValidTeachingChannels
		( size_t nSrcActChannels, size_t nDepthwise, size_t nTeachingChannels )
	{
		return	true ;
	}
	static inline float Loss
		( const float * pInAct, const float * pOutAct,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		float	loss = 0.0f ;
		for ( size_t i = 0; i < nCount; i ++ )
		{
			const float	x = pOutAct[i] ;	// == σ^2
			loss += (x - log(x) - 1.0f) * 0.5f ;
		}
		return	loss * lp.lossFactor ;
	}
	static inline void LossDelta
		( float * pLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		for ( size_t i = 0; i < nCount; i ++ )
		{
			const float	x = pOutput[i] ;	// == σ^2
			pLossDelta[i] = (0.5f - 0.5f / x) * lp.deltaFactor ;
		}
	}
	static inline __NN_CUDA_DEV__ float kernelLossDelta
		( size_t iDstLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount,
			size_t nDepthwise, const LossParam& lp )
	{
		const float	x = pOutput[iDstLossDelta] ;	// == σ^2
		return	(0.5f - 0.5f / x) * lp.deltaFactor ;
	}
	static inline void cudaLossDelta
		( float * pLossDelta, NNBufDim dimLossDelta,
			const float * pInAct, NNBufDim dimInAct,
			const float * pOutput, NNBufDim dimOutput,
			const float * pTeaching, NNBufDim dimTeaching,
			size_t nDepthwise, const LossParam& lp, cudaStream_t stream )
	{
		nncuda_LossDelta_VarianceForKLDivergence
			( pLossDelta, dimLossDelta,
				pInAct, dimInAct, pOutput, dimOutput,
				pTeaching, dimTeaching, (int) nDepthwise, lp, stream ) ;
	}
} ;



// 活性化関数 : exp(x)
//////////////////////////////////////////////////////////////////////////////
class	NNAFunctionExp	: public NNAFunctionLinear
{
public:
	constexpr static const char	FunctionName[] = "exp" ;

	static inline __NN_CUDA_DEV__ float kernelActivation
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	exp_sd( pSrc[iDst] ) ;
	}
	static inline void Activation
		( float * pDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t i = 0; i < nSrcCount; i ++ )
		{
			pDst[i] = exp_s( pSrc[i] ) ;
		}
	}
	static inline void cudaActivation
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			size_t xLeftBounds, size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_Exp
			( pDst, dimDst, pSrc, dimSrc, xLeftBounds, (int) nDepthwise, stream ) ;
	}
	static inline __NN_CUDA_DEV__ float kernelPreDifferential
		( const float * pSrc, const float * pActOut,
				size_t iSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	pActOut[iSrc] ; // == exp(x)
	}
	static inline __NN_CUDA_DEV__ float kernelDifferential
		( size_t iDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		return	pSrc[iDst] ;	// == exp(x)
	}
	static inline __NN_CUDA_DEV__ void Differential
		( float * pDst, const float * pSrc,
			const float * pActOut, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t i = 0; i < nSrcCount; i ++ )
		{
			pDst[i] = pActOut[i] ;	// == exp(x) ;
		}
	}
	static inline void cudaBackDelta
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pSrcAct, NNBufDim dimSrcAct,
			const float * pOutAct, NNBufDim dimOutAct,
			size_t nDepthwise, cudaStream_t stream )
	{
		nncuda_Activation_DeltaBack_Exp
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, (int) nDepthwise, stream ) ;
	}
} ;


}

#endif
