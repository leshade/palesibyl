
#ifndef	__NN_ACTIVATION_FUNC_H__
#define	__NN_ACTIVATION_FUNC_H__

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// 活性化関数
//////////////////////////////////////////////////////////////////////////////

class	NNActivationFunction
{
protected:
	static std::map
		< std::string,
			std::function
				< std::shared_ptr
					<NNActivationFunction>() > >	s_mapMakeFunc ;

public:
	// 関数生成準備
	static void InitMake( void ) ;
	// 関数生成
	static std::shared_ptr<NNActivationFunction> Make( const char * pszName ) ;
	// 登録
	template <class T> static void Register( const char * pszName )
	{
		s_mapMakeFunc.insert
			( std::make_pair(std::string(pszName),
							[]() { return std::make_shared<T>(); } ) ) ;
	}

public:
	// 関数名
	virtual const char * GetFunctionName( void) const = 0 ;
	// 最終層で使用できないか？
	virtual bool MustNotBeLastLayer( void ) const = 0 ;
	// 最終層でしか使用できないか？
	virtual bool MustBeLastLayer( void ) const = 0 ;
	// CUDA で利用可能なチャネル数か？
	virtual bool IsAcceptableChannelsForCuda( size_t chOutput, size_t chInput ) const ;
	// 最終層で教師データチャネル数と適合するか？
	virtual bool IsValidTeachingChannels
			( size_t chInput, size_t nDepthwise, size_t chTeaching ) const = 0 ;
	// 入力 -> 出力チャネル数
	virtual size_t CalcOutputChannels( size_t chInput, size_t nDepthwise ) const = 0 ;
	// 関数
	virtual void cpuFunction
		( float * pDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise ) = 0 ;
	virtual void cudaFunction
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			size_t xLeftBounds, size_t nDepthwise, cudaStream_t stream ) = 0 ;
	// 微分関数
	virtual void cpuDifferential
		( float * pDst, const float * pSrc,
			const float * pActOut, size_t nSrcCount, size_t nDepthwise ) = 0 ;
	// δ計算
	virtual void cpuLossDelta
		( float * pLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount, size_t nDepthwise ) = 0 ;
	virtual inline void cudaLossDelta
		( float * pLossDelta, NNBufDim dimLossDelta,
			const float * pInAct, NNBufDim dimInAct,
			const float * pOutput, NNBufDim dimOutput,
			const float * pTeaching, NNBufDim dimTeaching,
			size_t nDepthwise, cudaStream_t stream ) = 0 ;
	// δ逆伝播
	virtual void cpuDeltaBack
		( float * pDstDelta, const float * pSrcDelta,
			const float * pDiffAct, size_t nSrcCount, size_t nDepthwise ) = 0 ;
	virtual void cudaBackDelta
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pSrcAct, NNBufDim dimSrcAct,
			const float * pOutAct, NNBufDim dimOutAct,
			size_t nDepthwise, cudaStream_t stream ) = 0 ;
	// 損失計算
	virtual float cpuLoss
		( const float * pInAct, const float * pOutAct,
			const float * pTeaching, size_t nCount, size_t nDepthwise ) = 0 ;
	// シリアライズ
	virtual void Serialize( NNSerializer& ser ) = 0 ;
	// デシリアライズ
	virtual void Deserialize( NNDeserializer & dsr ) = 0 ;
} ;

template <class A, class L>
	class	NNActivation	: public NNActivationFunction
{
public:
	// 固有パラメータ
	typename L::LossParam	m_lossParam ;
	// 関数
	typedef	A	NNAFunc ;
	// 関数名
	virtual const char * GetFunctionName( void) const
	{
		return	A::FunctionName ;
	}
	// 最終層で使用できないか？
	virtual bool MustNotBeLastLayer( void ) const
	{
		return	A::MustNotBeLastLayer ;
	}
	// 最終層でしか使用できないか？
	virtual bool MustBeLastLayer( void ) const
	{
		return	A::MustBeLastLayer ;
	}
	// CUDA で利用可能なチャネル数か？
	virtual bool IsAcceptableChannelsForCuda( size_t chOutput, size_t chInput ) const
	{
		return	A::IsAcceptableChannelsForCuda( chOutput, chInput ) ;
	}
	// 最終層で教師データチャネル数と適合するか？
	virtual bool IsValidTeachingChannels
			( size_t chInput, size_t nDepthwise, size_t chTeaching ) const
	{
		return	L::IsValidTeachingChannels( chInput, nDepthwise, chTeaching ) ;
	}
	// 入力 -> 出力チャネル数
	virtual size_t CalcOutputChannels( size_t chInput, size_t nDepthwise ) const
	{
		return	A::CalcOutChannels( chInput, nDepthwise ) ;
	}
	// 関数
	virtual void cpuFunction
		( float * pDst, const float * pSrc, size_t nSrcCount, size_t nDepthwise )
	{
		A::Activation( pDst, pSrc, nSrcCount, nDepthwise ) ;
	}
	virtual void cudaFunction
		( float * pDst, NNBufDim dimDst,
			const float * pSrc, NNBufDim dimSrc,
			size_t xLeftBounds, size_t nDepthwise, cudaStream_t stream )
	{
		A::cudaActivation
			( pDst, dimDst, pSrc, dimSrc, xLeftBounds, nDepthwise, stream ) ;
	}
	// 微分関数
	virtual void cpuDifferential
		( float * pDst, const float * pSrc,
			const float * pActOut, size_t nSrcCount, size_t nDepthwise )
	{
		A::Differential( pDst, pSrc, pActOut, nSrcCount, nDepthwise ) ;
	}
	// δ計算
	virtual void cpuLossDelta
		( float * pLossDelta,
			const float * pInAct, const float * pOutput,
			const float * pTeaching, size_t nCount, size_t nDepthwise )
	{
		L::LossDelta
			( pLossDelta, pInAct, pOutput,
				pTeaching, nCount, nDepthwise, m_lossParam ) ;
	}
	virtual inline void cudaLossDelta
		( float * pLossDelta, NNBufDim dimLossDelta,
			const float * pInAct, NNBufDim dimInAct,
			const float * pOutput, NNBufDim dimOutput,
			const float * pTeaching, NNBufDim dimTeaching,
			size_t nDepthwise, cudaStream_t stream )
	{
		L::cudaLossDelta
			( pLossDelta, dimLossDelta,
				pInAct, dimInAct, pOutput, dimOutput,
				pTeaching, dimTeaching, nDepthwise, m_lossParam, stream ) ;
	}
	// δ逆伝播
	virtual void cpuDeltaBack
		( float * pDstDelta, const float * pSrcDelta,
			const float * pDiffAct, size_t nSrcCount, size_t nDepthwise )
	{
		for ( size_t i = 0; i < nSrcCount; i ++ )
		{
			pDstDelta[i] =
				A::BackDelta( i, pSrcDelta, pDiffAct, nSrcCount, nDepthwise ) ;
		}
	}
	virtual void cudaBackDelta
		( float * pDstDelta, NNBufDim dimDstDelta,
			const float * pSrcDelta, NNBufDim dimSrcDelta,
			const float * pSrcAct, NNBufDim dimSrcAct,
			const float * pOutAct, NNBufDim dimOutAct,
			size_t nDepthwise, cudaStream_t stream )
	{
		A::cudaBackDelta
			( pDstDelta, dimDstDelta,
				pSrcDelta, dimSrcDelta,
				pSrcAct, dimSrcAct,
				pOutAct, dimOutAct, nDepthwise, stream ) ;
	}
	// 損失計算
	virtual float cpuLoss
		( const float * pInAct, const float * pOutAct,
			const float * pTeaching, size_t nCount, size_t nDepthwise )
	{
		return	L::Loss( pInAct, pOutAct, pTeaching, nCount, nDepthwise, m_lossParam ) ;
	}
	// シリアライズ
	virtual void Serialize( NNSerializer& ser )
	{
		uint32_t	lpSize = sizeof(m_lossParam) ;
		ser.Write( &lpSize, sizeof(lpSize) ) ;
		ser.Write( &m_lossParam, sizeof(m_lossParam) ) ;
		uint32_t	apSize = 0 ;
		ser.Write( &apSize, sizeof(lpSize) ) ;
	}
	// デシリアライズ
	virtual void Deserialize( NNDeserializer & dsr )
	{
		uint32_t	lpSize = sizeof(m_lossParam) ;
		dsr.Read( &lpSize, sizeof(lpSize) ) ;
		dsr.Read( &m_lossParam, min(sizeof(m_lossParam),lpSize) ) ;
		dsr.Skip( lpSize - min(sizeof(m_lossParam),lpSize) ) ;
		uint32_t	apSize = 0 ;
		dsr.Read( &apSize, sizeof(lpSize) ) ;
	}
} ;

class	NNActivationLinear
	: public NNActivation<NNAFunctionLinear, NNLossFunctionMSE> {} ;
class	NNActivationLinearMAE
	: public NNActivation<NNAFunctionLinear, NNLossFunctionMAE> {} ;
class	NNActivationReLU
	: public NNActivation<NNAFunctionReLU, NNLossFunctionMSE> {} ;
class	NNActivationSigmoid
	: public NNActivation<NNAFunctionSigmoid, NNLossFunctionSigmoid> {} ;
class	NNActivationTanh
	: public NNActivation<NNAFunctionTanh, NNLossFunctionMSE> {} ;
class	NNActivationSoftmax
	: public NNActivation<NNAFunctionSoftmax, NNLossFunctionSoftmax> {} ;
class	NNActivationFastSoftmax
	: public NNActivation<NNAFunctionFastSoftmax, NNLossFunctionFastSoftmax> {} ;
class	NNActivationArgmax
	: public NNActivation<NNAFunctionArgmax, NNLossFunctionArgmax> {} ;
class	NNActivationFastArgmax
	: public NNActivation<NNAFunctionFastArgmax, NNLossFunctionFastArgmax> {} ;
class	NNActivationMaxPool
	: public NNActivation<NNAFunctionMaxPool, NNLossFunctionMSE> {} ;
class	NNActivationMultiply
	: public NNActivation<NNAFunctionMultiply, NNLossFunctionMSE> {} ;

}

#endif
