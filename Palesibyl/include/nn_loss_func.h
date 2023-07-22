
#ifndef	__NN_LOSS_FUNC_H__
#define	__NN_LOSS_FUNC_H__

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// 損失関数
//////////////////////////////////////////////////////////////////////////////

class	NNLossFunction
{
protected:
	static std::map
		< std::string,
			std::function
				< std::shared_ptr
					<NNLossFunction>() > >	s_mapMakeLossFunc ;

public:
	// 関数生成準備
	static void InitMakeLoss( void ) ;
	// 関数生成
	static std::shared_ptr<NNLossFunction> MakeLoss( const char * pszName ) ;
	// 登録
	template <class T> static void RegisterLoss( const char * pszName )
	{
		s_mapMakeLossFunc.insert
			( std::make_pair(std::string(pszName),
							[]() { return std::make_shared<T>(); } ) ) ;
	}
	template <class T> static void RegisterLoss( void )
	{
		s_mapMakeLossFunc.insert
			( std::make_pair(std::string(T::NNLFunc::FunctionName),
							[]() { return std::make_shared<T>(); } ) ) ;
	}

public:
	// 関数名
	virtual const char * GetFunctionName( void) const = 0 ;
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
	// 損失計算
	virtual float cpuLoss
		( const float * pInAct, const float * pOutAct,
			const float * pTeaching, size_t nCount, size_t nDepthwise ) = 0 ;
	// シリアライズ
	virtual void Serialize( NNSerializer& ser ) = 0 ;
	// デシリアライズ
	virtual void Deserialize( NNDeserializer & dsr ) = 0 ;

} ;

template <class L>	class	NNLoss	: public NNLossFunction
{
public:
	// 固有パラメータ
	typename L::LossParam	m_lossParam ;
	// 関数
	typedef	L	NNLFunc ;

	// 関数名
	virtual const char * GetFunctionName( void) const
	{
		return	L::FunctionName ;
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
	}
	// デシリアライズ
	virtual void Deserialize( NNDeserializer & dsr )
	{
		uint32_t	lpSize = sizeof(m_lossParam) ;
		dsr.Read( &lpSize, sizeof(lpSize) ) ;
		dsr.Read( &m_lossParam, __min(sizeof(m_lossParam),lpSize) ) ;
	}
} ;

class	NNLossMSE	: public NNLoss<NNFunctionLossMSE> {} ;
class	NNLossMAE	: public NNLoss<NNFunctionLossMAE> {} ;
class	NNLossBernoulliNLL	: public NNLoss<NNFunctionLossBernoulliNLL> {} ;


}

#endif

