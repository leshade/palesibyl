
#ifndef	__NN_EVALUATION_FUNC_H__
#define	__NN_EVALUATION_FUNC_H__

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// 評価指標
//////////////////////////////////////////////////////////////////////////////

class	NNEvaluationFunction
{
protected:
	static std::map
		< std::string,
			std::function
				< std::shared_ptr
					<NNEvaluationFunction>() > >	s_mapMakeFunc ;

public:
	// 関数生成準備
	static void InitMake( void ) ;
	// 関数生成
	static std::shared_ptr<NNEvaluationFunction> Make( const char * pszName ) ;
	// 登録
	template <class T> static void Register( void )
	{
		s_mapMakeFunc.insert
			( std::make_pair(std::string(T::FunctionName),
							[]() { return std::make_shared<T>(); } ) ) ;
	}

public:
	// 関数名
	virtual const char * GetFunctionName( void ) const = 0 ;
	// 関数表示名
	virtual const char * GetDisplayName( void ) const ;
	// 評価値計算
	virtual double Evaluate
		( const NNBuffer& bufPredicted, const NNBuffer& bufObserved ) const = 0 ;
	// シリアライズ
	virtual void Serialize( NNSerializer& ser ) const ;
	// デシリアライズ
	virtual void Deserialize( NNDeserializer & dsr ) ;

} ;


//////////////////////////////////////////////////////////////////////////////
// 評価指標 : R^2
//////////////////////////////////////////////////////////////////////////////

class	NNEvaluationR2Score	: public NNEvaluationFunction
{
public:
	constexpr static const char	FunctionName[] = "r2_score" ;
	constexpr static const char	DisplayName[] = "R2 score" ;

public:
	// 関数名
	virtual const char * GetFunctionName( void ) const ;
	// 関数表示名
	virtual const char * GetDisplayName( void ) const ;
	// 評価値計算
	virtual double Evaluate
		( const NNBuffer& bufPredicted, const NNBuffer& bufObserved ) const ;
} ;


//////////////////////////////////////////////////////////////////////////////
// 評価指標 : argmax 正解率
//////////////////////////////////////////////////////////////////////////////

class	NNEvaluationArgmaxAccuracy	: public NNEvaluationFunction
{
public:
	constexpr static const char	FunctionName[] = "argmax_accuracy" ;
	constexpr static const char	DisplayName[] = "accuracy" ;

public:
	// 関数名
	virtual const char * GetFunctionName( void ) const ;
	// 関数表示名
	virtual const char * GetDisplayName( void ) const ;
	// 評価値計算
	virtual double Evaluate
		( const NNBuffer& bufPredicted, const NNBuffer& bufObserved ) const ;
} ;


}

#endif
