
#include "nn_perceptron.h"

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// 評価指標
//////////////////////////////////////////////////////////////////////////////

std::map< std::string, std::function< std::shared_ptr<NNEvaluationFunction>() > >
	NNEvaluationFunction::s_mapMakeFunc ;

// 関数生成準備
//////////////////////////////////////////////////////////////////////////////
void NNEvaluationFunction::InitMake( void )
{
	s_mapMakeFunc.clear() ;
	Register<NNEvaluationMSE>() ;
	Register<NNEvaluationR2Score>() ;
	Register<NNEvaluationArgmaxAccuracy>() ;
}

// 関数生成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNEvaluationFunction> NNEvaluationFunction::Make( const char * pszName )
{
	decltype(s_mapMakeFunc)::iterator iter = s_mapMakeFunc.find(pszName) ;
	assert( iter != s_mapMakeFunc.end() ) ;
	if ( iter != s_mapMakeFunc.end() )
	{
		return	(iter->second)() ;
	}
	return	nullptr ;
}

// 関数表示名
//////////////////////////////////////////////////////////////////////////////
const char * NNEvaluationFunction::GetDisplayName( void ) const
{
	return	GetFunctionName() ;
}

// シリアライズ
//////////////////////////////////////////////////////////////////////////////
void NNEvaluationFunction::Serialize( NNSerializer& ser ) const
{
}

// デシリアライズ
//////////////////////////////////////////////////////////////////////////////
void NNEvaluationFunction::Deserialize( NNDeserializer & dsr )
{
}



//////////////////////////////////////////////////////////////////////////////
// 評価関数 : 平均二乗誤差 (Mean Squared Error)
//////////////////////////////////////////////////////////////////////////////

// 関数名
//////////////////////////////////////////////////////////////////////////////
const char * NNEvaluationMSE::GetFunctionName( void ) const
{
	return	FunctionName ;
}

// 関数表示名
//////////////////////////////////////////////////////////////////////////////
const char * NNEvaluationMSE::GetDisplayName( void ) const
{
	return	DisplayName ;
}

// 評価値計算
//////////////////////////////////////////////////////////////////////////////
double NNEvaluationMSE::Evaluate
	( const NNBuffer& bufPredicted, const NNBuffer& bufObserved ) const
{
	const NNBufDim	dimPredicted = bufPredicted.GetSize() ;
	const NNBufDim	dimObserved = bufObserved.GetSize() ;
	assert( dimObserved.n * dimObserved.z == dimPredicted.n * dimPredicted.z ) ;
	const size_t	nCount = __min( dimObserved.n * dimObserved.z,
									dimPredicted.n * dimPredicted.z ) ;
	if ( nCount == 0 )
	{
		return	0.0 ;
	}
	const float *	pPred = bufPredicted.GetConstBuffer() ;
	const float *	pObsr = bufObserved.GetConstBuffer() ;
	double			mse = 0.0 ;
	for ( size_t i = 0; i < nCount; i ++ )
	{
		const float	e = pObsr[i] - pPred[i] ;
		mse += e * e ;
	}
	return	mse / (double) nCount ;
}



//////////////////////////////////////////////////////////////////////////////
// 評価指標 : R^2
//////////////////////////////////////////////////////////////////////////////

// 関数名
//////////////////////////////////////////////////////////////////////////////
const char * NNEvaluationR2Score::GetFunctionName( void ) const
{
	return	FunctionName ;
}

// 関数表示名
//////////////////////////////////////////////////////////////////////////////
const char * NNEvaluationR2Score::GetDisplayName( void ) const
{
	return	DisplayName ;
}

// 評価値計算
//////////////////////////////////////////////////////////////////////////////
double NNEvaluationR2Score::Evaluate
	( const NNBuffer& bufPredicted, const NNBuffer& bufObserved ) const
{
	const NNBufDim	dimPredicted = bufPredicted.GetSize() ;
	const NNBufDim	dimObserved = bufObserved.GetSize() ;
	assert( dimObserved.n * dimObserved.z == dimPredicted.n * dimPredicted.z ) ;
	const size_t	nCount = __min( dimObserved.n * dimObserved.z,
									dimPredicted.n * dimPredicted.z ) ;
	if ( nCount == 0 )
	{
		return	0.0 ;
	}
	const float *	pPred = bufPredicted.GetConstBuffer() ;
	const float *	pObsr = bufObserved.GetConstBuffer() ;
	double			mean = 0.0 ;
	for ( size_t i = 0; i < nCount; i ++ )
	{
		mean += pObsr[i] ;
	}
	mean /= (double) nCount ;

	double	error = 0.0 ;
	double	dispersion = 0.0 ;
	for ( size_t i = 0; i < nCount; i ++ )
	{
		double	e = pObsr[i] - pPred[i] ;
		double	d = pObsr[i] - mean ;
		error += e * e ;
		dispersion += d * d ;
	}
	return	1.0 - error / __max( dispersion, 1.0e-8 ) ;
}



//////////////////////////////////////////////////////////////////////////////
// 評価指標 : argmax 正解率
//////////////////////////////////////////////////////////////////////////////

// 関数名
//////////////////////////////////////////////////////////////////////////////
const char * NNEvaluationArgmaxAccuracy::GetFunctionName( void ) const
{
	return	FunctionName ;
}

// 関数表示名
//////////////////////////////////////////////////////////////////////////////
const char * NNEvaluationArgmaxAccuracy::GetDisplayName( void ) const
{
	return	DisplayName ;
}

// 評価値計算
//////////////////////////////////////////////////////////////////////////////
double NNEvaluationArgmaxAccuracy::Evaluate
	( const NNBuffer& bufPredicted, const NNBuffer& bufObserved ) const
{
	const NNBufDim	dimPredicted = bufPredicted.GetSize() ;
	const NNBufDim	dimObserved = bufObserved.GetSize() ;
	assert( dimObserved.n == dimPredicted.n ) ;
	assert( dimPredicted.z > argmaxIndex ) ;
	const size_t	nCount = __min( dimObserved.n, dimPredicted.n ) ;
	if ( nCount == 0 )
	{
		return	0.0 ;
	}
	const float *	pPred = bufPredicted.GetConstBuffer() ;
	const float *	pObsr = bufObserved.GetConstBuffer() ;
	size_t			nCorrect = 0 ;
	for ( size_t i = 0; i < nCount; i ++ )
	{
		if ( floor(pPred[i + argmaxIndex]) == floor(pObsr[i]) )
		{
			nCorrect ++ ;
		}
		pPred += dimPredicted.z ;
		pObsr += dimObserved.z ;
	}
	return	(double) nCorrect / (double) nCount ;
}


