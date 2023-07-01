
#include "nn_perceptron.h"

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// 活性化関数
//////////////////////////////////////////////////////////////////////////////

std::map< std::string, std::function< std::shared_ptr<NNActivationFunction>() > >
	NNActivationFunction::s_mapMakeFunc ;

// 関数生成準備
//////////////////////////////////////////////////////////////////////////////
void NNActivationFunction::InitMake( void )
{
	s_mapMakeFunc.clear() ;
	Register<NNActivationLinear>( NNAFunctionLinear::FunctionName ) ;
	Register<NNActivationLinearMAE>( NNAFunctionLinearMAE::FunctionName ) ;
	Register<NNActivationReLU>( NNAFunctionReLU::FunctionName ) ;
	Register<NNActivationSigmoid>( NNAFunctionSigmoid::FunctionName ) ;
	Register<NNActivationTanh>( NNAFunctionTanh::FunctionName ) ;
	Register<NNActivationSoftmax>( NNAFunctionSoftmax::FunctionName ) ;
	Register<NNActivationFastSoftmax>( NNAFunctionFastSoftmax::FunctionName ) ;
	Register<NNActivationArgmax>( NNAFunctionArgmax::FunctionName ) ;
	Register<NNActivationFastArgmax>( NNAFunctionFastArgmax::FunctionName ) ;
	Register<NNActivationMaxPool>( NNAFunctionMaxPool::FunctionName ) ;
	Register<NNActivationMultiply>( NNAFunctionMultiply::FunctionName ) ;
}

// 関数生成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNActivationFunction>
	NNActivationFunction::Make( const char * pszName )
{
	decltype(s_mapMakeFunc)::iterator iter = s_mapMakeFunc.find(pszName) ;
	assert( iter != s_mapMakeFunc.end() ) ;
	if ( iter != s_mapMakeFunc.end() )
	{
		return	(iter->second)() ;
	}
	return	nullptr ;
}

// CUDA で利用可能なチャネル数か？
//////////////////////////////////////////////////////////////////////////////
bool NNActivationFunction::IsAcceptableChannelsForCuda( size_t chOutput, size_t chInput ) const
{
	return	nncuda_IsAcceptableActivationChannels( chOutput, chInput ) ;
}




