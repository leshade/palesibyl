
#include "nn_perceptron.h"

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// 活性化関数
//////////////////////////////////////////////////////////////////////////////

std::map< std::string, std::function< std::shared_ptr<NNActivationFunction>() > >
	NNActivationFunction::s_mapMakeActFunc ;

// 関数生成準備
//////////////////////////////////////////////////////////////////////////////
void NNActivationFunction::InitMake( void )
{
	s_mapMakeActFunc.clear() ;
	Register<NNActivationLinear>() ;
	Register<NNActivationLinearMAE>() ;
	Register<NNActivationReLU>() ;
	Register<NNActivationSigmoid>() ;
	Register<NNActivationTanh>() ;
	Register<NNActivationSoftmax>() ;
	Register<NNActivationFastSoftmax>() ;
	Register<NNActivationArgmax>() ;
	Register<NNActivationFastArgmax>() ;
	Register<NNActivationMaxPool>() ;
	Register<NNActivationMultiply>() ;
	Register<NNActivationExp>() ;
}

// 関数生成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNActivationFunction>
	NNActivationFunction::Make( const char * pszName )
{
	decltype(s_mapMakeActFunc)::iterator iter = s_mapMakeActFunc.find(pszName) ;
	assert( iter != s_mapMakeActFunc.end() ) ;
	if ( iter != s_mapMakeActFunc.end() )
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




