
#include "nn_perceptron.h"

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// 損失関数
//////////////////////////////////////////////////////////////////////////////

std::map< std::string, std::function< std::shared_ptr<NNLossFunction>() > >
	NNLossFunction::s_mapMakeLossFunc ;

// 関数生成準備
//////////////////////////////////////////////////////////////////////////////
void NNLossFunction::InitMakeLoss( void )
{
	s_mapMakeLossFunc.clear() ;
	RegisterLoss<NNLossMSE>() ;
	RegisterLoss<NNLossMAE>() ;
	RegisterLoss<NNLossBernoulliNLL>() ;
}

// 関数生成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNLossFunction>
	NNLossFunction::MakeLoss( const char * pszName )
{
	decltype(s_mapMakeLossFunc)::iterator iter = s_mapMakeLossFunc.find(pszName) ;
	assert( iter != s_mapMakeLossFunc.end() ) ;
	if ( iter != s_mapMakeLossFunc.end() )
	{
		return	(iter->second)() ;
	}
	return	nullptr ;
}


