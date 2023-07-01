
#ifndef	__NN_LOOP_STREAM_H__
#define	__NN_LOOP_STREAM_H__

#include "nn_multi_thread.h"


namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// CPU / CUDA 実行
//////////////////////////////////////////////////////////////////////////////

class	NNLoopStream
{
public:
	bool			m_useCuda ;
	CudaStream		m_cudaStream ;
	NNParallelLoop	m_ploop ;

public:
	NNLoopStream( void ) : m_useCuda(false) {}

} ;

}

#endif

