
#include "nn_cuda_util.h"

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// CUDA ストリーム・ラッパー
//////////////////////////////////////////////////////////////////////////////

// 構築
//////////////////////////////////////////////////////////////////////////////
CudaStream::CudaStream( void )
	: m_stream( nullptr )
{
}

// 消滅
//////////////////////////////////////////////////////////////////////////////
CudaStream::~CudaStream( void )
{
	if ( m_stream != nullptr )
	{
		Destroy() ;
	}
}

// 作成
//////////////////////////////////////////////////////////////////////////////
void CudaStream::Create( unsigned int flags )
{
	if ( m_stream == nullptr )
	{
		cudaVerify( cudaStreamCreateWithFlags( &m_stream, flags ) ) ;
	}
}

bool CudaStream::IsCreated( void ) const
{
	return	(m_stream != nullptr) ;
}

// 破棄
//////////////////////////////////////////////////////////////////////////////
void CudaStream::Destroy( void )
{
	if ( m_stream != nullptr )
	{
		cudaVerify( cudaStreamDestroy( m_stream ) ) ;
		m_stream = nullptr ;
	}
}

// 同期
//////////////////////////////////////////////////////////////////////////////
void CudaStream::Synchronize( void ) const
{
	assert( m_stream != nullptr ) ;
	cudaVerify( cudaStreamSynchronize( m_stream ) ) ;
}

void CudaStream::VerifySync( void ) const
{
	assert( m_stream != nullptr ) ;
#if	!defined(NDEBUG) && defined(_DEBUG)
	cudaVerify( cudaStreamSynchronize( m_stream ) ) ;
#endif
}

// cudaStream_t 取得
//////////////////////////////////////////////////////////////////////////////
cudaStream_t CudaStream::GetStream( void ) const
{
	return	m_stream ;
}

CudaStream::operator cudaStream_t ( void ) const
{
	return	m_stream ;
}



