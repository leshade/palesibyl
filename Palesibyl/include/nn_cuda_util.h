
#ifndef	__NN_CUDA_UTIL_H__
#define	__NN_CUDA_UTIL_H__

#if	defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
	// ※ GDI+ ヘッダがエラーとなるため #define NOMINMAX はしない
	#include <windows.h>
#endif

#include "nn_cuda_def.h"
#include <stdarg.h>

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// デバッグ出力
//////////////////////////////////////////////////////////////////////////////

#ifndef	TRACE
	#if defined(NDEBUG) || !defined(_DEBUG)
		inline void TRACE( const char * pszTrace, ... ) {}
	#else
		inline void TRACE( const char * pszTrace, ... )
		{
			va_list	vl ;
			va_start( vl, pszTrace ) ;

			#if	defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
				char	szBuf[0x1000] ;
				_vsnprintf_s( szBuf, 0x1000, 0xFFF, pszTrace, vl ) ;
				OutputDebugString( szBuf ) ;
			#else
				vprintf( pszTrace, vl ) ;
			#endif
		}
	#endif
#endif


//////////////////////////////////////////////////////////////////////////////
// 一時的なエラーハンドラー
//////////////////////////////////////////////////////////////////////////////

class	CudaErrorHandler
{
public:
	CudaErrorHandler( std::function<void(cudaError_t,const char*)> handler )
	{
		cudaSetErrorHandler( [handler]( cudaError_t error )
		{
			handler( error, cudaGetErrorName( error ) ) ;
		} ) ;
	}
	~CudaErrorHandler( void )
	{
		cudaPopErrorHandler() ;
	}
} ;



//////////////////////////////////////////////////////////////////////////////
// CUDA メモリ管理
//////////////////////////////////////////////////////////////////////////////

template <class T>	class	Cuda1DMemory
{
protected:
	T *		m_pHostMem ;
	T *		m_pDevMem ;
	size_t	m_nLength ;

public:
	// 構築関数
	Cuda1DMemory( void )
		: m_pHostMem( nullptr ), m_pDevMem( nullptr ), m_nLength( 0 ) { }
	~Cuda1DMemory( void )
	{
		Free() ;
	}
	// メモリ確保
	enum	AllocateFlag
	{
		allocDefault	= 0x0000,
		allocDeviceOnly	= 0x0001,
	} ;
	void Allocate( size_t nLength, uint32_t flags = allocDefault )
	{
		if ( m_nLength != nLength )
		{
			if ( m_nLength > 0 )
			{
				Free() ;
			}
			assert( m_pHostMem == nullptr ) ;
			assert( m_pDevMem == nullptr ) ;
			assert( m_nLength == 0 ) ;
			const size_t	nBytes = nLength * sizeof(T) ;
			if ( !(flags & allocDeviceOnly) )
			{
				cudaVerify( cudaMallocHost<T>( &m_pHostMem, nBytes ) ) ;
			}
			cudaVerify( cudaMalloc<T>( &m_pDevMem, nBytes ) ) ;
			m_nLength = nLength ;
			{
				std::lock_guard<std::mutex>	lock( g_cudaMutex ) ;
				g_cudaAllocDevMemory += nBytes ;
				if ( g_cudaAllocDevMemory > g_cudaMaxAllocDevMemory )
				{
					g_cudaMaxAllocDevMemory = g_cudaAllocDevMemory ;
					TRACE( "max used CUDA device memory : %d[MB]\r\n",
							(size_t) (g_cudaMaxAllocDevMemory / (1024*1024)) ) ;
				}
			}
		}
	}
	// メモリ解放
	void Free( void )
	{
		if ( m_pHostMem != nullptr )
		{
			cudaFreeHost( m_pHostMem ) ;
			m_pHostMem = nullptr ;
		}
		if ( m_pDevMem != nullptr )
		{
			const size_t	nBytes = m_nLength * sizeof(T) ;
			assert( g_cudaAllocDevMemory >= nBytes ) ;
			cudaFree( m_pDevMem ) ;
			m_pDevMem = nullptr ;
			{
				std::lock_guard<std::mutex>	lock( g_cudaMutex ) ;
				g_cudaAllocDevMemory -= nBytes ;
			}
		}
		m_nLength = 0 ;
	}
	// ポインタ取得
	T * GetArray( void ) const
	{
		return	m_pHostMem ;
	}
	T * GetDevicePtr( void ) const
	{
		return	m_pDevMem ;
	}
	// 配列長取得
	size_t GetLength( void ) const
	{
		return	m_nLength ;
	}
	// デバイスへ転送
	void AsyncToDevice( cudaStream_t stream )
	{
		assert( m_pDevMem != nullptr ) ;
		assert( m_pHostMem != nullptr ) ;
		assert( m_nLength != 0 ) ;
		cudaVerify
			( cudaMemcpyAsync
				( m_pDevMem, m_pHostMem,
					m_nLength * sizeof(T),
					cudaMemcpyHostToDevice, stream ) ) ;
	}
	void CopyToDevice( void )
	{
		assert( m_pDevMem != nullptr ) ;
		assert( m_pHostMem != nullptr ) ;
		assert( m_nLength != 0 ) ;
		cudaVerify
			( cudaMemcpy
				( m_pDevMem, m_pHostMem,
					m_nLength * sizeof(T),
					cudaMemcpyHostToDevice ) ) ;
	}
	// デバイスから転送
	void AsyncFromDevice( cudaStream_t stream )
	{
		assert( m_pDevMem != nullptr ) ;
		assert( m_pHostMem != nullptr ) ;
		assert( m_nLength != 0 ) ;
		cudaVerify
			( cudaMemcpyAsync
				( m_pHostMem, m_pDevMem,
					m_nLength * sizeof(T),
					cudaMemcpyDeviceToHost, stream ) ) ;
	}
	// データをデバイスへ転送
	void CopyAsyncFrom( const T * pSrc, size_t nLength, cudaStream_t stream )
	{
		assert( nLength <= m_nLength ) ;
		for ( size_t i = 0; i < nLength; i ++ )
		{
			m_pHostMem[i] = pSrc[i] ;
		}
		AsyncToDevice( stream ) ;
	}
	void CopyFrom( const T * pSrc, size_t nLength )
	{
		assert( nLength <= m_nLength ) ;
		for ( size_t i = 0; i < nLength; i ++ )
		{
			m_pHostMem[i] = pSrc[i] ;
		}
		CopyToDevice() ;
	}
	// デバイス間転送
	void CopyAsyncDeviceFrom
		( const Cuda1DMemory<T>& cmemSrc, cudaStream_t stream )
	{
		assert( m_pDevMem != nullptr ) ;
		assert( cmemSrc.m_pDevMem != nullptr ) ;
		assert( m_nLength == cmemSrc.m_nLength ) ;
		cudaVerify
			( cudaMemcpyAsync
				( m_pDevMem, cmemSrc.m_pDevMem,
					m_nLength * sizeof(T),
					cudaMemcpyDeviceToDevice, stream ) ) ;
	}
} ;

class	CudaFloat1DMemory	: public Cuda1DMemory<float>
{
} ;



//////////////////////////////////////////////////////////////////////////////
// CUDA ストリーム・ラッパー
//////////////////////////////////////////////////////////////////////////////

class	CudaStream
{
protected:
	cudaStream_t	m_stream ;

public:
	// 構築
	CudaStream( void ) ;
	// 消滅
	~CudaStream( void ) ;
	// 作成
	void Create( unsigned int flags = cudaStreamNonBlocking ) ;
	bool IsCreated( void ) const ;
	// 破棄
	void Destroy( void ) ;
	// 同期
	void Synchronize( void ) ;
	// 同期（デバッグ・コンパイル時のみ）
	void VerifySync( void ) ;
	// cudaStream_t 取得
	cudaStream_t GetStream( void ) const ;
	operator cudaStream_t ( void ) const ;

} ;

}

#endif
