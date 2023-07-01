
#include "nn_cuda_util.h"

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// CUDA ヘルパー関数
//////////////////////////////////////////////////////////////////////////////

std::mutex			Palesibyl::g_cudaMutex ;
int					Palesibyl::g_cudaDevice = -1 ;
unsigned long long	Palesibyl::g_cudaAllocDevMemory = 0 ;
unsigned long long	Palesibyl::g_cudaMaxAllocDevMemory = 0 ;
cudaDeviceProp		Palesibyl::g_cudaDevProp ;

static std::function<void(cudaError_t)>
					s_cudaErrorHandler = [](cudaError_t){} ;
static std::vector< std::function<void(cudaError_t)> >
					s_cudaErrorHandlerStack ;


// CUDA 初期化
//////////////////////////////////////////////////////////////////////////////
int Palesibyl::cudaInit( int devID )
{
	// デバイス数取得
	int	nDevCount = 0 ;
	cudaVerify( cudaGetDeviceCount(&nDevCount) ) ;

	if ( nDevCount == 0 )
	{
		TRACE( "Not found CUDA device.\n" ) ;
		return	-1 ;
	}
	if ( devID < 0 )
	{
		devID = 0 ;
	}
	if ( devID > nDevCount - 1 )
	{
		devID = 0 ;
	}

	// 機能確認
	int	computeMode = -1, major = 0, minor = 0;
	cudaVerify( cudaDeviceGetAttribute( &computeMode, cudaDevAttrComputeMode, devID ) ) ;
	cudaVerify( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, devID ) ) ;
	cudaVerify( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, devID ) ) ;
	if ( computeMode == cudaComputeModeProhibited )
	{
		TRACE( "Error: device is running in cudaComputeModeProhibited.\n" ) ;
		return	-1 ;
	}
	if ( major < 1 )
	{
		TRACE( "gpuDeviceInit(): GPU device does not support CUDA.\n" ) ;
		return	-1 ;
	}

	// デバイス選択
	cudaVerify( cudaSetDevice(devID) ) ;
	TRACE( "CUDA device [%d]: version %d.%d\n", devID, major, minor ) ;

	g_cudaDevice = devID ;

	if ( cudaVerify( cudaGetDeviceProperties( &g_cudaDevProp, devID ) ) )
	{
		TRACE( "    %s\n", g_cudaDevProp.name ) ;
		TRACE( "        VRAM global memory: %ld [MB]\n", (long)(g_cudaDevProp.totalGlobalMem/(1024*1204)) ) ;
		TRACE( "        VRAM const memory: %ld [KB]\n", (long) (g_cudaDevProp.totalConstMem/1024) ) ;
		TRACE( "        max shared memory: %d [KB]\n", (int) (g_cudaDevProp.sharedMemPerBlock/1024) ) ;
		TRACE( "        max thread count: %d\n", g_cudaDevProp.maxThreadsPerBlock ) ;
	}

	return	devID ;
}


// CUDA 無効化
//////////////////////////////////////////////////////////////////////////////
void Palesibyl::cudaDisable( void )
{
	g_cudaDevice = -1 ;
}


// CUDA を利用可能か？
//////////////////////////////////////////////////////////////////////////////
bool Palesibyl::cudaIsAvailable( void )
{
	return	(g_cudaDevice >= 0) ;
}


// CUDA 関数実行／エラー表示
//////////////////////////////////////////////////////////////////////////////
bool Palesibyl::cudaVerify( cudaError_t result )
{
	assert( result == cudaSuccess ) ;
	if ( result == cudaSuccess )
	{
		return	true ;
	}
	TRACE( "CUDA error: %08X : %s\n", result, cudaGetErrorName( result ) ) ;

	std::lock_guard<std::mutex>	lock( g_cudaMutex ) ;
	s_cudaErrorHandler( result ) ;

	return	false ;
}

// CUDA エラー表示関数設定
//////////////////////////////////////////////////////////////////////////////
void Palesibyl::cudaSetErrorHandler( std::function<void(cudaError_t)> handler )
{
	std::lock_guard<std::mutex>	lock( g_cudaMutex ) ;
	s_cudaErrorHandlerStack.push_back( s_cudaErrorHandler ) ;
	s_cudaErrorHandler = handler ;
}

void Palesibyl::cudaPopErrorHandler( void )
{
	std::lock_guard<std::mutex>	lock( g_cudaMutex ) ;
	assert( s_cudaErrorHandlerStack.size() >= 1 ) ;
	if ( s_cudaErrorHandlerStack.size() >= 1 )
	{
		s_cudaErrorHandler = s_cudaErrorHandlerStack.back() ;
		s_cudaErrorHandlerStack.pop_back() ;
	}
}

