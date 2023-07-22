
#ifndef	__NN_CUDA_DEFS_H__
#define	__NN_CUDA_DEFS_H__

#include <assert.h>
#include <mutex>
#include <functional>
#include <cuda_runtime.h>
//#include <curand.h>

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// CUDA 定数
//////////////////////////////////////////////////////////////////////////////

// 共有メモリは 48KB
constexpr size_t	cudaSharedMemorySize = 48 * 1024 ;

// 最大スレッド数
constexpr size_t	cudaMaxThreadCount	= 512 ;

// 最大ブロック数
constexpr size_t	cudaMaxBlockSize	= 65535 ;



//////////////////////////////////////////////////////////////////////////////
// CUDA 関数
//////////////////////////////////////////////////////////////////////////////

extern std::mutex			g_cudaMutex ;
extern int					g_cudaDevice ;
extern unsigned long long	g_cudaAllocDevMemory ;
extern unsigned long long	g_cudaMaxAllocDevMemory ;
extern cudaDeviceProp		g_cudaDevProp ;


// CUDA 初期化
int cudaInit( int devID = -1 ) ;

// CUDA 無効化
void cudaDisable( void ) ;

// CUDA を利用可能か？
bool cudaIsAvailable( void ) ;

// CUDA 関数実行／エラー表示
bool cudaVerify( cudaError_t result ) ;

// CUDA エラー表示関数設定
void cudaSetErrorHandler( std::function<void(cudaError_t)> handler ) ;
void cudaPopErrorHandler( void ) ;


}

#endif

