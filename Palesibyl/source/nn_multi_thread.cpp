
#include "nn_multi_thread.h"
#include <assert.h>

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// 並列ループ・スレッド
//////////////////////////////////////////////////////////////////////////////

NNParallelLoop::Thread::Thread( NNParallelLoop& ploop, size_t iThread )
	: m_ploop( ploop ), m_iThread( iThread ),
		m_flagQuit( false ), m_pDispatch( nullptr )
{
	m_pThread = std::make_unique<std::thread>( [this](){ Run(); } ) ;
}

NNParallelLoop::Thread::~Thread( void )
{
	Join() ;
}

void NNParallelLoop::Thread::Dispatch( std::function<void(size_t,size_t)>& func )
{
	assert( m_pDispatch == nullptr ) ;
	{
		std::lock_guard<std::mutex>	lock(m_mutex) ;
		m_pDispatch = &func ;
	}
	m_cvReady.notify_one() ;
}

void NNParallelLoop::Thread::Join( void )
{
	{
		std::lock_guard<std::mutex>	lock(m_mutex) ;
		m_flagQuit = true ;
	}
	m_cvReady.notify_one() ;
	m_pThread->join() ;
}

void NNParallelLoop::Thread::Run( void )
{
	while ( !m_flagQuit )
	{
		std::unique_lock<std::mutex>	lock(m_mutex) ;
		m_cvReady.wait( lock, [this] { return m_flagQuit || (m_pDispatch != nullptr) ; } ) ;

		if ( m_pDispatch != nullptr )
		{
			size_t	iLoop ;
			while ( m_ploop.NextLoopIndex( iLoop ) )
			{
				(*m_pDispatch)( m_iThread, iLoop ) ;
			}
			m_pDispatch = nullptr ;

			m_ploop.RecallDispatched() ;
		}
	}
}



//////////////////////////////////////////////////////////////////////////////
// 並列ループ
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNParallelLoop::NNParallelLoop( void )
	: m_nDispatched(0),
		m_spinLock(spinUnlocked),
		m_iLoopNext(0), m_iEndOfLoop(0)
{
}

// 消滅関数
//////////////////////////////////////////////////////////////////////////////
NNParallelLoop::~NNParallelLoop( void )
{
	EndThreads() ;
}

// スレッド準備
//////////////////////////////////////////////////////////////////////////////
void NNParallelLoop::BeginThreads( size_t nThreadCount )
{
	size_t	nLogicalProcessor = (size_t) std::thread::hardware_concurrency() ;
	if ( (nThreadCount == 0) || (nThreadCount > nLogicalProcessor) )
	{
		nThreadCount = nLogicalProcessor ;
	}
	nThreadCount = (nThreadCount >= 2) ? (nThreadCount - 1) : 0 ;

	for ( size_t i = m_threads.size(); i < nThreadCount; i ++ )
	{
		m_threads.push_back( std::make_shared<Thread>( *this, i + 1 ) ) ;
	}
}

// スレッド終了
//////////////////////////////////////////////////////////////////////////////
void NNParallelLoop::EndThreads( void )
{
	m_threads.clear() ;
}

// スレッド数（呼び出し元含む）取得
//////////////////////////////////////////////////////////////////////////////
size_t NNParallelLoop::GetThreadCount( void ) const
{
	return	m_threads.size() + 1 ;
}

// ループ実行
//////////////////////////////////////////////////////////////////////////////
void NNParallelLoop::Loop
	( size_t iFirstOfLoop, size_t iEndOfLoop,
		std::function<void(size_t,size_t)> func )
{
	m_iLoopNext = iFirstOfLoop ;
	m_iEndOfLoop = iEndOfLoop ;

	// スレッドにディスパッチ
	{
		std::lock_guard<std::mutex>	lock( m_mutex ) ;
		assert( m_nDispatched == 0 ) ;
		 for ( auto thread : m_threads )
		{
			++ m_nDispatched ;
			thread->Dispatch( func ) ;
		}
	}

	// このスレッドでも実行
	size_t	iLoop ;
	while ( NextLoopIndex( iLoop ) )
	{
		func( 0, iLoop ) ;
	}

	// 全てのスレッドの実行が完了するのを待つ
	{
		std::unique_lock<std::mutex>	lock(m_mutex) ;
		m_cvDone.wait( lock, [&](){ return (m_nDispatched == 0) ; } ) ;
	}
}

// 次のループカウンタ取得
//////////////////////////////////////////////////////////////////////////////
bool NNParallelLoop::NextLoopIndex( size_t& iLoop )
{
	bool	flagContinue = false ;
	SpinLock() ;
	if ( m_iLoopNext < m_iEndOfLoop )
	{
		iLoop = m_iLoopNext ;
		m_iLoopNext ++ ;
		flagContinue = true ;
	}
	SpinUnlock() ;
	return	flagContinue ;
}

// １つのスレッドがループを完了し待機状態になった
//////////////////////////////////////////////////////////////////////////////
void NNParallelLoop::RecallDispatched( void )
{
	bool	flagDone = false ;
	{
		std::lock_guard<std::mutex>	lock( m_mutex ) ;
		assert( m_nDispatched > 0 ) ;
		if ( -- m_nDispatched == 0 )
		{
			flagDone = true ;
		}
	}
	if ( flagDone )
	{
		m_cvDone.notify_one() ;
	}
}

// スピンロック
//////////////////////////////////////////////////////////////////////////////
void NNParallelLoop::SpinLock( void )
{
	while ( m_spinLock.exchange
		( spinLocked, std::memory_order_acquire ) == spinLocked )
	{
	}
}

void NNParallelLoop::SpinUnlock( void )
{
	m_spinLock.store( spinUnlocked, std::memory_order_release ) ;
}

