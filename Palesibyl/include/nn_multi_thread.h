
#ifndef	__NN_MULTI_THREAD_H__
#define	__NN_MULTI_THREAD_H__

#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// 並列ループ
//////////////////////////////////////////////////////////////////////////////

class	NNParallelLoop
{
protected:
	class	Thread
	{
	public:
		NNParallelLoop&							m_ploop ;
		size_t									m_iThread ;
		bool									m_flagQuit ;
		std::unique_ptr<std::thread>			m_pThread ;
		std::mutex								m_mutex ;
		std::condition_variable					m_cvReady ;
		std::function<void(size_t,size_t)> *	m_pDispatch ;
	public:
		Thread( NNParallelLoop& ploop, size_t iThread ) ;
		~Thread( void ) ;
		void Dispatch( std::function<void(size_t,size_t)>& func ) ;
		void Join( void ) ;
	private:
		void Run( void ) ;
	} ;
	friend class Thread ;

	std::vector< std::shared_ptr<Thread> >	m_threads ;	// 並列実行スレッド

	std::mutex					m_mutex ;
	std::condition_variable		m_cvDone ;
	size_t						m_nDispatched ;

	enum	SpinLockState
	{
		spinLocked,
		spinUnlocked,
	} ;
	std::atomic<SpinLockState>	m_spinLock ;
	size_t						m_iLoopNext ;
	size_t						m_iEndOfLoop ;

public:
	// 構築関数
	NNParallelLoop( void ) ;
	// 消滅関数
	~NNParallelLoop( void ) ;
	// スレッド準備（呼び出さない場合 DispatchLoop は単一スレッド）
	void BeginThreads( size_t nThreadCount = 0 ) ;
	// スレッド終了
	void EndThreads( void ) ;
	// スレッド数（呼び出し元含む）取得
	size_t GetThreadCount( void ) const ;
	// ループ実行
	void Loop( size_t iFirstOfLoop, size_t iEndOfLoop,
				std::function<void(size_t,size_t)> func ) ;

protected:
	// 次のループカウンタ取得
	bool NextLoopIndex( size_t& iLoop ) ;
	// １つのスレッドがループを完了し待機状態になった
	void RecallDispatched( void ) ;
	// スピンロック
	void SpinLock( void ) ;
	void SpinUnlock( void ) ;

} ;


}

#endif

