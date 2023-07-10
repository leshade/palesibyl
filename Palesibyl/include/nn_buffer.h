
#ifndef	__NN_BUFFER_H__
#define	__NN_BUFFER_H__

#include <vector>
#include "nn_cuda_util.h"
#include "nn_type_def.h"

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// 3次元バッファ
//////////////////////////////////////////////////////////////////////////////

class	NNBuffer
{
public:
	enum	AllocateMemoryFlag
	{
		cudaNoMemory		= 0x0000,
		cudaAllocate		= 0x0001,
		cudaDeviceOnly		= 0x0002,
		allocateWithCommit	= 0x0010,
	} ;

protected:
	NNBufDim								m_dimSize ;
	bool									m_commitBuf ;
	bool									m_commitCuda ;
	bool									m_invalidCuda ;
	uint32_t								m_cudaFlags ;
	std::shared_ptr< std::vector<float> >	m_buffer ;
	std::shared_ptr< CudaFloat1DMemory >	m_cudaMemory ;

public:
	// 構築関数
	NNBuffer( void ) ;
	// 構築関数（※同じバッファを参照する）
	NNBuffer( const NNBuffer& buf ) ;
	// 消滅関数
	~NNBuffer( void ) ;
	// バッファ作成（Commit も実行）
	void Create
		( size_t width, size_t height,
			size_t ch = 1, size_t nLength = 0,
			uint32_t cudaFlags = cudaNoMemory ) ;
	void Create
		( const NNBufDim& dim, uint32_t cudaFlags = cudaNoMemory ) ;
	// バッファ作成（サイズだけ設定（allocateWithCommit 指定しない場合））
	void Allocate
		( size_t width, size_t height,
			size_t ch = 1, size_t nLength = 0,
			uint32_t cudaFlags = cudaNoMemory ) ;
	void Allocate
		( const NNBufDim& dim, uint32_t cudaFlags = cudaNoMemory ) ;
	// 配列形状変換
	void TransformShape( size_t width, size_t height, size_t ch ) ;
	void TransformShape( const NNBufDim& dim ) ;
	// 所有バッファ入れ替え
	void SwapBuffer( NNBuffer& bufSwap ) ;
	// バッファ解放
	void Free( void ) ;
	// フィル
	void Fill( float fill = 0.0f ) ;
	// メモリ確保・解放 (Alloc 呼出し後)
	void Commit( void ) ;
	void Uncommit( void ) ;
	bool IsCommitted( void ) const ;
	// バッファ・オーバーラン・チェック（デバッグ・コンパイル時のみ）
	void CheckOverun( void ) ;
	// バッファサイズ
	const NNBufDim& GetSize( void ) const ;
	unsigned long long GetBufferBytes( void ) const ;
	unsigned long long GetCudaBufferBytes( void ) const ;
	// 画像データから変換
	void CopyFromImage
		( const uint8_t * pubImage,
			size_t width, size_t height, size_t depth,
			size_t stridePixel, int strideLine ) ;
	// 画像データへ変換
	void CopyToImage
		( uint8_t * pubImage,
			size_t width, size_t height, size_t depth,
			size_t stridePixel, int strideLine ) const ;
	// 複製
	void CopyFrom( const NNBuffer& nnSrcBuf ) ;
	void CopyFrom( const NNBuffer& nnSrcBuf, NNBufDim dimSrcOffset ) ;
	void CopyChannelFrom
		( size_t iDstChannel,
			const NNBuffer& nnSrcBuf,
			size_t iSrcChannel = 0, size_t nSrcChCount = 0 ) ;
	void ShiftCopyChannelFrom
		( size_t iDstChannel,
			const NNBuffer& nnSrcBuf,
			int xShiftSample, int yShitSample,
			size_t iSrcChannel = 0, size_t nSrcChCount = 0 ) ;
	// 要素値加算
	void AddChannelValue
		( size_t iDstChannel,
			const NNBuffer& nnSrcBuf, int xShift,
			size_t iSrcChannel, size_t nChannelCount, float scaleFactor ) ;
	// バッファポインタ
	float * GetBuffer( void ) ;
	float * GetBufferAt( size_t x, size_t y ) ;
	const float * GetConstBuffer( void ) const ;
	const float * GetConstBufferAt( size_t x, size_t y ) const ;

public:
	// CUDA メモリ確保・解放 (Alloc 呼出し後)
	bool CommitCuda( void ) ;
	void UncommitCuda( void ) ;
	bool IsCommittedCuda( void ) const ;
	// 次の CommitCuda でホストメモリから CUDA デバイスへ転送する
	void InvalidateCuda( void ) ;
	// フィル
	void CudaFill( float fill, cudaStream_t stream ) ;
	// CUDA デバイスへ転送
	void CudaAsyncToDevice( cudaStream_t stream ) ;
	// データを CUDA デバイスへ転送
	void CudaCopyAsyncFrom
		( const float * pSrc, size_t nLength, cudaStream_t stream ) ;
	// CUDA デバイスから転送
	void CudaAsyncFromDevice( cudaStream_t stream ) ;
	// CUDA デバイス間転送
	void CudaCopyFrom
		( const NNBuffer& nnSrcBuf, cudaStream_t stream ) ;
	void CudaCopyDeviceFrom
		( const CudaFloat1DMemory& cmemSrc, cudaStream_t stream ) ;
	void CudaCopyChannelFrom
		( size_t iDstChannel,
			const NNBuffer& nnSrcBuf, int xShift, int yShift,
			size_t iSrcChannel, size_t nChannelCount, cudaStream_t stream ) ;
	void CudaAddChannelFrom
		( size_t iDstChannel,
			const NNBuffer& nnSrcBuf, int xShift, int yShift,
			size_t iSrcChannel, size_t nChannelCount,
			float scaleSrc, cudaStream_t stream ) ;
	// CUDA メモリ
	CudaFloat1DMemory& CudaMemory( void ) ;
	const CudaFloat1DMemory& GetCudaMemory( void ) const ;
	float * GetCudaPtr( void ) const ;

} ;

}

#endif

