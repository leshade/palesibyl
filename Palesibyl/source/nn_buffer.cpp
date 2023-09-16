
#include "nn_buffer.h"
#include "nn_cuda_kernel.h"
#include <algorithm>

#ifdef	min
	#undef	min
#endif

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// 3次元バッファ
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNBuffer::NNBuffer( void )
	: m_dimSize(0,0,0), m_commitBuf(false),
		m_commitCuda(false), m_invalidCuda(false), m_cudaFlags(cudaNoMemory)
{
}

// 構築関数（※同じバッファを参照する）
//////////////////////////////////////////////////////////////////////////////
NNBuffer::NNBuffer( const NNBuffer& buf )
	: m_dimSize(buf.m_dimSize), m_commitBuf(buf.m_commitBuf),
		m_commitCuda(buf.m_commitCuda),
		m_invalidCuda(buf.m_invalidCuda), m_cudaFlags(buf.m_cudaFlags),
		m_buffer(buf.m_buffer), m_cudaMemory(buf.m_cudaMemory)
{
}

// 消滅関数
//////////////////////////////////////////////////////////////////////////////
NNBuffer::~NNBuffer( void )
{
	Free() ;
}

// バッファ作成（Commit も実行）
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::Create
	( size_t width, size_t height, size_t ch, size_t nLength, uint32_t cudaFlags )
{
	Allocate( width, height, ch, nLength, cudaFlags ) ;
	Commit() ;
}

void NNBuffer::Create( const NNBufDim& dim, uint32_t cudaFlags )
{
	Allocate( dim, cudaFlags ) ;
	Commit() ;
}

// バッファ作成（サイズだけ設定（allocateWithCommit 指定しない場合））
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::Allocate( const NNBufDim& dim, uint32_t cudaFlags )
{
	assert( !m_commitBuf ) ;
	m_dimSize = dim ;
	m_cudaFlags = cudaFlags & ~allocateWithCommit ;
	//
	if ( cudaFlags & allocateWithCommit )
	{
		Commit() ;
	}
}

void NNBuffer::Allocate
	( size_t width, size_t height, size_t ch, size_t nLength, uint32_t nCudaFlags )
{
	assert( !m_commitBuf ) ;
	m_dimSize = NNBufDim( width, height, ch ) ;
	assert( nLength <= m_dimSize.n ) ;
	if ( nLength != 0 )
	{
		m_dimSize.n = nLength ;
	}
	m_cudaFlags = nCudaFlags & ~allocateWithCommit ;
	//
	if ( nCudaFlags & allocateWithCommit )
	{
		Commit() ;
	}
}

// 配列形状変換
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::TransformShape( size_t width, size_t height, size_t ch )
{
	TransformShape( NNBufDim( width, height, ch ) ) ;
}

void NNBuffer::TransformShape( const NNBufDim& dim )
{
	assert( m_dimSize.n * m_dimSize.z >= dim.n * dim.z ) ;
	if ( m_dimSize.n * m_dimSize.z >= dim.n * dim.z )
	{
#if	!defined(NDEBUG) && defined(_DEBUG)
		const size_t	nLastSize = m_dimSize.n * m_dimSize.z ;
		const size_t	nNewSize = dim.n * dim.z ;
		if ( m_commitBuf && (nLastSize > nNewSize) )
		{
			assert( m_buffer != nullptr ) ;
			float *	pBuf = m_buffer->data() ;
			for ( size_t i = nNewSize; i < nLastSize; i ++ )
			{
				pBuf[i] = 0.0f ;
			}
		}
#endif
		m_dimSize = dim ;
	}
}

// 所有バッファ入れ替え
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::SwapBuffer( NNBuffer& bufSwap )
{
	assert( m_dimSize == bufSwap.m_dimSize ) ;
	assert( m_commitBuf == bufSwap.m_commitBuf ) ;
	assert( m_commitCuda == bufSwap.m_commitCuda ) ;

	bool		invalidCuda = m_invalidCuda ;
	uint32_t	cudaFlags = m_cudaFlags ;
	std::shared_ptr< std::vector<float> >
				buffer = m_buffer ;
	std::shared_ptr< CudaFloat1DMemory >
				cudaMemory = m_cudaMemory ;

	m_invalidCuda = bufSwap.m_invalidCuda ;
	m_cudaFlags = bufSwap.m_cudaFlags ;
	m_buffer = bufSwap.m_buffer ;
	m_cudaMemory = bufSwap.m_cudaMemory ;

	bufSwap.m_invalidCuda = m_invalidCuda ;
	bufSwap.m_cudaFlags = cudaFlags ;
	bufSwap.m_buffer = buffer ;
	bufSwap.m_cudaMemory = cudaMemory ;
}

// 所有バッファ複製
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::DuplicateBuffer( const NNBuffer& bufDup )
{
	m_dimSize = bufDup.m_dimSize ;
	m_commitBuf = bufDup.m_commitBuf ;
	m_commitCuda = bufDup.m_commitCuda ;
	m_invalidCuda = bufDup.m_invalidCuda ;
	m_cudaFlags = bufDup.m_cudaFlags ;
	m_buffer = bufDup.m_buffer ;
	m_cudaMemory = bufDup.m_cudaMemory ;
}

// 同一バッファ
//////////////////////////////////////////////////////////////////////////////
bool NNBuffer::IsEqualBuffer( const NNBuffer& buf ) const
{
	assert( ((m_buffer == buf.m_buffer) && (m_cudaMemory == buf.m_cudaMemory))
			|| (((m_buffer != buf.m_buffer) || (m_buffer == nullptr) || (buf.m_buffer == nullptr))
				&& ((m_cudaMemory != buf.m_cudaMemory) || (m_cudaMemory == nullptr) || (buf.m_cudaMemory == nullptr))) ) ;
	return	(m_buffer == buf.m_buffer) && (m_cudaMemory == buf.m_cudaMemory) ;
}

// バッファ解放
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::Free( void )
{
	Uncommit() ;
	UncommitCuda() ;
	m_dimSize = NNBufDim(0,0,0) ;
}

// フィル
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::Fill( float fill )
{
	assert( IsCommitted() ) ;
	float *			pBuf = GetBuffer() ;
	const size_t	nCount = m_dimSize.n * m_dimSize.z ;
	for ( size_t i = 0; i < nCount; i ++ )
	{
		pBuf[i] = fill ;
	}
}

// メモリ確保・解放 (Alloc 呼出し後)
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::Commit( void )
{
	if ( m_cudaFlags & cudaAllocate )
	{
		CommitCuda() ;
	}
	else if ( !m_commitBuf )
	{
		m_buffer = std::make_shared< std::vector<float> >() ;
#if	!defined(NDEBUG) && defined(_DEBUG)
		m_buffer->resize( (size_t) ((m_dimSize.n + m_dimSize.x) * m_dimSize.z) ) ;
		memset( m_buffer->data() + m_dimSize.n * m_dimSize.z,
					0, m_dimSize.x * m_dimSize.z * sizeof(float) ) ;
#else
		m_buffer->resize( (size_t) (m_dimSize.n * m_dimSize.z) ) ;
#endif
		m_commitBuf = true ;
	}
}

void NNBuffer::Uncommit( void )
{
	if ( m_cudaFlags & cudaAllocate )
	{
		UncommitCuda() ;
	}
	if ( m_commitBuf )
	{
		assert( m_buffer != nullptr ) ;
		CheckOverun() ;
		m_buffer = nullptr ;
		m_commitBuf = false ;
	}
}

bool NNBuffer::IsCommitted( void ) const
{
	return	m_commitBuf || ((m_cudaFlags & cudaAllocate) && m_commitCuda) ;
}

// バッファ・オーバーラン・チェック（デバッグ・コンパイル時のみ）
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::CheckOverun( void )
{
#if	!defined(NDEBUG) && defined(_DEBUG)
	if ( m_commitBuf )
	{
		assert( m_buffer != nullptr ) ;
		const uint8_t *	pBuf =
			reinterpret_cast<const uint8_t*>
				( m_buffer->data() + m_dimSize.n * m_dimSize.z ) ;
		size_t	nLineBytes = m_dimSize.x * m_dimSize.z * sizeof(float) ;
		for ( size_t i = 0; i < nLineBytes; i ++ )
		{
			assert( pBuf[i] == 0 ) ;
		}
	}
#endif
}

// バッファサイズ
//////////////////////////////////////////////////////////////////////////////
const NNBufDim& NNBuffer::GetSize( void ) const
{
	return	m_dimSize ;
}

unsigned long long NNBuffer::GetBufferBytes( void ) const
{
	return	(unsigned long long) m_dimSize.n * m_dimSize.z * sizeof(float) ;
}

unsigned long long NNBuffer::GetCudaBufferBytes( void ) const
{
	return	(m_cudaMemory != nullptr)
				? m_cudaMemory->GetLength() * sizeof(float) : 0 ;
}

// 画像データから変換
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::CopyFromImage
	( const uint8_t * pubImage,
		size_t width, size_t height, size_t depth,
		size_t stridePixel, int strideLine )
{
	assert( width <= m_dimSize.x ) ;
	assert( height <= m_dimSize.y ) ;
	assert( depth <= m_dimSize.z ) ;
	float *	pDst = GetBuffer() ;
	for ( size_t y = 0; y < height; y ++ )
	{
		float *			pDstNext = pDst + y * m_dimSize.x * m_dimSize.z ;
		const uint8_t *	pSrcNext = pubImage + (int) y * strideLine ;
		for ( size_t x = 0; x < width; x ++ )
		{
			for ( size_t c = 0; c < depth; c ++ )
			{
				pDstNext[c] = (float) pSrcNext[c] / 255.0f ;
			}
			pDstNext += m_dimSize.z ;
			pSrcNext += stridePixel ;
		}
	}
}

// 画像データへ変換
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::CopyToImage
	( uint8_t * pubImage,
		size_t width, size_t height, size_t depth,
		size_t stridePixel, int strideLine ) const
{
	assert( width <= m_dimSize.x ) ;
	assert( height <= m_dimSize.y ) ;
	assert( depth <= m_dimSize.z ) ;
	const float *	pSrc = GetConstBuffer() ;
	for ( size_t y = 0; y < height; y ++ )
	{
		const float *	pSrcNext = pSrc + y * m_dimSize.x * m_dimSize.z ;
		uint8_t *		pDstNext = pubImage + (int) y * strideLine ;
		for ( size_t x = 0; x < width; x ++ )
		{
			for ( size_t c = 0; c < depth; c ++ )
			{
				int	v = (int) (pSrcNext[c] * 255.0f + 0.5f) ;
				pDstNext[c] = (v <= 0xff) ? ((v > 0) ? (uint8_t) v : 0) : 0xff ;
			}
			pDstNext += stridePixel ;
			pSrcNext += m_dimSize.z ;
		}
	}
}

// 複製
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::CopyFrom( const NNBuffer& nnSrcBuf )
{
	assert( nnSrcBuf.IsCommitted() ) ;
	Commit() ;
	CopyFrom( nnSrcBuf, NNBufDim(0,0,0) ) ;
}

void NNBuffer::CopyFrom( const NNBuffer& nnSrcBuf, NNBufDim dimSrcOffset )
{
	assert( nnSrcBuf.IsCommitted() ) ;
	Commit() ;

	NNBufDim	dimSrc = nnSrcBuf.GetSize() ;
	NNBufDim	dimCopy = dimSrc ;
	if ( (dimCopy.x <= dimSrcOffset.x)
		|| (dimCopy.y <= dimSrcOffset.y)
		|| (dimCopy.z <= dimSrcOffset.z) )
	{
		return ;
	}
	dimCopy.x -= dimSrcOffset.x ;
	dimCopy.y -= dimSrcOffset.y ;
	dimCopy.z -= dimSrcOffset.z ;
	//
	dimCopy.x = std::min( dimCopy.x, m_dimSize.x ) ;
	dimCopy.y = std::min( dimCopy.y, m_dimSize.y ) ;
	dimCopy.z = std::min( dimCopy.z, m_dimSize.z ) ;
	//
	float *			pDstBuf = GetBuffer() ;
	const float *	pSrcBuf = nnSrcBuf.GetConstBuffer() ;
	const size_t	nDstLineStride = m_dimSize.x * m_dimSize.z ;
	const size_t	nSrcLineStride = dimSrc.x * dimSrc.z ;
	for ( size_t y = 0; y < dimCopy.y; y ++ )
	{
		size_t	ySrc = dimSrcOffset.y + y ;
		for ( size_t x = 0; x < dimCopy.x; x ++ )
		{
			size_t	xSrc = dimSrcOffset.x + x ;
			size_t	zSrc = dimSrcOffset.z ;
			for ( size_t z = 0; z < dimCopy.z; z ++, zSrc ++ )
			{
				pDstBuf[y * nDstLineStride + (x * m_dimSize.z) + z] =
					pSrcBuf[ySrc * nSrcLineStride + (xSrc * dimSrc.z) + zSrc] ;
			}
		}
	}
}

void NNBuffer::CopyChannelFrom
	( size_t iDstChannel,
		const NNBuffer& nnSrcBuf, size_t iSrcChannel, size_t nSrcChCount )
{
	assert( nnSrcBuf.IsCommitted() ) ;
	Commit() ;

	if ( m_dimSize.z <= iDstChannel )
	{
		return ;
	}
	NNBufDim	dimSrc = nnSrcBuf.GetSize() ;
	if ( dimSrc.z <= iSrcChannel )
	{
		return ;
	}
	NNBufDim	dimCopy = dimSrc ;
	dimCopy.x = std::min( dimCopy.x, m_dimSize.x ) ;
	dimCopy.y = std::min( dimCopy.y, m_dimSize.y ) ;
	dimCopy.z = std::min( dimSrc.z - iSrcChannel, m_dimSize.z - iDstChannel ) ;
	if ( (nSrcChCount != 0) && (nSrcChCount < dimCopy.z) )
	{
		dimCopy.z = nSrcChCount ;
	}
	//
	float *			pDstBuf = GetBuffer() ;
	const float *	pSrcBuf = nnSrcBuf.GetConstBuffer() ;
	const size_t	nDstLineStride = m_dimSize.x * m_dimSize.z ;
	const size_t	nSrcLineStride = dimSrc.x * dimSrc.z ;
	for ( size_t y = 0; y < dimCopy.y; y ++ )
	{
		for ( size_t x = 0; x < dimCopy.x; x ++ )
		{
			for ( size_t z = 0; z < dimCopy.z; z ++ )
			{
				pDstBuf[y * nDstLineStride + (x * m_dimSize.z) + iDstChannel + z] =
					pSrcBuf[y * nSrcLineStride + (x * dimSrc.z) + iSrcChannel + z] ;
			}
		}
	}
}

void NNBuffer::ShiftCopyChannelFrom
	( size_t iDstChannel,
		const NNBuffer& nnSrcBuf,
		int xShiftSample, int yShitSample,
		size_t iSrcChannel, size_t nSrcChCount )
{
	assert( nnSrcBuf.IsCommitted() ) ;
	Commit() ;

	assert( (this != &nnSrcBuf)
			|| ((xShiftSample <= 0) && (yShitSample <= 0)) ) ;

	if ( m_dimSize.z <= iDstChannel )
	{
		return ;
	}
	NNBufDim	dimSrc = nnSrcBuf.GetSize() ;
	if ( dimSrc.z <= iSrcChannel )
	{
		return ;
	}
	NNBufDim	dimCopy = dimSrc ;
	dimCopy.x = std::min( dimCopy.x, m_dimSize.x ) ;
	dimCopy.y = std::min( dimCopy.y, m_dimSize.y ) ;
	dimCopy.z = std::min( dimSrc.z - iSrcChannel, m_dimSize.z - iDstChannel ) ;
	if ( (nSrcChCount != 0) && (nSrcChCount < dimCopy.z) )
	{
		dimCopy.z = nSrcChCount ;
	}
	float *			pDstBuf = GetBuffer() ;
	const float *	pSrcBuf = nnSrcBuf.GetConstBuffer() ;
	const size_t	nDstLineStride = m_dimSize.x * m_dimSize.z ;
	const size_t	nSrcLineStride = dimSrc.x * dimSrc.z ;
	for ( size_t y = 0; y < dimCopy.y; y ++ )
	{
		int	ySrc = (int) y - yShitSample ;
		if ( (ySrc < 0) || ((size_t) ySrc >= dimSrc.y) )
		{
			ySrc = (int) y ;
		}
		float *			pDstLineBuf = pDstBuf + y * nDstLineStride ;
		const float *	pSrcLineBuf = pSrcBuf + ySrc * nSrcLineStride ;
		for ( size_t x = 0; x < dimCopy.x; x ++ )
		{
			int	iSrc =(int) x - xShiftSample ;
			if ( (iSrc < 0) || ((size_t) iSrc >= dimSrc.x) )
			{
				iSrc = (int) x ;
			}
			iSrc *= (int) dimSrc.z ;
			//
			for ( size_t z = 0; z < dimCopy.z; z ++ )
			{
				pDstLineBuf[iDstChannel + z] = pSrcLineBuf[iSrc + iSrcChannel + z] ;
			}
			pDstLineBuf += m_dimSize.z ;
		}
	}
}

// 要素値加算
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::AddChannelValue
	( size_t xDst, size_t yDst, size_t iDstChannel,
		const NNBuffer& nnSrcBuf, int xShift,
		size_t iSrcChannel, size_t nChannelCount,
		size_t nWidth, size_t nHeight, float scaleFactor )
{
	assert( nnSrcBuf.IsCommitted() ) ;
	Commit() ;

	if ( m_dimSize.z <= iDstChannel )
	{
		return ;
	}
	NNBufDim	dimSrc = nnSrcBuf.GetSize() ;
	if ( dimSrc.z <= iSrcChannel )
	{
		return ;
	}
	if ( nWidth == 0 )
	{
		assert( xDst < m_dimSize.x ) ;
		nWidth = m_dimSize.x - xDst ;
	}
	if ( nHeight == 0 )
	{
		assert( yDst < m_dimSize.y ) ;
		nHeight = m_dimSize.y - yDst ;
	}
	NNBufDim	dimCopy = dimSrc ;
	dimCopy.x = std::min( dimCopy.x, nWidth ) ;
	dimCopy.y = std::min( dimCopy.y, nHeight ) ;
	dimCopy.z = std::min( dimSrc.z - iSrcChannel, m_dimSize.z - iDstChannel ) ;
	dimCopy.z = std::min( dimCopy.z, nChannelCount ) ;

	float *			pDstBuf = GetBuffer() ;
	const float *	pSrcBuf = nnSrcBuf.GetConstBuffer() ;
	const size_t	nDstLineStride = m_dimSize.x * m_dimSize.z ;
	const size_t	nSrcLineStride = dimSrc.x * dimSrc.z ;
	for ( size_t y = 0; y < dimCopy.y; y ++ )
	{
		float *			pDstLineBuf = pDstBuf + (y + yDst) * nDstLineStride
												+ xDst * m_dimSize.z ;
		const float *	pSrcLineBuf = pSrcBuf + y * nSrcLineStride ;
		for ( size_t x = 0; x < dimCopy.x; x ++ )
		{
			int	iSrc = (int) x - xShift ;
			if ( (iSrc < 0) || ((size_t) iSrc >= dimSrc.x) )
			{
				iSrc = (int) x ;
			}
			iSrc *= (int) dimSrc.z ;
			//
			for ( size_t z = 0; z < dimCopy.z; z ++ )
			{
				pDstLineBuf[iDstChannel + z]
					+= pSrcLineBuf[iSrc + iSrcChannel + z] * scaleFactor ;
			}
			pDstLineBuf += m_dimSize.z ;
		}
	}
}

// バッファポインタ
//////////////////////////////////////////////////////////////////////////////
float * NNBuffer::GetBuffer( void )
{
	assert( IsCommitted() || IsCommittedCuda() ) ;
	return	m_commitCuda ? m_cudaMemory->GetArray() : m_buffer->data() ;
}

float * NNBuffer::GetBufferAt( size_t x, size_t y )
{
	assert( x < m_dimSize.x ) ;
	assert( y < m_dimSize.y ) ;
	return	GetBuffer() + (y * m_dimSize.x + x) * m_dimSize.z ;
}

const float * NNBuffer::GetConstBuffer( void ) const
{
	assert( IsCommitted() || IsCommittedCuda() ) ;
	return	m_commitCuda ? m_cudaMemory->GetArray() : m_buffer->data() ;
}

const float * NNBuffer::GetConstBufferAt( size_t x, size_t y ) const
{
	assert( x < m_dimSize.x ) ;
	assert( y < m_dimSize.y ) ;
	return	GetConstBuffer() + (y * m_dimSize.x + x) * m_dimSize.z ;
}

// CUDA メモリ確保・解放 (Alloc 呼出し後)
//////////////////////////////////////////////////////////////////////////////
bool NNBuffer::CommitCuda( void )
{
	if ( !cudaIsAvailable() )
	{
		return	false ;
	}
	if ( !m_commitCuda )
	{
		m_cudaMemory = std::make_shared<CudaFloat1DMemory>() ;
		m_cudaMemory->Allocate
			( m_dimSize.n * m_dimSize.z,
				((m_cudaFlags & cudaDeviceOnly) && !m_commitBuf)
					? CudaFloat1DMemory::allocDeviceOnly
					: CudaFloat1DMemory::allocDefault ) ;
		m_commitCuda = true ;
		m_invalidCuda = false ;
		//
		if ( m_commitBuf )
		{
			assert( m_buffer != nullptr ) ;
			assert( m_buffer->size() >= m_dimSize.n * m_dimSize.z ) ;
			m_cudaMemory->CopyFrom( m_buffer->data(), m_dimSize.n * m_dimSize.z ) ;
			m_buffer->clear() ;
			m_buffer = nullptr ;
			m_commitBuf = false ;
		}
	}
	else if ( m_invalidCuda )
	{
		assert( m_cudaMemory != nullptr ) ;
		m_cudaMemory->CopyToDevice() ;
		m_invalidCuda = false ;
	}
	return	true ;
}

bool NNBuffer::CommitCudaWithHost( void )
{
	if ( !cudaIsAvailable() )
	{
		return	false ;
	}
	m_cudaFlags &= ~cudaDeviceOnly ;
	if ( !m_commitCuda )
	{
		return	CommitCuda() ;
	}
	else
	{
		assert( m_cudaMemory != nullptr ) ;
		m_cudaMemory->AllocateHost() ;
	}
	return	true ;
}

void NNBuffer::UncommitCuda( void )
{
	if ( m_commitCuda )
	{
		assert( m_cudaMemory != nullptr ) ;
		m_cudaMemory = nullptr ;
		m_commitCuda = false ;
	}
}

bool NNBuffer::IsCommittedCuda( void ) const
{
	return	m_commitCuda ;
}

// 次の CommitCuda でホストメモリから CUDA デバイスへ転送する
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::InvalidateCuda( void )
{
	m_invalidCuda = true ;
}

// フィル
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::CudaFill( float fill, cudaStream_t stream )
{
	assert( m_commitCuda ) ;
	assert( m_cudaMemory != nullptr ) ;
	nncuda_FillMemory( m_cudaMemory->GetDevicePtr(), m_dimSize, fill, stream ) ;
}

// 矩形外側フィル
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::CudaFillExterior
	( size_t xLeft, size_t yTop,
		size_t xRight, size_t yBottom,
		float fill, cudaStream_t stream )
{
	assert( m_commitCuda ) ;
	assert( m_cudaMemory != nullptr ) ;
	if ( (xLeft > 0) || (yTop > 0)
		|| (xRight < m_dimSize.x) || (yBottom < m_dimSize.y) )
	{
		nncuda_FillExterior
			( m_cudaMemory->GetDevicePtr(), m_dimSize,
				xLeft, yTop, xRight, yBottom, fill, stream ) ;
	}
}

// CUDA デバイスへ転送
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::CudaAsyncToDevice( cudaStream_t stream )
{
	assert( m_commitCuda ) ;
	assert( m_cudaMemory != nullptr ) ;
	m_cudaMemory->AsyncToDevice( stream ) ;
}

// データを CUDA デバイスへ転送
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::CudaCopyAsyncFrom
	( const float * pSrc, size_t nLength, cudaStream_t stream )
{
	assert( m_commitCuda ) ;
	assert( m_cudaMemory != nullptr ) ;
	m_cudaMemory->CopyAsyncFrom( pSrc, nLength, stream ) ;
}

// CUDA デバイスから転送
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::CudaAsyncFromDevice( cudaStream_t stream )
{
	assert( m_commitCuda ) ;
	assert( m_cudaMemory != nullptr ) ;
	m_cudaMemory->AsyncFromDevice( stream ) ;
}

// CUDA デバイス間転送
//////////////////////////////////////////////////////////////////////////////
void NNBuffer::CudaCopyFrom
	( const NNBuffer& nnSrcBuf, cudaStream_t stream )
{
	assert( m_commitCuda ) ;
	assert( m_cudaMemory != nullptr ) ;
	assert( nnSrcBuf.m_commitCuda ) ;
	assert( nnSrcBuf.m_cudaMemory != nullptr ) ;
	assert( m_dimSize == nnSrcBuf.m_dimSize ) ;
	m_cudaMemory->CopyAsyncDeviceFrom( *(nnSrcBuf.m_cudaMemory), stream ) ;
}

void NNBuffer::CudaCopyDeviceFrom
	( const CudaFloat1DMemory& cmemSrc, cudaStream_t stream )
{
	assert( m_commitCuda ) ;
	assert( m_cudaMemory != nullptr ) ;
	assert( m_dimSize.n == cmemSrc.GetLength() ) ;
	m_cudaMemory->CopyAsyncDeviceFrom( cmemSrc, stream ) ;
}

void NNBuffer::CudaCopyChannelFrom
	( size_t xDst, size_t yDst, size_t iDstChannel,
		const NNBuffer& nnSrcBuf, int xShift, int yShift,
		size_t iSrcChannel, size_t nChannelCount,
		size_t nWidth, size_t nHeight, cudaStream_t stream )
{
	if ( nWidth == 0 )
	{
		assert( xDst < m_dimSize.x ) ;
		nWidth = m_dimSize.x - xDst ;
	}
	if ( nHeight == 0 )
	{
		assert( yDst < m_dimSize.y ) ;
		nHeight = m_dimSize.y - yDst ;
	}
	assert( m_commitCuda ) ;
	assert( m_cudaMemory != nullptr ) ;
	assert( nnSrcBuf.m_cudaMemory != nullptr ) ;
	assert( xDst + nWidth <= m_dimSize.x ) ;
	assert( yDst + nHeight <= m_dimSize.y ) ;
	assert( iDstChannel + nChannelCount <= m_dimSize.z ) ;
	assert( iSrcChannel + nChannelCount <= nnSrcBuf.m_dimSize.z ) ;
	nncuda_ShiftMoveMemory
		( m_cudaMemory->GetDevicePtr(), m_dimSize,
			xDst, yDst, iDstChannel, nWidth, nHeight,
			nnSrcBuf.m_cudaMemory->GetDevicePtr(), nnSrcBuf.m_dimSize,
			xShift, yShift, iSrcChannel, nChannelCount, 1.0f, stream ) ;
}

void NNBuffer::CudaAddChannelFrom
	( size_t xDst, size_t yDst, size_t iDstChannel,
		const NNBuffer& nnSrcBuf, int xShift, int yShift,
		size_t iSrcChannel, size_t nChannelCount,
		size_t nWidth, size_t nHeight, float scaleSrc, cudaStream_t stream )
{
	if ( nWidth == 0 )
	{
		assert( xDst < m_dimSize.x ) ;
		nWidth = m_dimSize.x - xDst ;
	}
	if ( nHeight == 0 )
	{
		assert( yDst < m_dimSize.y ) ;
		nHeight = m_dimSize.y - yDst ;
	}
	assert( m_commitCuda ) ;
	assert( m_cudaMemory != nullptr ) ;
	assert( nnSrcBuf.m_cudaMemory != nullptr ) ;
	assert( xDst + nWidth <= m_dimSize.x ) ;
	assert( yDst + nHeight <= m_dimSize.y ) ;
	assert( iDstChannel + nChannelCount <= m_dimSize.z ) ;
	assert( iSrcChannel + nChannelCount <= nnSrcBuf.m_dimSize.z ) ;
	nncuda_ShiftAddMemory
		( m_cudaMemory->GetDevicePtr(), m_dimSize,
			xDst, yDst, iDstChannel, nWidth, nHeight,
			nnSrcBuf.m_cudaMemory->GetDevicePtr(), nnSrcBuf.m_dimSize,
			xShift, yShift, iSrcChannel, nChannelCount, scaleSrc, stream ) ;
}

// CUDA メモリ
//////////////////////////////////////////////////////////////////////////////
CudaFloat1DMemory& NNBuffer::CudaMemory( void )
{
	assert( m_cudaMemory != nullptr ) ;
	return	*m_cudaMemory ;
}

const CudaFloat1DMemory& NNBuffer::GetCudaMemory( void ) const
{
	assert( m_cudaMemory != nullptr ) ;
	return	*m_cudaMemory ;
}

float * NNBuffer::GetCudaPtr( void ) const
{
	assert( m_commitCuda ) ;
	assert( m_cudaMemory != nullptr ) ;
	return	m_cudaMemory->GetDevicePtr() ;
}


