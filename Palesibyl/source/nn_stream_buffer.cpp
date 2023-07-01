
#include "nn_stream_buffer.h"

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// ストリーミング用バッファ
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNStreamBuffer::NNStreamBuffer( void )
	: m_xFilled( 0 )
{
}

// バッファ解放
//////////////////////////////////////////////////////////////////////////////
void NNStreamBuffer::Free( void )
{
	NNBuffer::Free() ;
	m_xFilled = 0 ;
}

// 出力蓄積数取得
//////////////////////////////////////////////////////////////////////////////
size_t NNStreamBuffer::GetCurrent( void ) const
{
	return	m_xFilled ;
}

// 空き空間を埋める
//////////////////////////////////////////////////////////////////////////////
void NNStreamBuffer::FillEmpty( float fill )
{
	if ( m_xFilled < m_dimSize.x )
	{
		const size_t	nCount = (m_dimSize.y - m_xFilled) * m_dimSize.z ;
		for ( size_t y = 0; y < m_dimSize.y; y ++ )
		{
			float *	pLine = GetBufferAt( m_xFilled, y ) ;
			for ( size_t i = 0; i < nCount; i ++ )
			{
				pLine[i] = fill ;
			}
		}
	}
}

// シフト（ｘ方向に左へシフトしデータを捨てる）
//////////////////////////////////////////////////////////////////////////////
size_t NNStreamBuffer::Shift( size_t xCount )
{
	const size_t	xShift = min( m_xFilled, xCount ) ;
	if ( xShift > 0 )
	{
		assert( m_xFilled <= m_dimSize.x ) ;
		if ( xShift < m_xFilled )
		{
			const size_t	xShiftCount = m_xFilled - xShift ;
			for ( size_t y = 0; y < m_dimSize.y; y ++ )
			{
				float *	pLine = GetBufferAt( 0, y ) ;
				memmove( pLine, pLine + (xShift * m_dimSize.z),
							xShiftCount * m_dimSize.z * sizeof(float) ) ;
			}
			assert( m_xFilled >= xShift ) ;
			m_xFilled -= xShift ;
		}
		else
		{
			m_xFilled = 0 ;
		}
	}
	return	xShift ;
}

// ストリーミング（ｘ方向に右から左へ流れていく）
//////////////////////////////////////////////////////////////////////////////
size_t NNStreamBuffer::Stream( const NNBuffer& bufSrc, size_t xSrc, size_t xCount )
{
	const NNBufDim	dimSrc = bufSrc.GetSize() ;
	assert( dimSrc.y == m_dimSize.y ) ;
	assert( xCount <= m_dimSize.x ) ;
	assert( xSrc + xCount <= dimSrc.x ) ;
	if ( xSrc >= dimSrc.x )
	{
		return	0 ;
	}
	NNBufDim	dimCopy( min( xCount, m_dimSize.x ),
							min( dimSrc.y, m_dimSize.y ),
							min( dimSrc.z, m_dimSize.z ) ) ;
	if ( xSrc + xCount > dimSrc.x )
	{
		dimCopy.x = dimSrc.x - xSrc ;
	}

	// バッファ要素を左に xCount シフト
	if ( m_xFilled + dimCopy.x > m_dimSize.x )
	{
		Shift( m_xFilled + dimCopy.x - m_dimSize.x ) ;
	}

	// バッファ要素をコピー
	for ( size_t y = 0; y < dimCopy.y; y ++ )
	{
		float *			pDstLine = GetBufferAt( m_xFilled, y ) ;
		const float *	pSrcLine = bufSrc.GetConstBufferAt( xSrc, y ) ;
		for ( size_t x = 0; x < dimCopy.x; x ++ )
		{
			for ( size_t z = 0; z < dimCopy.z; z ++ )
			{
				pDstLine[z] = pSrcLine[z] ;
			}
			pDstLine += m_dimSize.z ;
			pSrcLine += dimSrc.z ;
		}
	}

	m_xFilled += dimCopy.x ;
	assert( m_xFilled <= m_dimSize.x ) ;

	return	dimCopy.x ;
}

// 空き空間を切り落としてバッファサイズを変更する
//////////////////////////////////////////////////////////////////////////////
void NNStreamBuffer::Trim( void )
{
	if ( m_xFilled >= m_dimSize.x )
	{
		return ;
	}
	NNBuffer	bufTemp( *this ) ;
	NNBufDim	dimSize( m_xFilled, m_dimSize.y, m_dimSize.z ) ;
	Free() ;
	Create( dimSize, m_cudaFlags ) ;
	CopyFrom( bufTemp ) ;
	m_xFilled = dimSize.x ;
}


