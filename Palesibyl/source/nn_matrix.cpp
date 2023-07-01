
#include "nn_perceptron.h"
#include "nn_simd_util.h"

#include <random>

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// 行列
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMatrix::NNMatrix( void )
	: m_lines(0), m_columns(0)
{
}

NNMatrix::NNMatrix( size_t lines, size_t columns )
	: m_lines(0), m_columns(0)
{
	Create( lines, columns ) ;
}

NNMatrix::NNMatrix( const NNMatrix& matrix )
	: m_lines( matrix.m_lines ),
		m_columns( matrix.m_columns ),
		m_matrix( matrix.m_matrix )
{
}

// 行列設定
//////////////////////////////////////////////////////////////////////////////
void NNMatrix::Create( size_t lines, size_t columns )
{
	m_lines = lines ;
	m_columns = columns ;
	m_matrix.resize( lines * columns ) ;
}

// 単位行列 * s
//////////////////////////////////////////////////////////////////////////////
void NNMatrix::InitDiagonal( float s )
{
	const size_t	nLength = GetLength() ;
	float *			pArray = GetArray() ;
	for ( size_t i = 0; i < nLength; i ++ )
	{
		pArray[i] = 0.0f ;
	}
	const size_t	nSize = (m_lines < m_columns) ? m_lines : m_columns ;
	for ( size_t i = 0; i < nSize; i ++ )
	{
		At(i,i) = s ;
	}
}

// 一様乱数 [low,high)
//////////////////////////////////////////////////////////////////////////////
void NNMatrix::Randomize( float low, float high )
{
	std::random_device						random ;
	std::mt19937							engine( random() ) ;
	std::uniform_real_distribution<float>	dist(low, high) ;

	const size_t	nLength = GetLength() ;
	float *			pArray = GetArray() ;
	for ( size_t i = 0; i < nLength; i ++ )
	{
		pArray[i] = dist(engine) ;
	}
}

// 平均μ, 標準偏差σ (分散σ^2) に従う正規分布
//////////////////////////////////////////////////////////////////////////////
void NNMatrix::RandomizeNormalDist( float mu, float sig )
{
	std::random_device				random ;
	std::mt19937					engine( random() ) ;
	std::normal_distribution<float>	dist(mu, sig) ;

	const size_t	nLength = GetLength() ;
	float *			pArray = GetArray() ;
	for ( size_t i = 0; i < nLength; i ++ )
	{
		pArray[i] = dist(engine) ;
	}
}

// 代入
//////////////////////////////////////////////////////////////////////////////
const NNMatrix& NNMatrix::operator = ( const NNMatrix& src )
{
	m_lines = src.m_lines ;
	m_columns = src.m_columns ;
	m_matrix = src.m_matrix ;
	return	*this ;
}

// 乗算
//////////////////////////////////////////////////////////////////////////////
NNMatrix NNMatrix::operator * ( const NNMatrix& op2 ) const
{
	NNMatrix	mat ;
	mat.ProductOf( *this, op2 ) ;
	return	mat ;
}

NNMatrix NNMatrix::operator * ( float s ) const
{
	NNMatrix	mat = *this ;
	mat *= s ;
	return	mat ;
}

const NNMatrix& NNMatrix::ProductOf( const NNMatrix& op1, const NNMatrix& op2 )
{
	assert( op1.m_columns == op2.m_lines ) ;
	Create( op1.m_lines, op2.m_columns ) ;

	for ( size_t i = 0; i < m_lines; i ++ )
	{
		for ( size_t j = 0; j < m_columns; j ++ )
		{
			float	f = 0.0f ;
			for ( size_t k = 0; k < op1.m_columns; k ++ )
			{
				f += op1.GetAt(i,k) * op2.GetAt(k,j) ;
			}
			At(i,j) = f ;
		}
	}
	return	*this ;
}

const NNMatrix& NNMatrix::operator *= ( float s )
{
	const size_t	nLength = GetLength() ;
	float *			pArray = GetArray() ;
	for ( size_t i = 0; i < nLength; i ++ )
	{
		pArray[i] *= s ;
	}
	return	*this ;
}

// 除算
//////////////////////////////////////////////////////////////////////////////
NNMatrix NNMatrix::operator / ( float s ) const
{
	NNMatrix	mat = *this ;
	mat /= s ;
	return	mat ;
}

const NNMatrix& NNMatrix::operator /= ( float s )
{
	const size_t	nLength = GetLength() ;
	float *			pArray = GetArray() ;
	for ( size_t i = 0; i < nLength; i ++ )
	{
		pArray[i] /= s ;
	}
	return	*this ;
}

// ベクトル乗算
//////////////////////////////////////////////////////////////////////////////
void NNMatrix::ProductVector( float * pDst, const float * pSrc ) const
{
	const float *	pMatrix = m_matrix.data() ;
	const size_t	lines = m_lines ;
	const size_t	columns = m_columns ;

#ifdef	__NN_USE_SIMD__
	__m128	xmmZero = _mm_setzero_ps() ;

	const size_t	nLineBy4 = lines >> 2 ;
	const size_t	nLineAlign4 = nLineBy4 << 2 ;
	const size_t	nColBy4 = columns >> 2 ;
	const size_t	nColAlign4 = nColBy4 << 2 ;
	for ( size_t i4 = 0; i4 < nLineAlign4; i4 += 4 )
	{
		const float *	pMatLine0 = pMatrix + i4 * columns ;
		const float *	pMatLine1 = pMatLine0 + columns ;
		const float *	pMatLine2 = pMatLine1 + columns ;
		const float *	pMatLine3 = pMatLine2 + columns ;

		__m128	xmm0 = xmmZero ;
		__m128	xmm1 = xmmZero ;
		__m128	xmm2 = xmmZero ;
		__m128	xmm3 = xmmZero ;
		for ( size_t j4 = 0; j4 < nColAlign4; j4 += 4 )
		{
			__m128	xmmSrc = _mm_loadu_ps( pSrc + j4 ) ;
			xmm0 = _mm_add_ps( xmm0,
					_mm_mul_ps( xmmSrc, _mm_loadu_ps( pMatLine0 + j4 ) ) ) ;
			xmm1 = _mm_add_ps( xmm1,
					_mm_mul_ps( xmmSrc, _mm_loadu_ps( pMatLine1 + j4 ) ) ) ;
			xmm2 = _mm_add_ps( xmm2,
					_mm_mul_ps( xmmSrc, _mm_loadu_ps( pMatLine2 + j4 ) ) ) ;
			xmm3 = _mm_add_ps( xmm3,
					_mm_mul_ps( xmmSrc, _mm_loadu_ps( pMatLine3 + j4 ) ) ) ;
		}
		__m128	xmmf0 = _mm_add_ps( _mm_unpacklo_ps( xmm0, xmm1 ),
									_mm_unpackhi_ps( xmm0, xmm1 ) ) ;
		__m128	xmmf2 = _mm_add_ps( _mm_unpacklo_ps( xmm2, xmm3 ),
									_mm_unpackhi_ps( xmm2, xmm3 ) ) ;
		_mm_storeu_ps
			( (pDst + i4),
				_mm_add_ps( _mm_movelh_ps( xmmf0, xmmf2 ),
							_mm_movehl_ps( xmmf2, xmmf0 ) ) ) ;
		//
		if ( nColAlign4 < columns )
		{
			float	d0 = pDst[i4] ;
			float	d1 = pDst[i4 + 1] ;
			float	d2 = pDst[i4 + 2] ;
			float	d3 = pDst[i4 + 3] ;
			for ( size_t j = nColAlign4; j < columns; j ++ )
			{
				d0 += pSrc[j] * pMatLine0[j] ;
				d1 += pSrc[j] * pMatLine1[j] ;
				d2 += pSrc[j] * pMatLine2[j] ;
				d3 += pSrc[j] * pMatLine3[j] ;
			}
			pDst[i4]     = d0 ;
			pDst[i4 + 1] = d1 ;
			pDst[i4 + 2] = d2 ;
			pDst[i4 + 3] = d3 ;
		}
	}
	for ( size_t i = nLineAlign4; i < lines; i ++ )
	{
		const float *	pMatLine = pMatrix + i * columns ;

		__m128	xmmf = xmmZero ;
		for ( size_t j4 = 0; j4 < nColAlign4; j4 += 4 )
		{
			xmmf = _mm_add_ps( xmmf,
					_mm_mul_ps( _mm_loadu_ps( pSrc + j4 ),
								_mm_loadu_ps( pMatLine + j4 ) ) ) ;
		}
		xmmf = _mm_add_ps( _mm_movehl_ps( xmmf, xmmf ), xmmf ) ;
		_mm_store_ss( pDst + i, _mm_add_ss( _mm_shuffle_ps( xmmf, xmmf, 1 ), xmmf ) ) ;
		//
		if ( nColAlign4 < columns )
		{
			float	d = pDst[i] ;
			for ( size_t j = nColAlign4; j < columns; j ++ )
			{
				d += pSrc[j] * pMatLine[j] ;
			}
			pDst[i] = d ;
		}
	}
#else
	for ( size_t i = 0; i < lines; i ++ )
	{
		const float *	pMatLine = pMatrix + i * columns ;

		float	d = 0.0f ;
		for ( size_t j = 0; j < columns; j ++ )
		{
			d += pSrc[j] * pMatLine[j] ;
		}
		pDst[i] = d ;
	}
#endif
}

void NNMatrix::DepthwiseProductVector
	( float * pDst, const float * pSrc, size_t depthwise, size_t iBias ) const
{
	const float *	pMatrix = m_matrix.data() ;
	const size_t	lines = m_lines ;
	const size_t	columns = m_columns ;
	assert( iBias <= columns ) ;

	for ( size_t i = 0; i < lines; i ++ )
	{
		const float *	pMatLine = pMatrix + i * columns ;

		float	d = 0.0f ;
		for ( size_t j = (i % depthwise); j < iBias; j += depthwise )
		{
			d += pSrc[j] * pMatLine[j] ;
		}
		for ( size_t j = iBias; j < columns; j ++ )
		{
			d += pSrc[j] * pMatLine[j] ;
		}
		pDst[i] = d ;
	}
}

void NNMatrix::ProductVectorLines
	( float * pDst, size_t iDstBase, size_t nDstCount,
		const float * pSrc, size_t depthwise, size_t iBias ) const
{
	const float *	pMatrix = m_matrix.data() ;
	const size_t	lines = m_lines ;
	const size_t	columns = m_columns ;
	assert( iDstBase + nDstCount <= lines ) ;
	assert( iBias <= columns ) ;

	for ( size_t i = 0; i < nDstCount; i ++ )
	{
		const size_t	iLine = iDstBase + i ;
		const float *	pMatLine = pMatrix + iLine * columns ;

		float	d = 0.0f ;
		for ( size_t j = (iLine % depthwise); j < iBias; j += depthwise )
		{
			d += pSrc[j] * pMatLine[j] ;
		}
		for ( size_t j = iBias; j < columns; j ++ )
		{
			d += pSrc[j] * pMatLine[j] ;
		}
		pDst[i] = d ;
	}
}

// 加減算
//////////////////////////////////////////////////////////////////////////////
NNMatrix NNMatrix::operator + ( const NNMatrix& op2 ) const
{
	NNMatrix	mat = *this ;
	mat += op2 ;
	return	mat ;
}

NNMatrix NNMatrix::operator - ( const NNMatrix& op2 ) const
{
	NNMatrix	mat = *this ;
	mat -= op2 ;
	return	mat ;
}

NNMatrix NNMatrix::operator - ( void ) const
{
	NNMatrix	mat = *this ;
	for ( size_t i = 0; i < mat.GetLength(); i ++ )
	{
		mat.ArrayAt(i) = - mat.ArrayAt(i) ;
	}
	return	mat ;
}

const NNMatrix& NNMatrix::operator += ( const NNMatrix& op2 )
{
	assert( m_lines == op2.m_lines ) ;
	assert( m_columns == op2.m_columns ) ;
	for ( size_t i = 0; i < GetLength(); i ++ )
	{
		ArrayAt(i) += op2.GetArrayAt(i) ;
	}
	return	*this ;
}

const NNMatrix& NNMatrix::operator -= ( const NNMatrix& op2 )
{
	assert( m_lines == op2.m_lines ) ;
	assert( m_columns == op2.m_columns ) ;
	for ( size_t i = 0; i < GetLength(); i ++ )
	{
		ArrayAt(i) -= op2.GetArrayAt(i) ;
	}
	return	*this ;
}

// 転置
//////////////////////////////////////////////////////////////////////////////
NNMatrix NNMatrix::Transpose( void ) const
{
	NNMatrix	mat ;
	mat.TransposeOf( *this ) ;
	return	mat ;
}

const NNMatrix& NNMatrix::TransposeOf( const NNMatrix& src )
{
	Create( src.m_columns, src.m_lines ) ;
	for ( size_t i = 0; i < src.m_lines; i ++ )
	{
		for ( size_t j = 0; j < src.m_columns; j ++ )
		{
			At(j,i) = src.GetAt(i,j) ;
		}
	}
	return	*this ;
}

// ノルム計算
//////////////////////////////////////////////////////////////////////////////
float NNMatrix::FrobeniusNorm( void ) const
{
	const size_t	nLength = GetLength() ;
	const float *	pArray = GetConstArray() ;
	float			norm2 = 0.0f ;
	for ( size_t i = 0; i < nLength; i ++ )
	{
		norm2 += pArray[i] * pArray[i] ;
	}
	return	(float) sqrt( norm2 ) ;
}

// 行列サイズ
//////////////////////////////////////////////////////////////////////////////
size_t NNMatrix::GetLineCount( void ) const
{
	return	m_lines ;
}

size_t NNMatrix::GetColumnCount( void ) const
{
	return	m_columns ;
}

size_t NNMatrix::GetLength( void ) const
{
	return	m_matrix.size() ;
}

// 要素
//////////////////////////////////////////////////////////////////////////////
float& NNMatrix::At( size_t i, size_t j )
{
	assert( i < m_lines ) ;
	assert( j < m_columns ) ;
	return	m_matrix.at( i * m_columns + j ) ;
}

float NNMatrix::GetAt( size_t i, size_t j ) const
{
	assert( i < m_lines ) ;
	assert( j < m_columns ) ;
	return	m_matrix.at( i * m_columns + j ) ;
}

float& NNMatrix::ArrayAt( size_t i )
{
	return	m_matrix.at( i ) ;
}

float NNMatrix::GetArrayAt( size_t i ) const
{
	return	m_matrix.at( i ) ;
}

// バッファアクセス
//////////////////////////////////////////////////////////////////////////////
float * NNMatrix::GetArray( void )
{
	return	m_matrix.data() ;
}

float * NNMatrix::GetLineArray( size_t i )
{
	return	m_matrix.data() + i * m_columns ;
}

const float * NNMatrix::GetConstArray( void ) const
{
	return	m_matrix.data() ;
}

const float * NNMatrix::GetConstLineAt( size_t i ) const
{
	return	m_matrix.data() + i * m_columns ;
}



