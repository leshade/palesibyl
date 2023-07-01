
#ifndef	__NN_MATRIX_H__
#define	__NN_MATRIX_H__

#include <vector>

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// 行列
//////////////////////////////////////////////////////////////////////////////

class	NNMatrix
{
protected:
	size_t				m_lines ;		// 行数
	size_t				m_columns ;		// 列数
	std::vector<float>	m_matrix ;

public:
	// 構築関数
	NNMatrix( void ) ;
	NNMatrix( size_t lines, size_t columns ) ;
	NNMatrix( const NNMatrix& matrix ) ;
	// 行列設定
	void Create( size_t lines, size_t columns ) ;
	// 単位行列 * s
	void InitDiagonal( float s = 1.0f ) ;
	// 一様乱数 [low,high)
	void Randomize( float low, float high ) ;
	// 平均μ, 標準偏差σ (分散σ^2) に従う正規分布
	void RandomizeNormalDist( float mu, float sig ) ;
	// 代入
	const NNMatrix& operator = ( const NNMatrix& src ) ;
	// 乗算
	NNMatrix operator * ( const NNMatrix& op2 ) const ;
	NNMatrix operator * ( float s ) const ;
	const NNMatrix& ProductOf( const NNMatrix& op1, const NNMatrix& op2 ) ;
	const NNMatrix& operator *= ( float s ) ;
	// 除算
	NNMatrix operator / ( float s ) const ;
	const NNMatrix& operator /= ( float s ) ;
	// ベクトル乗算
	void ProductVector( float * pDst, const float * pSrc ) const ;
	void DepthwiseProductVector
		( float * pDst, const float * pSrc, size_t depthwise, size_t iBias ) const ;
	void ProductVectorLines
		( float * pDst, size_t iDstBase, size_t nDstCount,
			const float * pSrc, size_t depthwise, size_t iBias ) const ;
	// 加減算
	NNMatrix operator + ( const NNMatrix& op2 ) const ;
	NNMatrix operator - ( const NNMatrix& op2 ) const ;
	NNMatrix operator - ( void ) const ;
	const NNMatrix& operator += ( const NNMatrix& op2 ) ;
	const NNMatrix& operator -= ( const NNMatrix& op2 ) ;
	// 転置
	NNMatrix Transpose( void ) const ;
	const NNMatrix& TransposeOf( const NNMatrix& src ) ;
	// ノルム計算
	float FrobeniusNorm( void ) const ;
	// 行列サイズ
	size_t GetLineCount( void ) const ;
	size_t GetColumnCount( void ) const ;
	size_t GetLength( void ) const ;
	// 要素
	float& At( size_t i, size_t j ) ;
	float GetAt( size_t i, size_t j ) const ;
	float& ArrayAt( size_t i ) ;
	float GetArrayAt( size_t i ) const ;
	// バッファアクセス
	float * GetArray( void ) ;
	float * GetLineArray( size_t i ) ;
	const float * GetConstArray( void ) const ;
	const float * GetConstLineAt( size_t i ) const ;

} ;


}

#endif

