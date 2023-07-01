
#ifndef	__NN_STREAM_BUFFER_H__
#define	__NN_STREAM_BUFFER_H__

#include "nn_buffer.h"

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// ストリーミング用バッファ
//////////////////////////////////////////////////////////////////////////////

class	NNStreamBuffer	: public NNBuffer
{
protected:
	size_t	m_xFilled ;

public:
	// 構築関数
	NNStreamBuffer( void ) ;
	// バッファ解放
	void Free( void ) ;
	// 出力蓄積数取得
	size_t GetCurrent( void ) const ;
	// 空き空間を埋める
	void FillEmpty( float fill = 0.0f ) ;
	// シフト（ｘ方向に左へシフトしデータを捨てる）
	size_t Shift( size_t xCount ) ;
	// ストリーミング（ｘ方向に右から左へ流れていく）
	size_t Stream( const NNBuffer& bufSrc, size_t xSrc, size_t xCount ) ;
	// 空き空間を切り落としてバッファサイズを変更する
	void Trim( void ) ;

} ;


}

#endif

