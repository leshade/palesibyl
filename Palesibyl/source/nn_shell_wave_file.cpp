
#include "nn_shell_wave_file.h"

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// RIFF (Resource Interchange File Format) ファイル
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
RIFFFile::RIFFFile( void )
	: m_mode( modeNotOpened )
{
}

// 消滅関数
//////////////////////////////////////////////////////////////////////////////
RIFFFile::~RIFFFile( void )
{
	if ( m_mode != modeNotOpened )
	{
		Close() ;
	}
}

// ファイル開く（読み込み用）
//////////////////////////////////////////////////////////////////////////////
bool RIFFFile::Open( const char * pszFilePath )
{
	assert( m_mode == modeNotOpened ) ;
	try
	{
		m_ifs = std::make_unique<std::ifstream>
					( pszFilePath, std::ios_base::in | std::ios_base::binary ) ;
		if ( !m_ifs->is_open() )
		{
			return	false ;
		}
	}
	catch ( const std::exception& e )
	{
		TRACE( "exception at RIFFFile::Open: %s\n", e.what() ) ;
		return	false ;
	}

	m_mode = modeRead ;
	if ( DescendChunk() != ChunkID('R','I','F','F') )
	{
		return	false ;
	}
	return	true ;
}

// ファイル開く（新規作成／書き込み用）
//////////////////////////////////////////////////////////////////////////////
bool RIFFFile::Create( const char * pszFilePath )
{
	assert( m_mode == modeNotOpened ) ;
	try
	{
		m_ofs = std::make_unique<std::ofstream>
					( pszFilePath, std::ios_base::out | std::ios_base::binary ) ;
		if ( !m_ofs->is_open() )
		{
			return	false ;
		}
	}
	catch ( const std::exception& e )
	{
		TRACE( "exception at RIFFFile::Create: %s\n", e.what() ) ;
		return	false ;
	}

	m_mode = modeWrite ;
	if ( DescendChunk( ChunkID('R','I','F','F') ) != ChunkID('R','I','F','F') )
	{
		return	false ;
	}
	return	true ;
}

// ファイルを閉じる
//////////////////////////////////////////////////////////////////////////////
bool RIFFFile::Close( void )
{
	bool	flagIsGood = true ;
	if ( m_mode == modeWrite )
	{
		while ( m_chunks.size() > 0 )
		{
			AscendChunk() ;
		}
		flagIsGood = m_ofs->good() ;
		m_ofs = nullptr ;
	}
	else if ( m_mode == modeRead )
	{
		m_chunks.clear() ;
		flagIsGood = m_ifs->good() ;
		m_ifs = nullptr ;
	}
	m_mode = modeNotOpened ;
	return	flagIsGood ;
}

// チャンクへ入る
//////////////////////////////////////////////////////////////////////////////
uint32_t RIFFFile::DescendChunk( uint32_t idChunk )
{
	ChunkStack	chunk ;
	if ( m_mode == modeWrite )
	{
		assert( idChunk != 0 ) ;
		chunk.sposHeader = m_ofs->tellp() ;
		chunk.chdr.id = idChunk ;
		chunk.chdr.bytes = 0 ;

		Write( &(chunk.chdr), sizeof(chunk.chdr) ) ;
	}
	else if ( m_mode == modeRead )
	{
		for ( ; ; )
		{
			chunk.sposHeader = m_ifs->tellg() ;

			Read( &chunk.chdr, sizeof(chunk.chdr) ) ;

			if ( !m_ifs->good() )
			{
				return	0 ;
			}
			if ( (idChunk == 0) || (idChunk == chunk.chdr.id) )
			{
				break ;
			}
			m_ifs->seekg( (std::streampos) chunk.chdr.bytes, std::ios_base::cur ) ;
			if ( m_ifs->eof() )
			{
				return	0 ;
			}
		}
	}
	else
	{
		assert( m_mode != modeNotOpened ) ;
		return	0 ;
	}

	chunk.soffInBytes = chunk.sposHeader ;
	chunk.soffInBytes += sizeof(ChunkHeader) ;

	m_chunks.push_back( chunk ) ;

	return	chunk.chdr.id ;
}

// チャンクから出る
//////////////////////////////////////////////////////////////////////////////
bool RIFFFile::AscendChunk( void )
{
	assert( m_chunks.size() > 0 ) ;
	if ( m_chunks.size() == 0 )
	{
		return	false ;
	}
	bool	flagIsGood = true ;
	if ( m_mode == modeWrite )
	{
		std::streampos	fposEnd = m_ofs->tellp() ;
		std::streamoff	offEnd = fposEnd ;
		//
		ChunkStack	chunk = m_chunks.back() ;
		chunk.chdr.bytes = (uint32_t) (offEnd - chunk.soffInBytes) ;
		//
		m_ofs->seekp( chunk.sposHeader ) ;
		Write( &(chunk.chdr), sizeof(chunk.chdr) ) ;
		m_ofs->seekp( fposEnd ) ;
		flagIsGood = m_ofs->good() ;
	}
	else if ( m_mode == modeRead )
	{
		ChunkStack	chunk = m_chunks.back() ;
		m_ifs->seekg( chunk.sposHeader ) ;
		m_ifs->seekg
			( (std::streampos) (chunk.chdr.bytes
								+ sizeof(ChunkHeader)), std::ios_base::cur ) ;
	}
	m_chunks.pop_back() ;
	return	flagIsGood ;
}

// 現在のチャンク長（バイト単位）取得
//////////////////////////////////////////////////////////////////////////////
uint32_t RIFFFile::GetChunkLength( void ) const
{
	assert( m_chunks.size() > 0 ) ;
	if ( m_chunks.size() == 0 )
	{
		return	0 ;
	}
	return	m_chunks.back().chdr.bytes ;
}

// ファイルへ書き込む
//////////////////////////////////////////////////////////////////////////////
bool RIFFFile::Write( const void * buf, size_t bytes )
{
	assert( m_mode == modeWrite ) ;
	assert( m_ofs != nullptr ) ;
	m_ofs->write( (const char*) buf, bytes ) ;
	return	m_ofs->good() ;
}

// ファイルから読み込む
//////////////////////////////////////////////////////////////////////////////
bool RIFFFile::Read( void * buf, size_t bytes )
{
	assert( m_mode == modeRead ) ;
	assert( m_ifs != nullptr ) ;
	m_ifs->read( (char*) buf, bytes ) ;
	return	m_ifs->good() ;
}



//////////////////////////////////////////////////////////////////////////////
// Windows Wave Form ファイル
//////////////////////////////////////////////////////////////////////////////

// Wave ファイル読み込み
//////////////////////////////////////////////////////////////////////////////
bool WaveFile::LoadFile( const char * pszFilePath, size_t nReqChannels )
{
	RIFFFile	riff ;
	if ( !riff.Open( pszFilePath ) )
	{
		return	false ;
	}
	uint32_t	hdrWave = 0 ;
	if ( !riff.Read( &hdrWave, sizeof(hdrWave) )
		|| (hdrWave != RIFFFile::ChunkID('W','A','V','E')) )
	{
		return	false ;
	}
	const uint32_t	id_fmt = RIFFFile::ChunkID('f','m','t',' ') ;
	const uint32_t	id_data = RIFFFile::ChunkID('d','a','t','a') ;

	// フォーマット読み込み
	if ( riff.DescendChunk( id_fmt ) != id_fmt )
	{
		return	false ;
	}
	riff.Read( &m_format, sizeof(m_format) ) ;
	riff.AscendChunk() ;

	if ( (m_format.wFormatTag != formatWavePCM)
		|| (m_format.wChannels == 0)
		|| ((m_format.wBitsPerSample != 16)
				&& (m_format.wBitsPerSample != 8)) )
	{
		// 非対応フォーマット
		return	false ;
	}

	// データ読み込み
	if ( riff.DescendChunk( id_data ) != id_data )
	{
		return	false ;
	}
	if ( m_format.wBitsPerSample == 16 )
	{
		std::vector<int16_t>	buf ;
		const size_t	nLength = riff.GetChunkLength() / sizeof(int16_t) ;
		buf.resize( nLength ) ;
		riff.Read( buf.data(), nLength * sizeof(int16_t) ) ;
		//
		m_pWave = MakeFromPCM16bits
					( buf.data(), m_format.wChannels,
						nLength / m_format.wChannels, nReqChannels ) ;
	}
	else
	{
		assert( m_format.wBitsPerSample == 8 ) ;
		std::vector<uint8_t>	buf ;
		const size_t	nLength = riff.GetChunkLength() / sizeof(uint8_t) ;
		buf.resize( nLength ) ;
		riff.Read( buf.data(), nLength * sizeof(uint8_t) ) ;
		//
		m_pWave = MakeFromPCM8bits
					( buf.data(), m_format.wChannels,
						nLength / m_format.wChannels, nReqChannels ) ;
	}
	riff.AscendChunk() ;

	return	true ;
}

// Wave ファイル書き出し
//////////////////////////////////////////////////////////////////////////////
bool WaveFile::SaveFile( const char * pszFilePath )
{
	assert( m_pWave != nullptr ) ;
	if ( m_pWave == nullptr )
	{
		return	false ;
	}
	if ( (m_format.wFormatTag != formatWavePCM)
		|| (m_format.wChannels == 0)
		|| ((m_format.wBitsPerSample != 16)
				&& (m_format.wBitsPerSample != 8)) )
	{
		// 非対応フォーマット
		return	false ;
	}

	RIFFFile	riff ;
	if ( !riff.Create( pszFilePath ) )
	{
		return	false ;
	}
	uint32_t	hdrWave = RIFFFile::ChunkID('W','A','V','E') ;
	riff.Write( &hdrWave, sizeof(hdrWave) ) ;

	riff.DescendChunk( RIFFFile::ChunkID('f','m','t',' ') ) ;
	riff.Write( &m_format, sizeof(m_format) ) ;
	riff.AscendChunk() ;

	riff.DescendChunk( RIFFFile::ChunkID('d','a','t','a') ) ;
	if ( m_format.wBitsPerSample == 16 )
	{
		std::vector<int16_t>	bufWave ;
		MakePCM16bitsFrom( bufWave, m_pWave ) ;
		riff.Write( bufWave.data(), bufWave.size() * sizeof(int16_t) ) ;
	}
	else
	{
		assert( m_format.wBitsPerSample == 8 ) ;
		std::vector<uint8_t>	bufWave ;
		MakePCM8bitsFrom( bufWave, m_pWave ) ;
		riff.Write( bufWave.data(), bufWave.size() * sizeof(uint8_t) ) ;
	}
	riff.AscendChunk() ;

	return	riff.Close() ;
}

// フォーマット取得
//////////////////////////////////////////////////////////////////////////////
const WaveFile::WAVEFORMAT& WaveFile::GetFormat( void ) const
{
	return	m_format ;
}

// フォーマット設定
//////////////////////////////////////////////////////////////////////////////
void WaveFile::SetFormat( const WaveFile::WAVEFORMAT& format )
{
	m_format = format ;
}

void WaveFile::SetFormat( size_t nFrequency, size_t nChannels, size_t bitsPerSample )
{
	m_format.wFormatTag = formatWavePCM ;
	m_format.wChannels = (uint16_t) nChannels ;
	m_format.nFrequency = (uint32_t) nFrequency ;
	m_format.wBitsPerSample = (uint16_t) bitsPerSample ;
	m_format.wBlockAlign = (m_format.wChannels * m_format.wBitsPerSample) / 8 ;
	m_format.nBytesPerSec = m_format.nFrequency * m_format.wBlockAlign ;
}

// 周波数変換
//////////////////////////////////////////////////////////////////////////////
void WaveFile::ResampleFrequency( uint32_t nFrequency )
{
	if ( (m_pWave == nullptr)
		|| (m_format.nFrequency == nFrequency) )
	{
		return ;
	}
	const NNBufDim	dimSrcWave = m_pWave->GetSize() ;
	const NNBufDim	dimDstWave( dimSrcWave.x * nFrequency / m_format.nFrequency,
								dimSrcWave.y, dimSrcWave.z ) ;
	std::shared_ptr<NNBuffer>	pDstBuf = std::make_shared<NNBuffer>() ;
	pDstBuf->Create( dimDstWave ) ;

	for ( size_t y = 0; y < dimDstWave.y; y ++ )
	{
		for ( size_t x = 0; x < dimDstWave.x; x ++ )
		{
			size_t	xSrc = x * m_format.nFrequency / nFrequency ;
			float	dec = (float) (x * m_format.nFrequency % nFrequency)
													/ (float) nFrequency ;
			assert( xSrc < dimSrcWave.x ) ;
			const float *	pSrcWave = m_pWave->GetConstBufferAt( xSrc, y ) ;
			float *			pDstWave = pDstBuf->GetBufferAt( x, y ) ;
			if ( xSrc + 1 < dimSrcWave.x )
			{
				for ( size_t z = 0; z < dimDstWave.z; z ++ )
				{
					pDstWave[z] = pSrcWave[z] * (1.0f - dec)
									+ pSrcWave[dimDstWave.z + z] * dec ;
				}
			}
			else
			{
				for ( size_t z = 0; z < dimDstWave.z; z ++ )
				{
					pDstWave[z] = pSrcWave[z] ;
				}
			}
		}
	}

	m_pWave = pDstBuf ;
	m_format.nFrequency = nFrequency ;
}

// データ取得
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> WaveFile::GetBuffer( void ) const
{
	return	m_pWave ;
}

// データ設定
//////////////////////////////////////////////////////////////////////////////
void WaveFile::SetBuffer( std::shared_ptr<NNBuffer> pWave )
{
	m_pWave = pWave ;
}

// 16bit PCM -> Buffer
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> WaveFile::MakeFromPCM16bits
	( const int16_t * pWave, size_t nChannels, size_t nLength, size_t nReqChannels )
{
	std::shared_ptr<NNBuffer>	pBuffer = std::make_shared<NNBuffer>() ;
	if ( nReqChannels == 0 )
	{
		nReqChannels = nChannels ;
	}
	pBuffer->Create( nLength, 1, nReqChannels ) ;
	//
	float *			pBufNext = pBuffer->GetBuffer() ;
	const int16_t * pWaveNext = pWave ;
	for ( size_t i = 0; i < nLength; i ++ )
	{
		for ( size_t ch = 0; (ch < nReqChannels) && (ch < nChannels); ch ++ )
		{
			pBufNext[ch] = (float) pWaveNext[ch] * (1.0f / 32768.0f) ;
		}
		pBufNext += nReqChannels ;
		pWaveNext += nChannels ;
	}
	return	pBuffer ;
}

// 8bit PCM -> Buffer
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> WaveFile::MakeFromPCM8bits
	( const uint8_t * pWave, size_t nChannels, size_t nLength, size_t nReqChannels )
{
	std::shared_ptr<NNBuffer>	pBuffer = std::make_shared<NNBuffer>() ;
	if ( nReqChannels == 0 )
	{
		nReqChannels = nChannels ;
	}
	pBuffer->Create( nLength, 1, nReqChannels ) ;
	//
	float *			pBufNext = pBuffer->GetBuffer() ;
	const uint8_t * pWaveNext = pWave ;
	for ( size_t i = 0; i < nLength; i ++ )
	{
		for ( size_t ch = 0; (ch < nReqChannels) && (ch < nChannels); ch ++ )
		{
			pBufNext[ch] = ((float)pWaveNext[ch] - 128.0f) * (1.0f / 128.0f) ;
		}
		pBufNext += nReqChannels ;
		pWaveNext += nChannels ;
	}
	return	pBuffer ;
}

// 16bit PCM <- Buffer
//////////////////////////////////////////////////////////////////////////////
void WaveFile::MakePCM16bitsFrom
	( std::vector<int16_t>& bufWave, std::shared_ptr<NNBuffer> pBuffer )
{
	NNBufDim	dimBuf = pBuffer->GetSize() ;
	bufWave.resize( dimBuf.n * dimBuf.z ) ;

	const float *	pBuf = pBuffer->GetConstBuffer() ;
	int16_t *		pWave = bufWave.data() ;
	const size_t	nCount = dimBuf.n * dimBuf.z ;
	for ( size_t i = 0; i < nCount; i ++ )
	{
		pWave[i] = min( max( (int16_t) (pBuf[i] * 32768.0f), -0x7FFF ), 0x7FFF ) ;
	}
}

// 8bit PCM <- Buffer
//////////////////////////////////////////////////////////////////////////////
void WaveFile::MakePCM8bitsFrom
	( std::vector<uint8_t>& bufWave, std::shared_ptr<NNBuffer> pBuffer )
{
	NNBufDim	dimBuf = pBuffer->GetSize() ;
	bufWave.resize( dimBuf.n * dimBuf.z ) ;

	const float *	pBuf = pBuffer->GetConstBuffer() ;
	uint8_t *		pWave = bufWave.data() ;
	const size_t	nCount = dimBuf.n * dimBuf.z ;
	for ( size_t i = 0; i < nCount; i ++ )
	{
		pWave[i] = min( max( (uint8_t) (pBuf[i] * 128.0f), -0x7F ), 0x7F ) ;
	}
}




//////////////////////////////////////////////////////////////////////////////
// WAVE ファイル入出力
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellWaveIterator::NNMLPShellWaveIterator
	( const char * pszSourceDir,
		const char * pszPairDir,
		bool flagOutputPair, size_t nReqChannels,
		size_t nPackSamples, size_t nUnpackSamples, size_t nReqFrequency )
	: NNMLPShellFileIterator( pszSourceDir, pszPairDir, flagOutputPair ),
		m_nReqChannels( nReqChannels ),
		m_nPackSamples( nPackSamples ),
		m_nUnpackSamples( nUnpackSamples ),
		m_nReqFrequency( nReqFrequency ),
		m_nLoadedFrequency( 44100 )
{
}

// 消滅関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellWaveIterator::~NNMLPShellWaveIterator( void )
{
}

// ファイルを読み込んでバッファに変換する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer>
	NNMLPShellWaveIterator::LoadFromFile( const std::filesystem::path& path )
{
	WaveFile	wavfile ;
	if ( !wavfile.LoadFile( path.string().c_str(), m_nReqChannels ) )
	{
		return	nullptr ;
	}
	m_nLoadedFrequency = wavfile.GetFormat().nFrequency ;
	//
	if ( (m_nReqFrequency != 0) && (m_nLoadedFrequency != m_nReqFrequency) )
	{
		wavfile.ResampleFrequency( (uint32_t) m_nReqFrequency ) ;
		m_nLoadedFrequency = wavfile.GetFormat().nFrequency ;
	}
	//
	std::shared_ptr<NNBuffer>
				pWave = std::make_shared<NNBuffer>( *(wavfile.GetBuffer()) ) ;
	NNBufDim	dimWave = pWave->GetSize() ;
	dimWave.x /= m_nPackSamples ;
	dimWave.n = dimWave.x * dimWave.y ;
	dimWave.z *= m_nPackSamples ;
	pWave->TransformShape( dimWave ) ;
	return	pWave ;
}

// バッファを形式変換してファイルに書き込む
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellWaveIterator::SaveToFile
	( const std::filesystem::path& path, const NNBuffer& bufOutput )
{
	std::shared_ptr<NNBuffer>
				pWave = std::make_shared<NNBuffer>( bufOutput ) ;
	NNBufDim	dimWave = pWave->GetSize() ;
	assert( (dimWave.z % m_nUnpackSamples) == 0 ) ;
	dimWave.x *= m_nUnpackSamples ;
	dimWave.n = dimWave.x * dimWave.y ;
	dimWave.z /= m_nUnpackSamples ;
	pWave->TransformShape( dimWave ) ;

	WaveFile	wavfile ;
	wavfile.SetFormat( m_nLoadedFrequency, dimWave.z ) ;
	wavfile.SetBuffer( pWave ) ;
	return	wavfile.SaveFile( path.string().c_str() ) ;
}



//////////////////////////////////////////////////////////////////////////////
// WAVE ファイル固定サイズ切り出し（学習専用）
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellWaveCropper::NNMLPShellWaveCropper
	( const char * pszSourceDir,
		const char * pszPairDir, const NNBufDim& dimCrop,
		size_t nPackSamples, size_t nUnpackSamples )
	: NNMLPShellWaveIterator
			( pszSourceDir, pszPairDir, false,
				dimCrop.z, nPackSamples, nUnpackSamples ),
		m_engine( m_random() ),
		m_dimCrop( dimCrop )
{
}

// 読み込んだバッファを処理して次のデータとして設定する
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellWaveCropper::SetNextDataOnLoaded
			( std::shared_ptr<NNBuffer> pSource,
				std::shared_ptr<NNBuffer> pTeaching )
{
	if ( pTeaching == nullptr )
	{
		return	false ;
	}
	NNBufDim	dimSource = pSource->GetSize() ;
	NNBufDim	dimTeaching = pTeaching->GetSize() ;
	dimSource.x = (dimSource.x > dimTeaching.x) ? dimTeaching.x : dimSource.x ;
	//
	std::vector<size_t>	samplesIndecies ;
	samplesIndecies.resize( m_dimCrop.y ) ;
	for ( size_t i = 0; i < m_dimCrop.y; i ++ )
	{
		samplesIndecies[i] = (dimSource.x <= m_dimCrop.x) ? 0
							: (m_engine() % (dimSource.x - m_dimCrop.x)) ;
	}

	std::shared_ptr<NNBuffer>
			pCropSource = CropSourceData( pSource, samplesIndecies ) ;
	std::shared_ptr<NNBuffer>
			pCropTeaching = CropTeachingData( pTeaching, samplesIndecies ) ;
	//
	return	NNMLPShellFileIterator::SetNextDataOnLoaded( pCropSource, pCropTeaching ) ;
}

// ソースデータの切り出し
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellWaveCropper::CropSourceData
	( std::shared_ptr<NNBuffer> pSource,
		const std::vector<size_t>& samplesIndecies )
{
	return	CropWaveData( pSource, samplesIndecies ) ;
}

// ソースの切り出し位置に対応する教師データを切り出す
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellWaveCropper::CropTeachingData
	( std::shared_ptr<NNBuffer> pTeaching,
		const std::vector<size_t>& samplesIndecies )
{
	return	CropWaveData( pTeaching, samplesIndecies ) ;
}

// 単純に WAVE を切り出す
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellWaveCropper::CropWaveData
	( std::shared_ptr<NNBuffer> pWave,
		const std::vector<size_t>& samplesIndecies )
{
	std::shared_ptr<NNBuffer>	pCrop = std::make_shared<NNBuffer>() ;
	NNBufDim	dimCrop( m_dimCrop.x, m_dimCrop.y, m_dimCrop.z * m_nPackSamples ) ;
	pCrop->Create( dimCrop ) ;

	NNBufDim	dimWave = pWave->GetSize() ;
	for ( size_t i = 0; (i < dimCrop.y) && (i < samplesIndecies.size()); i ++ )
	{
		const size_t	xSamples = samplesIndecies[i] ;
		const size_t	zChannels = min( dimCrop.z, dimWave.z ) ;
		float *			pDstWave = pCrop->GetBufferAt( 0, i ) ;
		const float *	pSrcWave = pWave->GetBufferAt( xSamples, 0 ) ;
		for ( size_t j = 0; (j < dimCrop.x) && (xSamples + j < dimWave.x); j ++ )
		{
			for ( size_t ch = 0; ch < zChannels; ch ++ )
			{
				pDstWave[ch] = pSrcWave[ch] ;
			}
			pDstWave += dimCrop.z ;
			pSrcWave += dimWave.z ;
		}
	}
	pCrop->CheckOverun() ;
	return	pCrop ;
}

