
#ifndef	__NN_SHELL_WAVE_FILE_H__
#define	__NN_SHELL_WAVE_FILE_H__

#include <fstream>
#include <iostream>
#include <random>

#include "nn_mlp_shell.h"

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// RIFF (Resource Interchange File Format) ファイル
//////////////////////////////////////////////////////////////////////////////

class	RIFFFile
{
public:
	// チャンクID
	static inline uint32_t ChunkID
		( unsigned char a, unsigned char b, unsigned char c, unsigned char d )
	{
		return	(uint32_t) a | (((uint32_t) b) << 8)
				| (((uint32_t) c) << 16) | (((uint32_t) d) << 24) ;
	}

	// チャックヘッダ
	struct	ChunkHeader
	{
		uint32_t	id ;
		uint32_t	bytes ;
	} ;

	// 書き込み／読み取りモード
	enum	FileMode
	{
		modeNotOpened,
		modeRead,
		modeWrite,
	} ;

protected:
	// チャンク情報
	struct	ChunkStack
	{
		ChunkHeader		chdr ;
		std::streampos	sposHeader ;
		std::streamoff	soffInBytes ;
	} ;

	FileMode						m_mode ;
	std::unique_ptr<std::ofstream>	m_ofs ;
	std::unique_ptr<std::ifstream>	m_ifs ;
	std::vector<ChunkStack>			m_chunks ;

public:
	// 構築関数
	RIFFFile( void ) ;
	// 消滅関数
	~RIFFFile( void ) ;
	// ファイル開く（読み込み用）
	bool Open( const char * pszFilePath ) ;
	// ファイル開く（新規作成／書き込み用）
	bool Create( const char * pszFilePath ) ;
	// ファイルを閉じる
	bool Close( void ) ;
	// チャンクへ入る
	uint32_t DescendChunk( uint32_t idChunk = 0 ) ;
	// チャンクから出る
	bool AscendChunk( void ) ;
	// 現在のチャンク長（バイト単位）取得
	uint32_t GetChunkLength( void ) const ;
	// ファイルへ書き込む
	bool Write( const void * buf, size_t bytes ) ;
	// ファイルから読み込む
	bool Read( void * buf, size_t bytes ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// Windows Wave Form ファイル
//////////////////////////////////////////////////////////////////////////////

class	WaveFile
{
public:
	struct	WAVEFORMAT
	{
		uint16_t	wFormatTag ;
		uint16_t	wChannels ;
		uint32_t	nFrequency ;
		uint32_t	nBytesPerSec ;
		uint16_t	wBlockAlign ;
		uint16_t	wBitsPerSample ;
	} ;
	enum	WaveFormat
	{
		formatWavePCM	= 1,
	} ;

protected:
	WAVEFORMAT					m_format ;
	std::shared_ptr<NNBuffer>	m_pWave ;

public:
	// Wave ファイル読み込み
	bool LoadFile( const char * pszFilePath, size_t nReqChannels = 0 ) ;
	// Wave ファイル書き出し
	bool SaveFile( const char * pszFilePath ) ;
	// フォーマット取得
	const WAVEFORMAT& GetFormat( void ) const ;
	// フォーマット設定
	void SetFormat( const WAVEFORMAT& format ) ;
	void SetFormat( size_t nFrequency, size_t nChannels, size_t bitsPerSample = 16 ) ;
	// 周波数変換
	void ResampleFrequency( uint32_t nFrequency ) ;
	// データ取得
	std::shared_ptr<NNBuffer> GetBuffer( void ) const ;
	// データ設定
	void SetBuffer( std::shared_ptr<NNBuffer> pWave ) ;

public:
	// 16bit PCM -> Buffer
	static std::shared_ptr<NNBuffer> MakeFromPCM16bits
		( const int16_t * pWave, size_t nChannels,
				size_t nLength, size_t nReqChannels = 0 ) ;
	// 8bit PCM -> Buffer
	static std::shared_ptr<NNBuffer> MakeFromPCM8bits
		( const uint8_t * pWave, size_t nChannels,
				size_t nLength, size_t nReqChannels = 0 ) ;
	// 16bit PCM <- Buffer
	static void MakePCM16bitsFrom
		( std::vector<int16_t>& bufWave, std::shared_ptr<NNBuffer> pBuffer ) ;
	// 8bit PCM <- Buffer
	static void MakePCM8bitsFrom
		( std::vector<uint8_t>& bufWave, std::shared_ptr<NNBuffer> pBuffer ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// WAVE ファイル入出力
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShellWaveIterator	: public NNMLPShellFileIterator
{
protected:
	size_t	m_nReqChannels ;
	size_t	m_nPackSamples ;
	size_t	m_nUnpackSamples ;
	size_t	m_nReqFrequency ;
	size_t	m_nLoadedFrequency ;

public:
	// 構築関数
	NNMLPShellWaveIterator
		( const char * pszSourceDir,
			const char * pszPairDir,
			bool flagOutputPair, size_t nReqChannels = 0,
			size_t nPackSamples = 1, size_t nUnpackSamples = 1,
			size_t nReqFrequency = 0,
			bool flagRandValidation = false, double rateValidation = 0.25 ) ;
	// 消滅関数
	~NNMLPShellWaveIterator( void ) ;

public:
	// ファイルを読み込んでバッファに変換する
	virtual std::shared_ptr<NNBuffer>
				LoadFromFile( const std::filesystem::path& path ) ;
	// バッファを形式変換してファイルに書き込む
	virtual bool SaveToFile
		( const std::filesystem::path& path, const NNBuffer& bufOutput ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// WAVE ファイル固定サイズ切り出し（学習専用）
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShellWaveCropper	: public NNMLPShellWaveIterator
{
public:
	// 境界範囲外の切り出し
	enum	CropOutOfBounds
	{
		cropPadZero,	// 範囲外は 0.0 で埋める
		cropWrap,		// 先頭から繰り返し
		cropEdge,		// 端の値の延長
	} ;

protected:
	std::random_device	m_random ;
	std::mt19937		m_engine ;
	NNBufDim			m_dimCrop ;
	CropOutOfBounds		m_cobCrop ;

public:
	// 構築関数
	NNMLPShellWaveCropper
		( const char * pszSourceDir, const char * pszPairDir,
			const NNBufDim& dimCrop, CropOutOfBounds cob = cropPadZero,
			size_t nPackSamples = 1, size_t nUnpackSamples = 1,
			size_t nReqFrequency = 0,
			bool flagRandValidation = false, double rateValidation = 0.25 ) ;
	// 読み込んだバッファを処理して次のデータとして設定する
	virtual bool SetNextDataOnLoaded
				( std::shared_ptr<NNBuffer> pSource,
					std::shared_ptr<NNBuffer> pTeaching ) ;
	// ソースデータの切り出し
	virtual std::shared_ptr<NNBuffer> CropSourceData
		( std::shared_ptr<NNBuffer> pSource,
			const std::vector<size_t>& samplesIndecies ) ;
	// ソースの切り出し位置に対応する教師データを切り出す
	virtual std::shared_ptr<NNBuffer> CropTeachingData
		( std::shared_ptr<NNBuffer> pTeaching,
			const std::vector<size_t>& samplesIndecies ) ;
	// 単純に WAVE を切り出す
	std::shared_ptr<NNBuffer> CropWaveData
		( std::shared_ptr<NNBuffer> pWave,
			const std::vector<size_t>& samplesIndecies,
			CropOutOfBounds cob = cropPadZero ) ;

} ;



}

#endif
