
#ifndef	__NN_SHELL_IMAGE_FILE_H__
#define	__NN_SHELL_IMAGE_FILE_H__

#include "nn_mlp_shell.h"
#include <random>

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// 画像ファイル・コーデック
//////////////////////////////////////////////////////////////////////////////

class	NNImageCodec
{
public:
	// 利用するライブラリの初期化
	static void InitializeLib( void ) ;
	// 利用するライブラリの終了処理
	static void ReleaseLib( void ) ;

public:
	// ファイルを読み込んでバッファに変換する
	static std::shared_ptr<NNBuffer>
		LoadFromFile( const std::filesystem::path& path, size_t nReqDepth = 0 ) ;
	// バッファを形式変換してファイルに書き込む
	static bool SaveToFile
		( const std::filesystem::path& path, const NNBuffer& bufOutput ) ;
} ;



//////////////////////////////////////////////////////////////////////////////
// 画像ファイル入出力
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShellImageIterator	: public NNMLPShellFileIterator
{
protected:
	size_t	m_nReqDepth ;

public:
	// 構築関数
	NNMLPShellImageIterator
		( const char * pszSourceDir,
			const char * pszPairDir,
			bool flagOutputPair, size_t nReqDepth = 0 ) ;
	// 消滅関数
	~NNMLPShellImageIterator( void ) ;

public:
	// ファイルを読み込んでバッファに変換する
	virtual std::shared_ptr<NNBuffer>
				LoadFromFile( const std::filesystem::path& path ) ;
	// バッファを形式変換してファイルに書き込む
	virtual bool SaveToFile
		( const std::filesystem::path& path, const NNBuffer& bufOutput ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// 画像ファイル固定サイズ切り出し（学習専用）
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShellImageCropper	: public NNMLPShellImageIterator
{
protected:
	std::random_device	m_random ;
	std::mt19937		m_engine ;
	NNBufDim			m_dimCrop ;
	int					m_xMarginLeft ;
	int					m_xMarginRight ;
	int					m_yMarginTop ;
	int					m_yMarginBottom ;

public:
	// 構築関数
	NNMLPShellImageCropper
		( const char * pszSourceDir,
			const char * pszPairDir, const NNBufDim& dimCrop,
			int xMarginLeft = 0, int xMarginRight = 0,
			int yMarginTop = 0, int yMarginBottom = 0 ) ;
	// 読み込んだバッファを処理して次のデータとして設定する
	virtual bool SetNextDataOnLoaded
				( std::shared_ptr<NNBuffer> pSource,
					std::shared_ptr<NNBuffer> pTeaching ) ;
	// ソースデータの切り出し
	virtual std::shared_ptr<NNBuffer> CropSourceData
		( std::shared_ptr<NNBuffer> pSource, const NNBufDim& dimCropOffset ) ;
	// ソースの切り出し位置に対応する教師データを切り出す
	virtual std::shared_ptr<NNBuffer> CropTeachingData
		( std::shared_ptr<NNBuffer> pTeaching, const NNBufDim& dimCropOffset ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// 画像分類器ファイル入力
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShellImageClassifier	: public NNMLPShellFileClassIterator
{
protected:
	size_t	m_nReqDepth ;

public:
	// 構築関数
	NNMLPShellImageClassifier
		( const char * pszSourceDir,
			bool flagPrediction,
			const char * pszClassDir = nullptr, size_t nReqDepth = 0 ) ;
	// 消滅関数
	~NNMLPShellImageClassifier( void ) ;

public:
	// ファイルを読み込んでバッファに変換する
	virtual std::shared_ptr<NNBuffer>
				LoadFromFile( const std::filesystem::path& path ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// 画像生成器ファイル出力
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShellImageGenerativeIterator	: public NNMLPShellGenerativeIterator
{
public:
	// 構築関数
	NNMLPShellImageGenerativeIterator
		( const char * pszSourceDir,
			const char * pszOutputDir, const char * pszClassDir ) ;
	// ファイルを読み込んでバッファに変換する
	virtual std::shared_ptr<NNBuffer>
				LoadFromFile( const std::filesystem::path& path ) ;
	// バッファを形式変換してファイルに書き込む
	virtual bool SaveToFile
		( const std::filesystem::path& path, const NNBuffer& bufOutput ) ;
	// 出力ファイル名を生成する
	virtual std::filesystem::path
		MakeOutputPathOf( const std::filesystem::path& pathSource ) ;

} ;


}

#endif
