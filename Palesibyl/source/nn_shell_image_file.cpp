
#include "nn_shell_image_file.h"

#include <locale>
#include <atomic>

#if	!defined(__USE_GDIPLUS__) && !defined(__USE_OPENCV__)
	#if	defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
		#define	__USE_GDIPLUS__
	#else
		#define	__USE_OPENCV__
	#endif
#endif

#ifdef	__USE_GDIPLUS__
	#include <gdiplus.h>
	#pragma comment(lib, "gdiplus.lib")
#else
	#include <opencv2/opencv.hpp>
#endif

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// 画像ファイル・コーデック
//////////////////////////////////////////////////////////////////////////////

#ifdef	__USE_GDIPLUS__
static std::atomic<int>				s_gdpiRefCount = 0 ;
static ULONG_PTR					s_gdipToken ;
static Gdiplus::GdiplusStartupInput	s_gdipStartupInput ;
#endif

// 利用するライブラリの初期化
//////////////////////////////////////////////////////////////////////////////
void NNImageCodec::InitializeLib( void )
{
#ifdef	__USE_GDIPLUS__
	if ( s_gdpiRefCount.fetch_add( 1 ) == 0 )
	{
		Gdiplus::GdiplusStartup( &s_gdipToken, &s_gdipStartupInput, 0 ) ;
	}
#endif
}

// 利用するライブラリの終了処理
//////////////////////////////////////////////////////////////////////////////
void NNImageCodec::ReleaseLib( void )
{
#ifdef	__USE_GDIPLUS__
	if ( s_gdpiRefCount.fetch_add( -1 ) == 1 )
	{
		Gdiplus::GdiplusShutdown( s_gdipToken ) ;
	}
#endif
}

// ファイルを読み込んでバッファに変換する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer>
		NNImageCodec::LoadFromFile
			( const std::filesystem::path& path, size_t nReqDepth )
{
#ifdef	__USE_GDIPLUS__
	Gdiplus::Bitmap *	pImage = Gdiplus::Bitmap::FromFile( path.wstring().c_str() ) ;
	if ( pImage == nullptr )
	{
		return	nullptr ;
	}
	const UINT	nWidth = pImage->GetWidth() ;
	const UINT	nHeight = pImage->GetHeight() ;
	const Gdiplus::PixelFormat
				pxfmt = pImage->GetPixelFormat() ;
	size_t		nChannels, nPixelStride ;
	switch ( pxfmt )
	{
	case	PixelFormat8bppIndexed:
		nChannels = 1 ;
		nPixelStride = 1 ;
		break ;
	case	PixelFormat24bppRGB:
		nChannels = 3 ;
		nPixelStride = 3 ;
		break ;
	case	PixelFormat32bppRGB:
		nChannels = 3 ;
		nPixelStride = 4 ;
		break ;
	case	PixelFormat32bppARGB:
		nChannels = 4 ;
		nPixelStride = 4 ;
		break ;
	default:
		return	nullptr ;
	}
	if ( nReqDepth == 0 )
	{
		nReqDepth = nChannels ;
	}

	Gdiplus::BitmapData	bmpData ;
	Gdiplus::Rect		rect( 0, 0, nWidth, nHeight ) ;
	pImage->LockBits( &rect, Gdiplus::ImageLockModeRead, pxfmt, &bmpData ) ;
	//
	std::shared_ptr<NNBuffer>	pData = std::make_shared<NNBuffer>() ;
	pData->Create( nWidth, nHeight, nReqDepth ) ;
	pData->CopyFromImage
		( (const uint8_t*) bmpData.Scan0,
			nWidth, nHeight,
			((nReqDepth > nChannels) ? nChannels : nReqDepth),
			nPixelStride, bmpData.Stride ) ;
	//
	pImage->UnlockBits( &bmpData ) ;
	delete	pImage ;

	return	pData ;

#elif defined(__USE_OPENCV__)
	cv::Mat	image = cv::imread( path.string().c_str() ) ;
	if ( image.empty() )
	{
		return	nullptr ;
	}
	const size_t	nWidth = (size_t) image.size().width  ;
	const size_t	nHeight = (size_t) image.size().height ;
	const size_t	nChannels = (size_t) image.channels() ;
	const size_t	nPixelStride = (size_t) image.elemSize1() * nChannels ;
	const int		nLineStride = (int) image.step ;
	switch ( image.type() )
	{
	case	CV_8UC1:
	case	CV_8UC3:
	case	CV_8UC4:
		break ;
	default:
		return	nullptr ;
	}
	if ( nReqDepth == 0 )
	{
		nReqDepth = (size_t) nChannels ;
	}
	std::shared_ptr<NNBuffer>	pData = std::make_shared<NNBuffer>() ;
	pData->Create( nWidth, nHeight, nReqDepth ) ;
	pData->CopyFromImage
		( (const uint8_t*) image.data,
			nWidth, nHeight,
			((nReqDepth > nChannels) ? nChannels : nReqDepth),
			nPixelStride, nLineStride ) ;

	return	pData ;

#else
	#error no image codec
	return	nullptr ;

#endif
}

// バッファを形式変換してファイルに書き込む
//////////////////////////////////////////////////////////////////////////////
bool NNImageCodec::SaveToFile
	( const std::filesystem::path& path, const NNBuffer& bufOutput )
{
#ifdef	__USE_GDIPLUS__
	// 大文字小文字を区別しない比較
	auto	EqualCharNoCase = []( const wchar_t& c0, const wchar_t& c1 )
	{
		std::locale	locale ;
		return std::tolower(c0, locale) == std::tolower(c1, locale) ;
	} ;
	auto	EqualStrNoCase = [&]( const std::wstring& str0, const std::wstring& str1 )
	{
		return	(str0.size() == str1.size())
			&& std::equal( str0.cbegin(), str0.cend(), str1.cbegin(), EqualCharNoCase ) ;
	} ;

	// GDI+ エンコーダーを列挙
	UINT	nEncoderCount, nEncoderSize ;
	if ( Gdiplus::GetImageEncodersSize
		( &nEncoderCount, &nEncoderSize ) != Gdiplus::Ok )
	{
		return	false ;
	}
	std::vector<uint8_t>	buffer ;
	buffer.resize( nEncoderSize ) ;
	Gdiplus::ImageCodecInfo *	pCodecInfo =
		reinterpret_cast<Gdiplus::ImageCodecInfo*>( buffer.data() ) ;
	//
	if ( Gdiplus::GetImageEncoders
		( nEncoderCount, nEncoderSize, pCodecInfo ) != Gdiplus::Ok )
	{
		return	false ;
	}

	// エンコーダー CLSID 取得
	CLSID *			clsidEncoder = nullptr ;
	std::wstring	wstrPathExt = path.extension().wstring() ;
	for ( UINT i = 0; i < nEncoderCount; i ++ )
	{
		const WCHAR *	pwchFileExt = pCodecInfo[i].FilenameExtension ;
		size_t			iNextExt = 0 ;
		while ( pwchFileExt[iNextExt] != 0 )
		{
			size_t	iEndOfExt = iNextExt ;
			while ( (pwchFileExt[iEndOfExt] != 0)
				&& (pwchFileExt[iEndOfExt] != L';') )
			{
				iEndOfExt ++ ;
			}
			std::filesystem::path	pathFilter = std::wstring( pwchFileExt + iNextExt, iEndOfExt - iNextExt ) ;
			std::wstring			strFilterExt = pathFilter.extension().wstring() ;
			if ( EqualStrNoCase( wstrPathExt, strFilterExt ) )
			{
				clsidEncoder = &(pCodecInfo[i].Clsid) ;
				break ;
			}
			if ( pwchFileExt[iEndOfExt] == L';' )
			{
				iEndOfExt ++ ;
			}
			iNextExt = iEndOfExt ;
		}
	}
	if ( clsidEncoder == nullptr )
	{
		return	false ;
	}

	// ピクセルフォーマット
	const NNBufDim			dimBuf = bufOutput.GetSize() ;
	Gdiplus::PixelFormat	pxfmt ;
	size_t					nChannels = dimBuf.z ;
	switch ( nChannels )
	{
	case	0:
		return	false ;

	case	1:
	case	2:
		pxfmt = PixelFormat8bppIndexed ;
		nChannels = 1 ;
		break ;

	case	3:
		pxfmt = PixelFormat24bppRGB ;
		break ;

	case	4:
	default:
		pxfmt = PixelFormat32bppARGB ;
		nChannels = 4 ;
		break ;
	}

	// GDI+ イメージに変換
	Gdiplus::Bitmap	image( (INT) dimBuf.x, (INT) dimBuf.y, pxfmt ) ;

	if ( pxfmt == PixelFormat8bppIndexed )
	{
		std::vector<uint8_t>	bufPalette ;
		bufPalette.resize( sizeof(Gdiplus::ColorPalette)
							+ sizeof(Gdiplus::ARGB) * 0x100 ) ;
		//
		Gdiplus::ColorPalette *	pClrPalette =
			reinterpret_cast<Gdiplus::ColorPalette*>( bufPalette.data() ) ;
		//
		pClrPalette->Flags = Gdiplus::PaletteFlagsGrayScale ;
		pClrPalette->Count = 0x100 ;
		for ( size_t i = 0; i < 0x100; i ++ )
		{
			pClrPalette->Entries[i] =
				(Gdiplus::ARGB) ((0xFF << ALPHA_SHIFT) | (i << RED_SHIFT)
								| (i << GREEN_SHIFT) | (i << BLUE_SHIFT)) ;
		}
		image.SetPalette( pClrPalette ) ;
	}

	Gdiplus::BitmapData	bmpData ;
	Gdiplus::Rect		rect( 0, 0, (INT) dimBuf.x, (INT) dimBuf.y ) ;
	image.LockBits( &rect, Gdiplus::ImageLockModeWrite, pxfmt, &bmpData ) ;
	//
	bufOutput.CopyToImage
		( (uint8_t*) bmpData.Scan0,
			dimBuf.x, dimBuf.y, nChannels, nChannels, bmpData.Stride ) ;
	//
	image.UnlockBits( &bmpData ) ;

	// 保存
	if ( image.Save( path.wstring().c_str(), clsidEncoder ) != Gdiplus::Ok )
	{
		return	false ;
	}
	return	true ;

#elif defined(__USE_OPENCV__)
	// ピクセルフォーマット
	const NNBufDim	dimBuf = bufOutput.GetSize() ;
	size_t			nChannels = dimBuf.z ;
	int				type ;
	switch ( nChannels )
	{
	case	0:
		return	false ;

	case	1:
	case	2:
		type = CV_8UC1 ;
		nChannels = 1 ;
		break ;

	case	3:
		type = CV_8UC3 ;
		break ;

	case	4:
	default:
		type = CV_8UC4 ;
		nChannels = 4 ;
		break ;
	}

	// cv::Mat に変換
	cv::Mat	image( (int) dimBuf.y, (int) dimBuf.x, type ) ;
	bufOutput.CopyToImage
		( (uint8_t*) image.data,
			dimBuf.x, dimBuf.y, nChannels,
			nChannels, (int) image.step ) ;

	// 保存
	return	cv::imwrite( path.string().c_str(), image ) ;

#else
	#error no image codec
	return	false ;
#endif
}



//////////////////////////////////////////////////////////////////////////////
// 画像ファイル入出力
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellImageIterator::NNMLPShellImageIterator
	( const char * pszSourceDir,
		const char * pszPairDir, bool flagOutputPair, size_t nReqDepth,
		bool flagRandValidation, double rateValidation )
	: NNMLPShellFileIterator
		( pszSourceDir, pszPairDir,
			flagOutputPair, flagRandValidation, rateValidation ),
		m_nReqDepth( nReqDepth )
{
	NNImageCodec::InitializeLib() ;
}

// 消滅関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellImageIterator::~NNMLPShellImageIterator( void )
{
	NNImageCodec::ReleaseLib() ;
}

// ファイルを読み込んでバッファに変換する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer>
	NNMLPShellImageIterator::LoadFromFile( const std::filesystem::path& path )
{
	return	NNImageCodec::LoadFromFile( path, m_nReqDepth ) ;
}

// バッファを形式変換してファイルに書き込む
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellImageIterator::SaveToFile
	( const std::filesystem::path& path, const NNBuffer& bufOutput )
{
	return	NNImageCodec::SaveToFile( path, bufOutput ) ;
}



//////////////////////////////////////////////////////////////////////////////
// 画像ファイル固定サイズ切り出し
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellImageCropper::NNMLPShellImageCropper
	( const char * pszSourceDir,
		const char * pszPairDir, const NNBufDim& dimCrop,
		int xMarginLeft, int xMarginRight,
			int yMarginTop, int yMarginBottom )
	: NNMLPShellImageIterator( pszSourceDir, pszPairDir, false ),
		m_engine( m_random() ),
		m_dimCrop( dimCrop ),
		m_xMarginLeft( xMarginLeft ),
		m_xMarginRight( xMarginRight ),
		m_yMarginTop( yMarginTop ),
		m_yMarginBottom( yMarginBottom )
{
	assert( m_xMarginLeft >= 0 ) ;
	assert( m_xMarginRight >= 0 ) ;
	assert( m_yMarginTop >= 0 ) ;
	assert( m_yMarginBottom >= 0 ) ;
}

// 読み込んだバッファを処理して次のデータとして設定する
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellImageCropper::SetNextDataOnLoaded
	( std::shared_ptr<NNBuffer> pSource, std::shared_ptr<NNBuffer> pTeaching )
{
	if ( pTeaching == nullptr )
	{
		return	false ;
	}
	NNBufDim	dimSource = pSource->GetSize() ;
	NNBufDim	dimTeaching = pTeaching->GetSize() ;
	dimSource.x = (dimSource.x > dimTeaching.x) ? dimTeaching.x : dimSource.x ;
	dimSource.y = (dimSource.y > dimTeaching.y) ? dimTeaching.y : dimSource.y ;
	//
	NNBufDim	dimCropOffset( 100, 100, 0 ) ;
	dimCropOffset.x = (dimSource.x <= m_dimCrop.x) ? 0
						: (m_engine() % (dimSource.x - m_dimCrop.x)) ;
	dimCropOffset.y = (dimSource.y <= m_dimCrop.y) ? 0
						: (m_engine() % (dimSource.y - m_dimCrop.y)) ;

	std::shared_ptr<NNBuffer>
			pCropSource = CropSourceData( pSource, dimCropOffset ) ;
	std::shared_ptr<NNBuffer>
			pCropTeaching = CropTeachingData( pTeaching, dimCropOffset ) ;
	//
	return	NNMLPShellFileIterator::SetNextDataOnLoaded( pCropSource, pCropTeaching ) ;
}

// ソースデータの切り出し
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer> NNMLPShellImageCropper::CropSourceData
	( std::shared_ptr<NNBuffer> pSource, const NNBufDim& dimCropOffset )
{
	std::shared_ptr<NNBuffer>	pCropSource = std::make_shared<NNBuffer>() ;
	pCropSource->Create( m_dimCrop ) ;
	pCropSource->CopyFrom( *pSource, dimCropOffset ) ;
	return	pCropSource ;
}

// ソースの切り出し位置に対応する教師データを切り出す
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer>
	NNMLPShellImageCropper::CropTeachingData
		( std::shared_ptr<NNBuffer> pTeaching, const NNBufDim& dimCropOffset )
{
	std::shared_ptr<NNBuffer>	pCropTeaching = std::make_shared<NNBuffer>() ;
	NNBufDim	dimCrop = m_dimCrop ;
	NNBufDim	dimOffset = dimCropOffset ;
	assert( dimCrop.x > (size_t) (m_xMarginLeft + m_xMarginRight) ) ;
	assert( dimCrop.y > (size_t) (m_yMarginTop + m_yMarginBottom) ) ;
	dimCrop.x -= m_xMarginLeft + m_xMarginRight ;
	dimCrop.y -= m_yMarginTop + m_yMarginBottom ;
	dimOffset.x += m_xMarginLeft ;
	dimOffset.y += m_yMarginTop ;
	pCropTeaching->Create( dimCrop ) ;
	pCropTeaching->CopyFrom( *pTeaching, dimOffset ) ;
	return	pCropTeaching ;
}



//////////////////////////////////////////////////////////////////////////////
// 画像分類器ファイル入力
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellImageClassifier::NNMLPShellImageClassifier
	( const char * pszSourceDir,
		bool flagPrediction, const char * pszClassDir, size_t nReqDepth,
		bool formatIndex, bool flagRandValidation, double rateValidation )
	: NNMLPShellFileClassIterator
		( pszSourceDir, flagPrediction, pszClassDir,
			formatIndex, flagRandValidation, rateValidation ),
		m_nReqDepth( nReqDepth )
{
	NNImageCodec::InitializeLib() ;
}

// 消滅関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellImageClassifier::~NNMLPShellImageClassifier( void )
{
	NNImageCodec::ReleaseLib() ;
}

// ファイルを読み込んでバッファに変換する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer>
	NNMLPShellImageClassifier::LoadFromFile( const std::filesystem::path& path )
{
	return	NNImageCodec::LoadFromFile( path, m_nReqDepth ) ;
}



//////////////////////////////////////////////////////////////////////////////
// 画像生成器ファイル出力
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMLPShellImageGenerativeIterator::NNMLPShellImageGenerativeIterator
	( const char * pszSourceDir,
		const char * pszOutputDir, const char * pszClassDir )
	: NNMLPShellGenerativeIterator( pszSourceDir, pszOutputDir, pszClassDir )
{
}

// ファイルを読み込んでバッファに変換する
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNBuffer>
	NNMLPShellImageGenerativeIterator::LoadFromFile( const std::filesystem::path& path )
{
	return	NNMLPShellFileClassifier::ParseFile( path ) ;
}

// バッファを形式変換してファイルに書き込む
//////////////////////////////////////////////////////////////////////////////
bool NNMLPShellImageGenerativeIterator::SaveToFile
	( const std::filesystem::path& path, const NNBuffer& bufOutput )
{
	return	NNImageCodec::SaveToFile( path, bufOutput ) ;
}

// 出力ファイル名を生成する
//////////////////////////////////////////////////////////////////////////////
std::filesystem::path
	NNMLPShellImageGenerativeIterator::MakeOutputPathOf
		( const std::filesystem::path& pathSource )
{
	std::filesystem::path	pathOutput = m_pathPairDir ;
	pathOutput /= pathSource.stem().string() + ".png" ;
	return	pathOutput ;
}


