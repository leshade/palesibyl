

//////////////////////////////////////////////////////////////////////////////
// アップサンプラー画像ファイル固定サイズ切り出し（学習専用）
//////////////////////////////////////////////////////////////////////////////

class	UpsamplerImageCropper : public NNMLPShellImageCropper
{
protected:
	size_t	m_nUpScale ;

public:
	// 構築関数
	UpsamplerImageCropper
		( const char * pszSourceDir,
			const char * pszPairDir,
			const NNBufDim& dimCrop, size_t nUpScale ) ;
	// ソースの切り出し位置に対応する教師データを切り出す
	virtual std::shared_ptr<NNBuffer> CropTeachingData
		( std::shared_ptr<NNBuffer> pTeaching, const NNBufDim& dimCropOffset ) ;
} ;

