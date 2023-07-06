
#ifndef	__NN_MLP_SHELL_H__
#define	__NN_MLP_SHELL_H__

#include "nn_multi_layer.h"
#include "nn_stream_buffer.h"

#include <chrono>
#include <filesystem>

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// マルチ・レイヤー・パーセプトロン処理ラッパ
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShell
{
public:
	// 訓練データー反復器
	class	Iterator
	{
	public:
		// 初めから
		virtual void ResetIterator( void ) = 0 ;
		// 次の入力データを取得する
		virtual std::shared_ptr<NNBuffer> NextSource( void ) = 0 ;
		// 次の検証データを取得する
		virtual std::shared_ptr<NNBuffer> NextValidation( void ) = 0 ;
		// 最後に取得した{入力|検証}データに対応する教師データを取得する
		virtual std::shared_ptr<NNBuffer> GetTeachingData( void ) = 0 ;
		// 最後に取得した入力データに対応する予測データを出力する
		virtual bool OutputPrediction( const NNBuffer& bufOutput ) = 0 ;
		// {NextSource|NextValidation} で取得した入力データのパスを取得する
		virtual std::string GetSourcePath( void ) = 0 ;
		// GetTeachingData で取得できる教師データのパスを取得する
		virtual std::string GetTeachingDataPath( void ) = 0 ;
		// OutputPrediction で出力するパスを取得する
		virtual std::string GetOutputPath( void ) = 0 ;
	} ;

	// ストリーミング進捗
	class	Streamer
	{
	public:
		// ストリーム進行度
		// ※psbOutput から入力データへ反映させたり、psbOutput を Shift してもよい
		// 　OutputPrediction には最終的に Trim された psbOutput が渡される
		virtual void OnProgress
			( NNMLPShell& shell, size_t current, size_t total,
				NNStreamBuffer* psbOutput, size_t xLastStream ) = 0 ;
	} ;

	// 分類器
	class	Classifier
	{
	public:
		enum	ClassIndex
		{
			classInvalid	= -1,
			classFalse		= 0,	// 偽判定用（分類名は空文字列）
			classFirstIndex	= 1,
		} ;
		// 分類名取得
		virtual const std::string& GetClassNameAt( size_t iClass ) const = 0 ;
		virtual size_t GetClassCount( void ) const = 0 ;
		// 分類インデックス
		virtual int GetClassIndexOf( const char * pszClassName ) const = 0 ;
	} ;

	// GAN 用訓練データ反復器
	class	GANIterator
	{
	public:
		// 分類器への偽物教師データを取得する（GAN用）
		virtual std::shared_ptr<NNBuffer> GetCounterfeitData( void ) = 0 ;
		// 分類器用の反復器取得
		virtual std::shared_ptr<Iterator> GetClassifierIterator( void ) = 0 ;
	} ;

	// 時間計測
	class	TimeMeasure
	{
	private:
		std::chrono::system_clock::time_point	m_tpStart ;
	public:
		TimeMeasure( void )
		{
			Start() ;
		}
		void Start( void )
		{
			m_tpStart = std::chrono::system_clock::now() ;
		}
		long int MeasureMilliSec( void ) const
		{
			return	(long int)
				std::chrono::duration_cast<std::chrono::milliseconds>
					( std::chrono::system_clock::now() - m_tpStart ).count() ;
		}
	} ;

	// 設定
	enum	BehaviorFlag
	{
		behaviorPrintLearningFile	= 0x0001,	// 学習中のファイル名を表示しない
		behaviorLineFeedByMiniBatch	= 0x0002,	// 学習中のミニバッチ毎に表示行を送る
		behaviorNoDropout			= 0x0004,	// 学習時のドロップアウトは行わない
	} ;
	struct	ShellConfig
	{
		uint32_t	flagsBehavior ;		// enum BehaviorFlag の組み合わせ
		size_t		nBatchThreads ;		// ミニバッチ並列処理スレッド数

		ShellConfig( void )
			: flagsBehavior(behaviorLineFeedByMiniBatch), nBatchThreads(1) { }
	} ;

	// 学習パラメータ
	enum	LearningFlag
	{
	} ;
	struct	LearningParameter
	{
		uint32_t	flagsLearning ;		// enum LearningFlag の組み合わせ
		size_t		nEpochCount ;		// エポック数
		size_t		nSubLoopCount ;		// ミニバッチ反復回数
		size_t		nMiniBatchCount ;	// ミニバッチ構成数
		float		deltaRate0 ;		// 学習速度係数（初期値）
		float		deltaRate1 ;		// 学習速度係数（最終値）

		LearningParameter( void )
			: flagsLearning(0), nEpochCount(100),
				nSubLoopCount(1), nMiniBatchCount(10),
				deltaRate0(0.1f), deltaRate1(0.01f) {}
	} ;

	// 進捗情報
	enum	LearningProgressFlag
	{
		learningGAN				= 0x00000001,		// GAN 生成器
		learningGANClassifier	= 0x00000002,		// GAN 分類器
	} ;
	struct	LearningProgressInfo
	{
		uint32_t			flags ;			// enum LearningProgressFlag の組み合わせ
		size_t				iGANLoop ;
		size_t				nGANCount ;
		size_t				iLoopEpoch ;
		size_t				nEpochCount ;
		size_t				iInBatch ;
		size_t				nCountInBatch ;
		size_t				iSubLoop ;
		size_t				nSubLoopCount ;
		size_t				iMiniBatch ;
		size_t				nMiniBatchCount ;
		NNBuffer *			pTraining ;
		size_t				iValidation ;
		size_t				nValidationCount ;
		NNBuffer *			pValidation ;
		double				lossLearn ;
		double				lossValid ;
		double				evalLearn ;
		double				evalValid ;
		long int			msecLearn ;
		float				deltaRate ;
		std::vector<float>	gradNorms ;
		size_t				nGradNorm ;

		LearningProgressInfo( void )
			: flags(0), iGANLoop(0), nGANCount(0),
				iLoopEpoch(0), nEpochCount(0), iInBatch(0), nCountInBatch(0),
				iSubLoop(0), nSubLoopCount(0),
				iMiniBatch(0), nMiniBatchCount(0), pTraining(nullptr),
				iValidation(0), nValidationCount(0), pValidation(nullptr),
				lossLearn(0.0), lossValid(0.0),
				evalLearn(0.0), evalValid(0.0),
				msecLearn(0), deltaRate(0.0f), nGradNorm(0) {}
	} ;

	// 学習コンテキスト
	struct	LearningContext
	{
		std::vector< std::shared_ptr<NNBuffer> >
									sources ;			// 入力データ
		std::vector< std::shared_ptr<NNBuffer> >
									teachers ;			// 教師データ

		size_t						nBufCount ;			// ミニバッチ並列スレッド数
		NNParallelLoop				ploop ;				// 並列処理用
		std::vector
			< std::shared_ptr
				<NNMultiLayerPerceptron::BufferArrays> >
									bufArraysArray ;	// スレッド毎のバッファ配列
		NNMultiLayerPerceptron::LossAndGradientArray
									lagArray ;			// 統合された更新用勾配と損失合計

		std::vector<double>			vEvalArray ;		// スレッド毎の評価値の合計
		std::vector<size_t>			vEvalSummed ;		// スレッド毎の評価値の合計回数

		std::vector<NNBufDim>		dimSourceArray ;	// 入力データサイズ配列
		uint32_t					flagsBuf ;			// バッファフラグ (enum NNMultiLayerPerceptron::PrepareBufferFlag)

		uint32_t					flagsLearning ;		// enum LearningFlag の組み合わせ
		double						lossLearning ;		// 損失値
		double						lossCurLoop ;
		double						lossLastLoop ;
		double						lossSummed ;
		size_t						nLossSummed ;
		double						lossValidation ;
		double						evalLearning ;		// 評価値（合計）
		double						evalSummed ;
		size_t						nEvalSummed ;
		float						deltaRate ;			// 学習速度係数
		float						deltaCurRate ;
		bool						flagEndOfIter ;		// 訓練データ終端フラグ
		bool						flagCanceled ;		// キャンセルフラグ
	} ;

protected:
	NNMultiLayerPerceptron					m_mlp ;
	ShellConfig								m_config ;
	NNMultiLayerPerceptron::BufferConfig	m_bufConfig ;

	std::mutex								m_mutex ;

	int										m_nLastProgress ;

public:
	// 関連初期化処理
	static void StaticInitialize( void ) ;
	// 関連初期化処理
	static void StaticRelase( void ) ;

public:
	// 構築関数
	NNMLPShell( void ) ;

	// モデル読み込み
	virtual bool LoadModel( const char * pszFilePath ) ;
	// モデル書き出し
	virtual bool SaveModel( const char * pszFilePath ) ;
	// モデル取得
	NNMultiLayerPerceptron& MLP( void ) ;
	const NNMultiLayerPerceptron& GetMLP( void ) const ;

	// コンフィグ
	const ShellConfig& GetShellConfig( void ) const ;
	void SetShellConfig( const ShellConfig& cfg ) ;

	const NNMultiLayerPerceptron::BufferConfig& GetMLPConfig( void ) const ;
	void SetMLPConfig( const NNMultiLayerPerceptron::BufferConfig& cfg ) ;

	// 学習
	virtual void DoLearning
		( Iterator& iter, const LearningParameter& param ) ;
	// 学習（GAN）
	virtual void DoLearningGAN
		( Iterator& iterGAN,
			NNMLPShell& mlpClassifier, Iterator& iterClassifier,
			size_t nGANLoop, const LearningParameter& param ) ;

	// 予測変換
	virtual void DoPrediction( Iterator& iter ) ;

	// 学習準備
	virtual void PrepareLearning
		( LearningContext& context, const LearningParameter& param, uint32_t flagsBuf ) ;
	// １エポック学習
	virtual void DoLearningEpoch
		( LearningContext& context, Iterator& iter,
			LearningProgressInfo& lpi,
			const LearningParameter& param,
			NNMultiLayerPerceptron * pForwardMLP = nullptr,
			LearningContext * pForwardContext = nullptr ) ;
	// エポック開始時
	virtual void OnBeginEpoch
		( LearningContext& context,
			LearningProgressInfo& lpi, Iterator& iter ) ;
	// エポック終了時
	virtual void OnEndEpoch
		( LearningContext& context, LearningProgressInfo& lpi ) ;
	// ミニバッチ訓練データ収集
	virtual bool LoadTrainingData
		( LearningContext& context, Iterator& iter, size_t nBatchCount ) ;
	// バッファ準備
	virtual void PrepareBuffer( LearningContext& context ) ;
	virtual void PrepareBuffer
		( NNMultiLayerPerceptron& mlp,
			LearningContext& context, const NNBufDim& dimSource ) ;
	virtual void PrepareForwardBuffer
		( NNMultiLayerPerceptron& mlpForward,
			LearningContext& lcForward, LearningContext& context ) ;
	// バッファのミニバッチ用準備処理（ドロップアウト用）
	virtual void PreapreForMiniBatch( LearningContext& context, bool flagLearning ) ;
	// ミニバッチループ開始時
	virtual void OnBeginMiniBatch( LearningContext& context, bool flagLearning ) ;
	// ミニバッチループ終了時
	virtual void OnEndMiniBatch( LearningContext& context ) ;
	// １回学習
	virtual void LearnOnce
		( LearningContext& context, LearningProgressInfo& lpi,
			NNMultiLayerPerceptron * pForwardMLP = nullptr,
			LearningContext * pForwardContext = nullptr ) ;
	// 損失値と勾配を合計し、平均損失を返す
	virtual double IntegrateLossAndGradient( LearningContext& context ) ;
	// 勾配反映
	virtual void GradientReflection
		( LearningContext& context, LearningProgressInfo& lpi ) ;
	// 検証
	virtual bool ValidateLearning
		( LearningContext& context,
			LearningProgressInfo& lpi, Iterator& iter ) ;

	// モデルとデータの形状の検証
	virtual NNMultiLayerPerceptron::VerifyError
		VerifyDataShape
			( const NNMultiLayerPerceptron::BufferArrays& bufArrays,
				const NNBufDim& dimTeaching, const NNBufDim& dimSource0 ) const ;
	virtual NNMultiLayerPerceptron::VerifyError
		VerifyDataShape
			( const NNMultiLayerPerceptron::BufferArrays& bufArrays,
				const NNBufDim& dimSource0 ) const ;

public:
	// 学習進捗表示
	enum	LearningEvent
	{
		learningStartMiniBatch,
		learningOneData,
		learningEndMiniBatch,
		learningEndSubLoop,
		learningValidation,
		learningEndEpoch,
	} ;
	virtual void OnLearningProgress
		( LearningEvent le, const LearningProgressInfo& lpi ) ;

	// ストリーム出力進捗表示
	virtual void OnStreamingProgress
		( Streamer * pStreamer,
			size_t current, size_t total,
			NNStreamBuffer * psbOutput, size_t xLastStream ) ;

	// メッセージ出力
	virtual void Print( const char * pszFormat, ... ) const ;
	// 処理中断
	virtual bool IsCancel( void ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// GAN 分類器用反復器
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShellGANClassifierIterator
			: public NNMLPShell::Iterator, public NNMLPShell::Classifier
{
protected:
	NNMLPShell::Iterator&						m_iterClassifier ;
	NNMLPShell::Classifier *					m_pClassifier ;

	NNMultiLayerPerceptron&						m_mlpGenerator ;
	const NNMultiLayerPerceptron::BufferConfig&	m_cfgGenerator ;
	NNMLPShell::LearningContext&				m_lcGenerator ;
	NNMLPShell::Iterator&						m_iterGAN ;
	NNMLPShell::GANIterator *					m_pGANIter ;

	size_t										m_iIterNext ;
	bool										m_flagDataGen ;
	std::shared_ptr<NNBuffer>					m_pTeachingGen ;

public:
	// 構築関数
	NNMLPShellGANClassifierIterator
		( NNMLPShell::Iterator& iterClassifier,
			NNMultiLayerPerceptron& mlpGenerator,
			const NNMultiLayerPerceptron::BufferConfig& cfgGenerator,
			NNMLPShell::LearningContext& lcGenerator,
			NNMLPShell::Iterator& iterGAN ) ;

public:	// NNMLPShell::Iterator
	// 初めから
	virtual void ResetIterator( void ) ;
	// 次の入力データを取得する
	virtual std::shared_ptr<NNBuffer> NextSource( void ) ;
	// 次の検証データを取得する
	virtual std::shared_ptr<NNBuffer> NextValidation( void ) ;
	// 最後に取得した{入力|検証}データに対応する教師データを取得する
	virtual std::shared_ptr<NNBuffer> GetTeachingData( void ) ;
	// 最後に取得した入力データに対応する予測データを出力する
	virtual bool OutputPrediction( const NNBuffer& bufOutput ) ;
	// {NextSource|NextValidation} で取得した入力データのパスを取得する
	virtual std::string GetSourcePath( void ) ;
	// GetTeachingData で取得できる教師データのパスを取得する
	virtual std::string GetTeachingDataPath( void ) ;
	// OutputPrediction で出力するパスを取得する
	virtual std::string GetOutputPath( void ) ;

public:
	// NNMLPShell::Classifier
	// 分類名取得
	virtual const std::string& GetClassNameAt( size_t iClass ) const ;
	virtual size_t GetClassCount( void ) const ;
	// 分類インデックス
	virtual int GetClassIndexOf( const char * pszClassName ) const ;
} ;



//////////////////////////////////////////////////////////////////////////////
// ファイル入出力反復器基底
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShellGenFileIterator	: public NNMLPShell::Iterator
{
protected:
	size_t						m_iValidation ;		// 検証用データ指標
	bool						m_flagShuffle ;		// ファイル順序シャッフル
	std::vector
		<std::filesystem::path>	m_files ;			// [0,m_iValidation) が学習用
													// [m_iValidation,n) が検証用

	size_t						m_iNextSource ;		// 次の入力元
	size_t						m_iNextValidation ;	// 次の検証用
	std::shared_ptr<NNBuffer>	m_pSource ;
	std::shared_ptr<NNBuffer>	m_pTeaching ;
	std::filesystem::path		m_pathSourceFile ;

public:
	// 構築関数
	NNMLPShellGenFileIterator( bool flagShuffle ) ;

	// ファイル列挙
	static void EnumerateFiles
		( const char * pszDir,
			std::function<void(const std::filesystem::path&)> func ) ;
	// ディレクトリ列挙
	static void EnumerateDirectories
		( const char * pszDir,
			std::function<void(const std::filesystem::path&)> func ) ;
	// 汎用シャッフル関数
	static void Shuffle
		( size_t iFirst, size_t nCount,
			std::function<void(size_t,size_t)> funcShuffle ) ;

public:
	// 初めから
	virtual void ResetIterator( void ) ;
	// 次の入力データを取得する
	virtual std::shared_ptr<NNBuffer> NextSource( void ) ;
	// 次の検証データを取得する
	virtual std::shared_ptr<NNBuffer> NextValidation( void ) ;
	// 最後に取得した{入力|検証}データに対応する教師データを取得する
	virtual std::shared_ptr<NNBuffer> GetTeachingData( void ) ;
	// {NextSource|NextValidation} で取得した入力データのパスを取得する
	virtual std::string GetSourcePath( void ) ;

public:
	// 入力ファイルをシャッフルする（指定指標のファイルを入れ替える）
	virtual void ShuffleFile( size_t iFile1, size_t iFile2 ) ;
	// 次のデータを用意する
	virtual bool PrepareNextDataAt( size_t iFile ) = 0 ;
	// ファイルを読み込んでバッファに変換する
	virtual std::shared_ptr<NNBuffer>
				LoadFromFile( const std::filesystem::path& path ) = 0 ;
	// バッファを形式変換してファイルに書き込む
	virtual bool SaveToFile
		( const std::filesystem::path& path, const NNBuffer& bufOutput ) = 0 ;
	// 読み込んだバッファを処理して次のデータとして設定する
	virtual bool SetNextDataOnLoaded
				( std::shared_ptr<NNBuffer> pSource,
					std::shared_ptr<NNBuffer> pTeaching ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// ファイル入出力反復器
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShellFileIterator	: public NNMLPShellGenFileIterator
{
protected:
	std::filesystem::path		m_pathSourceDir ;	// 入力元
	std::filesystem::path		m_pathPairDir ;		// 教師又は出力先
	bool						m_flagOutputPair ;	// 出力モード
	std::filesystem::path		m_pathPairFile ;

public:
	// 構築関数
	NNMLPShellFileIterator
		( const char * pszSourceDir,
			const char * pszPairDir, bool flagOutputPair ) ;

public:
	// 最後に取得した入力データに対応する予測データを出力する
	virtual bool OutputPrediction( const NNBuffer& bufOutput ) ;
	// GetTeachingData で取得できる教師データのパスを取得する
	virtual std::string GetTeachingDataPath( void ) ;
	// OutputPrediction で出力するパスを取得する
	virtual std::string GetOutputPath( void ) ;

public:
	// 次のデータを用意する
	virtual bool PrepareNextDataAt( size_t iFile ) ;
	// 教師ファイル名を生成する
	virtual std::filesystem::path
		MakeTeacherPathOf( const std::filesystem::path& pathSource ) ;
	// 出力ファイル名を生成する
	virtual std::filesystem::path
		MakeOutputPathOf( const std::filesystem::path& pathSource ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// ファイル分類
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShellFileClassifier	: public NNMLPShell::Classifier
{
protected:
	std::vector<std::string>	m_classNames ;	// 分類名（ディレクトリ）
	bool						m_formatIndex ;	// one-hot は1チャネルのインデックス表現

public:
	// 構築関数
	NNMLPShellFileClassifier
		( const char * pszClassDir = nullptr, bool formatIndex = false ) ;
	// 予測モードでサブディレクトリを分類名として設定する
	void DirectoryAsClassName( const char * pszClassDir ) ;

public:
	// 分類名取得
	virtual const std::string& GetClassNameAt( size_t iClass ) const ;
	virtual size_t GetClassCount( void ) const ;
	// 分類インデックス
	virtual int GetClassIndexOf( const char * pszClassName ) const ;

public:
	// One-Hot データ作成
	std::shared_ptr<NNBuffer> MakeOneHot( size_t iClass ) const ;
	// One-Hot インデックス形式か？
	bool IsOneHotIndexFormat( void ) const ;
	// One-Hot 表現をインデックス形式に設定
	void SetOneHotToIndexFormat( bool modeIndex ) ;

public:
	// 分類名記述ファイル読み込み
	std::shared_ptr<NNBuffer>
			ParseFile( const std::filesystem::path& path ) ;
	
} ;



//////////////////////////////////////////////////////////////////////////////
// 分類器ファイル入力反復器
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShellFileClassIterator
				: public NNMLPShellGenFileIterator,
					public NNMLPShellFileClassifier
{
protected:
	std::filesystem::path		m_pathSourceDir ;	// 入力元
	bool						m_flagPrediction ;	// 分類予測モード
	std::vector<size_t>			m_classIndices ;	// ファイル毎の分類
	std::string					m_strPairData ;

public:
	// 構築関数
	NNMLPShellFileClassIterator
		( const char * pszSourceDir, bool flagPrediction,
			const char * pszClassDir = nullptr, bool formatIndex = false ) ;

public:
	// 最後に取得した入力データに対応する予測データを出力する
	virtual bool OutputPrediction( const NNBuffer& bufOutput ) ;
	// GetTeachingData で取得できる教師データのパスを取得する
	virtual std::string GetTeachingDataPath( void ) ;
	// OutputPrediction で出力するパスを取得する
	virtual std::string GetOutputPath( void ) ;

public:
	// 入力ファイルをシャッフルする（指定指標のファイルを入れ替える）
	virtual void ShuffleFile( size_t iFile1, size_t iFile2 ) ;
	// 次のデータを用意する
	virtual bool PrepareNextDataAt( size_t iFile ) ;
	// バッファを形式変換してファイルに書き込む
	virtual bool SaveToFile
		( const std::filesystem::path& path, const NNBuffer& bufOutput ) ;
	virtual void WriteToFile
		( std::ostream& ofs, const NNBuffer& bufOutput ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// GAN 学習用反復器
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShellGANIterator	: public NNMLPShell::Iterator,
									public NNMLPShell::GANIterator
{
protected:
	std::shared_ptr<NNMLPShellFileClassIterator>	m_pClassifier ;

	std::shared_ptr<NNBuffer>	m_pTeaching ;
	std::shared_ptr<NNBuffer>	m_pCounterfeit ;

	std::vector<size_t>	m_classes ;		// 分類インデックス配列（シャッフル）
	size_t				m_iNext ;

public:
	// 構築関数
	NNMLPShellGANIterator
		( std::shared_ptr<NNMLPShellFileClassIterator> pClassifier ) ;

public:
	// 初めから
	virtual void ResetIterator( void ) ;
	// 次の入力データを取得する
	virtual std::shared_ptr<NNBuffer> NextSource( void ) ;
	// 次の検証データを取得する
	virtual std::shared_ptr<NNBuffer> NextValidation( void ) ;
	// 最後に取得した{入力|検証}データに対応する教師データを取得する
	virtual std::shared_ptr<NNBuffer> GetTeachingData( void ) ;
	// 最後に取得した入力データに対応する予測データを出力する
	virtual bool OutputPrediction( const NNBuffer& bufOutput ) ;
	// {NextSource|NextValidation} で取得した入力データのパスを取得する
	virtual std::string GetSourcePath( void ) ;
	// GetTeachingData で取得できる教師データのパスを取得する
	virtual std::string GetTeachingDataPath( void ) ;
	// OutputPrediction で出力するパスを取得する
	virtual std::string GetOutputPath( void ) ;

public:
	// 分類器への偽物教師データを取得する（GAN用）
	virtual std::shared_ptr<NNBuffer> GetCounterfeitData( void ) ;
	// 分類器用の反復器取得
	virtual std::shared_ptr<Iterator> GetClassifierIterator( void ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// 生成器用抽象反復器
//////////////////////////////////////////////////////////////////////////////

class	NNMLPShellGenerativeIterator
			: public NNMLPShellFileIterator, public NNMLPShellFileClassifier
{
public:
	// 構築関数
	NNMLPShellGenerativeIterator
		( const char * pszSourceDir,
			const char * pszOutputDir,
			const char * pszClassDir, bool formatIndex = false ) ;
} ;


}


#endif

