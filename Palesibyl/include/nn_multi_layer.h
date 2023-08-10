
#ifndef	__NN_MULTI_LAYER_H__
#define	__NN_MULTI_LAYER_H__

#include "nn_perceptron.h"

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// 定数
//////////////////////////////////////////////////////////////////////////////

// 畳み込みパッディング
enum	ConvolutionPadding
{
	convNoPad,
	convPadZero,
	convPadBorder,
	convPadWrap,
	convNoPad_sparse,
	convPadZero_sparse,
} ;

// 活性化関数
constexpr static const char *	activLinear			= NNAFunctionLinear::FunctionName ;		// "linear"
constexpr static const char *	activLinearMAE		= NNAFunctionLinearMAE::FunctionName ;	// "l1_loss"
constexpr static const char *	activReLU			= NNAFunctionReLU::FunctionName ;		// "relu"
constexpr static const char *	activSigmoid		= NNAFunctionSigmoid::FunctionName ;	// "sigmoid"
constexpr static const char *	activTanh			= NNAFunctionTanh::FunctionName ;		// "tanh"
constexpr static const char *	activSoftmax		= NNAFunctionSoftmax::FunctionName ;	// "softmax"
constexpr static const char *	activFastSoftmax	= NNAFunctionFastSoftmax::FunctionName ;// "fast_softmax"
constexpr static const char *	activArgmax			= NNAFunctionArgmax::FunctionName ;		// "argmax"
constexpr static const char *	activFastArgmax		= NNAFunctionFastArgmax::FunctionName ;	// "fast_argmax"
constexpr static const char *	activMaxPool		= NNAFunctionMaxPool::FunctionName ;	// "maxpool"
constexpr static const char *	activMultiply		= NNAFunctionMultiply::FunctionName ;	// "multiply"



//////////////////////////////////////////////////////////////////////////////
// ポインタ型定義
//////////////////////////////////////////////////////////////////////////////

typedef	std::shared_ptr<NNPerceptron>	NNPerceptronPtr ;



//////////////////////////////////////////////////////////////////////////////
// レイヤー入力情報
//////////////////////////////////////////////////////////////////////////////

class	NNLayerConnection	: public NNPerceptron::Connection
{
public:
	NNPerceptronPtr	m_pFrom ;
public:
	NNLayerConnection( void ) { }
	NNLayerConnection( const NNLayerConnection& lc )
		: NNPerceptron::Connection(lc), m_pFrom(lc.m_pFrom) { }
	NNLayerConnection
		( NNPerceptronPtr pFrom,
			int iDelay = 0, size_t iChannel = 0, size_t nChannels = 0 )
		: NNPerceptron::Connection( 0, iDelay, iChannel, nChannels ), m_pFrom(pFrom) { }
} ;

class	NNLayerConnections	: public std::vector<NNLayerConnection>
{
public:
	NNLayerConnections( void ) { }
	NNLayerConnections( const NNLayerConnections& lc )
		: std::vector<NNLayerConnection>(lc) {}
	NNLayerConnections( const NNLayerConnection& lc )
	{
		std::vector<NNLayerConnection>::push_back( lc ) ;
	}
	NNLayerConnections
		( NNPerceptronPtr pFrom,
			int iDelay = 0, size_t iChannel = 0, size_t nChannels = 0 )
	{
		Append( pFrom, iDelay, iChannel, nChannels ) ;
	}
	NNLayerConnections&
		Append( NNPerceptronPtr pFrom,
			int iDelay = 0, size_t iChannel = 0, size_t nChannels = 0 )
	{
		std::vector<NNLayerConnection>::push_back
			( NNLayerConnection( pFrom, iDelay, iChannel, nChannels ) ) ;
		return	*this ;
	}
} ;



//////////////////////////////////////////////////////////////////////////////
// マルチ・レイヤー・パーセプトロン（基底）
//////////////////////////////////////////////////////////////////////////////

class	NNPerceptronArray
{
public:
	// バッファ
	struct	BufferArray
	{
		uint32_t								flags ;		// enum PrepareBufferFlag の組み合わせ
		int										iDelta2 ;	// δ逆伝播２パス目開始レイヤー
		NNPerceptron::BufferArray				buffers ;
		NNPerceptron::CPUWorkArrayArray			works ;
		std::vector<NNPerceptron::InputBuffer>	inBufs ;

		BufferArray( void ) : flags(0), iDelta2(-1) { }
	} ;
	class	LossAndGradientArray
				: public std::vector<NNPerceptron::LossAndGradientBuf>
	{
	public:
		float	bufNormMax ;
	public:
		LossAndGradientArray( void ) : bufNormMax(0.0f) {}
	} ;

	// レイヤー・コンテキスト
	struct	LayerContext
	{
		NNPerceptronArray *				pMLP ;
		NNPerceptron::BufferArray *		pBufArray ;
		NNPerceptron *					pLayer ;
		NNBuffer *						pOutput ;
		NNPerceptron::Buffer *			pBuffer ;
		NNPerceptron::CPUWorkArray *	pWorks ;

		LayerContext( void ) 
			: pMLP(nullptr), pBufArray(nullptr), pLayer(nullptr),
				pOutput(nullptr), pBuffer(nullptr), pWorks(nullptr) { }
	} ;

protected:
	std::string						m_id ;
	std::vector<NNPerceptronPtr>	m_mlp ;
	std::shared_ptr<NNLossFunction>	m_loss ;

public:
	// 識別子
	const std::string& GetIdentity( void ) const ;
	void SetIdentity( const char * pszId ) ;

	// データ初期化
	void ClearAll( void ) ;

	// 損失関数（デフォルトは出力レイヤーの活性化関数）
	std::shared_ptr<NNLossFunction> GetLossFunction( void ) const ;
	void SetLossFunction( std::shared_ptr<NNLossFunction> loss ) ;

	// レイヤー追加
	NNPerceptronPtr AppendLayer
		( size_t nDstChannels, size_t nSrcChannels, size_t nBias = 1,
			const char * pszActivation = activLinear,
			std::shared_ptr<NNSamplingFilter> sampler = nullptr ) ;
	NNPerceptronPtr AppendLayer
		( size_t nDstChannels, size_t nSrcChannels, size_t nBias,
			std::shared_ptr<NNActivationFunction> activation,
			std::shared_ptr<NNSamplingFilter> sampler = nullptr ) ;
	size_t AppendLayer( NNPerceptronPtr pLayer ) ;
	size_t InsertLayer( size_t iLayer, NNPerceptronPtr pLayer ) ;

	// 畳み込みレイヤー追加
	NNPerceptronPtr AppendConvLayer
		( size_t nDstChannels, size_t nSrcChannels,
			int xConv, int yConv,
			ConvolutionPadding padding = convPadZero, size_t nBias = 1,
			const char * pszActivation = activLinear,
			int xStride = 1, int yStride = 1, int xOffset = 0, int yOffset = 0 ) ;
	NNPerceptronPtr AppendConvLayer
		( size_t nDstChannels, size_t nSrcChannels,
			int xConv, int yConv,
			ConvolutionPadding padding, size_t nBias,
			std::shared_ptr<NNActivationFunction> activation,
			int xStride = 1, int yStride = 1, int xOffset = 0, int yOffset = 0 ) ;

	// チャネル毎の畳み込みレイヤー追加
	NNPerceptronPtr AppendDepthwiseConv
		( size_t nDstChannels, size_t nSrcChannels, size_t nDepthwise,
			int xConv, int yConv,
			ConvolutionPadding padding = convPadZero, size_t nBias = 1,
			const char * pszActivation = activLinear,
			int xStride = 1, int yStride = 1, int xOffset = 0, int yOffset = 0 ) ;
	NNPerceptronPtr AppendDepthwiseConv
		( size_t nDstChannels, size_t nSrcChannels, size_t nDepthwise,
			int xConv, int yConv,
			ConvolutionPadding padding, size_t nBias,
			std::shared_ptr<NNActivationFunction> activation,
			int xStride = 1, int yStride = 1, int xOffset = 0, int yOffset = 0 ) ;

	// 疎行列な結合レイヤー追加
	NNPerceptronPtr AppendDepthwiseLayer
		( size_t nDstChannels, size_t nSrcChannels,
			size_t nDepthwise, size_t nBias = 1,
			const char * pszActivation = activLinear ) ;
	NNPerceptronPtr AppendDepthwiseLayer
		( size_t nDstChannels, size_t nSrcChannels,
			size_t nDepthwise, size_t nBias,
			std::shared_ptr<NNActivationFunction> activation,
			std::shared_ptr<NNSamplingFilter> sampler = nullptr ) ;

	// アップサンプリング・レイヤー追加
	NNPerceptronPtr AppendUpsamplingLayer
		( size_t nDstChannels, size_t nSrcChannels,
			int xUpsampling, int yUpsampling, size_t nBias = 1,
			const char * pszActivation = activLinear ) ;
	NNPerceptronPtr AppendUpsamplingLayer
		( size_t nDstChannels, size_t nSrcChannels,
			int xUpsampling, int yUpsampling, size_t nBias,
			std::shared_ptr<NNActivationFunction> activation ) ;
	NNPerceptronPtr AppendUp2x2Layer
		( size_t nDstChannels, size_t nSrcChannels, size_t nBias = 1,
			const char * pszActivation = activLinear ) ;
	NNPerceptronPtr AppendUp2x2Layer
		( size_t nDstChannels, size_t nSrcChannels, size_t nBias,
			std::shared_ptr<NNActivationFunction> activation ) ;

	// アップサンプリング（パススルー）レイヤー追加
	NNPerceptronPtr AppendUpsamplingFixLayer
		( size_t nDstChannels, size_t nSrcChannels,
			int xUpsampling, int yUpsampling,
			const char * pszActivation = activLinear ) ;

	// One-Hot 相当１チャネル（インデックス値）入力レイヤー追加
	NNPerceptronPtr AppendLayerAsOneHot
		( size_t nDstChannels, size_t nSrcChannels,
			const char * pszActivation = activLinear ) ;

	// Softmax 高速化用レイヤー追加
	NNPerceptronPtr AppendFastSoftmax
		( size_t nDstChannels, size_t nSrcChannels, size_t nBias = 1,
			const char * pszActivation = activFastArgmax ) ;

	// チャネル毎の畳み込み MaxPool レイヤー追加
	NNPerceptronPtr AppendMaxPoolLayer
		( size_t nSrcChannels, size_t xPoolWidth, size_t yPoolHeight ) ;

	// ゲート付き活性化関数レイヤー追加
	std::tuple< NNPerceptronPtr,		// 乗算ゲート
				NNPerceptronPtr,		// [pszActivation]
				NNPerceptronPtr >		// [pszGateActivation]
		AppendGatedLayer
			( size_t nDstChannels, size_t nSrcChannels, size_t nBias,
				const char * pszActivation = activLinear,
				const char * pszGateActivation = activSigmoid,
				const NNLayerConnections lcInputs = NNLayerConnections() ) ;

	// チャネル毎の加算結合レイヤー追加
	NNPerceptronPtr AppendPointwiseAdd
		( size_t nDstChannels,
			int iLayerOffset1 = 1, int iDelay1 = 0,
			int iLayerOffset2 = 2, int iDelay2 = 0,
			int xOffset2 = 0, int yOffset2 = 0,
			const char * pszActivation = activLinear ) ;
	NNPerceptronPtr AppendPointwiseAdd
		( size_t nDstChannels,
			NNPerceptronPtr pLayer1, int iDelay1,
			NNPerceptronPtr pLayer2, int iDelay2,
			int xOffset2 = 0, int yOffset2 = 0,
			const char * pszActivation = activLinear ) ;

	// チャネル毎の乗算結合レイヤー追加
	NNPerceptronPtr AppendPointwiseMul
		( size_t nDstChannels,
			int iLayerOffset1 = 1, int iDelay1 = 0,
			int iLayerOffset2 = 2, int iDelay2 = 0,
			int xOffset2 = 0, int yOffset2 = 0 ) ;
	NNPerceptronPtr AppendPointwiseMul
		( size_t nDstChannels,
			NNPerceptronPtr pLayer1, int iDelay1,
			NNPerceptronPtr pLayer2, int iDelay2,
			int xOffset2 = 0, int yOffset2 = 0 ) ;

	// μ, log(σ^2) から乱数 z～N(μ,σ) を生成する
	NNPerceptronPtr AppendGaussianLayer
		( size_t nDstChannels,
			NNPerceptronPtr pLayerMean,		// μ
			NNPerceptronPtr pLayerLnVar,	// log(σ^2)
			const char * pszActivation = activLinear ) ;

	// レイヤー数取得
	size_t GetLayerCount( void ) const ;

	// レイヤー取得
	NNPerceptronPtr GetLayerAt( size_t iLayer ) const ;
	NNPerceptronPtr GetLayerAs( const char * pszId ) const ;
	LayerContext GetLayerContextAt
			( size_t iLayer, const BufferArray& bufArray ) const ;

	// レイヤー削除
	NNPerceptronPtr RemoveLayerAt( size_t iLayer ) ;
	NNPerceptronPtr RemoveLayerOf( NNPerceptronPtr pLayer ) ;
	void RemoveAllLayers( void ) ;

	// レイヤー入力情報設定
	void SetLayerInput
		( NNPerceptronPtr pLayer, const NNLayerConnections lcInputs ) ;

	// レイヤー検索（見つからない場合 -1 ）
	int FindLayer( NNPerceptronPtr pLayer ) const ;
	int FindLayer( NNPerceptron * pLayer ) const ;
	int FindLayerAs( const char * pszId, size_t iFirst = 0 ) const ;
	// レイヤー検索（最終レイヤーから pLayer へ AddConnection する時のレイヤーオフセット）
	int LayerOffsetOf( NNPerceptronPtr pLayer ) const ;
	int LayerOffsetOf( NNPerceptron * pLayer ) const ;
	// レイヤー検索（pFromLayer から pLayer へ AddConnection する時のレイヤーオフセット）
	// （pLayer == nullptr の時には、pLayer は入力データ）
	int LayerOffsetFrom
		( NNPerceptronPtr pLayer, NNPerceptronPtr pFromLayer ) const ;
	int LayerOffsetFrom
		( NNPerceptron * pLayer, NNPerceptron * pFromLayer ) const ;

public:
	// シリアライズ用チャンク
	constexpr static const uint32_t	CHHDRID_MLP_ID = NNCHUNKID('M','L','I','D') ;
	constexpr static const uint32_t	CHHDRID_LOSS = NNCHUNKID('L','O','S','S') ;
	constexpr static const uint32_t	CHHDRID_LAYER = NNCHUNKID('L','A','Y','R') ;

	// シリアライズ
	virtual void Serialize( NNSerializer& ser ) ;
	// デシリアライズ
	virtual bool Deserialize( NNDeserializer & dsr ) ;
	bool DeserializeChunk( NNDeserializer & dsr, uint32_t idChunk ) ;
	bool DeserializeLayer( NNDeserializer & dsr ) ;

public:
	// 学習の前に事前予測処理が必要な回数
	size_t CountOfPrePrediction( void ) const ;
	// 出力サイズ計算
	NNBufDim CalcOutputSize( const NNBufDim& dimInput ) const ;
	void PrepareOutputDimArray
		( std::vector<NNBufDim>& dimArray, const NNBufDim& dimInput ) const ;

	// バッファ準備コンフィグ
	struct	BufferConfig
	{
		bool	flagMemoryCommit ;
		bool	flagUseCuda ;
		size_t	nThreadCount ;

		BufferConfig( void )
			: flagMemoryCommit( false ),
				flagUseCuda( true ), nThreadCount( 0 ) {}
	} ;
	// バッファ準備フラグ
	enum	PrepareBufferFlag
	{
		bufferForLearning		= NNPerceptron::bufferForLearning,		// 学習用
		bufferPropagateDelta	= NNPerceptron::bufferPropagateDelta,	// 入力元へデルタを伝播させる
		bufferNoDropout			= NNPerceptron::bufferNoDropout,		// ドロップアウトは行わない
	} ;
	// バッファ準備
	void PrepareBuffer
		( BufferArray& bufArray,
			const NNBufDim& dimInput,
			uint32_t flagsBuffer,		// enum PrepareBufferFlag の組み合わせ
			const BufferConfig& bufConfig,
			const NNLoopStream& stream ) ;
	void PrepareLossAndGradientArray( LossAndGradientArray& lagArray ) ;
	// 出力サイズ取得
	NNBufDim GetOutputSize( const BufferArray& bufArray ) const ;
	// 勾配リセット
	void ResetWorkInBatch( BufferArray& bufArray ) ;
	void ResetLossAndGrandient( LossAndGradientArray& lagArray ) ;
	// エポック開始時処理
	void OnBeginEpoch( void ) ;
	// エポック終了時処理
	void OnEndEpoch( BufferArray& bufArray ) ;
	// 損失と勾配を合計する
	void AddLossAndGradient
		( LossAndGradientArray& lagArray, const BufferArray& bufArray ) ;
	// ミニバッチ毎の処理
	void PrepareForMiniBatch
		( BufferArray& bufArray,
			uint32_t flagsBuffer,
			std::random_device::result_type rndSeed, NNLoopStream& stream ) const ;
	// ストリーミングに連動してバッファをシフトする
	void ShiftBufferWithStreaming
		( BufferArray& bufArray, size_t xShift, NNLoopStream& stream ) ;
	// 出力層が再帰入力されている場合、学習の前に遅延バッファに教師データを入れる
	void PrepareOutputDelay
		( BufferArray& bufArray, NNBuffer& bufTeacher, NNLoopStream& stream ) ;

public:
	// モデルとデータの形状の検証
	enum	VerifyError
	{
		verifyNormal,				// 正常
		outOfRangeInputLayer,		// 入力レイヤーが範囲外
		mustBeFirstInputLayer,		// 入力層でなければならいサンプラーがそれ以外で使用されている
		mustBeLastLayer,			// 最終層でしか使用できない活性化関数が使用されている
		mustNotBeLastLayer,			// 最終層で使用できない活性化関数が使用されている
		mismatchInputSize,			// レイヤーへの入力サイズが不正
		mismatchInputChannel,		// レイヤーへの入力チャネル数が不正
		mismatchSourceSize,			// 入力データのサイズが不正
		mismatchSourceChannel,		// 入力データのチャネル数が不正
		mismatchTeachingSize,		// 教師データのサイズが不正
		mismatchTeachingChannel,	// 教師データのチャネル数が不正
		lowCudaMemory,				// CUDA メモリが不足
		tooHugeMatrixForCuda,		// CUDA で実行するには大きすぎる行列サイズ
	} ;
	struct	VerifyResult
	{
		VerifyError	verfError ;
		size_t		iLayer ;
		size_t		iConnection ;
	} ;
	bool VerifyDataShape
		( VerifyResult& verfResult,
			const BufferArray& bufArrays,
			const NNBufDim& dimTeaching,
			const NNBufDim& dimSource0, bool flagCuda ) const ;

public:
	// 予測処理
	NNBuffer * Prediction
		( BufferArray& bufArray, NNLoopStream& stream,
			NNBuffer& bufInput, size_t xBoundary = 0,
			size_t iFirstLayer = 0, size_t iEndLayer = 0,
			bool flagForLearning = false, bool flagLowMemory = false ) ;
	// 予測値の損失計算
	double CalcLoss
		( BufferArray& bufArray, NNLoopStream& stream, NNBuffer& bufTeacher ) ;

public:
	// 学習１回
	double Learning
		( BufferArray& bufArray, NNLoopStream& stream,
			NNBuffer& bufTeacher, NNBuffer& bufInput,
			const LayerContext * pInputLayer = nullptr,
			NNPerceptronArray * pForwardMLP = nullptr,
			BufferArray * pForwardBufArrays = nullptr,
			std::function<void()> funcAddLossDelta = [](){} ) ;
	// 指定レイヤーのδ逆伝播処理
	void DeltaBackLayerAt
		( size_t iLayer, bool flagOutputLayer,
			const LayerContext * pInputLayer,
			BufferArray& bufArrays, NNLoopStream& stream ) ;
	// 勾配反映
	void GradientReflection
		( LossAndGradientArray& lagArray, float deltaRate ) ;
	// 平均損失値取得
	double GetAverageLoss( const BufferArray& bufArray ) const ;
	double GetAverageLoss( const LossAndGradientArray& lagArray ) const ;

} ;



//////////////////////////////////////////////////////////////////////////////
// マルチ・レイヤー・パーセプトロン
//////////////////////////////////////////////////////////////////////////////

class	NNMultiLayerPerceptron	: public NNPerceptronArray
{
public:
	// 特殊レイヤー指標（入力バッファ指定）
	enum	SpecialLayerIndex
	{
		layerTeacher	= -2,
		layerSource		= -1,
		layerMLPFirst	= 0,
	} ;

	// 追加的な MLP パス
	struct	PassDescription
	{
		int32_t	iSourceLayer ;		// 入力レイヤー指標
		int32_t	iTeachingLayer ;	// 教師レイヤー指標
	} ;
	class	Pass	: public NNPerceptronArray
	{
	public:
		PassDescription	m_dsc ;
	public:
		virtual void Serialize( NNSerializer& ser ) ;
		virtual bool Deserialize( NNDeserializer & dsr ) ;
	} ;

	// バッファ
	struct	BufferArrays	: public NNPerceptronArray::BufferArray
	{
		size_t			iFirstLayer ;	// 開始レイヤー（一部のみ使用する場合）
		size_t			iEndLayer ;		// 終了レイヤー＋１（一部のみ使用する場合）
		size_t			xBoundary ;		// ストリーミングでの開始位置
		NNLoopStream	stream ;		// 実行ストリーム

		std::vector
			< std::shared_ptr
				<NNPerceptronArray::BufferArray> >
						subpass ;		// 追加的なパス用

		BufferArrays( void )
			: iFirstLayer(0), iEndLayer(0), xBoundary(0) { }
	} ;

	class	LossAndGradientArrays	: public NNPerceptronArray::LossAndGradientArray
	{
	public:
		std::vector<NNPerceptronArray::LossAndGradientArray>	subpass ;
	} ;

	// シリアライズ用チャンク
	constexpr static const uint32_t	CHHDRID_HEADER = NNCHUNKID('M','L','P','H') ;
	constexpr static const uint32_t	CHHDRID_MLP_BODY = NNCHUNKID('M','L','P','B') ;
	constexpr static const uint32_t	CHHDRID_EVALUATION = NNCHUNKID('E','V','A','L') ;
	constexpr static const uint32_t	CHHDRID_SUBPASS = NNCHUNKID('S','U','B','P') ;
	constexpr static const uint32_t	CHHDRID_SUBPASS_HEADER = NNCHUNKID('S','U','B','H') ;

	// サイズ情報
	struct	LayerDim
	{
		uint32_t	x, y, z ;
	} ;

	// ヘッダ情報
	struct	FileHeader
	{
		uint32_t	flagsHeader ;	// enum FileHeaderFlag の組み合わせ
		uint32_t	nLayerCount ;	// レイヤー数
		uint32_t	flagsMLP ;		// enum MLPFlag の組み合わせ
		uint32_t	nReserved ;		// = 0
		LayerDim	dimInShape ;
		LayerDim	dimInUnit ;
	} ;

	// ヘッダフラグ
	enum	FileHeaderFlag
	{
		hdrFlagChunkedLayer	= 0x0001,
	} ;

	// MLP フラグ
	enum	MLPFlag
	{
		mlpFlagStream		= 0x0001,	// RNN などの可変長入力
	} ;

protected:
	std::vector< std::shared_ptr<Pass> >	m_subpass ;
	std::shared_ptr<NNEvaluationFunction>	m_evaluation ;

	uint32_t	m_flagsMLP ;		// enum MLPFlag の組み合わせ
	NNBufDim	m_dimInShape ;		// デフォルト入力サイズ（mlpFlagStream 時）
	NNBufDim	m_dimInUnit ;		// 入力単位（mlpFlagStream 時）

public:
	// 構築関数
	NNMultiLayerPerceptron( void ) ;
	// データ初期化
	void ClearAll( void ) ;
	// 入力サイズ設定
	void SetInputShape
		( uint32_t flagsMLP,
			const NNBufDim& dimShape, const NNBufDim& dimUnit ) ;
	// フラグ取得
	uint32_t GetMLPFlags( void ) const ;
	// 入力サイズ取得
	const NNBufDim& GetInputShape( void ) const ;
	const NNBufDim& GetInputUnit( void ) const ;
	// 学習の前に事前予測処理が必要な回数
	size_t CountOfPrePrediction( void ) const ;
	// 評価関数
	void SetEvaluationFunction( std::shared_ptr<NNEvaluationFunction> pEvaluation ) ;
	std::shared_ptr<NNEvaluationFunction> GetEvaluationFunction( void ) const ;
	// 追加的なパス
	void AddSubpass( std::shared_ptr<Pass> pass ) ;
	std::shared_ptr<Pass> GetSubpassAs( const char * pszId ) const ;
	std::shared_ptr<Pass> GetSubpassAt( size_t iPass ) const ;
	int FindSubpass( const char * pszId ) const ;
	size_t GetSubpassCount( void ) const ;
	bool RemoveSubpass( std::shared_ptr<Pass> pass ) ;
	std::shared_ptr<Pass> RemoveSubpassAt( size_t iPass ) ;
	void RemoveAllSubpass( void ) ;

	// μ, log(σ^2) の Gaussian KL Divergence 損失関数追加
	void AddLossGaussianKLDivergence
		( NNPerceptronPtr pLayerMean,
			NNPerceptronPtr pLayerLnVar,
			float lossFactor = 1.0f, float deltaFactor = 1.0f ) ;

public:
	// シリアライズ
	virtual void Serialize( NNSerializer& ser ) ;
	// デシリアライズ
	virtual bool Deserialize( NNDeserializer & dsr ) ;

public:
	// バッファ準備
	virtual void PrepareBuffer
		( BufferArrays& bufArrays,
			const NNBufDim& dimInput,
			uint32_t flagsBuffer,		// enum PrepareBufferFlag の組み合わせ
			const BufferConfig& bufConfig ) ;
	virtual void PrepareLossAndGradientArrays( LossAndGradientArrays& lagArrays ) ;
	// 勾配リセット
	virtual void ResetWorkInBatch( BufferArrays& bufArrays ) ;
	virtual void ResetLossAndGrandient( LossAndGradientArrays& lagArrays ) ;
	// エポック開始時処理
	virtual void OnBeginEpoch( void ) ;
	// エポック終了時処理
	virtual void OnEndEpoch( BufferArrays& bufArrays ) ;
	// 損失と勾配を合計する
	virtual void AddLossAndGradient
		( LossAndGradientArrays& lagArrays, const BufferArrays& bufArrays ) ;
	// ミニバッチ毎の処理
	virtual void PrepareForMiniBatch
		( BufferArrays& bufArrays,
			uint32_t flagsBuffer, std::random_device::result_type rndSeed ) const ;
	// ストリーミングに連動してバッファをシフトする
	virtual void ShiftBufferWithStreaming( BufferArrays& bufArrays, size_t xShift ) ;
	// 出力層が再帰入力されている場合、学習の前に遅延バッファに教師データを入れる
	virtual void PrepareOutputDelay
		( BufferArrays& bufArrays, NNBuffer& bufTeacher ) ;

protected:
	// 保護オーバーロード（ミス防止用）
	void PrepareLossAndGradientArray( LossAndGradientArray& lagArray ) ;
	void ResetWorkInBatch( BufferArray& bufArray ) ;
	void ResetLossAndGrandient( LossAndGradientArray& lagArray ) ;
	void OnEndEpoch( BufferArray& bufArrays ) ;
	void AddLossAndGradient
		( LossAndGradientArray& lagArray, const BufferArray& bufArray ) ;

public:
	// モデルとデータの形状の検証
	struct	VerifyResult	: public NNPerceptronArray::VerifyResult
	{
		int	iSubpass ;		// 主 MLP の時は -1
	} ;
	virtual bool VerifyDataShape
		( VerifyResult& verfResult,
			const BufferArrays& bufArrays,
			const NNBufDim& dimTeaching, const NNBufDim& dimSource0 ) const ;
	// レイヤー出力バッファサイズ取得
	virtual NNBufDim GetLayerOutputSize
		( int iLayer, const BufferArrays& bufArrays,
			const NNBufDim& dimTeaching, const NNBufDim& dimSource0 ) const ;
	// レイヤー出力バッファ取得
	virtual NNBuffer * GetLayerOutputBuffer
		( int iLayer, const BufferArrays& bufArrays,
			NNBuffer * pbufTeacher, NNBuffer * pbufSource ) const ;
	// レイヤーコンテキスト取得
	virtual LayerContext * GetLayerContext
		( LayerContext& ctxLayer, int iLayer,
			const BufferArrays& bufArrays,
			NNBuffer * pbufTeacher, NNBuffer * pbufSource ) const ;

public:
	// 予測処理
	virtual NNBuffer * Prediction
		( BufferArrays& bufArrays, NNBuffer& bufInput,
			bool flagForLearning = false, bool flagLowMemory = false ) ;
	// 予測値の損失計算
	double CalcLoss
		( BufferArrays& bufArrays, NNBuffer& bufTeacher ) ;

public:
	// 学習１回
	virtual double Learning
		( BufferArrays& bufArrays,
			NNBuffer& bufTeacher, NNBuffer& bufInput,
			NNMultiLayerPerceptron * pForwardMLP = nullptr,
			BufferArrays * pForwardBufArrays = nullptr ) ;
	// 勾配反映
	virtual void GradientReflection
		( LossAndGradientArrays& lagArrays, float deltaRate ) ;
protected:
	virtual void GradientReflection
		( LossAndGradientArray& lagArray, float deltaRate ) ;

public:
	// 平均損失値取得
	virtual double GetAverageLoss( const BufferArrays& bufArrays ) const ;
	virtual double GetAverageLoss( const LossAndGradientArrays& lagArrays ) const ;
protected:
	double GetAverageLoss( const BufferArray& bufArray ) const ;
	double GetAverageLoss( const LossAndGradientArray& lagArray ) const ;

} ;


}

#endif


