
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
// マルチ・レイヤー・パーセプトロン
//////////////////////////////////////////////////////////////////////////////

class	NNMultiLayerPerceptron
{
public:
	// バッファ
	struct	BufferArrays
	{
		uint32_t								flags ;		// enum PrepareBufferFlag の組み合わせ
		int										iDelta2 ;	// δ逆伝播２パス目開始レイヤー
		NNLoopStream							stream ;
		NNPerceptron::BufferArray				buffers ;
		NNPerceptron::CPUWorkArrayArray			works ;
		std::vector<NNPerceptron::InputBuffer>	inBufs ;

		BufferArrays( void ) : flags(0), iDelta2(-1) { }
	} ;
	class	LossAndGradientArray
				: public std::vector<NNPerceptron::LossAndGradientBuf>
	{
	public:
		float	bufNormMax ;
	public:
		LossAndGradientArray( void ) : bufNormMax(0.0f) {}
	} ;

	// シリアライズ用チャンク
	constexpr static const uint32_t	CHHDRID_HEADER = NNCHUNKID('M','L','P','H') ;

	// サイズ情報
	struct	LayerDim
	{
		uint32_t	x, y, z ;
	} ;

	// ヘッダ情報
	struct	FileHeader
	{
		uint32_t	flagsHeader ;	// = 0
		uint32_t	nLayerCount ;	// レイヤー数
		uint32_t	flagsMLP ;		// enum MLPFlag の組み合わせ
		uint32_t	nReserved ;		// = 0
		LayerDim	dimInShape ;
		LayerDim	dimInUnit ;
	} ;

	// フラグ
	enum	MLPFlag
	{
		mlpFlagStream	= 0x0001,	// RNN
	} ;

protected:
	std::vector<NNPerceptronPtr>	m_mlp ;

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

public:
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

	// レイヤー数取得
	size_t GetLayerCount( void ) const ;

	// レイヤー取得
	NNPerceptronPtr GetLayerAt( size_t iLayer ) const ;
	NNPerceptronPtr GetLayerAs( const char * pszId ) const ;

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
	// シリアライズ
	virtual void Serialize( NNSerializer& ser ) ;
	// デシリアライズ
	virtual bool Deserialize( NNDeserializer & dsr ) ;

public:
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
	virtual void PrepareBuffer
		( BufferArrays& bufArrays,
			const NNBufDim& dimInput,
			uint32_t flagsBuffer,		// enum PrepareBufferFlag の組み合わせ
			const BufferConfig& bufConfig ) ;
	virtual void PrepareLossAndGradientArray( LossAndGradientArray& lagArray ) ;
	// 出力サイズ取得
	NNBufDim GetOutputSize( const BufferArrays& bufArrays ) const ;
	// 出力サイズ計算
	NNBufDim CalcOutputSize( const NNBufDim& dimInput ) const ;
	void PrepareOutputDimArray
		( std::vector<NNBufDim>& dimArray, const NNBufDim& dimInput ) const ;
	// ミニバッチ毎の処理
	virtual void PrepareForMiniBatch
		( BufferArrays& bufArrays,
			uint32_t flagsBuffer, std::random_device::result_type rndSeed ) const ;
	// 勾配リセット
	virtual void ResetWorkInBatch( BufferArrays& bufArrays ) ;
	virtual void ResetLossAndGrandient( LossAndGradientArray& lagArray ) ;
	// エポック開始時処理
	virtual void OnBeginEpoch( void ) ;
	// エポック終了時処理
	virtual void OnEndEpoch( BufferArrays& bufArrays ) ;
	// 損失と勾配を合計する
	virtual void AddLossAndGradient
		( LossAndGradientArray& lagArray, const BufferArrays& bufArrays ) ;
	// ストリーミングに連動してバッファをシフトする
	virtual void ShiftBufferWithStreaming( BufferArrays& bufArrays, size_t xShift ) ;
	// 出力層が再帰入力されている場合、学習の前に遅延バッファに教師データを入れる
	virtual void PrepareOutputDelay
		( BufferArrays& bufArrays, NNBuffer& bufTearcher ) ;

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
	virtual bool VerifyDataShape
		( VerifyResult& verfResult,
			const BufferArrays& bufArrays,
			const NNBufDim& dimTeaching, const NNBufDim& dimSource0 ) const ;

public:
	// 予測処理
	virtual NNBuffer * Prediction
		( BufferArrays& bufArrays, NNBuffer& bufInput,
			bool flagForLearning = false, bool flagLowMemory = false ) ;
	// 予測値の損失計算
	double CalcLoss( BufferArrays& bufArrays, NNBuffer& bufTearcher ) ;

public:
	// 学習１回
	virtual double Learning
		( BufferArrays& bufArrays,
			NNBuffer& bufTearcher, NNBuffer& bufInput,
			NNMultiLayerPerceptron * pForwardMLP = nullptr,
			BufferArrays * pForwardBufArrays = nullptr ) ;
	// 指定レイヤーのδ逆伝播処理
	virtual void DeltaBackLayerAt
		( size_t iLayer, bool flagOutputLayer, BufferArrays& bufArrays ) ;
	// 勾配反映
	virtual void GradientReflection
		( BufferArrays& bufArrays, float deltaRate ) ;
	virtual void GradientReflection
		( LossAndGradientArray& lagArray, float deltaRate ) ;
	// 平均損失値取得
	double GetAverageLoss( const BufferArrays& bufArrays ) const ;
	double GetAverageLoss( const LossAndGradientArray& lagArray ) const ;

} ;


}

#endif


