
#ifndef	__NN_PERCEPTRON_H__
#define	__NN_PERCEPTRON_H__

#include <vector>
#include <map>
#include <functional>
#include <random>
#include "nn_matrix.h"
#include "nn_function.h"
#include "nn_buffer.h"
#include "nn_serializer.h"
#include "nn_loop_stream.h"
#include "nn_normalization.h"
#include "nn_sampling_filter.h"
#include "nn_activation_func.h"
#include "nn_evaluation_func.h"

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// パーセプトロン
//////////////////////////////////////////////////////////////////////////////

class	NNPerceptron
{
public:
	// 入力情報
	struct	Connection
	{
		int32_t		iLayer ;	// 入力レイヤーオフセット（1 で直前レイヤー）
		int32_t		iDelay ;	// 遅延入力サンプルオフセット（1 で直前サンプル）
		uint32_t	iChannel ;	// 入力レイヤーチャネル指標
		uint32_t	nChannels ;	// 入力レイヤーチャネル数（0 の時全チャネル）
		int32_t		xOffset ;	// 入力レイヤーｘオフセット
		int32_t		yOffset ;	// 入力レイヤーｙオフセット

		Connection( void )
			: iLayer(1), iDelay(0),
				iChannel(0), nChannels(0), xOffset(0), yOffset(0) { }
		Connection( const Connection& cn )
			: iLayer(cn.iLayer), iDelay(cn.iDelay),
				iChannel(cn.iChannel), nChannels(cn.nChannels),
				xOffset(cn.xOffset), yOffset(cn.yOffset) { }
		Connection( int layer, int delay,
					size_t channel, size_t channels, int xoffset = 0, int yoffset = 0 )
			: iLayer((int32_t)layer), iDelay((int32_t)delay),
				iChannel((uint32_t)channel), nChannels((uint32_t)channels),
				xOffset((int32_t)xoffset), yOffset((int32_t)yoffset) { }
	} ;

	// 中間バッファ
	enum	InputSourceType
	{
		inputSrcOutput,		// 入力元レイヤーの出力バッファ (bufOutput)
		inputTemporary,		// 一時バッファ（複数のソースを結合する場合）
	} ;
	struct	Buffer
	{
		InputSourceType	inSrc ;				// 入力バッファ
		bool			forLearning ;		// 学習用
		bool			reqDelay ;			// ディレイバッファへの出力が必要
		bool			reqDelta2 ;			// 逆伝播が前レイヤーから行われる（2パス必要）
		bool			transMatrix ;		// bufMatrix へ行列転送済み
		size_t			iThisLayer ;		// このレイヤー番号
		size_t			nRefOut ;			// 出力先レイヤーの数
		size_t			nRefOut2 ;
		size_t			iRefMostRearLayer ;	// 参照元の再後背レイヤー
		NNBuffer		bufMatrix ;			// 行列 (CUDA用)
		NNBuffer		bufDropoutMask ;	// ドロップアウト用
		NNBuffer		bufInput ;			// 入力用テンポラリバッファ
		NNBuffer		bufInAct ;			// 活性化関数入力（行列出力）
		NNBuffer		bufOutput ;			// 活性化関数出力  = a(bufInAct)
		NNBuffer		bufDelay ;			// 遅延入力バッファ
		NNBuffer		bufPrevDelta ;		// 逆伝播用δベクトル（後レイヤーから入力）
		NNBuffer		bufPrevDelta2 ;		// 逆伝播用δベクトル（前レイヤーから入力）
		NNBuffer		bufInDelta ;		// 逆伝播用δベクトル（活性化関数から逆伝播）
		NNBuffer		bufOutDelta ;		// 逆伝播用δベクトル（前レイヤーへ）
		NNBuffer		bufGradient ;		// 勾配バッファ (CUDA 用)
		size_t			xGradientBlock ;
		size_t			yGradientBlock ;
		NNNormalizationFilter::WorkBuf
						normWorkBuf ;		// 正規化用バッファ

		size_t GetBufferBytes( void ) const ;
		size_t GetCudaBufferBytes( void ) const ;
		size_t EstimateCudaBufferBytes( void ) const ;
	} ;
	class	BufferArray	: public std::vector< std::shared_ptr<Buffer> >
	{
	public:
		unsigned long long GetTotalBufferBytes( void ) const ;
		unsigned long long GetTotalCudaBufferBytes( void ) const ;
	} ;

	// 入力バッファ情報
	struct	InputBuffer
	{
		NNBuffer *	pInput ;
	} ;

	// 損失値と行列勾配の合計
	struct	LossAndGradient
	{
		NNMatrix	matGradient ;	// 勾配用バッファ
		size_t		nGradient ;		// 勾配合計サンプル数

		double		fpLoss ;		// 損失合計
		size_t		nLossSamples ;	// 損失合計サンプル数

		LossAndGradient( void )
			: nGradient(0), fpLoss(0.0), nLossSamples(0) {}
		void ResetLoss( void )
		{
			fpLoss = 0.0 ;
			nLossSamples = 0 ;
		}
		void ResetGradient( void )
		{
			matGradient.InitDiagonal( 0.0f ) ;
			nGradient = 0 ;
		}
	} ;

	struct	LossAndGradientBuf	: public LossAndGradient
	{
		NNMatrix	matAdaOpt[2] ;	// 最適化用

		NNNormalizationFilter::GradientBuf
					normGrad ;		// 正規化パラメータ用勾配

		void ResetGradient( void )
		{
			LossAndGradient::ResetGradient() ;
			normGrad.ResetGradient() ;
		}
	} ;

	// CPU 処理用作業バッファ
	struct	CPUWorkBuf	: public LossAndGradient
	{
		std::vector<float>	vecSrc ;		// 行列入力ベクトル用バッファ
		std::vector<float>	vecDst ;		// 行列出力ベクトル用バッファ
		std::vector<float>	vecDiff ;		// 微分 a'
		std::vector<float>	vecOutDelta ;	// 行列逆伝播δ
	} ;
	class	CPUWorkArray	: public LossAndGradientBuf,
								public std::vector<CPUWorkBuf>
	{
	public:
	} ;
	class	CPUWorkArrayArray
				: public std::vector< std::shared_ptr<CPUWorkArray> > { } ;

	// シリアライズ用チャンク
	constexpr static const uint32_t	CHHDRID_TYPE = NNCHUNKID('P','C','L','T') ;
	constexpr static const uint32_t	CHHDRID_BODY = NNCHUNKID('P','C','L','B') ;
	constexpr static const uint32_t	CHHDRID_MATRIX = NNCHUNKID('M','A','T','X') ;
	constexpr static const uint32_t	CHHDRID_PARAM = NNCHUNKID('X','P','R','M') ;
	constexpr static const uint32_t	CHHDRID_EXTEND = NNCHUNKID('P','C','X','I') ;
	constexpr static const uint32_t	CHHDRID_SAMPLER = NNCHUNKID('S','M','P','F') ;
	constexpr static const uint32_t	CHHDRID_ACTIVATION = NNCHUNKID('A','C','T','F') ;
	constexpr static const uint32_t	CHHDRID_NORM = NNCHUNKID('N','O','R','M') ;
	constexpr static const uint32_t	CHHDRID_CONNECTION = NNCHUNKID('I','N','C','T') ;

	// シリアライズ拡張フラグ
	enum	SerializeExtendInfoFlag
	{
		extendInfoAdaptiveOptimization	= 0x00000001,
		extendInfoL2regularization		= 0x00000002,
		extendInfoDeltaFactor			= 0x00000004,
		extendInfoGradientFactor		= 0x00000008,
		extendInfoDropout				= 0x00000010,
		extendInfoIdentity				= 0x80000000,
	} ;

	// 動作フラグ
	enum	BehaviorFlag
	{
		behaviorFixed			= 0x00000001,	// 行列を更新しない（学習しない）
		behaviorCutOff			= 0x00000002,	// 誤差δ逆伝播を遮断する
		behaviorCutOffCon0		= 0x00010000,	// 入力 #0 への誤差δ逆伝播を遮断する
		behaviorCutOffCon1		= 0x00020000,	// 入力 #1 への誤差δ逆伝播を遮断する
		behaviorCutOffCon2		= 0x00040000,	// 入力 #2 への誤差δ逆伝播を遮断する
		behaviorCutOffCon3		= 0x00080000,	// 入力 #3 への誤差δ逆伝播を遮断する
		behaviorCutOffConMask	= 0x000F0000,
		behaviorNoDropout		= 0x00000004,	// ドロップアウトしない
	} ;

	// 勾配更新最適化
	enum	AdaptiveOptimization
	{
		adaOptNo,			// 適応的最適化無し (SGD)
		adaOptMomentum,		// モメンタム
		adaOptRMSProp,		// RMSProp
		adaOptAdam,			// Adam
	} ;
	struct	AdaptiveHyperparameter
	{
		float	alpha ;		// [0,1) : 1次パラメータ減衰 (adaOptMomentum, adaOptAdam)
		float	beta ;		// [0,1) : 2次パラメータ減衰 (adaOptRMSProp, adaOptAdam)
		float	delta ;		// 学習速度係数

		AdaptiveHyperparameter( void )
			: alpha( 0.9f ), beta( 0.999f ), delta( 0.001f ) {}
	} ;

public:
	uint32_t								m_behavior ;	// 動作フラグ（enum BehaviorFlag）（※揮発性）
	std::string								m_id ;			// 識別子
	NNMatrix								m_matrix ;		// 行列
	size_t									m_bias ;		// バイアス項
	size_t									m_depthwise ;	// 疎行列（通常の行列は 1）
	float									m_deltaFactor ;	// δ逆伝播係数（レイヤー毎に調整したい場合）
	float									m_gradFactor ;	// 学習速度係数（レイヤー毎に調整したい場合）
	AdaptiveOptimization					m_adaOpt ;		// 勾配更新最適化法
	AdaptiveHyperparameter					m_adaParam ;
	float									m_l2reg ;		// L2 正則化パラメータ
	float									m_dropout ;		// ドロップアウト率 [0,1)
	std::shared_ptr<NNSamplingFilter>		m_sampler ;		// サンプリング・フィルタ
	std::shared_ptr<NNActivationFunction>	m_activation ;	// 活性化関数
	std::shared_ptr<NNNormalizationFilter>	m_normalizer ;	// 正規化
	std::vector<Connection>					m_connection ;	// 入力元情報

public:
	// 構築関数
	NNPerceptron( void ) ;
	NNPerceptron
		( size_t nDstCount, size_t nSrcCount,
			size_t nDepthwise, size_t nBias,
			std::shared_ptr<NNSamplingFilter> sampler,
			std::shared_ptr<NNActivationFunction> activation ) ;
	NNPerceptron( const NNPerceptron& nnp ) ;

	// 作成
	virtual void Create
		( size_t nDstCount, size_t nSrcCount,
			size_t nDepthwise, size_t nBias,
			std::shared_ptr<NNSamplingFilter> sampler,
			std::shared_ptr<NNActivationFunction> activation ) ;
	// 行列の特殊化
	virtual void Specialize( void ) ;

	// 入力情報追加
	NNPerceptron * AddConnection( const Connection& cn ) ;
	NNPerceptron * AddConnection
		( int iLayer = 1, int iDelay = 0,
			size_t iChannel = 0, size_t nChCount = 0,
			int xOffset = 0, int yOffset = 0 ) ;
	// 入力情報取得
	const std::vector<Connection>& GetConnection( void ) const ;
	// 入力情報クリア
	void ClearAllConnection( void ) ;

public:
	// 行列
	NNMatrix& Matrix( void ) ;
	const NNMatrix& GetMatrix( void ) const ;
	// バイアス項
	size_t GetBias( void ) const ;
	// 対角化単位
	virtual size_t GetDepthwise( void ) const ;
	virtual size_t GetActivationDepthwise( void ) const ;
	// 勾配更新最適化
	AdaptiveOptimization GetAdaptiveOptimization( void ) const ;
	const AdaptiveHyperparameter& GetAdaptiveHyperparameter( void ) const ;
	NNPerceptron * SetAdaptiveOptimization
		( AdaptiveOptimization adaOpt, const AdaptiveHyperparameter& adaParam ) ;
	// δ逆伝播係数（※レイヤー毎に調整したい場合）
	float GetDeltaFactor( void ) const ;
	NNPerceptron * SetDeltaFactor( float delta ) ;
	// 学習速度係数（※レイヤー毎に調整したい場合）
	float GetGradientFactor( void ) const ;
	NNPerceptron * SetGradientFactor( float grad ) ;
	// L2 正則化パラメータ
	float GetRidgeParameter( void ) const ;
	NNPerceptron * SetRidgeParameter( float l2reg ) ;
	// ドロップアウト率
	float GetDropoutRate( void ) const ;
	NNPerceptron * SetDropoutRate( float dropout ) ;
	// 識別子
	const std::string& GetIdentity( void ) const ;
	NNPerceptron * SetIdentity( const char * pszId ) ;
	// 動作フラグ
	void SetBehaviorFlags( uint32_t flags ) ;
	uint32_t GetBehaviorFlags( void ) const ;
	bool IsMatrixFixed( void ) const ;
	bool IsDeltaCutOff( void ) const ;
	bool IsNoDropout( void ) const ;
	// 正規化
	NNPerceptron * SetNormalization( std::shared_ptr<NNNormalizationFilter> pNorm ) ;
	std::shared_ptr<NNNormalizationFilter> GetNormalization( void ) const ;
	// サンプラー
	std::shared_ptr<NNSamplingFilter> GetSampler( void ) const ;
	// 活性化関数
	std::shared_ptr<NNActivationFunction> GetActivation( void ) const ;

public:
	// シリアライズ
	virtual void Serialize( NNSerializer& ser ) ;
	virtual void SerializeExtendInfo( NNSerializer& ser ) ;
	// デシリアライズ
	virtual bool Deserialize( NNDeserializer & dsr ) ;
	virtual bool DeserializeExtendInfo( NNDeserializer & dsr ) ;

protected:
	static std::map
		< std::string,
			std::function
				< std::shared_ptr<NNPerceptron>() > >	s_mapMakePerceptron ;

public:
	// 関数生成準備
	static void InitMake( void ) ;
	// 関数生成
	static std::shared_ptr<NNPerceptron> Make( const char * pszName ) ;
	// 登録
	template <class T> static void Register( void )
	{
		s_mapMakePerceptron.insert
			( std::make_pair(std::string(T::PERCEPTRON_TYPE),
							[]() { return std::make_shared<T>(); } ) ) ;
	}

public:
	// パーセプトロンタイプ
	static constexpr const char	PERCEPTRON_TYPE[] = "full" ;
	virtual const char * GetPerceptronType( void ) const
	{
		return	PERCEPTRON_TYPE ;
	}

public:
	// 出力バッファサイズの計算
	virtual size_t CalcOutputChannels( void ) const ;
	virtual NNBufDim CalcOutputDim( const NNBufDim& dimSrc ) const ;
	virtual NNBufDim CalcInnerDim( const NNBufDim& dimSrc ) const ;
	virtual NNBufDim CalcMatrixPointDim( const NNBufDim& dimSrc ) const ;
	// 入力バッファサイズの計算
	virtual NNBufDim CalcInputDim
		( const BufferArray& bufArray,
			size_t iThisLayer, const NNBufDim& dimSrc0 ) const ;
	virtual NNBufDim CalcInputDim
		( const std::vector<NNBufDim>& dimArray,
			size_t iThisLayer, const NNBufDim& dimSrc0 ) const ;
	virtual NNBufDim CalcInputDim
		( size_t iThisLayer, const NNBufDim& dimSrc0,
			size_t nLayerCount,
			std::function<NNBufDim(size_t)> funcGetDim ) const ;
	// 中間バッファの準備
	enum	PrepareBufferFlag
	{
		bufferForLearning		= 0x0001,	// 学習用
		bufferPropagateDelta	= 0x0002,	// 入力元へデルタを伝播させる
		bufferNoDropout			= 0x0004,	// ドロップアウトは行わない
	} ;
	virtual void ResetBuffer( Buffer& bufThis, size_t iThisLayer ) ;
	virtual void PrepareBuffer
		( Buffer& bufThis, const NNBufDim& dimSrc,
			const BufferArray& bufArray,
			const NNLoopStream& stream,
			size_t iThisLayer,
			uint32_t flagsBuffer,	// enum PrepareBufferFlag の組み合わせ
			bool flagMemoryCommit = true ) const ;
	// 省メモリモードでの予測処理で不要なバッファの解放
	virtual void LowMemoryBuffer( Buffer& bufThis, size_t iPredictedLayer ) ;
	// 作業バッファの準備
	virtual void PrepareWorkArray( CPUWorkArray& bufWorks, size_t nCount ) const ;
	virtual void PrepareWorkBuf( CPUWorkBuf& bufWork ) const ;
	void PrepareLossAndGradientBuf( LossAndGradientBuf& lagb ) const ;
	void PrepareLossAndGradient( LossAndGradient& lag ) const ;
	// ミニバッチ毎の処理
	virtual void PrepareForMiniBatch
		( Buffer& bufThis, uint32_t flagsBuffer /* enum PrepareBufferFlag */,
			std::random_device::result_type rndSeed, NNLoopStream& stream ) const ;
	// 勾配反映後のバッファ処理
	virtual void ResetBufferInBatch( Buffer& bufThis ) const ;
	// 行列勾配・損失合計値初期化
	virtual void ResetWorkArrayInBatch( CPUWorkArray& bufWorks ) const ;
	virtual void ResetWorkBufInBatch( CPUWorkBuf& bufWork ) const ;
	void ResetLossAndGradient( LossAndGradient& lag ) const ;
	// エポック開始時処理
	virtual void OnBeginEpoch( void ) ;
	// エポック終了時処理
	virtual void OnEndEpoch( Buffer& bufThis, CPUWorkArray& bufWorks ) ;
	// 損失・勾配加算
	void AddLossAndGradient
		( LossAndGradientBuf& lagDst, LossAndGradientBuf& lagSrc ) const ;

public:
	// 入力バッファの準備
	virtual InputBuffer PrepareInput
		( const BufferArray& bufArray,
			size_t iThisLayer, NNBuffer& bufInput0, NNLoopStream& stream ) ;
	// 予測処理
	virtual void Prediction
		( CPUWorkArray& bufWorks, Buffer& bufThis,
			const InputBuffer bufInput, NNLoopStream& stream ) ;
	virtual void cpuPrediction
		( CPUWorkArray& bufWorks, Buffer& bufThis,
			const InputBuffer bufInput, NNLoopStream& stream ) ;
	virtual void cudaPrediction
		( CPUWorkArray& bufWorks, Buffer& bufThis,
			const InputBuffer bufInput, NNLoopStream& stream ) ;
	// 損失計算
	virtual double cpuCalcLoss
		( CPUWorkArray& bufWorks, Buffer& bufThis,
			const NNBuffer& bufTeaching, NNLoopStream& stream ) ;
	// 出力を遅延バッファにコピー
	virtual void CopyToDelayBuffer( Buffer& bufThis, NNLoopStream& stream ) ;
	virtual void cpuCopyToDelayBuffer( Buffer& bufThis ) ;
	virtual void cudaCopyToDelayBuffer( Buffer& bufThis, NNLoopStream& stream ) ;
	// 遅延バッファへシフトコピー
	virtual void ShiftBufferWithStreaming
		( Buffer& bufThis, size_t xShift, NNLoopStream& stream ) ;
	virtual void cpuShiftBufferWithStreaming
		( Buffer& bufThis, size_t xShift ) ;
	virtual void cudaShiftBufferWithStreaming
		( Buffer& bufThis, size_t xShift, NNLoopStream& stream ) ;
	// 遅延バッファに教師データをコピー
	virtual void CopyTeachingDataToDelayBuffer
		( Buffer& bufThis, NNBuffer& bufTeaching, NNLoopStream& stream ) ;
	virtual void cpuCopyTeachingDataToDelayBuffer
		( Buffer& bufThis, const NNBuffer& bufTeaching ) ;
	virtual void cudaCopyTeachingDataToDelayBuffer
		( Buffer& bufThis, NNBuffer& bufTeaching, NNLoopStream& stream ) ;

public:
	// 逆伝播用バッファクリア
	virtual void ClearDeltaBuffer( Buffer& bufThis, NNLoopStream& stream ) ;
	virtual void cpuClearDeltaBuffer( Buffer& bufThis ) ;
	virtual void cudaClearDeltaBuffer( Buffer& bufThis, NNLoopStream& stream ) ;
	// 逆伝播用バッファを２パス用からコピー／又はクリア
	virtual void SwitchDeltaSecondaryBuffer( Buffer& bufThis, NNLoopStream& stream ) ;
	virtual void cpuSwitchDeltaSecondaryBuffer( Buffer& bufThis ) ;
	virtual void cudaSwitchDeltaSecondaryBuffer( Buffer& bufThis, NNLoopStream& stream ) ;
	// 損失関数δ計算
	virtual double LossDelta
		( CPUWorkArray& bufWorks, Buffer& bufThis,
			NNBuffer& bufTeaching, NNLoopStream& stream ) ;
	virtual double cpuLossDelta
		( CPUWorkArray& bufWorks, Buffer& bufThis,
			const NNBuffer& bufTeaching, NNLoopStream& stream ) ;
	virtual double cudaLossDelta
		( CPUWorkArray& bufWorks, Buffer& bufThis,
			NNBuffer& bufTeaching, NNLoopStream& stream ) ;
	// 活性化関数のδ逆伝播処理
	virtual void ActivationDeltaBack
		( CPUWorkArray& bufWorks, Buffer& bufThis, NNLoopStream& stream ) ;
	virtual void cpuActivationDeltaBack
		( CPUWorkArray& bufWorks, Buffer& bufThis, NNLoopStream& stream ) ;
	virtual void cudaActivationDeltaBack
		( CPUWorkArray& bufWorks, Buffer& bufThis, NNLoopStream& stream ) ;
	// 行列更新用勾配計算
	virtual void CalcMatrixGradient
		( CPUWorkArray& bufWorks, Buffer& bufThis,
			const InputBuffer bufInput, NNLoopStream& stream ) ;
	virtual void cpuCalcMatrixGradient
		( CPUWorkArray& bufWorks, Buffer& bufThis,
			const InputBuffer bufInput, NNLoopStream& stream ) ;
	virtual void cudaCalcMatrixGradient
		( CPUWorkArray& bufWorks, Buffer& bufThis,
			const InputBuffer bufInput, NNLoopStream& stream ) ;
	// 更新用行列勾配を統合する
	virtual void IntegrateMatrixGradient
		( CPUWorkArray& bufWorks, Buffer& bufThis, NNLoopStream& stream ) ;
	virtual void cpuIntegrateMatrixGradient
		( CPUWorkArray& bufWorks, Buffer& bufThis ) ;
	virtual void cudaIntegrateMatrixGradient
		( CPUWorkArray& bufWorks, Buffer& bufThis ) ;
	// 行列のδ逆伝播処理
	virtual void MatrixDeltaBack
		( CPUWorkArray& bufWorks, Buffer& bufThis, NNLoopStream& stream ) ;
	virtual void cpuMatrixDeltaBack
		( CPUWorkArray& bufWorks, Buffer& bufThis, NNLoopStream& stream ) ;
	virtual void cudaMatrixDeltaBack
		( CPUWorkArray& bufWorks, Buffer& bufThis, NNLoopStream& stream ) ;
	// 入力元レイヤーへδを逆伝播
	virtual void LayerDeltaBack
		( const BufferArray& bufArray,
			const Buffer& bufThis, NNLoopStream& stream ) ;
	// 入力元レイヤーへδを逆伝播（別の MLP 最終レイヤーへ損失関数δとして）
	virtual void LayerDeltaBackTo
		( Buffer& bufDst, const Buffer& bufThis, NNLoopStream& stream ) ;
	// 勾配を行列に更新する
	virtual void AddMatrixGradient
		( LossAndGradientBuf& laGradient,
			float deltaRate = 0.01f, float scaleGradient = 1.0f ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// Depthwise 用パーセプトロン
//		以下のような疎な行列の形（depthwise=3, bias=1 の例示）
//		[ w_11   0    0   w_14    0     0   ...   0   bias_1 ]
//		[   0  w_22   0     0   w_25    0   ...   0   bias_2 ]
//		[   0    0  w_33    0     0   w_36  ... w_3m  bias_3 ]
//		[ w_30   0    0   w_33    0     0   ...   0   bias_4 ]
//		[               ..                  ...   0   bias_i ]
//		[   0    0  w_n3    0     0   w_n6  ... w_nm  bias_n ]
//////////////////////////////////////////////////////////////////////////////

class	NNDepthwisePerceptron : public NNPerceptron
{
public:
	// 構築関数
	NNDepthwisePerceptron( void ) ;
	NNDepthwisePerceptron
		( size_t nDstCount, size_t nSrcCount,
			size_t nDepthwise, size_t nBias,
			std::shared_ptr<NNSamplingFilter> sampler,
			std::shared_ptr<NNActivationFunction> activation ) ;
	NNDepthwisePerceptron( const NNDepthwisePerceptron& dwp ) ;

	// 行列の特殊化
	virtual void Specialize( void ) ;

	// パーセプトロンタイプ
	static constexpr const char	PERCEPTRON_TYPE[] = "depthwise" ;
	virtual const char * GetPerceptronType( void ) const
	{
		return	PERCEPTRON_TYPE ;
	}

} ;



//////////////////////////////////////////////////////////////////////////////
// 固定値行列パーセプトロン基底
//////////////////////////////////////////////////////////////////////////////

class	NNFixedPerceptron : public NNPerceptron
{
public:
	// 構築関数
	NNFixedPerceptron( void ) ;
	NNFixedPerceptron
		( size_t nDstCount, size_t nSrcCount,
			size_t nDepthwise, size_t nBias,
			std::shared_ptr<NNSamplingFilter> sampler,
			std::shared_ptr<NNActivationFunction> activation ) ;
	NNFixedPerceptron( const NNFixedPerceptron& fxp ) ;

	// パーセプトロンタイプ
	static constexpr const char	PERCEPTRON_TYPE[] = "fixed" ;
	virtual const char * GetPerceptronType( void ) const
	{
		return	PERCEPTRON_TYPE ;
	}

public:
	// 勾配反映後のバッファ処理
	virtual void ResetBufferInBatch( Buffer& bufThis ) const ;
	// 行列更新用勾配計算
	virtual void CalcMatrixGradient
		( CPUWorkArray& bufWorks, Buffer& bufThis,
			const InputBuffer bufInput, NNLoopStream& stream ) ;
	virtual void cpuCalcMatrixGradient
		( CPUWorkArray& bufWorks, Buffer& bufThis,
			const InputBuffer bufInput, NNLoopStream& stream ) ;
	virtual void cudaCalcMatrixGradient
		( CPUWorkArray& bufWorks, Buffer& bufThis,
			const InputBuffer bufInput, NNLoopStream& stream ) ;
	// 更新用行列勾配を統合する
	virtual void IntegrateMatrixGradient
		( CPUWorkArray& bufWorks, Buffer& bufThis, NNLoopStream& stream ) ;
	virtual void cpuIntegrateMatrixGradient
		( CPUWorkArray& bufWorks, Buffer& bufThis ) ;
	virtual void cudaIntegrateMatrixGradient
		( CPUWorkArray& bufWorks, Buffer& bufThis ) ;
} ;



//////////////////////////////////////////////////////////////////////////////
// 単位行列パーセプトロン
//		単位行列 I を含む以下のような行列の形
//		                           [  I  ]
//		                           [  I  ]
//		[ I  I  I ... I ]   又は   [  I  ]
//		                           [ ... ]
//		                           [  I  ]
// ※depthwise は行列の形には影響しない（活性化関数のパラメータとしてのみ機能）
//////////////////////////////////////////////////////////////////////////////

class	NNIdentityPerceptron : public NNFixedPerceptron
{
public:
	// 構築関数
	NNIdentityPerceptron( void ) ;
	NNIdentityPerceptron
		( size_t nDstCount, size_t nSrcCount, size_t nDepthwise,
			std::shared_ptr<NNSamplingFilter> sampler,
			std::shared_ptr<NNActivationFunction> activation ) ;
	NNIdentityPerceptron( const NNIdentityPerceptron& idp ) ;

	// 対角化単位
	virtual size_t GetDepthwise( void ) const ;
	// 行列の特殊化
	virtual void Specialize( void ) ;

	// パーセプトロンタイプ
	static constexpr const char	PERCEPTRON_TYPE[] = "identity" ;
	virtual const char * GetPerceptronType( void ) const
	{
		return	PERCEPTRON_TYPE ;
	}

} ;



}

#endif

