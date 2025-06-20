
#ifndef	__NN_NORMALIZATION_H__
#define	__NN_NORMALIZATION_H__

#include <vector>
#include <map>
#include <functional>
#include <random>
#include "nn_buffer.h"
#include "nn_serializer.h"
#include "nn_loop_stream.h"

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// 正規化基底
//////////////////////////////////////////////////////////////////////////////

class	NNNormalizationFilter
{
public:
	struct	Parameter
	{
		float	scale ;
		float	shift ;
	} ;

	struct	Aggregation
	{
		float	sum ;		// Σx(i)
		float	sum2 ;		// Σx(i)^2
		float	num ;		// N
	} ;

	struct	MeanAndVariance
	{
		float	mean ;			// 平均 μ = sum/N
		//float	variance ;		// 分散 σ^2 = sum2/N - μ^2
		float	rcpvar ;		// 分散逆数 1 / (σ^2 + ϵ)
	} ;

	struct	Gradient	: public Parameter
	{
		size_t	num ;		// 合計サンプル数
	} ;

	struct	GradientBuf
	{
		std::vector<Gradient>	vecGradients ;

		void ResetGradient( void ) ;
		void AddGradient( const GradientBuf& bufSrc ) ;
	} ;

	struct	WorkBuf
	{
		size_t							nThreads ;		// スレッド数
		size_t							nChannels ;		// 分布の数
		bool							forLearning ;
		bool							transParam ;
		NNBuffer						bufParameter ;
		NNBuffer						bufVariance ;
		size_t							iAggregate ;
		NNBufDim						dimAggregate ;
		NNBuffer						bufAggregation[3] ;
		NNBuffer						bufGradient ;
		NNBufDim						dimSample ;
		std::vector<MeanAndVariance>	vecVariance ;
		std::vector<Aggregation>		vecAggregation ;
		std::vector<Gradient>			vecGradients ;

		size_t GetBufferBytes( void ) const ;
		size_t GetCudaBufferBytes( void ) const ;
	} ;

	enum	HyperparamFlags
	{
		flagZeroBias	= 0x00000001,
	} ;
	struct	Hyperparameter
	{
		uint32_t	flags ;			// enum HyperparamFlags
		float		alpha ;			// [0,1) : 集計値のミニバッチ毎のスケール
		float		beta ;			// [0,1) : 集計値のエポック毎のスケール
		float		delta ;			// アフィン・パラメータ学習速度 : η*delta + deltac
		float		deltac ;

		Hyperparameter( void )
			: flags(0), alpha(0.9f), beta(0.5f),
					delta(1.0f), deltac(0.00001f) { }
	} ;

	std::vector<Parameter>		m_params ;		// アフィン・パラメータ
	std::vector<Aggregation>	m_aggregation ;	// 集計値
	Hyperparameter				m_hyparam ;		// ハイパー・パラメータ
	std::vector<uint32_t>		m_vecIndices ;	// チャネル指標 → 分布指標

protected:
	static std::map
		< std::string,
			std::function
				< std::shared_ptr
					<NNNormalizationFilter>() > >	s_mapMakeFilter ;

public:
	// 関数生成準備
	static void InitMake( void ) ;
	// 関数生成
	static std::shared_ptr<NNNormalizationFilter> Make( const char * pszName ) ;
	// 登録
	template <class T> static void Register( void )
	{
		s_mapMakeFilter.insert
			( std::make_pair(std::string(T::FilterName),
								[]() { return std::make_shared<T>(); } ) ) ;
	}

public:
	// 構築関数
	NNNormalizationFilter( void ) ;
	NNNormalizationFilter( const Hyperparameter& hyparam ) ;
	// フィルタ名
	virtual const char * GetFilterName( void) const = 0 ;
	// ハイパーパラメータ
	const Hyperparameter& GetHyperparameter( void ) const ;
	void SetHyperparameter( const Hyperparameter& hyparam ) ;
	// 生成
	virtual void CreateFilter( size_t zChannels ) ;
	// アフィンパラメータ乗算
	void ScaleParameter( float scale ) ;
	// 作業バッファ準備
	virtual void PrepareWorkBuf
		( WorkBuf& bufWork, const NNBufDim& dimSample,
			bool forLearning, const NNLoopStream& stream ) const ;
	virtual void PrepareGradBuf( GradientBuf& bufGrad ) const ;
	// 正規化チャネル数
	virtual size_t NormalizeChannelCount( size_t zChannels ) const ;
	// 標本範囲チャネル数
	virtual size_t SamplingChannels( size_t zChannels ) const = 0 ;
	// 作業バッファ初期化
	virtual void ResetWorkBuf( WorkBuf& bufWork ) const ;
	// 分布を計算して正規化
	virtual void cpuNormalize
		( NNBuffer& bufSample, WorkBuf& bufWork,
			NNLoopStream& stream, size_t xLeftBounds = 0 ) const ;
	virtual void cudaNormalize
		( NNBuffer& bufSample, WorkBuf& bufWork,
			NNLoopStream& stream, size_t xLeftBounds = 0 ) const ;
	// サンプルを集計
	virtual void cpuAggregateSample
		( WorkBuf& bufWork, const NNBuffer& bufSample, NNLoopStream& stream ) ;
	virtual void cudaAggregateSample
		( WorkBuf& bufWork, const NNBuffer& bufSample, NNLoopStream& stream ) ;
	// δ逆伝播と勾配計算
	virtual void cpuDeltaBack
		( WorkBuf& bufWork, NNBuffer& bufDelta,
			const NNBuffer& bufDstSample, NNLoopStream& stream ) const ;
	virtual void cudaDeltaBack
		( WorkBuf& bufWork, NNBuffer& bufDelta,
			const NNBuffer& bufDstSample, NNLoopStream& stream ) const ;
	// 更新用行列勾配を統合する
	virtual void cpuIntegrateGradient
		( GradientBuf& bufGrad, WorkBuf& bufWork ) ;
	virtual void cudaIntegrateGradient
		( GradientBuf& bufGrad, WorkBuf& bufWork ) ;
	// 勾配をパラメータに更新する
	virtual void AddGradient
		( const GradientBuf& bufWork,
			float deltaRate = 0.01f, float l2reg = 0.0f ) ;
	// エポック開始時の集計データスケーリング
	virtual void OnBeginEpoch( void ) ;
	// エポック終了時
	virtual void OnEndEpoch( WorkBuf& bufWork ) ;
	// 集計値のスケーリング
	virtual void ScaleAggregate( float scale ) ;
	// シリアライズ
	virtual void Serialize( NNSerializer& ser ) ;
	// デシリアライズ
	virtual bool Deserialize( NNDeserializer & dsr ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// レイヤー正規化
//////////////////////////////////////////////////////////////////////////////

class	NNLayerNormalization	: public NNNormalizationFilter
{
public:
	constexpr static const char		FilterName[] = "layer" ;

	// 構築関数
	NNLayerNormalization( void ) {}
	NNLayerNormalization( const Hyperparameter& hyparam )
		: NNNormalizationFilter( hyparam ) { }
	// フィルタ名
	virtual const char * GetFilterName( void) const
	{
		return	FilterName ;
	}
	// 標本範囲チャネル数
	virtual size_t SamplingChannels( size_t zChannels ) const ;

} ;



//////////////////////////////////////////////////////////////////////////////
// グループ正規化
//////////////////////////////////////////////////////////////////////////////

class	NNGroupNormalization	: public NNLayerNormalization
{
public:
	size_t	m_zSampling ;

public:
	constexpr static const char		FilterName[] = "group" ;

	// 構築関数
	NNGroupNormalization( size_t zSampling = 2 )
		: m_zSampling( zSampling )
	{
		assert( m_zSampling >= 1 ) ;
	}
	NNGroupNormalization( const Hyperparameter& hyparam, size_t zSampling = 2 )
		: NNLayerNormalization( hyparam ), m_zSampling( zSampling )
	{
		assert( m_zSampling >= 1 ) ;
	}
	// フィルタ名
	virtual const char * GetFilterName( void) const
	{
		return	FilterName ;
	}
	// 標本範囲チャネル数
	virtual size_t SamplingChannels( size_t zChannels ) const ;
	// シリアライズ
	virtual void Serialize( NNSerializer& ser ) ;
	// デシリアライズ
	virtual bool Deserialize( NNDeserializer & dsr ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// インスタンス正規化
//////////////////////////////////////////////////////////////////////////////

class	NNInstanceNormalization	: public NNGroupNormalization
{
public:
	constexpr static const char		FilterName[] = "instance" ;

	// 構築関数
	NNInstanceNormalization( void )
		: NNGroupNormalization( 1 ) {}
	NNInstanceNormalization( const Hyperparameter& hyparam )
		: NNGroupNormalization( hyparam, 1 ) {}
	// フィルタ名
	virtual const char * GetFilterName( void) const
	{
		return	FilterName ;
	}

} ;


}

#endif

