
#include "nn_multi_layer.h"

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// マルチ・レイヤー・パーセプトロン
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNMultiLayerPerceptron::NNMultiLayerPerceptron( void )
	: m_flagsMLP( 0 ),
		m_dimInShape( 0, 0, 0 ),
		m_dimInUnit( 0, 0, 0 )
{
}

// データ初期化
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::ClearAll( void )
{
	m_flagsMLP = 0 ;
	m_dimInShape = NNBufDim( 0, 0, 0 ) ;
	m_dimInUnit = NNBufDim( 0, 0, 0 ) ;

	RemoveAllLayers() ;
}

// 入力サイズ設定
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::SetInputShape
	( uint32_t flagsMLP,
		const NNBufDim& dimShape, const NNBufDim& dimUnit )
{
	m_flagsMLP = flagsMLP ;
	m_dimInShape = dimShape ;
	m_dimInUnit = dimUnit ;
}

// フラグ取得
//////////////////////////////////////////////////////////////////////////////
uint32_t NNMultiLayerPerceptron::GetMLPFlags( void ) const
{
	return	m_flagsMLP ;
}

// 入力サイズ取得
//////////////////////////////////////////////////////////////////////////////
const NNBufDim& NNMultiLayerPerceptron::GetInputShape( void ) const
{
	return	m_dimInShape ;
}

const NNBufDim& NNMultiLayerPerceptron::GetInputUnit( void ) const
{
	return	m_dimInUnit ;
}

// 学習の前に事前予測処理が必要な回数
//////////////////////////////////////////////////////////////////////////////
size_t NNMultiLayerPerceptron::CountOfPrePrediction( void ) const
{
	if ( (m_flagsMLP & mlpFlagStream)
		&& (m_dimInShape.x != 0) && (m_dimInUnit.x != 0) )
	{
		return	max( m_dimInShape.x / m_dimInUnit.x, 1 ) ;
	}
	return	1 ;
}

// レイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNMultiLayerPerceptron::AppendLayer
	( size_t nDstChannels, size_t nSrcChannels, size_t nBias,
		const char * pszActivation, std::shared_ptr<NNSamplingFilter> sampler )
{
	if ( sampler == nullptr )
	{
		sampler = std::make_shared<NNSamplerInjection>() ;
	}
	return	AppendLayer
		( nDstChannels, nSrcChannels, nBias,
			NNActivationFunction::Make( pszActivation ), sampler ) ;
}

NNPerceptronPtr
	NNMultiLayerPerceptron::AppendLayer
		( size_t nDstChannels, size_t nSrcChannels, size_t nBias,
			std::shared_ptr<NNActivationFunction> activation,
			std::shared_ptr<NNSamplingFilter> sampler )
{
	if ( activation == nullptr )
	{
		activation = std::make_shared<NNActivationLinear>() ;
	}
	if ( sampler == nullptr )
	{
		sampler = std::make_shared<NNSamplerInjection>() ;
	}
	NNPerceptronPtr	pLayer =
		std::make_shared<NNPerceptron>
			( nDstChannels, nSrcChannels, 1, nBias, sampler, activation ) ;
	AppendLayer( pLayer ) ;
	return	pLayer ;
}

size_t NNMultiLayerPerceptron::AppendLayer( NNPerceptronPtr pLayer )
{
	size_t	iLayer = m_mlp.size() ;
	m_mlp.push_back( pLayer ) ;
	return	iLayer ;
}

size_t NNMultiLayerPerceptron::InsertLayer( size_t iLayer, NNPerceptronPtr pLayer )
{
	if ( iLayer > m_mlp.size() )
	{
		iLayer = m_mlp.size() ;
	}
	m_mlp.insert( m_mlp.begin() + iLayer, pLayer ) ;
	return	iLayer ;
}

// 畳み込みレイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNMultiLayerPerceptron::AppendConvLayer
	( size_t nDstChannels, size_t nSrcChannels,
		int xConv, int yConv, ConvolutionPadding padding, size_t nBias,
		const char * pszActivation,
		int xStride, int yStride, int xOffset, int yOffset )
{
	return	AppendConvLayer
		( nDstChannels, nSrcChannels,
			xConv, yConv, padding, nBias,
			NNActivationFunction::Make( pszActivation ),
			xStride, yStride, xOffset, yOffset ) ;
}

NNPerceptronPtr NNMultiLayerPerceptron::AppendConvLayer
	( size_t nDstChannels, size_t nSrcChannels,
		int xConv, int yConv, ConvolutionPadding padding, size_t nBias,
		std::shared_ptr<NNActivationFunction> activation,
		int xStride, int yStride, int xOffset, int yOffset )
{
	std::shared_ptr<NNSamplingFilter>	sampler ;
	switch ( padding )
	{
	case	convPadBorder:
		sampler = std::make_shared<NNConvClampFilter>
					( xConv, yConv, true,
							xStride, yStride, xOffset, yOffset ) ;
		break ;
	case	convNoPad:
	case	convPadZero:
	default:
		sampler = std::make_shared<NNConvEdgeFilter>
					( xConv, yConv, (padding != convNoPad),
							xStride, yStride, xOffset, yOffset ) ;
		break ;
	case	convNoPad_sparse:
	case	convPadZero_sparse:
		sampler = std::make_shared<NNSparseConvFilter>
					( xConv, yConv, (padding != convNoPad_sparse),
							xStride, yStride, xOffset, yOffset ) ;
		break ;
	}
	return	AppendLayer
		( nDstChannels, nSrcChannels * xConv * yConv, nBias, activation, sampler ) ;
}

// チャネル毎の畳み込みレイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNMultiLayerPerceptron::AppendDepthwiseConv
	( size_t nDstChannels, size_t nSrcChannels, size_t nDepthwise,
		int xConv, int yConv, ConvolutionPadding padding, size_t nBias,
		const char * pszActivation,
		int xStride, int yStride, int xOffset, int yOffset )
{
	return	AppendDepthwiseConv
		( nDstChannels, nSrcChannels, nDepthwise,
			xConv, yConv, padding, nBias,
			NNActivationFunction::Make( pszActivation ),
			xStride, yStride, xOffset, yOffset ) ;
}

NNPerceptronPtr NNMultiLayerPerceptron::AppendDepthwiseConv
	( size_t nDstChannels, size_t nSrcChannels, size_t nDepthwise,
		int xConv, int yConv, ConvolutionPadding padding, size_t nBias,
		std::shared_ptr<NNActivationFunction> activation,
		int xStride, int yStride, int xOffset, int yOffset )
{
	assert( (nSrcChannels % nDepthwise) == 0 ) ;
	if ( activation == nullptr )
	{
		activation = std::make_shared<NNActivationLinear>() ;
	}
	std::shared_ptr<NNSamplingFilter>	sampler ;
	if ( padding == convPadBorder )
	{
		sampler = std::make_shared<NNConvClampFilter>
					( xConv, yConv, true,
							xStride, yStride, xOffset, yOffset ) ;
	}
	else
	{
		sampler = std::make_shared<NNConvEdgeFilter>
					( xConv, yConv, (padding != convNoPad),
							xStride, yStride, xOffset, yOffset ) ;
	}
	NNPerceptronPtr	pLayer =
		std::make_shared<NNDepthwisePerceptron>
			( nDstChannels, nSrcChannels * xConv * yConv,
					nDepthwise, nBias, sampler, activation ) ;
	AppendLayer( pLayer ) ;
	return	pLayer ;
}

// 疎行列な結合レイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr
	NNMultiLayerPerceptron::AppendDepthwiseLayer
		( size_t nDstChannels, size_t nSrcChannels,
			size_t nDepthwise, size_t nBias, const char * pszActivation )
{
	return	AppendDepthwiseLayer
		( nDstChannels, nSrcChannels, nDepthwise, nBias,
			NNActivationFunction::Make( pszActivation ) ) ;
}

NNPerceptronPtr NNMultiLayerPerceptron::AppendDepthwiseLayer
	( size_t nDstChannels, size_t nSrcChannels,
		size_t nDepthwise, size_t nBias,
		std::shared_ptr<NNActivationFunction> activation,
		std::shared_ptr<NNSamplingFilter> sampler )
{
	if ( activation == nullptr )
	{
		activation = std::make_shared<NNActivationLinear>() ;
	}
	if ( sampler == nullptr )
	{
		sampler = std::make_shared<NNSamplerInjection>() ;
	}
	NNPerceptronPtr	pLayer =
		std::make_shared<NNDepthwisePerceptron>
			( nDstChannels, nSrcChannels, nDepthwise, nBias, sampler, activation ) ;
	AppendLayer( pLayer ) ;
	return	pLayer ;
}

// アップサンプリング・レイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNMultiLayerPerceptron::AppendUpsamplingLayer
	( size_t nDstChannels, size_t nSrcChannels,
		int xUpsampling, int yUpsampling, size_t nBias, const char * pszActivation )
{
	return	AppendUpsamplingLayer
				( nDstChannels, nSrcChannels,
					xUpsampling, yUpsampling, nBias, 
					NNActivationFunction::Make( pszActivation ) ) ;
}

NNPerceptronPtr NNMultiLayerPerceptron::AppendUpsamplingLayer
	( size_t nDstChannels, size_t nSrcChannels,
		int xUpsampling, int yUpsampling, size_t nBias,
		std::shared_ptr<NNActivationFunction> activation )
{
	if ( activation == nullptr )
	{
		activation = std::make_shared<NNActivationLinear>() ;
	}
	NNPerceptronPtr	pLayer =
		std::make_shared<NNPerceptron>
			( nDstChannels * (xUpsampling*yUpsampling), nSrcChannels, 1, nBias,
				std::make_shared<NNSamplerUpSampler>(xUpsampling,yUpsampling), activation ) ;
	AppendLayer( pLayer ) ;
	return	pLayer ;
}

NNPerceptronPtr NNMultiLayerPerceptron::AppendUp2x2Layer
	( size_t nDstChannels, size_t nSrcChannels, size_t nBias, const char * pszActivation )
{
	return	AppendUp2x2Layer
				( nDstChannels, nSrcChannels, nBias, 
					NNActivationFunction::Make( pszActivation ) ) ;
}

NNPerceptronPtr NNMultiLayerPerceptron::AppendUp2x2Layer
	( size_t nDstChannels, size_t nSrcChannels, size_t nBias,
			std::shared_ptr<NNActivationFunction> activation )
{
	if ( activation == nullptr )
	{
		activation = std::make_shared<NNActivationLinear>() ;
	}
	NNPerceptronPtr	pLayer =
		std::make_shared<NNPerceptron>
			( nDstChannels * (2*2), nSrcChannels, 1, nBias,
				std::make_shared<NNSamplerUp2x2>(), activation ) ;
	AppendLayer( pLayer ) ;
	return	pLayer ;
}

// アップサンプリング（パススルー）レイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNMultiLayerPerceptron::AppendUpsamplingFixLayer
	( size_t nDstChannels, size_t nSrcChannels,
		int xUpsampling, int yUpsampling, const char * pszActivation )
{
	std::shared_ptr<NNActivationFunction>
			activation = NNActivationFunction::Make( pszActivation ) ;
	if ( activation == nullptr )
	{
		activation = std::make_shared<NNActivationLinear>() ;
	}
	NNPerceptronPtr	pLayer =
		std::make_shared<NNIdentityPerceptron>
			( nDstChannels * (xUpsampling*yUpsampling), nSrcChannels, nSrcChannels,
				std::make_shared<NNSamplerUpSampler>(xUpsampling,yUpsampling), activation ) ;
	AppendLayer( pLayer ) ;
	return	pLayer ;
}

// One-Hot 相当１チャネル（インデックス値）入力レイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNMultiLayerPerceptron::AppendLayerAsOneHot
	( size_t nDstChannels, size_t nSrcChannels, const char * pszActivation )
{
	return	AppendLayer
		( nDstChannels, nSrcChannels, 0,
			pszActivation, std::make_shared<NNSamplerOneHot>() ) ;
}

// Softmax 高速化用レイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNMultiLayerPerceptron::AppendFastSoftmax
	( size_t nDstChannels, size_t nSrcChannels, size_t nBias, const char * pszActivation )
{
	return	AppendLayer
		( nDstChannels, nSrcChannels, nBias,
			pszActivation, std::make_shared<NNSamplerSparse>() ) ;
}

// チャネル毎の MaxPool レイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNMultiLayerPerceptron::AppendMaxPoolLayer
	( size_t nSrcChannels, size_t xPoolWidth, size_t yPoolHeight )
{
	NNPerceptronPtr	pLayer =
		std::make_shared<NNIdentityPerceptron>
			( nSrcChannels * xPoolWidth * yPoolHeight,
				nSrcChannels * xPoolWidth * yPoolHeight, nSrcChannels,
				std::make_shared<NNConvClampFilter>
					( (int) xPoolWidth, (int) yPoolHeight, false,
						(int) xPoolWidth, (int) yPoolHeight ),
				std::make_shared<NNActivationMaxPool>() ) ;
	AppendLayer( pLayer ) ;
	return	pLayer ;
}

// ゲート付き活性化関数レイヤー追加
//////////////////////////////////////////////////////////////////////////////
std::tuple< NNPerceptronPtr,
			NNPerceptronPtr,
			NNPerceptronPtr >
	NNMultiLayerPerceptron::AppendGatedLayer
		( size_t nDstChannels, size_t nSrcChannels, size_t nBias,
			const char * pszActivation, const char * pszGateActivation,
			const NNLayerConnections lcInputs )
{
	std::shared_ptr<NNActivationFunction>
			activation = NNActivationFunction::Make( pszActivation ) ;
	if ( activation == nullptr )
	{
		activation = std::make_shared<NNActivationLinear>() ;
	}
	std::shared_ptr<NNActivationFunction>
			gate = NNActivationFunction::Make( pszGateActivation ) ;
	if ( gate == nullptr )
	{
		gate = std::make_shared<NNActivationSigmoid>() ;
	}
	NNPerceptronPtr	pSigLayer =
		std::make_shared<NNPerceptron>
			( nDstChannels, nSrcChannels, 1, nBias,
				std::make_shared<NNSamplerInjection>(), gate ) ;
	NNPerceptronPtr	pLayer =
		std::make_shared<NNPerceptron>
			( nDstChannels, nSrcChannels, 1, nBias,
				std::make_shared<NNSamplerInjection>(), activation ) ;

	AppendLayer( pSigLayer ) ;
	AppendLayer( pLayer ) ;

	if ( lcInputs.size() >= 1 )
	{
		SetLayerInput( pSigLayer, lcInputs ) ;
		SetLayerInput( pLayer, lcInputs ) ;
	}
	else
	{
		pLayer->AddConnection( 2, 0, 0, nSrcChannels ) ;
	}

	NNPerceptronPtr	pGateLayer =
		AppendPointwiseMul( nDstChannels, 1, 0, 2, 0 ) ;

	return	std::make_tuple( pGateLayer, pLayer, pSigLayer ) ;
}

// チャネル毎の加算結合レイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr
	NNMultiLayerPerceptron::AppendPointwiseAdd
		( size_t nDstChannels,
			int iLayerOffset1, int iDelay1,
			int iLayerOffset2, int iDelay2,
			int xOffset2, int yOffset2,
			const char * pszActivation )
{
	std::shared_ptr<NNActivationFunction>
			activation = NNActivationFunction::Make( pszActivation ) ;
	if ( activation == nullptr )
	{
		activation = std::make_shared<NNActivationLinear>() ;
	}
	NNPerceptronPtr	pLayer =
		std::make_shared<NNIdentityPerceptron>
			( nDstChannels, nDstChannels * 2, nDstChannels,
				std::make_shared<NNSamplerInjection>(), activation ) ;
	AppendLayer( pLayer ) ;

	pLayer->AddConnection
		( iLayerOffset1, iDelay1, 0, nDstChannels ) ;
	pLayer->AddConnection
		( iLayerOffset2, iDelay2, 0, nDstChannels, xOffset2, yOffset2 ) ;

	return	pLayer ;
}

NNPerceptronPtr
	NNMultiLayerPerceptron::AppendPointwiseAdd
		( size_t nDstChannels,
			NNPerceptronPtr pLayer1, int iDelay1,
			NNPerceptronPtr pLayer2, int iDelay2,
			int xOffset2, int yOffset2,
			const char * pszActivation )
{
	std::shared_ptr<NNActivationFunction>
			activation = NNActivationFunction::Make( pszActivation ) ;
	if ( activation == nullptr )
	{
		activation = std::make_shared<NNActivationLinear>() ;
	}
	NNPerceptronPtr	pLayer =
		std::make_shared<NNIdentityPerceptron>
			( nDstChannels, nDstChannels * 2, nDstChannels,
				std::make_shared<NNSamplerInjection>(), activation ) ;
	AppendLayer( pLayer ) ;

	pLayer->AddConnection
		( LayerOffsetOf(pLayer1), iDelay1, 0, nDstChannels ) ;
	pLayer->AddConnection
		( LayerOffsetOf(pLayer2), iDelay2, 0, nDstChannels, xOffset2, yOffset2 ) ;

	return	pLayer ;
}

// チャネル毎の乗算結合レイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr
	NNMultiLayerPerceptron::AppendPointwiseMul
		( size_t nDstChannels,
			int iLayerOffset1, int iDelay1,
			int iLayerOffset2, int iDelay2, int xOffset2, int yOffset2 )
{
	NNPerceptronPtr	pLayer =
		std::make_shared<NNIdentityPerceptron>
			( nDstChannels * 2, nDstChannels * 2, nDstChannels,
				std::make_shared<NNSamplerInjection>(),
				std::make_shared<NNActivationMultiply>() ) ;
	AppendLayer( pLayer ) ;

	pLayer->AddConnection
		( iLayerOffset1, iDelay1, 0, nDstChannels ) ;
	pLayer->AddConnection
		( iLayerOffset2, iDelay2, 0, nDstChannels, xOffset2, yOffset2 ) ;

	return	pLayer ;
}

NNPerceptronPtr
	NNMultiLayerPerceptron::AppendPointwiseMul
		( size_t nDstChannels,
			NNPerceptronPtr pLayer1, int iDelay1,
			NNPerceptronPtr pLayer2, int iDelay2, int xOffset2, int yOffset2 )
{
	NNPerceptronPtr	pLayer =
		std::make_shared<NNIdentityPerceptron>
			( nDstChannels * 2, nDstChannels * 2, nDstChannels,
				std::make_shared<NNSamplerInjection>(),
				std::make_shared<NNActivationMultiply>() ) ;
	AppendLayer( pLayer ) ;

	pLayer->AddConnection
		( LayerOffsetOf(pLayer1), iDelay1, 0, nDstChannels ) ;
	pLayer->AddConnection
		( LayerOffsetOf(pLayer2), iDelay2, 0, nDstChannels, xOffset2, yOffset2 ) ;

	return	pLayer ;
}

// レイヤー数取得
//////////////////////////////////////////////////////////////////////////////
size_t NNMultiLayerPerceptron::GetLayerCount( void ) const
{
	return	m_mlp.size() ;
}

// レイヤー取得
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNMultiLayerPerceptron::GetLayerAt( size_t iLayer ) const
{
	assert( iLayer < m_mlp.size() ) ;
	return	m_mlp.at( iLayer ) ;
}

NNPerceptronPtr NNMultiLayerPerceptron::GetLayerAs( const char * pszId ) const
{
	int	iLayer = FindLayerAs( pszId ) ;
	if ( iLayer < 0 )
	{
		return	nullptr ;
	}
	return	GetLayerAt( (size_t) iLayer ) ;
}

// レイヤー削除
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNMultiLayerPerceptron::RemoveLayerAt( size_t iLayer )
{
	assert( iLayer < m_mlp.size() ) ;
	NNPerceptronPtr	pRemove ;
	if ( iLayer < m_mlp.size() )
	{
		pRemove = m_mlp.at( iLayer ) ;
		m_mlp.erase( m_mlp.begin() + iLayer ) ;
	}
	return	pRemove ;
}

NNPerceptronPtr NNMultiLayerPerceptron::RemoveLayerOf( NNPerceptronPtr pLayer )
{
	int	iLayer = FindLayer( pLayer ) ;
	if ( iLayer < 0 )
	{
		return	nullptr ;
	}
	return	RemoveLayerAt( (size_t) iLayer ) ;
}

void NNMultiLayerPerceptron::RemoveAllLayers( void )
{
	m_mlp.clear() ;
}

// レイヤー入力情報設定
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::SetLayerInput
	( NNPerceptronPtr pLayer, const NNLayerConnections lcInputs )
{
	assert( pLayer != nullptr ) ;
	assert( FindLayer(pLayer) >= 0 ) ;
	const int	iLayer = FindLayer( pLayer ) ;
	if ( iLayer < 0 )
	{
		return ;
	}
	pLayer->ClearAllConnection() ;

	for ( size_t i = 0; i < lcInputs.size(); i ++ )
	{
		const NNLayerConnection&	lc = lcInputs.at(i) ;
		if ( lc.m_pFrom != nullptr )
		{
			assert( FindLayer(lc.m_pFrom) >= 0 ) ;
			assert( lc.iChannel + lc.nChannels <= lc.m_pFrom->CalcOutputChannels() ) ;
			const int	iFrom = FindLayer( lc.m_pFrom ) ;
			pLayer->AddConnection
				( iLayer - iFrom, lc.iDelay, lc.iChannel,
					(lc.nChannels != 0) ? lc.nChannels
						: (lc.m_pFrom->CalcOutputChannels() - lc.iChannel) ) ;
		}
		else
		{
			pLayer->AddConnection
				( iLayer + 1, lc.iDelay, lc.iChannel, lc.nChannels ) ;
		}
	}
}

// レイヤー検索（見つからない場合 -1 ）
//////////////////////////////////////////////////////////////////////////////
int NNMultiLayerPerceptron::FindLayer( NNPerceptronPtr pLayer ) const
{
	return	FindLayer( pLayer.get() ) ;
}

int NNMultiLayerPerceptron::FindLayer( NNPerceptron * pLayer ) const
{
	if ( pLayer == nullptr )
	{
		return	-1 ;
	}
	for ( size_t i = 0; i < m_mlp.size(); i ++ )
	{
		if ( m_mlp.at(i).get() == pLayer )
		{
			return	(int) i ;
		}
	}
	return	-1 ;
}

int NNMultiLayerPerceptron::FindLayerAs( const char * pszId, size_t iFirst ) const
{
	for ( size_t i = iFirst; i < m_mlp.size(); i ++ )
	{
		if ( m_mlp.at(i)->GetIdentity() == pszId )
		{
			return	(int) i ;
		}
	}
	return	-1 ;
}

// レイヤー検索（最終レイヤーから pLayer へ AddConnect する時のレイヤーオフセット）
//////////////////////////////////////////////////////////////////////////////
int NNMultiLayerPerceptron::LayerOffsetOf( NNPerceptronPtr pLayer ) const
{
	return	LayerOffsetOf( pLayer.get() ) ;
}

int NNMultiLayerPerceptron::LayerOffsetOf( NNPerceptron* pLayer ) const
{
	return	(int) m_mlp.size() - 1 - FindLayer( pLayer ) ;
}

// レイヤー検索（pFromLayer から pLayer へ AddConnect する時のレイヤーオフセット）
// （pLayer == nullptr の時には、pLayer は入力データ）
//////////////////////////////////////////////////////////////////////////////
int NNMultiLayerPerceptron::LayerOffsetFrom
	( NNPerceptronPtr pLayer, NNPerceptronPtr pFromLayer ) const
{
	return	LayerOffsetFrom( pLayer.get(), pFromLayer.get() ) ;
}

int NNMultiLayerPerceptron::LayerOffsetFrom
	( NNPerceptron* pLayer, NNPerceptron* pFromLayer ) const
{
	assert( FindLayer( pFromLayer ) >= 0 ) ;
	return	FindLayer( pFromLayer ) - FindLayer( pLayer ) ;
}

// シリアライズ
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::Serialize( NNSerializer& ser )
{
	FileHeader	fhdr ;
	memset( &fhdr, 0, sizeof(fhdr) ) ;
	fhdr.nLayerCount = (uint32_t) GetLayerCount() ;
	fhdr.flagsMLP = m_flagsMLP ;
	fhdr.dimInShape.x = (uint32_t) m_dimInShape.x ;
	fhdr.dimInShape.y = (uint32_t) m_dimInShape.y ;
	fhdr.dimInShape.z = (uint32_t) m_dimInShape.z ;
	fhdr.dimInUnit.x = (uint32_t) m_dimInUnit.x ;
	fhdr.dimInUnit.y = (uint32_t) m_dimInUnit.y ;
	fhdr.dimInUnit.z = (uint32_t) m_dimInUnit.z ;
	//
	ser.Descend( CHHDRID_HEADER ) ;
	ser.Write( &fhdr, sizeof(fhdr) ) ;
	ser.Ascend() ;

	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		NNPerceptronPtr	pLayer = GetLayerAt( i ) ;
		ser.WriteString( pLayer->GetPerceptronType() ) ;
		pLayer->Serialize( ser ) ;
	}
}

// デシリアライズ
//////////////////////////////////////////////////////////////////////////////
bool NNMultiLayerPerceptron::Deserialize( NNDeserializer & dsr )
{
	ClearAll() ;

	if ( dsr.Descend( CHHDRID_HEADER ) != CHHDRID_HEADER )
	{
		return	false ;
	}
	FileHeader	fhdr ;
	memset( &fhdr, 0, sizeof(fhdr) ) ;
	dsr.Read( &fhdr, sizeof(fhdr) ) ;
	dsr.Ascend() ;

	m_flagsMLP = fhdr.flagsMLP ;
	m_dimInShape = NNBufDim( fhdr.dimInShape.x, fhdr.dimInShape.y, fhdr.dimInShape.z ) ;
	m_dimInUnit = NNBufDim( fhdr.dimInUnit.x, fhdr.dimInUnit.y, fhdr.dimInUnit.z ) ;

	for ( size_t i = 0; i < fhdr.nLayerCount; i ++ )
	{
		std::string		strType = dsr.ReadString() ;
		NNPerceptronPtr	pLayer = NNPerceptron::Make( strType.c_str() ) ;
		if ( pLayer == nullptr )
		{
			return	false ;
		}
		if ( !pLayer->Deserialize( dsr ) )
		{
			return	false ;
		}
		AppendLayer( pLayer ) ;
	}
	return	true ;
}

// バッファ準備
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::PrepareBuffer
	( NNMultiLayerPerceptron::BufferArrays& bufArrays,
		const NNBufDim& dimInput, uint32_t flagsBuffer,
		const NNMultiLayerPerceptron::BufferConfig& bufConfig )
{
	// CUDA ストリーム／CPU マルチスレッド処理の準備
	if ( cudaIsAvailable() && bufConfig.flagUseCuda )
	{
		bufArrays.stream.m_useCuda = true ;
		if ( !bufArrays.stream.m_cudaStream.IsCreated() )
		{
			bufArrays.stream.m_cudaStream.Create() ;
		}
	}
	else
	{
		bufArrays.stream.m_useCuda = false ;
		bufArrays.stream.m_ploop.BeginThreads( bufConfig.nThreadCount ) ;
	}

	// バッファ・配列の初期化
	bufArrays.flags = flagsBuffer ;
	bufArrays.iDelta2 = -1 ;
	bufArrays.buffers.clear() ;
	bufArrays.works.clear() ;
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		bufArrays.buffers.push_back( std::make_shared<NNPerceptron::Buffer>() ) ;
		bufArrays.works.push_back( std::make_shared<NNPerceptron::CPUWorkArray>() ) ;
		//
		GetLayerAt(i)->ResetBuffer( *(bufArrays.buffers.at(i)), i ) ;
	}
	bufArrays.inBufs.resize( GetLayerCount() ) ;

	// 各レイヤー出力チャネル数事前計算
	std::vector<NNBufDim>	dimArray ;
	PrepareOutputDimArray( dimArray, dimInput ) ;

	// 各レイヤーのバッファ準備
	for ( size_t iLayer = 0; iLayer < GetLayerCount(); iLayer ++ )
	{
		NNPerceptronPtr							pLayer = GetLayerAt(iLayer) ;
		std::shared_ptr<NNPerceptron::Buffer>	pBuf = bufArrays.buffers.at(iLayer) ;
		NNBufDim	dimSrc = pLayer->CalcInputDim( dimArray, iLayer, dimInput ) ;
		pLayer->PrepareBuffer
			( *pBuf, dimSrc,
				bufArrays.buffers,
				bufArrays.stream, iLayer,
				flagsBuffer, bufConfig.flagMemoryCommit ) ;
		//
		pLayer->PrepareWorkArray
			( *(bufArrays.works.at(iLayer)),
				bufArrays.stream.m_ploop.GetThreadCount() ) ;
	}

	// δ逆伝播２パス目が必要なレイヤーを検索
	for ( size_t i = 0; i < bufArrays.buffers.size(); i ++ )
	{
		if ( bufArrays.buffers.at(i)->reqDelta2 )
		{
			bufArrays.iDelta2 = (int) i ;
		}
	}

	// 作業バッファの初期化
	ResetWorkInBatch( bufArrays ) ;
}

void NNMultiLayerPerceptron::PrepareLossAndGradientArray
		( NNMultiLayerPerceptron::LossAndGradientArray& lagArray )
{
	lagArray.resize( GetLayerCount() ) ;

	lagArray.bufNormMax = 0.0f ;

	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->PrepareLossAndGradientBuf( lagArray.at(i) ) ;
	}
}

// 出力サイズ取得
//////////////////////////////////////////////////////////////////////////////
NNBufDim NNMultiLayerPerceptron::GetOutputSize
	( const NNMultiLayerPerceptron::BufferArrays& bufArrays ) const
{
	assert( bufArrays.buffers.size() == GetLayerCount() ) ;
	assert( bufArrays.buffers.size() >= 1 ) ;
	return	bufArrays.buffers.back()->bufOutput.GetSize() ;
}

// 出力サイズ計算
//////////////////////////////////////////////////////////////////////////////
NNBufDim NNMultiLayerPerceptron::CalcOutputSize( const NNBufDim& dimInput ) const
{
	assert( GetLayerCount() >= 1 ) ;
	if ( GetLayerCount() == 0 )
	{
		return	NNBufDim( 0, 0, 0 ) ;
	}
	std::vector<NNBufDim>	dimArray ;
	PrepareOutputDimArray( dimArray, dimInput ) ;
	return	dimArray.back() ;
}

void NNMultiLayerPerceptron::PrepareOutputDimArray
	( std::vector<NNBufDim>& dimArray, const NNBufDim& dimInput ) const
{
	assert( GetLayerCount() >= 1 ) ;
	if ( GetLayerCount() == 0 )
	{
		return ;
	}
	dimArray.resize( GetLayerCount() ) ;

	// １パス目（出力チャネル数を確定する）
	for ( size_t iLayer = 0; iLayer < GetLayerCount(); iLayer ++ )
	{
		NNPerceptronPtr	pLayer = GetLayerAt(iLayer) ;
		dimArray.at(iLayer) = NNBufDim( 0, 0, pLayer->CalcOutputChannels() ) ;
	}

	// ２パス目（出力サンプル数を確定する）
	for ( size_t iLayer = 0; iLayer < GetLayerCount(); iLayer ++ )
	{
		NNPerceptronPtr	pLayer = GetLayerAt(iLayer) ;
		NNBufDim		dimSrc = pLayer->CalcInputDim( dimArray, iLayer, dimInput ) ;
		NNBufDim		dimOut = pLayer->CalcOutputDim( dimSrc ) ;
		dimArray.at(iLayer) = dimOut ;
	}
}

// ミニバッチ毎の処理
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::PrepareForMiniBatch
	( NNMultiLayerPerceptron::BufferArrays& bufArrays,
		uint32_t flagsBuffer, std::random_device::result_type rndSeed ) const
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->PrepareForMiniBatch
			( *(bufArrays.buffers.at(i)),
				flagsBuffer, rndSeed, bufArrays.stream ) ;
	}
}

// 勾配リセット
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::ResetWorkInBatch
	( NNMultiLayerPerceptron::BufferArrays& bufArrays )
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->ResetBufferInBatch( *(bufArrays.buffers.at(i)) ) ;
		GetLayerAt(i)->ResetWorkArrayInBatch( *(bufArrays.works.at(i)) ) ;
	}
}

void NNMultiLayerPerceptron::ResetLossAndGrandient
		( NNMultiLayerPerceptron::LossAndGradientArray& lagArray )
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->ResetLossAndGradient( lagArray.at(i) ) ;
	}
}

// エポック開始時処理
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::OnBeginEpoch( void )
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->OnBeginEpoch() ;
	}
}

// エポック終了時処理
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::OnEndEpoch
	( NNMultiLayerPerceptron::BufferArrays& bufArrays )
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->OnEndEpoch
			( *(bufArrays.buffers.at(i)), *(bufArrays.works.at(i)) ) ;
	}
}

// 損失と勾配を合計する
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::AddLossAndGradient
	( NNMultiLayerPerceptron::LossAndGradientArray& lagArray,
		const NNMultiLayerPerceptron::BufferArrays& bufArrays )
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->AddLossAndGradient( lagArray.at(i), *(bufArrays.works.at(i)) ) ;
	}
}

// ストリーミングに連動してバッファをシフトする
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::ShiftBufferWithStreaming
	( NNMultiLayerPerceptron::BufferArrays& bufArrays, size_t xShift )
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->ShiftBufferWithStreaming
			( *(bufArrays.buffers.at(i)), xShift, bufArrays.stream ) ;
	}
}

// 出力層が再帰入力されている場合、学習の前に遅延バッファに教師データを入れる
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::PrepareOutputDelay
	( NNMultiLayerPerceptron::BufferArrays& bufArrays, NNBuffer& bufTearcher )
{
	if ( (GetLayerCount() >= 1)
		&& bufArrays.buffers.back()->reqDelay )
	{
		GetLayerAt(GetLayerCount()-1)->CopyTeachingDataToDelayBuffer
			( *(bufArrays.buffers.back()), bufTearcher, bufArrays.stream ) ;
	}
}

// モデルとデータの形状の検証
//////////////////////////////////////////////////////////////////////////////
bool NNMultiLayerPerceptron::VerifyDataShape
	( NNMultiLayerPerceptron::VerifyResult& verfResult,
		const BufferArrays& bufArrays,
		const NNBufDim& dimTeaching, const NNBufDim& dimSource0 ) const
{
	verfResult.verfError = verifyNormal ;
	verfResult.iLayer = 0 ;
	verfResult.iConnection = 0 ;

	unsigned long long	nUseCudaMemory = 0 ;
	for ( size_t iLayer = 0; iLayer < GetLayerCount(); iLayer ++ )
	{
		NNPerceptronPtr							pLayer = GetLayerAt( iLayer ) ;
		std::shared_ptr<NNSamplingFilter>		pSampler = pLayer->GetSampler() ;
		std::shared_ptr<NNActivationFunction>	pActivation = pLayer->GetActivation() ;
		const NNBufDim	dimInput =
			pLayer->CalcInputDim( bufArrays.buffers, iLayer, dimSource0 ) ;
		//
		if ( iLayer == 0 )
		{
			if ( (m_dimInShape.z != 0)
				&& (m_dimInShape.z != dimSource0.z) )
			{
				verfResult.verfError = mismatchSourceChannel ;
				return	false ;
			}
			/*
			if ( ((m_dimInShape.x != 0) && (m_dimInShape.x != dimSource0.x))
				|| ((m_dimInShape.y != 0) && (m_dimInShape.y != dimSource0.y)) )
			{
				verfResult.verfError = mismatchSourceSize ;
				return	false ;
			}
			*/
		}
		else if ( pSampler->MustBeInputLayer() )
		{
			verfResult.verfError = mustBeFirstInputLayer ;
			verfResult.iLayer = iLayer ;
			return	false ;
		}
		const size_t	xMatrix = pLayer->GetMatrix().GetColumnCount() - pLayer->GetBias() ;
		if ( (xMatrix != pSampler->ConvChannelCount( dimInput.z, xMatrix ) ) )
		{
			verfResult.verfError = (iLayer == 0) ? mismatchSourceChannel
													: mismatchInputChannel ;
			verfResult.iLayer = iLayer ;
			return	false ;
		}
		for ( size_t iCon = 0; iCon < pLayer->GetConnection().size(); iCon ++ )
		{
			const NNPerceptron::Connection
						cn = pLayer->GetConnection().at(iCon) ;
			NNBufDim	dimRef ;
			if ( (((int) iLayer - cn.iLayer) < -1)
				|| ((int)(iLayer - cn.iLayer) >= (int) bufArrays.buffers.size()) )
			{
				verfResult.verfError = outOfRangeInputLayer ;
				verfResult.iLayer = iLayer ;
				verfResult.iConnection = iCon ;
				return	false ;
			}
			if ( cn.iLayer <= (int) iLayer )
			{
				size_t	iRefLayer = iLayer - cn.iLayer ;
				dimRef = bufArrays.buffers.at(iRefLayer)->bufOutput.GetSize() ;
			}
			else
			{
				dimRef = dimSource0 ;
			}
			if ( (dimInput.x + cn.xOffset*2 != dimRef.x)
				|| (dimInput.y + cn.yOffset*2 != dimRef.y) )
			{
				if ( iCon == 0 )
				{
					verfResult.verfError = mismatchInputSize ;
					verfResult.iLayer = iLayer ;
					verfResult.iConnection = iCon ;
					return	false ;
				}
			}
		}
		if ( iLayer + 1 == GetLayerCount() )
		{
			if ( pActivation->MustNotBeLastLayer() )
			{
				verfResult.verfError = mustNotBeLastLayer ;
				verfResult.iLayer = iLayer ;
				return	false ;
			}
			NNBufDim	dimInAct = bufArrays.buffers.at(iLayer)->bufInAct.GetSize() ;
			NNBufDim	dimOutput = bufArrays.buffers.at(iLayer)->bufOutput.GetSize() ;
			if ( ((dimTeaching.x != 0) && (dimOutput.x > dimTeaching.x))
				|| ((dimTeaching.x != 0) && (dimOutput.y > dimTeaching.y)) )
			{
				verfResult.verfError = mismatchTeachingSize ;
				verfResult.iLayer = iLayer ;
				return	false ;
			}
			if ( (dimTeaching.z != 0)
				&& !pActivation->IsValidTeachingChannels
						( dimInAct.z, pLayer->GetDepthwise(), dimTeaching.z ) )
			{
				verfResult.verfError = mismatchTeachingChannel ;
				verfResult.iLayer = iLayer ;
				return	false ;
			}
		}
		else
		{
			if ( pActivation->MustBeLastLayer() )
			{
				verfResult.verfError = mustBeLastLayer ;
				verfResult.iLayer = iLayer ;
				return	false ;
			}
		}
		if ( bufArrays.stream.m_useCuda )
		{
			std::shared_ptr<NNPerceptron::Buffer>
								pBuf = bufArrays.buffers.at(iLayer) ;
			NNBufDim	dimInAct = pBuf->bufInAct.GetSize() ;
			NNBufDim	dimOutput = pBuf->bufOutput.GetSize() ;
			nUseCudaMemory += pBuf->GetBufferBytes() ;
			if ( nUseCudaMemory >= g_cudaDevProp.totalGlobalMem )
			{
				// 即復帰せず、他のエラーがある場合にはそちらを返す
				verfResult.verfError = lowCudaMemory ;
			}
			else if ( !nncuda_IsAcceptableMatrixSize
						( pLayer->GetMatrix().GetColumnCount(),
							pLayer->GetMatrix().GetLineCount(), dimInput.z )
				|| !pActivation->IsAcceptableChannelsForCuda( dimOutput.z, dimInAct.z ) )
			{
				// 即復帰せず、他のエラーがある場合にはそちらを返す
				verfResult.verfError = tooHugeMatrixForCuda ;
			}
		}
	}
	return	(verfResult.verfError == verifyNormal) ;
}

// 予測処理
//////////////////////////////////////////////////////////////////////////////
NNBuffer * NNMultiLayerPerceptron::Prediction
	( NNMultiLayerPerceptron::BufferArrays& bufArrays,
		NNBuffer& bufInput, bool flagForLearning, bool flagLowMemory )
{
	NNBuffer *	pOutput = nullptr ;
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		NNPerceptronPtr	pLayer = GetLayerAt(i) ;
		if ( flagLowMemory )
		{
			bufArrays.stream.m_cudaStream.Synchronize() ;
			pLayer->LowMemoryBuffer( *(bufArrays.buffers.at(i)), i ) ;
		}
		bufArrays.inBufs.at(i) =
			pLayer->PrepareInput
				( bufArrays.buffers, i, bufInput, bufArrays.stream ) ;
		pLayer->Prediction
			( *(bufArrays.works.at(i)),
				*(bufArrays.buffers.at(i)),
				bufArrays.inBufs.at(i), bufArrays.stream ) ;
		pOutput = &(bufArrays.buffers.at(i)->bufOutput) ;

	}
	if ( bufArrays.stream.m_useCuda )
	{
		if ( flagForLearning && (GetLayerCount() >= 1) )
		{
			bufArrays.buffers.back()->
				bufInAct.CudaAsyncFromDevice( bufArrays.stream.m_cudaStream ) ;
		}
		if ( pOutput != nullptr )
		{
			pOutput->CudaAsyncFromDevice( bufArrays.stream.m_cudaStream ) ;
		}
		bufArrays.stream.m_cudaStream.Synchronize() ;
	}
	return	pOutput ;
}

// 予測値の損失計算
//////////////////////////////////////////////////////////////////////////////
double NNMultiLayerPerceptron::CalcLoss
	( NNMultiLayerPerceptron::BufferArrays& bufArrays, NNBuffer& bufTearcher )
{
	const size_t	iLastLayer = GetLayerCount() - 1 ;
	const double	loss =
		GetLayerAt(iLastLayer)->LossDelta
			( *(bufArrays.works.at(iLastLayer)),
				*(bufArrays.buffers.at(iLastLayer)),
				bufTearcher, bufArrays.stream ) ;

	return	loss ;
}

// 学習１回
//////////////////////////////////////////////////////////////////////////////
double NNMultiLayerPerceptron::Learning
	( NNMultiLayerPerceptron::BufferArrays& bufArrays,
		NNBuffer& bufTearcher, NNBuffer& bufInput,
		NNMultiLayerPerceptron * pForwardMLP,
		NNMultiLayerPerceptron::BufferArrays * pForwardBufArrays )
{
	assert( GetLayerCount() >= 1 ) ;
	if ( GetLayerCount() == 0 )
	{
		return	0.0 ;
	}

	// 初めに予測（損失計算と遅延バッファに値を入れるため）
	NNBuffer *		pOutput = nullptr ;
	const size_t	nPrePrediction = CountOfPrePrediction() ;
	assert( nPrePrediction != 0 ) ;
	for ( size_t i = 0; i < nPrePrediction; i ++ )
	{
		// 出力層が再帰入力されている場合、遅延バッファに教師データを入れる
		PrepareOutputDelay( bufArrays, bufTearcher ) ;

		// 予測
		pOutput = Prediction( bufArrays, bufInput, true ) ;
	}
	assert( pOutput != nullptr ) ;

	double	loss = 0.0 ;
	if ( pForwardMLP != nullptr )
	{
		// 前方 MLP を学習
		assert( pForwardBufArrays != nullptr ) ;
		loss = pForwardMLP->Learning( *pForwardBufArrays, bufTearcher, *pOutput ) ;
	}

	// δ用バッファクリア
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		NNPerceptronPtr			pLayer = GetLayerAt(i) ;
		std::shared_ptr<NNPerceptron::Buffer>	pBuf = bufArrays.buffers.at(i) ;
		pLayer->ClearDeltaBuffer( *pBuf, bufArrays.stream ) ;
	}

	const size_t	iLastLayer = GetLayerCount() - 1 ;
	if ( pForwardMLP == nullptr )
	{
		// 損失関数
		loss = GetLayerAt(iLastLayer)->LossDelta
					( *(bufArrays.works.at(iLastLayer)),
						*(bufArrays.buffers.at(iLastLayer)),
						bufTearcher, bufArrays.stream ) ;
	}
	else
	{
		// 前方 MLP からδ逆伝播
		assert( pForwardBufArrays->stream.m_useCuda == bufArrays.stream.m_useCuda ) ;
		NNPerceptronPtr							pForwardLayer0 = pForwardMLP->GetLayerAt(0) ;
		NNPerceptronPtr							pLastLayer = GetLayerAt(iLastLayer) ;
		NNPerceptron::CPUWorkArray&				bufLastWorks = *(bufArrays.works.at(iLastLayer)) ;
		std::shared_ptr<NNPerceptron::Buffer>	pLastBuf = bufArrays.buffers.at(iLastLayer) ;

		pForwardLayer0->LayerDeltaBackTo
			( *pLastBuf, *(pForwardBufArrays->buffers.at(0)), bufArrays.stream ) ;

		// 活性化関数のδ逆伝播
		pLastLayer->ActivationDeltaBack( bufLastWorks, *pLastBuf, bufArrays.stream ) ;

		// 損失値を付け替える
		NNPerceptron::CPUWorkArray&
			workDstLayer = *(bufArrays.works.at(iLastLayer)) ;
		NNPerceptron::CPUWorkArray&
			workSrcLayer = *(pForwardBufArrays->works.back()) ;
		workDstLayer.fpLoss += workSrcLayer.fpLoss ;
		workDstLayer.nLossSamples += workSrcLayer.nLossSamples ;
		workSrcLayer.fpLoss = 0.0 ;
		workSrcLayer.nLossSamples = 0 ;
	}

	// δ逆伝播
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		DeltaBackLayerAt( iLastLayer - i, (i == 0), bufArrays ) ;
	}

	// δ逆伝播の２パス目
	if ( bufArrays.iDelta2 >= 0 )
	{
		if ( bufArrays.stream.m_useCuda )
		{
			bufArrays.stream.m_cudaStream.Synchronize() ;
		}
		// 勾配を一度集計する
		for ( size_t i = 0; i < GetLayerCount(); i ++ )
		{
			GetLayerAt(i)->IntegrateMatrixGradient
				( *(bufArrays.works.at(i)), *(bufArrays.buffers.at(i)), bufArrays.stream ) ;
		}

		// δ逆伝播をセカンダリバッファからコピー
		for ( size_t i = 0; i <= (size_t) bufArrays.iDelta2; i ++ )
		{
			GetLayerAt(i)->SwitchDeltaSecondaryBuffer
				( *(bufArrays.buffers.at(i)), bufArrays.stream ) ;
		}

		// δ逆伝播２パス目
		for ( size_t i = 0; i <= (size_t) bufArrays.iDelta2; i ++ )
		{
			DeltaBackLayerAt( (size_t) bufArrays.iDelta2 - i, false, bufArrays ) ;
		}
	}

	// 遅延バッファ
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		NNPerceptronPtr							pLayer = GetLayerAt(i) ;
		std::shared_ptr<NNPerceptron::Buffer>	pBuf = bufArrays.buffers.at(i) ;
		pLayer->CopyToDelayBuffer( *pBuf, bufArrays.stream ) ;
	}

	// 勾配統合
	if ( bufArrays.stream.m_useCuda )
	{
		bufArrays.stream.m_cudaStream.Synchronize() ;
	}
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->IntegrateMatrixGradient
			( *(bufArrays.works.at(i)), *(bufArrays.buffers.at(i)), bufArrays.stream ) ;
	}
	return	loss ;
}

// 指定レイヤーのδ逆伝播処理
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::DeltaBackLayerAt
	( size_t iLayer, bool flagOutputLayer,
			NNMultiLayerPerceptron::BufferArrays& bufArrays )
{
	NNPerceptronPtr							pLayer = GetLayerAt(iLayer) ;
	std::shared_ptr<NNPerceptron::Buffer>	pBuf = bufArrays.buffers.at(iLayer) ;
	NNPerceptron::CPUWorkArray&				bufWorks = *(bufArrays.works.at(iLayer)) ;

	if ( !flagOutputLayer )
	{
		// 活性化関数を逆伝播
		pLayer->ActivationDeltaBack( bufWorks, *pBuf, bufArrays.stream ) ;
	}

	// 勾配計算
	pLayer->CalcMatrixGradient
		( bufWorks, *pBuf, bufArrays.inBufs.at(iLayer), bufArrays.stream ) ;

	if ( (bufArrays.flags & bufferPropagateDelta)
			|| pBuf->reqDelay || (iLayer > 0)
			|| (pLayer->GetConnection().size() > 1) )
	{
		// 行列を逆伝播
		pLayer->MatrixDeltaBack( bufWorks, *pBuf, bufArrays.stream ) ;

		// 次のレイヤーへ伝播
		pLayer->LayerDeltaBack( bufArrays.buffers, *pBuf, bufArrays.stream ) ;
	}
}

// 勾配反映
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::GradientReflection
	( NNMultiLayerPerceptron::BufferArrays& bufArrays, float deltaRate )
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->AddMatrixGradient( *(bufArrays.works.at(i)), deltaRate ) ;
	}
}

void NNMultiLayerPerceptron::GradientReflection
	( NNMultiLayerPerceptron::LossAndGradientArray& lagArray, float deltaRate )
{
	// 全てのレイヤーの勾配の Frobenius ノルム計算
	float	maxNorm = 0.0f ;
	size_t	nCount = 0 ;
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		const size_t	nGradient = lagArray.at(i).nGradient ;
		if ( nGradient > 0 )
		{
			const float	norm = (lagArray.at(i).matGradient
										/ (float) nGradient).FrobeniusNorm() ;
			maxNorm = max( maxNorm, norm ) ;
			nCount ++ ;
		}
	}
	// 勾配が大きい場合、暴れてしまうのを抑制（特に 適用最適化無しの SGD の場合）
	lagArray.bufNormMax = max( maxNorm,
								(lagArray.bufNormMax * 0.9f + maxNorm * 0.1f) ) ;

	// 勾配反映
	const float	scaleGrad = 1.0f / max( lagArray.bufNormMax, 1.0f ) ;
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->AddMatrixGradient( lagArray.at(i), deltaRate, scaleGrad ) ;
	}
}

// 平均損失値取得
//////////////////////////////////////////////////////////////////////////////
double NNMultiLayerPerceptron::GetAverageLoss
	( const NNMultiLayerPerceptron::BufferArrays& bufArrays ) const
{
	if ( GetLayerCount() == 0 )
	{
		return	0.0 ;
	}
	const NNPerceptron::CPUWorkArray&	bufWorks = *(bufArrays.works.back()) ;
	if ( bufWorks.nLossSamples == 0 )
	{
		return	0.0 ;
	}
	return	bufWorks.fpLoss / (double) bufWorks.nLossSamples ;
}

double NNMultiLayerPerceptron::GetAverageLoss
	( const NNMultiLayerPerceptron::LossAndGradientArray& lagArray ) const
{
	if ( GetLayerCount() == 0 )
	{
		return	0.0 ;
	}
	const NNPerceptron::LossAndGradient&	lag = lagArray.back() ;
	if ( lag.nLossSamples == 0 )
	{
		return	0.0 ;
	}
	return	lag.fpLoss / (double) lag.nLossSamples ;
}


