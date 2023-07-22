
#include "nn_multi_layer.h"
#include <string.h>

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// マルチ・レイヤー・パーセプトロン（基底）
//////////////////////////////////////////////////////////////////////////////

// 識別子
//////////////////////////////////////////////////////////////////////////////
const std::string& NNPerceptronArray::GetIdentity( void ) const
{
	return	m_id ;
}

void NNPerceptronArray::SetIdentity( const char * pszId )
{
	m_id = pszId ;
}

// データ初期化
//////////////////////////////////////////////////////////////////////////////
void NNPerceptronArray::ClearAll( void )
{
	m_id.clear() ;
	m_loss = nullptr ;

	RemoveAllLayers() ;
}

// 損失関数（デフォルトは出力レイヤーの活性化関数）
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNLossFunction> NNPerceptronArray::GetLossFunction( void ) const
{
	return	m_loss ;
}

void NNPerceptronArray::SetLossFunction( std::shared_ptr<NNLossFunction> loss )
{
	m_loss = loss ;
}

// レイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNPerceptronArray::AppendLayer
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
	NNPerceptronArray::AppendLayer
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

size_t NNPerceptronArray::AppendLayer( NNPerceptronPtr pLayer )
{
	size_t	iLayer = m_mlp.size() ;
	m_mlp.push_back( pLayer ) ;
	return	iLayer ;
}

size_t NNPerceptronArray::InsertLayer( size_t iLayer, NNPerceptronPtr pLayer )
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
NNPerceptronPtr NNPerceptronArray::AppendConvLayer
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

NNPerceptronPtr NNPerceptronArray::AppendConvLayer
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
NNPerceptronPtr NNPerceptronArray::AppendDepthwiseConv
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

NNPerceptronPtr NNPerceptronArray::AppendDepthwiseConv
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
	NNPerceptronArray::AppendDepthwiseLayer
		( size_t nDstChannels, size_t nSrcChannels,
			size_t nDepthwise, size_t nBias, const char * pszActivation )
{
	return	AppendDepthwiseLayer
		( nDstChannels, nSrcChannels, nDepthwise, nBias,
			NNActivationFunction::Make( pszActivation ) ) ;
}

NNPerceptronPtr NNPerceptronArray::AppendDepthwiseLayer
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
NNPerceptronPtr NNPerceptronArray::AppendUpsamplingLayer
	( size_t nDstChannels, size_t nSrcChannels,
		int xUpsampling, int yUpsampling, size_t nBias, const char * pszActivation )
{
	return	AppendUpsamplingLayer
				( nDstChannels, nSrcChannels,
					xUpsampling, yUpsampling, nBias, 
					NNActivationFunction::Make( pszActivation ) ) ;
}

NNPerceptronPtr NNPerceptronArray::AppendUpsamplingLayer
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

NNPerceptronPtr NNPerceptronArray::AppendUp2x2Layer
	( size_t nDstChannels, size_t nSrcChannels, size_t nBias, const char * pszActivation )
{
	return	AppendUp2x2Layer
				( nDstChannels, nSrcChannels, nBias, 
					NNActivationFunction::Make( pszActivation ) ) ;
}

NNPerceptronPtr NNPerceptronArray::AppendUp2x2Layer
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
NNPerceptronPtr NNPerceptronArray::AppendUpsamplingFixLayer
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
NNPerceptronPtr NNPerceptronArray::AppendLayerAsOneHot
	( size_t nDstChannels, size_t nSrcChannels, const char * pszActivation )
{
	return	AppendLayer
		( nDstChannels, nSrcChannels, 0,
			pszActivation, std::make_shared<NNSamplerOneHot>() ) ;
}

// Softmax 高速化用レイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNPerceptronArray::AppendFastSoftmax
	( size_t nDstChannels, size_t nSrcChannels, size_t nBias, const char * pszActivation )
{
	return	AppendLayer
		( nDstChannels, nSrcChannels, nBias,
			pszActivation, std::make_shared<NNSamplerSparse>() ) ;
}

// チャネル毎の MaxPool レイヤー追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNPerceptronArray::AppendMaxPoolLayer
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
	NNPerceptronArray::AppendGatedLayer
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
	NNPerceptronArray::AppendPointwiseAdd
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
	NNPerceptronArray::AppendPointwiseAdd
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
	NNPerceptronArray::AppendPointwiseMul
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
	NNPerceptronArray::AppendPointwiseMul
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
size_t NNPerceptronArray::GetLayerCount( void ) const
{
	return	m_mlp.size() ;
}

// レイヤー取得
//////////////////////////////////////////////////////////////////////////////
NNPerceptronPtr NNPerceptronArray::GetLayerAt( size_t iLayer ) const
{
	assert( iLayer < m_mlp.size() ) ;
	return	m_mlp.at( iLayer ) ;
}

NNPerceptronPtr NNPerceptronArray::GetLayerAs( const char * pszId ) const
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
NNPerceptronPtr NNPerceptronArray::RemoveLayerAt( size_t iLayer )
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

NNPerceptronPtr NNPerceptronArray::RemoveLayerOf( NNPerceptronPtr pLayer )
{
	int	iLayer = FindLayer( pLayer ) ;
	if ( iLayer < 0 )
	{
		return	nullptr ;
	}
	return	RemoveLayerAt( (size_t) iLayer ) ;
}

void NNPerceptronArray::RemoveAllLayers( void )
{
	m_mlp.clear() ;
}

// レイヤー入力情報設定
//////////////////////////////////////////////////////////////////////////////
void NNPerceptronArray::SetLayerInput
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
int NNPerceptronArray::FindLayer( NNPerceptronPtr pLayer ) const
{
	return	FindLayer( pLayer.get() ) ;
}

int NNPerceptronArray::FindLayer( NNPerceptron * pLayer ) const
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

int NNPerceptronArray::FindLayerAs( const char * pszId, size_t iFirst ) const
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
int NNPerceptronArray::LayerOffsetOf( NNPerceptronPtr pLayer ) const
{
	return	LayerOffsetOf( pLayer.get() ) ;
}

int NNPerceptronArray::LayerOffsetOf( NNPerceptron* pLayer ) const
{
	return	(int) m_mlp.size() - 1 - FindLayer( pLayer ) ;
}

// レイヤー検索（pFromLayer から pLayer へ AddConnect する時のレイヤーオフセット）
// （pLayer == nullptr の時には、pLayer は入力データ）
//////////////////////////////////////////////////////////////////////////////
int NNPerceptronArray::LayerOffsetFrom
	( NNPerceptronPtr pLayer, NNPerceptronPtr pFromLayer ) const
{
	return	LayerOffsetFrom( pLayer.get(), pFromLayer.get() ) ;
}

int NNPerceptronArray::LayerOffsetFrom
	( NNPerceptron* pLayer, NNPerceptron* pFromLayer ) const
{
	assert( FindLayer( pFromLayer ) >= 0 ) ;
	return	FindLayer( pFromLayer ) - FindLayer( pLayer ) ;
}

// シリアライズ
//////////////////////////////////////////////////////////////////////////////
void NNPerceptronArray::Serialize( NNSerializer& ser )
{
	ser.Descend( CHHDRID_MLP_ID ) ;
	ser.WriteString( m_id.c_str() ) ;
	ser.Ascend() ;

	if ( m_loss != nullptr )
	{
		ser.Descend( CHHDRID_LOSS ) ;
		ser.WriteString( m_loss->GetFunctionName() ) ;
		m_loss->Serialize( ser ) ;
		ser.Ascend() ;
	}

	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		ser.Descend( CHHDRID_LAYER ) ;
		NNPerceptronPtr	pLayer = GetLayerAt( i ) ;
		ser.WriteString( pLayer->GetPerceptronType() ) ;
		pLayer->Serialize( ser ) ;
		ser.Ascend() ;
	}
}

// デシリアライズ
//////////////////////////////////////////////////////////////////////////////
bool NNPerceptronArray::Deserialize( NNDeserializer & dsr )
{
	for ( ; ; )
	{
		uint32_t	idChunk = dsr.Descend() ;
		if ( idChunk == 0 )
		{
			break ;
		}
		if ( !DeserializeChunk( dsr, idChunk ) )
		{
			dsr.Ascend() ;
			return	false ;
		}
		dsr.Ascend() ;
	}
	return	true ;
}

bool NNPerceptronArray::DeserializeChunk( NNDeserializer & dsr, uint32_t idChunk )
{
	if ( idChunk == CHHDRID_LAYER )
	{
		// レイヤー
		if ( !DeserializeLayer( dsr ) )
		{
			return	false ;
		}
	}
	else if ( idChunk == CHHDRID_LOSS )
	{
		// 損失関数
		std::string	strLoss = dsr.ReadString() ;
		m_loss = NNLossFunction::MakeLoss( strLoss.c_str() ) ;
		if ( m_loss != nullptr )
		{
			m_loss->Deserialize( dsr ) ;
		}
	}
	else if ( idChunk == CHHDRID_MLP_ID )
	{
		// 識別子
		m_id = dsr.ReadString() ;
	}
	return	true ;
}

bool NNPerceptronArray::DeserializeLayer( NNDeserializer & dsr )
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
	return	true ;
}

// 学習の前に事前予測処理が必要な回数
//////////////////////////////////////////////////////////////////////////////
size_t NNPerceptronArray::CountOfPrePrediction( void ) const
{
	size_t	nCount = 1 ;
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		NNPerceptronPtr	pLayer = GetLayerAt(i) ;
		assert( pLayer != nullptr ) ;
		for ( auto con : pLayer->GetConnection() )
		{
			if ( (con.iLayer <= 0) || (con.iDelay >= 1) )
			{
				nCount ++ ;
				break ;
			}
		}
	}
	return	nCount ;
}

// 出力サイズ計算
//////////////////////////////////////////////////////////////////////////////
NNBufDim NNPerceptronArray::CalcOutputSize( const NNBufDim& dimInput ) const
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

void NNPerceptronArray::PrepareOutputDimArray
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

// バッファ準備
//////////////////////////////////////////////////////////////////////////////
void NNPerceptronArray::PrepareBuffer
	( NNPerceptronArray::BufferArray& bufArray,
		const NNBufDim& dimInput, uint32_t flagsBuffer,
		const NNPerceptronArray::BufferConfig& bufConfig,
		const NNLoopStream& stream )
{
	// バッファ・配列の初期化
	bufArray.flags = flagsBuffer ;
	bufArray.iDelta2 = -1 ;
	bufArray.buffers.clear() ;
	bufArray.works.clear() ;
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		bufArray.buffers.push_back( std::make_shared<NNPerceptron::Buffer>() ) ;
		bufArray.works.push_back( std::make_shared<NNPerceptron::CPUWorkArray>() ) ;
		//
		GetLayerAt(i)->ResetBuffer( *(bufArray.buffers.at(i)), i ) ;
	}
	bufArray.inBufs.resize( GetLayerCount() ) ;

	// 各レイヤー出力チャネル数事前計算
	std::vector<NNBufDim>	dimArray ;
	PrepareOutputDimArray( dimArray, dimInput ) ;

	// 各レイヤーのバッファ準備
	for ( size_t iLayer = 0; iLayer < GetLayerCount(); iLayer ++ )
	{
		NNPerceptronPtr							pLayer = GetLayerAt(iLayer) ;
		std::shared_ptr<NNPerceptron::Buffer>	pBuf = bufArray.buffers.at(iLayer) ;
		NNBufDim	dimSrc = pLayer->CalcInputDim( dimArray, iLayer, dimInput ) ;
		pLayer->PrepareBuffer
			( *pBuf, dimSrc,
				bufArray.buffers,
				stream, iLayer,
				flagsBuffer, bufConfig.flagMemoryCommit ) ;
		//
		pLayer->PrepareWorkArray
			( *(bufArray.works.at(iLayer)),
				stream.m_ploop.GetThreadCount() ) ;
	}

	// δ逆伝播２パス目が必要なレイヤーを検索
	for ( size_t i = 0; i < bufArray.buffers.size(); i ++ )
	{
		if ( bufArray.buffers.at(i)->reqDelta2 )
		{
			bufArray.iDelta2 = (int) i ;
		}
	}

	// 作業バッファの初期化
	ResetWorkInBatch( bufArray ) ;
}

void NNPerceptronArray::PrepareLossAndGradientArray
		( NNPerceptronArray::LossAndGradientArray& lagArray )
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
NNBufDim NNPerceptronArray::GetOutputSize
	( const NNPerceptronArray::BufferArray& bufArray ) const
{
	assert( bufArray.buffers.size() == GetLayerCount() ) ;
	assert( bufArray.buffers.size() >= 1 ) ;
	return	bufArray.buffers.back()->bufOutput.GetSize() ;
}

// 勾配リセット
//////////////////////////////////////////////////////////////////////////////
void NNPerceptronArray::ResetWorkInBatch
	( NNPerceptronArray::BufferArray& bufArray )
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->ResetBufferInBatch( *(bufArray.buffers.at(i)) ) ;
		GetLayerAt(i)->ResetWorkArrayInBatch( *(bufArray.works.at(i)) ) ;
	}
}

void NNPerceptronArray::ResetLossAndGrandient
		( NNPerceptronArray::LossAndGradientArray& lagArray )
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->ResetLossAndGradient( lagArray.at(i) ) ;
	}
}

// エポック開始時処理
//////////////////////////////////////////////////////////////////////////////
void NNPerceptronArray::OnBeginEpoch( void )
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->OnBeginEpoch() ;
	}
}

// エポック終了時処理
//////////////////////////////////////////////////////////////////////////////
void NNPerceptronArray::OnEndEpoch
	( NNPerceptronArray::BufferArray& bufArray )
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->OnEndEpoch
			( *(bufArray.buffers.at(i)), *(bufArray.works.at(i)) ) ;
	}
}

// 損失と勾配を合計する
//////////////////////////////////////////////////////////////////////////////
void NNPerceptronArray::AddLossAndGradient
	( NNPerceptronArray::LossAndGradientArray& lagArray,
		const NNPerceptronArray::BufferArray& bufArray )
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->AddLossAndGradient( lagArray.at(i), *(bufArray.works.at(i)) ) ;
	}
}

// ミニバッチ毎の処理
//////////////////////////////////////////////////////////////////////////////
void NNPerceptronArray::PrepareForMiniBatch
	( NNPerceptronArray::BufferArray& bufArray,
		uint32_t flagsBuffer,
		std::random_device::result_type rndSeed, NNLoopStream& stream ) const
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->PrepareForMiniBatch
			( *(bufArray.buffers.at(i)), flagsBuffer, rndSeed, stream ) ;
	}
}

// ストリーミングに連動してバッファをシフトする
//////////////////////////////////////////////////////////////////////////////
void NNPerceptronArray::ShiftBufferWithStreaming
	( NNPerceptronArray::BufferArray& bufArray,
		size_t xShift, NNLoopStream& stream )
{
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->ShiftBufferWithStreaming
			( *(bufArray.buffers.at(i)), xShift, stream ) ;
	}
}

// 出力層が再帰入力されている場合、学習の前に遅延バッファに教師データを入れる
//////////////////////////////////////////////////////////////////////////////
void NNPerceptronArray::PrepareOutputDelay
	( NNPerceptronArray::BufferArray& bufArray,
		NNBuffer& bufTeacher, NNLoopStream& stream )
{
	if ( (GetLayerCount() >= 1)
		&& bufArray.buffers.back()->reqDelay )
	{
		GetLayerAt(GetLayerCount()-1)->CopyTeachingDataToDelayBuffer
			( *(bufArray.buffers.back()), bufTeacher, stream ) ;
	}
}

// モデルとデータの形状の検証
//////////////////////////////////////////////////////////////////////////////
bool NNPerceptronArray::VerifyDataShape
	( NNPerceptronArray::VerifyResult& verfResult,
		const NNPerceptronArray::BufferArray& bufArray,
		const NNBufDim& dimTeaching, const NNBufDim& dimSource0, bool flagCuda ) const
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
			pLayer->CalcInputDim( bufArray.buffers, iLayer, dimSource0 ) ;
		//
		if ( (iLayer != 0) && pSampler->MustBeInputLayer() )
		{
			if ( (pLayer->GetConnection().size() != 1)
				|| (pLayer->GetConnection().at(0).iLayer <= (int) iLayer) )
			{
				verfResult.verfError = mustBeFirstInputLayer ;
				verfResult.iLayer = iLayer ;
				return	false ;
			}
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
				|| ((int)(iLayer - cn.iLayer) >= (int) bufArray.buffers.size()) )
			{
				verfResult.verfError = outOfRangeInputLayer ;
				verfResult.iLayer = iLayer ;
				verfResult.iConnection = iCon ;
				return	false ;
			}
			if ( cn.iLayer <= (int) iLayer )
			{
				size_t	iRefLayer = iLayer - cn.iLayer ;
				dimRef = bufArray.buffers.at(iRefLayer)->bufOutput.GetSize() ;
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
			NNBufDim	dimInAct = bufArray.buffers.at(iLayer)->bufInAct.GetSize() ;
			NNBufDim	dimOutput = bufArray.buffers.at(iLayer)->bufOutput.GetSize() ;
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
		if ( flagCuda )
		{
			std::shared_ptr<NNPerceptron::Buffer>
								pBuf = bufArray.buffers.at(iLayer) ;
			NNBufDim	dimInAct = pBuf->bufInAct.GetSize() ;
			NNBufDim	dimOutput = pBuf->bufOutput.GetSize() ;
			nUseCudaMemory += pBuf->EstimateCudaBufferBytes() ;
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
NNBuffer * NNPerceptronArray::Prediction
	( NNPerceptronArray::BufferArray& bufArray, NNLoopStream& stream,
		NNBuffer& bufInput, size_t xBoundary,
		size_t iFirstLayer, size_t iEndLayer,
		bool flagForLearning, bool flagLowMemory )
{
	NNBuffer *	pOutput = nullptr ;
	if ( iEndLayer == 0 )
	{
		iEndLayer = GetLayerCount() ;
	}
	for ( size_t i = iFirstLayer; i < iEndLayer; i ++ )
	{
		NNPerceptronPtr	pLayer = GetLayerAt(i) ;
		if ( flagLowMemory )
		{
			stream.m_cudaStream.Synchronize() ;
			pLayer->LowMemoryBuffer( *(bufArray.buffers.at(i)), i ) ;
		}
		bufArray.inBufs.at(i) =
			pLayer->PrepareInput( bufArray.buffers, i, bufInput, stream ) ;
		pLayer->Prediction
			( *(bufArray.works.at(i)),
				*(bufArray.buffers.at(i)),
				bufArray.inBufs.at(i), stream, xBoundary ) ;
		pOutput = &(bufArray.buffers.at(i)->bufOutput) ;
	}
	if ( stream.m_useCuda )
	{
		if ( flagForLearning && (iEndLayer >= 1) )
		{
			bufArray.buffers.at(iEndLayer - 1)->
				bufInAct.CudaAsyncFromDevice( stream.m_cudaStream ) ;
		}
		if ( pOutput != nullptr )
		{
			pOutput->CudaAsyncFromDevice( stream.m_cudaStream ) ;
		}
		stream.m_cudaStream.Synchronize() ;
	}
	return	pOutput ;
}

// 予測値の損失計算
//////////////////////////////////////////////////////////////////////////////
double NNPerceptronArray::CalcLoss
	( NNPerceptronArray::BufferArray& bufArray,
		NNLoopStream& stream, NNBuffer& bufTeacher )
{
	const size_t	iLastLayer = GetLayerCount() - 1 ;
	const double	loss =
		GetLayerAt(iLastLayer)->cpuCalcLoss
			( *(bufArray.works.at(iLastLayer)),
				*(bufArray.buffers.at(iLastLayer)),
				bufTeacher, stream, m_loss.get() ) ;
	return	loss ;
}

// 学習１回
//////////////////////////////////////////////////////////////////////////////
double NNPerceptronArray::Learning
	( NNPerceptronArray::BufferArray& bufArray, NNLoopStream& stream,
		NNBuffer& bufTeacher, NNBuffer& bufInput,
		NNPerceptronArray * pForwardMLP,
		NNPerceptronArray::BufferArray * pForwardBufArrays )
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
		PrepareOutputDelay( bufArray, bufTeacher, stream ) ;

		// 予測
		pOutput = Prediction( bufArray, stream, bufInput, 0, 0, 0, true ) ;
	}
	assert( pOutput != nullptr ) ;

	double	loss = 0.0 ;
	if ( pForwardMLP != nullptr )
	{
		// 前方 MLP を学習
		assert( pForwardBufArrays != nullptr ) ;
		loss = pForwardMLP->Learning( *pForwardBufArrays, stream, bufTeacher, *pOutput ) ;
	}

	// δ用バッファクリア
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		NNPerceptronPtr	pLayer = GetLayerAt(i) ;
		std::shared_ptr<NNPerceptron::Buffer>
						pBuf = bufArray.buffers.at(i) ;
		pLayer->ClearDeltaBuffer( *pBuf, stream ) ;
	}

	const size_t	iLastLayer = GetLayerCount() - 1 ;
	if ( pForwardMLP == nullptr )
	{
		// 損失関数
		loss = GetLayerAt(iLastLayer)->LossDelta
					( *(bufArray.works.at(iLastLayer)),
						*(bufArray.buffers.at(iLastLayer)),
						bufTeacher, stream, m_loss.get() ) ;
	}
	else
	{
		// 前方 MLP からδ逆伝播
		NNPerceptronPtr							pForwardLayer0 = pForwardMLP->GetLayerAt(0) ;
		NNPerceptronPtr							pLastLayer = GetLayerAt(iLastLayer) ;
		NNPerceptron::CPUWorkArray&				bufLastWorks = *(bufArray.works.at(iLastLayer)) ;
		std::shared_ptr<NNPerceptron::Buffer>	pLastBuf = bufArray.buffers.at(iLastLayer) ;

		pForwardLayer0->LayerDeltaBackTo
			( *pLastBuf, *(pForwardBufArrays->buffers.at(0)), stream ) ;

		// 活性化関数のδ逆伝播
		pLastLayer->ActivationDeltaBack( bufLastWorks, *pLastBuf, stream ) ;

		// 損失値を付け替える
		NNPerceptron::CPUWorkArray&
			workDstLayer = *(bufArray.works.at(iLastLayer)) ;
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
		DeltaBackLayerAt( iLastLayer - i, (i == 0), bufArray, stream ) ;
	}

	// δ逆伝播の２パス目
	if ( bufArray.iDelta2 >= 0 )
	{
		if ( stream.m_useCuda )
		{
			stream.m_cudaStream.Synchronize() ;
		}
		// 勾配を一度集計する
		for ( size_t i = 0; i < GetLayerCount(); i ++ )
		{
			GetLayerAt(i)->IntegrateMatrixGradient
				( *(bufArray.works.at(i)), *(bufArray.buffers.at(i)), stream ) ;
		}

		// δ逆伝播をセカンダリバッファからコピー
		for ( size_t i = 0; i <= (size_t) bufArray.iDelta2; i ++ )
		{
			GetLayerAt(i)->SwitchDeltaSecondaryBuffer
						( *(bufArray.buffers.at(i)), stream ) ;
		}

		// δ逆伝播２パス目
		for ( size_t i = 0; i <= (size_t) bufArray.iDelta2; i ++ )
		{
			DeltaBackLayerAt
				( (size_t) bufArray.iDelta2 - i, false, bufArray, stream ) ;
		}
	}

	// 遅延バッファ
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		NNPerceptronPtr							pLayer = GetLayerAt(i) ;
		std::shared_ptr<NNPerceptron::Buffer>	pBuf = bufArray.buffers.at(i) ;
		pLayer->CopyToDelayBuffer( *pBuf, stream ) ;
	}

	// 勾配統合
	if ( stream.m_useCuda )
	{
		stream.m_cudaStream.Synchronize() ;
	}
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->IntegrateMatrixGradient
			( *(bufArray.works.at(i)), *(bufArray.buffers.at(i)), stream ) ;
	}
	return	loss ;
}

// 指定レイヤーのδ逆伝播処理
//////////////////////////////////////////////////////////////////////////////
void NNPerceptronArray::DeltaBackLayerAt
	( size_t iLayer, bool flagOutputLayer,
		NNPerceptronArray::BufferArray& bufArray, NNLoopStream& stream )
{
	NNPerceptronPtr							pLayer = GetLayerAt(iLayer) ;
	std::shared_ptr<NNPerceptron::Buffer>	pBuf = bufArray.buffers.at(iLayer) ;
	NNPerceptron::CPUWorkArray&				bufWorks = *(bufArray.works.at(iLayer)) ;

	if ( !flagOutputLayer )
	{
		// 活性化関数を逆伝播
		pLayer->ActivationDeltaBack( bufWorks, *pBuf, stream ) ;
	}

	// 勾配計算
	pLayer->CalcMatrixGradient
		( bufWorks, *pBuf, bufArray.inBufs.at(iLayer), stream ) ;

	if ( (bufArray.flags & bufferPropagateDelta)
			|| pBuf->reqDelay || (iLayer > 0)
			|| (pLayer->GetConnection().size() > 1) )
	{
		// 行列を逆伝播
		pLayer->MatrixDeltaBack( bufWorks, *pBuf, stream ) ;

		// 次のレイヤーへ伝播
		pLayer->LayerDeltaBack( bufArray.buffers, *pBuf, stream ) ;
	}
}

// 勾配反映
//////////////////////////////////////////////////////////////////////////////
void NNPerceptronArray::GradientReflection
	( NNPerceptronArray::LossAndGradientArray& lagArray, float deltaRate )
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
			maxNorm = __max( maxNorm, norm * GetLayerAt(i)->GetGradientFactor() ) ;
			nCount ++ ;
		}
	}
	// 勾配が大きい場合、暴れてしまうのを抑制（特に 適用最適化無しの SGD の場合）
	lagArray.bufNormMax = __max( maxNorm,
								(lagArray.bufNormMax * 0.9f + maxNorm * 0.1f) ) ;

	// 勾配反映
	const float	scaleGrad = 1.0f / __max( lagArray.bufNormMax, 1.0f ) ;
	for ( size_t i = 0; i < GetLayerCount(); i ++ )
	{
		GetLayerAt(i)->AddMatrixGradient( lagArray.at(i), deltaRate, scaleGrad ) ;
	}
}

// 平均損失値取得
//////////////////////////////////////////////////////////////////////////////
double NNPerceptronArray::GetAverageLoss
	( const NNPerceptronArray::BufferArray& bufArray ) const
{
	if ( GetLayerCount() == 0 )
	{
		return	0.0 ;
	}
	const NNPerceptron::CPUWorkArray&	bufWorks = *(bufArray.works.back()) ;
	if ( bufWorks.nLossSamples == 0 )
	{
		return	0.0 ;
	}
	return	bufWorks.fpLoss / (double) bufWorks.nLossSamples ;
}

double NNPerceptronArray::GetAverageLoss
	( const NNPerceptronArray::LossAndGradientArray& lagArray ) const
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
	m_subpass.clear() ;
	m_evaluation = nullptr ;

	m_flagsMLP = 0 ;
	m_dimInShape = NNBufDim( 0, 0, 0 ) ;
	m_dimInUnit = NNBufDim( 0, 0, 0 ) ;

	NNPerceptronArray::ClearAll() ;
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
//		size_t	nCount = __max( m_dimInShape.x / m_dimInUnit.x, 1 ) ;
		return	NNPerceptronArray::CountOfPrePrediction() ;
	}
	return	1 ;
}

// 評価関数
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::SetEvaluationFunction
	( std::shared_ptr<NNEvaluationFunction> pEvaluation )
{
	m_evaluation = pEvaluation ;
}

std::shared_ptr<NNEvaluationFunction>
	NNMultiLayerPerceptron::GetEvaluationFunction( void ) const
{
	return	m_evaluation ;
}

// 追加的なパス
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::AddSubpass( std::shared_ptr<NNMultiLayerPerceptron::Pass> pass )
{
	m_subpass.push_back( pass ) ;
}

std::shared_ptr<NNMultiLayerPerceptron::Pass>
	NNMultiLayerPerceptron::GetSubpassAs( const char * pszId ) const
{
	int	iPass = FindSubpass( pszId ) ;
	if ( iPass < 0 )
	{
		return	nullptr ;
	}
	return	m_subpass.at( (size_t) iPass ) ;
}

std::shared_ptr<NNMultiLayerPerceptron::Pass>
	NNMultiLayerPerceptron::GetSubpassAt( size_t iPass ) const
{
	return	m_subpass.at( iPass ) ;
}

int NNMultiLayerPerceptron::FindSubpass( const char * pszId ) const
{
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		if ( m_subpass.at(i)->GetIdentity() == pszId )
		{
			return	(int) i ;
		}
	}
	return	-1 ;
}

size_t NNMultiLayerPerceptron::GetSubpassCount( void ) const
{
	return	m_subpass.size() ;
}

bool NNMultiLayerPerceptron::RemoveSubpass( std::shared_ptr<Pass> pass )
{
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		if ( m_subpass.at(i) == pass )
		{
			m_subpass.erase( m_subpass.begin() + i ) ;
			return	true ;
		}
	}
	return	false ;
}

std::shared_ptr<NNMultiLayerPerceptron::Pass>
	NNMultiLayerPerceptron::RemoveSubpassAt( size_t iPass )
{
	std::shared_ptr<Pass>	pass = m_subpass.at( iPass ) ;
	m_subpass.erase( m_subpass.begin() + iPass ) ;
	return	pass ;
}

void NNMultiLayerPerceptron::RemoveAllSubpass( void )
{
	m_subpass.clear() ;
}

// シリアライズ
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::Serialize( NNSerializer& ser )
{
	// ファイルヘッダ
	FileHeader	fhdr ;
	memset( &fhdr, 0, sizeof(fhdr) ) ;
	fhdr.flagsHeader = hdrFlagChunkedLayer ;
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

	// 評価関数
	if ( m_evaluation != nullptr )
	{
		ser.Descend( CHHDRID_EVALUATION ) ;
		ser.WriteString( m_evaluation->GetFunctionName() ) ;
		m_evaluation->Serialize( ser ) ;
		ser.Ascend() ;
	}

	// レイヤー配列
	ser.Descend( CHHDRID_MLP_BODY ) ;
	NNPerceptronArray::Serialize( ser ) ;
	ser.Ascend() ;

	// サブパス
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		ser.Descend( CHHDRID_SUBPASS ) ;
		m_subpass.at(i)->Serialize( ser ) ;
		ser.Ascend() ;
	}
}

void NNMultiLayerPerceptron::Pass::Serialize( NNSerializer& ser )
{
	uint32_t	zeroHeader = 0 ;
	uint32_t	bytesDesc = sizeof(PassDescription) ;

	ser.Descend( CHHDRID_SUBPASS_HEADER ) ;
	ser.Write( &zeroHeader, sizeof(zeroHeader) ) ;
	ser.Write( &bytesDesc, sizeof(bytesDesc) ) ;
	ser.Write( &m_dsc, sizeof(m_dsc) ) ;
	ser.Ascend() ;

	NNPerceptronArray::Serialize( ser ) ;
}

// デシリアライズ
//////////////////////////////////////////////////////////////////////////////
bool NNMultiLayerPerceptron::Deserialize( NNDeserializer & dsr )
{
	ClearAll() ;

	// ファイルヘッダ
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

	if ( !(fhdr.flagsHeader & hdrFlagChunkedLayer) )
	{
		// レイヤー配列（バイナリ互換性のため）
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
	}
	else
	{
		for ( ; ; )
		{
			uint32_t	idChunk = dsr.Descend() ;
			if ( idChunk == 0 )
			{
				break ;
			}
			if ( idChunk == CHHDRID_EVALUATION )
			{
				// 評価関数
				std::string	strEvaluation = dsr.ReadString() ;
				m_evaluation = NNEvaluationFunction::Make( strEvaluation.c_str() ) ;
				if ( m_evaluation != nullptr )
				{
					m_evaluation->Deserialize( dsr ) ;
				}
			}
			else if ( idChunk == CHHDRID_MLP_BODY )
			{
				// レイヤー配列
				if ( !NNPerceptronArray::Deserialize( dsr ) )
				{
					return	false ;
				}
			}
			else if ( idChunk == CHHDRID_SUBPASS )
			{
				// サブパス
				std::shared_ptr<Pass>	pass = std::make_shared<Pass>() ;
				if ( !pass->Deserialize( dsr ) )
				{
					return	false ;
				}
				m_subpass.push_back( pass ) ;
			}
			else if ( idChunk == CHHDRID_LAYER )
			{
				// レイヤー（バイナリ互換性のため）
				if ( !NNPerceptronArray::DeserializeLayer( dsr ) )
				{
					return	false ;
				}
			}
			dsr.Ascend() ;
		}
	}
	return	true ;
}

bool NNMultiLayerPerceptron::Pass::Deserialize( NNDeserializer & dsr )
{
	if ( dsr.Descend( CHHDRID_SUBPASS_HEADER ) != CHHDRID_SUBPASS_HEADER )
	{
		return	false ;
	}
	uint32_t	zeroHeader = 0 ;
	uint32_t	bytesDesc = sizeof(PassDescription) ;
	dsr.Read( &zeroHeader, sizeof(zeroHeader) ) ;
	dsr.Read( &bytesDesc, sizeof(bytesDesc) ) ;
	dsr.Read( &m_dsc, __min( bytesDesc, sizeof(m_dsc) ) ) ;
	dsr.Ascend() ;

	return	NNPerceptronArray::Deserialize( dsr ) ;
}

// バッファ準備
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::PrepareBuffer
	( NNMultiLayerPerceptron::BufferArrays& bufArrays,
		const NNBufDim& dimInput, uint32_t flagsBuffer,
		const NNPerceptronArray::BufferConfig& bufConfig )
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

	// バッファの準備
	NNPerceptronArray::PrepareBuffer
		( bufArrays, dimInput, flagsBuffer, bufConfig, bufArrays.stream ) ;

	// サブパス用バッファ準備
	bufArrays.subpass.clear() ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		std::shared_ptr<BufferArray>
				pBufArray = std::make_shared<BufferArray>() ;
		m_subpass.at(i)->PrepareBuffer
			( *pBufArray, dimInput, flagsBuffer, bufConfig, bufArrays.stream ) ;
		bufArrays.subpass.push_back( pBufArray ) ;
	}
}

void NNMultiLayerPerceptron::PrepareLossAndGradientArrays
	( NNMultiLayerPerceptron::LossAndGradientArrays& lagArrays )
{
	NNPerceptronArray::PrepareLossAndGradientArray( lagArrays ) ;

	lagArrays.subpass.resize( m_subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		m_subpass.at(i)->PrepareLossAndGradientArray( lagArrays.subpass.at(i) ) ;
	}
}

// 勾配リセット
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::ResetWorkInBatch
	( NNMultiLayerPerceptron::BufferArrays& bufArrays )
{
	NNPerceptronArray::ResetWorkInBatch( bufArrays ) ;

	assert( m_subpass.size() == bufArrays.subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		m_subpass.at(i)->ResetWorkInBatch( *(bufArrays.subpass.at(i)) ) ;
	}
}

void NNMultiLayerPerceptron::ResetLossAndGrandient
	( NNMultiLayerPerceptron::LossAndGradientArrays& lagArrays )
{
	NNPerceptronArray::ResetLossAndGrandient( lagArrays ) ;

	assert( m_subpass.size() == lagArrays.subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		m_subpass.at(i)->ResetLossAndGrandient( lagArrays.subpass.at(i) ) ;
	}
}

// エポック開始時処理
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::OnBeginEpoch( void )
{
	NNPerceptronArray::OnBeginEpoch() ;

	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		m_subpass.at(i)->OnBeginEpoch() ;
	}
}

// エポック終了時処理
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::OnEndEpoch
	( NNMultiLayerPerceptron::BufferArrays& bufArrays )
{
	NNPerceptronArray::OnEndEpoch( bufArrays ) ;

	assert( m_subpass.size() == bufArrays.subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		m_subpass.at(i)->OnEndEpoch( *(bufArrays.subpass.at(i)) ) ;
	}
}

// 損失と勾配を合計する
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::AddLossAndGradient
	( NNMultiLayerPerceptron::LossAndGradientArrays& lagArrays,
		const NNMultiLayerPerceptron::BufferArrays& bufArrays )
{
	NNPerceptronArray::AddLossAndGradient( lagArrays, bufArrays ) ;

	assert( m_subpass.size() == lagArrays.subpass.size() ) ;
	assert( m_subpass.size() == bufArrays.subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		m_subpass.at(i)->AddLossAndGradient
			( lagArrays.subpass.at(i), *(bufArrays.subpass.at(i)) ) ;
	}
}

// ミニバッチ毎の処理
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::PrepareForMiniBatch
	( NNMultiLayerPerceptron::BufferArrays& bufArrays,
		uint32_t flagsBuffer, std::random_device::result_type rndSeed ) const
{
	NNPerceptronArray::PrepareForMiniBatch
		( bufArrays, flagsBuffer, rndSeed, bufArrays.stream ) ;

	assert( m_subpass.size() == bufArrays.subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		m_subpass.at(i)->PrepareForMiniBatch
			( *(bufArrays.subpass.at(i)), flagsBuffer, rndSeed, bufArrays.stream ) ;
	}
}

// ストリーミングに連動してバッファをシフトする
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::ShiftBufferWithStreaming
	( NNMultiLayerPerceptron::BufferArrays& bufArrays, size_t xShift )
{
	NNPerceptronArray::ShiftBufferWithStreaming( bufArrays, xShift, bufArrays.stream ) ;

	assert( m_subpass.size() == bufArrays.subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		m_subpass.at(i)->ShiftBufferWithStreaming
			( *(bufArrays.subpass.at(i)), xShift, bufArrays.stream ) ;
	}
}

// 出力層が再帰入力されている場合、学習の前に遅延バッファに教師データを入れる
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::PrepareOutputDelay
	( NNMultiLayerPerceptron::BufferArrays& bufArrays, NNBuffer& bufTeacher )
{
	NNPerceptronArray::PrepareOutputDelay( bufArrays, bufTeacher, bufArrays.stream ) ;

	assert( m_subpass.size() == bufArrays.subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		m_subpass.at(i)->PrepareOutputDelay
			( *(bufArrays.subpass.at(i)), bufTeacher, bufArrays.stream ) ;
	}
}

void NNMultiLayerPerceptron::PrepareLossAndGradientArray
	( NNPerceptronArray::LossAndGradientArray& lagArray )
{
	NNPerceptronArray::PrepareLossAndGradientArray( lagArray ) ;
}

void NNMultiLayerPerceptron::ResetWorkInBatch( NNPerceptronArray::BufferArray& bufArray )
{
	NNPerceptronArray::ResetWorkInBatch( bufArray ) ;
}

void NNMultiLayerPerceptron::ResetLossAndGrandient
	( NNPerceptronArray::LossAndGradientArray& lagArray )
{
	NNPerceptronArray::ResetLossAndGrandient( lagArray ) ;
}

void NNMultiLayerPerceptron::OnEndEpoch( NNPerceptronArray::BufferArray& bufArray )
{
	NNPerceptronArray::OnEndEpoch( bufArray ) ;
}

void NNMultiLayerPerceptron::AddLossAndGradient
	( NNPerceptronArray::LossAndGradientArray& lagArray,
			const NNPerceptronArray::BufferArray& bufArray )
{
	NNPerceptronArray::AddLossAndGradient( lagArray, bufArray ) ;
}

// モデルとデータの形状の検証
//////////////////////////////////////////////////////////////////////////////
bool NNMultiLayerPerceptron::VerifyDataShape
	( NNMultiLayerPerceptron::VerifyResult& verfResult,
		const NNMultiLayerPerceptron::BufferArrays& bufArrays,
		const NNBufDim& dimTeaching, const NNBufDim& dimSource0 ) const
{
	verfResult.verfError = verifyNormal ;
	verfResult.iLayer = 0 ;
	verfResult.iConnection = 0 ;
	verfResult.iSubpass = -1 ;

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

	if ( !NNPerceptronArray::VerifyDataShape
		( verfResult, bufArrays, dimTeaching, dimSource0, bufArrays.stream.m_useCuda ) )
	{
		return	false ;
	}

	assert( m_subpass.size() == bufArrays.subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		const NNBufDim	dimSubTeaching =
			GetLayerOutputSize
				( m_subpass.at(i)->m_dsc.iTeachingLayer,
						bufArrays, dimTeaching, dimSource0 ) ;
		const NNBufDim	dimSubSource =
			GetLayerOutputSize
				( m_subpass.at(i)->m_dsc.iSourceLayer,
						bufArrays, dimTeaching, dimSource0 ) ;
		if ( !m_subpass.at(i)->VerifyDataShape
			( verfResult, *(bufArrays.subpass.at(i)),
				dimSubTeaching, dimSubSource, bufArrays.stream.m_useCuda ) )
		{
			verfResult.iSubpass = (int) i ;
			return	false ;
		}
	}
	return	(verfResult.verfError == verifyNormal) ;
}

// レイヤー出力バッファサイズ取得
//////////////////////////////////////////////////////////////////////////////
NNBufDim NNMultiLayerPerceptron::GetLayerOutputSize
	( int iLayer, const NNMultiLayerPerceptron::BufferArrays& bufArrays,
		const NNBufDim& dimTeaching, const NNBufDim& dimSource0 ) const
{
	if ( iLayer == layerTeacher )
	{
		return	dimTeaching ;
	}
	else if ( iLayer == layerSource )
	{
		return	dimSource0 ;
	}
	else if ( (size_t) iLayer < bufArrays.buffers.size() )
	{
		return	bufArrays.buffers.at((size_t)iLayer)->bufOutput.GetSize() ;
	}
	return	NNBufDim( 0, 0, 0 ) ;
}

// レイヤー出力バッファ取得
//////////////////////////////////////////////////////////////////////////////
NNBuffer * NNMultiLayerPerceptron::GetLayerOutputBuffer
	( int iLayer, const NNMultiLayerPerceptron::BufferArrays& bufArrays,
		NNBuffer * pbufTeacher, NNBuffer * pbufSource ) const
{
	if ( iLayer == layerTeacher )
	{
		return	pbufTeacher ;
	}
	else if ( iLayer == layerSource )
	{
		return	pbufSource ;
	}
	else if ( (size_t) iLayer < bufArrays.buffers.size() )
	{
		return	&(bufArrays.buffers.at((size_t)iLayer)->bufOutput) ;
	}
	return	nullptr ;
}

// 予測処理
//////////////////////////////////////////////////////////////////////////////
NNBuffer * NNMultiLayerPerceptron::Prediction
	( NNMultiLayerPerceptron::BufferArrays& bufArrays,
		NNBuffer& bufInput, bool flagForLearning, bool flagLowMemory )
{
	NNBuffer *	pOutput =
		NNPerceptronArray::Prediction
				( bufArrays, bufArrays.stream,
					bufInput, bufArrays.xBoundary,
					bufArrays.iFirstLayer, bufArrays.iEndLayer,
					flagForLearning, flagLowMemory ) ;

	if ( flagForLearning )
	{
		assert( m_subpass.size() == bufArrays.subpass.size() ) ;
		for ( size_t i = 0; i < m_subpass.size(); i ++ )
		{
			NNBuffer *	pSubInput =
				GetLayerOutputBuffer
					( m_subpass.at(i)->m_dsc.iSourceLayer,
								bufArrays, nullptr, &bufInput ) ;
			assert( pSubInput != nullptr ) ;
			m_subpass.at(i)->Prediction
				( *(bufArrays.subpass.at(i)), bufArrays.stream,
					(pSubInput != nullptr) ? *pSubInput : bufInput,
					bufArrays.xBoundary, 0, 0, flagForLearning, flagLowMemory ) ;
		}
	}
	return	pOutput ;
}

// 予測値の損失計算
//////////////////////////////////////////////////////////////////////////////
double NNMultiLayerPerceptron::CalcLoss
	( NNMultiLayerPerceptron::BufferArrays& bufArrays, NNBuffer& bufTeacher )
{
	double	loss = NNPerceptronArray::CalcLoss
					( bufArrays, bufArrays.stream, bufTeacher ) ;

	assert( m_subpass.size() == bufArrays.subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		NNBuffer *	pSubTeacher =
			GetLayerOutputBuffer
				( m_subpass.at(i)->m_dsc.iTeachingLayer,
							bufArrays, &bufTeacher, nullptr ) ;
		assert( pSubTeacher != nullptr ) ;
		loss += m_subpass.at(i)->CalcLoss
			( *(bufArrays.subpass.at(i)), bufArrays.stream,
				(pSubTeacher != nullptr) ? *pSubTeacher : bufTeacher ) ;
	}
	return	loss ;
}

// 学習１回
//////////////////////////////////////////////////////////////////////////////
double NNMultiLayerPerceptron::Learning
	( NNMultiLayerPerceptron::BufferArrays& bufArrays,
		NNBuffer& bufTeacher, NNBuffer& bufInput,
		NNMultiLayerPerceptron * pForwardMLP,
		NNMultiLayerPerceptron::BufferArrays * pForwardBufArrays )
{
	double	loss = NNPerceptronArray::Learning
				( bufArrays, bufArrays.stream,
					bufTeacher, bufInput, pForwardMLP, pForwardBufArrays ) ;

	assert( m_subpass.size() == bufArrays.subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		NNBuffer *	pSubTeacher =
			GetLayerOutputBuffer
				( m_subpass.at(i)->m_dsc.iTeachingLayer,
							bufArrays, &bufTeacher, &bufInput ) ;
		NNBuffer *	pSubInput =
			GetLayerOutputBuffer
				( m_subpass.at(i)->m_dsc.iSourceLayer,
							bufArrays, &bufTeacher, &bufInput ) ;
		assert( pSubTeacher != nullptr ) ;
		assert( pSubInput != nullptr ) ;
		loss += m_subpass.at(i)->Learning
			( *(bufArrays.subpass.at(i)), bufArrays.stream,
				(pSubTeacher != nullptr) ? *pSubTeacher : bufTeacher,
				(pSubInput != nullptr) ? *pSubInput : bufInput ) ;
	}
	return	loss ;
}

// 勾配反映
//////////////////////////////////////////////////////////////////////////////
void NNMultiLayerPerceptron::GradientReflection
	( NNMultiLayerPerceptron::LossAndGradientArrays& lagArrays, float deltaRate )
{
	NNPerceptronArray::GradientReflection( lagArrays, deltaRate ) ;

	assert( m_subpass.size() == lagArrays.subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		m_subpass.at(i)->GradientReflection( lagArrays.subpass.at(i), deltaRate ) ;
	}
}

void NNMultiLayerPerceptron::GradientReflection
	( NNPerceptronArray::LossAndGradientArray& lagArray, float deltaRate )
{
	NNPerceptronArray::GradientReflection( lagArray, deltaRate ) ;
}

// 平均損失値取得
//////////////////////////////////////////////////////////////////////////////
double NNMultiLayerPerceptron::GetAverageLoss
	( const NNMultiLayerPerceptron::BufferArrays& bufArrays ) const
{
	double	loss = NNPerceptronArray::GetAverageLoss( bufArrays ) ;

	assert( m_subpass.size() == bufArrays.subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		loss += m_subpass.at(i)->GetAverageLoss( *(bufArrays.subpass.at(i)) ) ;
	}
	return	loss ;
}

double NNMultiLayerPerceptron::GetAverageLoss
	( const NNMultiLayerPerceptron::LossAndGradientArrays& lagArrays ) const
{
	double	loss = NNPerceptronArray::GetAverageLoss( lagArrays ) ;

	assert( m_subpass.size() == lagArrays.subpass.size() ) ;
	for ( size_t i = 0; i < m_subpass.size(); i ++ )
	{
		loss += m_subpass.at(i)->GetAverageLoss( lagArrays.subpass.at(i) ) ;
	}
	return	loss ;
}

double NNMultiLayerPerceptron::GetAverageLoss
	( const NNPerceptronArray::BufferArray& bufArray ) const
{
	return	NNPerceptronArray::GetAverageLoss( bufArray ) ;
}

double NNMultiLayerPerceptron::GetAverageLoss
	( const NNPerceptronArray::LossAndGradientArray& lagArray ) const
{
	return	NNPerceptronArray::GetAverageLoss( lagArray ) ;
}


