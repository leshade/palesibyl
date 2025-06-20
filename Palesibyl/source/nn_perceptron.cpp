
#include "nn_perceptron.h"

using namespace Palesibyl ;



//////////////////////////////////////////////////////////////////////////////
// 処理中間バッファ
//////////////////////////////////////////////////////////////////////////////

size_t NNPerceptron::Buffer::GetBufferBytes( void ) const
{
	return	bufMatrix.GetBufferBytes()
			+ bufDropoutMask.GetBufferBytes()
			+ bufInput.GetBufferBytes()
			+ bufInAct.GetBufferBytes()
			+ bufOutput.GetBufferBytes()
			+ bufDelay.GetBufferBytes()
			+ bufPrevDelta.GetBufferBytes()
			+ bufPrevDelta2.GetBufferBytes()
			+ bufInDelta.GetBufferBytes()
			+ bufOutDelta.GetBufferBytes()
			+ bufGradient.GetBufferBytes()
			+ normWorkBuf.GetBufferBytes() ;
}

size_t NNPerceptron::Buffer::GetCudaBufferBytes( void ) const
{
	size_t	nBytes =
			bufMatrix.GetCudaBufferBytes()
			+ bufDropoutMask.GetCudaBufferBytes()
			+ bufInput.GetCudaBufferBytes()
			+ bufInAct.GetCudaBufferBytes()
			+ bufOutput.GetCudaBufferBytes()
			+ bufDelay.GetCudaBufferBytes()
			+ bufPrevDelta.GetCudaBufferBytes()
			+ bufPrevDelta2.GetCudaBufferBytes()
			+ bufInDelta.GetCudaBufferBytes()
			+ bufOutDelta.GetCudaBufferBytes()
			+ bufGradient.GetCudaBufferBytes()
			+ normWorkBuf.GetCudaBufferBytes() ;
	if ( linearActivation )
	{
		nBytes -= bufOutput.GetBufferBytes() ;
		nBytes -= bufInDelta.GetBufferBytes() ;
	}
	if ( sharedInputBuf )
	{
		nBytes -= bufInput.GetBufferBytes() ;
	}
	return	nBytes ;
}

size_t NNPerceptron::Buffer::EstimateCudaBufferBytes( void ) const
{
	size_t	nBytes = GetBufferBytes() ;
	if ( fixedMatrix )
	{
		nBytes -= bufGradient.GetBufferBytes() ;
	}
	if ( linearActivation )
	{
		nBytes -= bufOutput.GetBufferBytes() ;
		nBytes -= bufInDelta.GetBufferBytes() ;
	}
	if ( sharedInputBuf )
	{
		nBytes -= bufInput.GetBufferBytes() ;
	}
	if ( !reqDelay )
	{
		nBytes -= bufDelay.GetBufferBytes() ;
	}
	if ( !reqDelta2 )
	{
		nBytes -= bufPrevDelta2.GetBufferBytes() ;
	}
	return	nBytes ;
}

unsigned long long NNPerceptron::BufferArray::GetTotalBufferBytes( void ) const
{
	unsigned long long	nTotalBytes = 0 ;
	for ( size_t i = 0; i < size(); i ++ )
	{
		nTotalBytes += at(i)->GetBufferBytes() ;
	}
	return	nTotalBytes ;
}

unsigned long long NNPerceptron::BufferArray::GetTotalCudaBufferBytes( void ) const
{
	unsigned long long	nTotalBytes = 0 ;
	for ( size_t i = 0; i < size(); i ++ )
	{
		nTotalBytes += at(i)->GetCudaBufferBytes() ;
	}
	return	nTotalBytes ;
}



//////////////////////////////////////////////////////////////////////////////
// パーセプトロン
//////////////////////////////////////////////////////////////////////////////

std::map< std::string, std::function< std::shared_ptr<NNPerceptron>() > >
	NNPerceptron::s_mapMakePerceptron ;

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNPerceptron::NNPerceptron( void )
	: m_behavior( 0 ),
		m_xInvalidLeft( 0 ), m_yInvalidTop( 0 ),
		m_xInvalidRight( 0 ), m_yInvalidBottom( 0 ),
		m_bias( 0 ),
		m_depthwise( 1 ), m_depthActiv( 1 ),
		m_deltaFactor( 1.0f ), m_gradFactor( 1.0f ),
		m_dropout( 0.0f ), m_adaOpt(adaOptNo), m_l2reg( 0.0f )
{
}

NNPerceptron::NNPerceptron
	( size_t nDstCount, size_t nSrcCount,
		size_t nDepthwise, size_t nBias,
		std::shared_ptr<NNSamplingFilter> sampler,
		std::shared_ptr<NNActivationFunction> activation )
	: m_behavior( 0 ),
		m_xInvalidLeft( 0 ), m_yInvalidTop( 0 ),
		m_xInvalidRight( 0 ), m_yInvalidBottom( 0 ),
		m_bias( 0 ),
		m_depthwise( 1 ), m_depthActiv( 1 ),
		m_deltaFactor( 1.0f ), m_gradFactor( 1.0f ),
		m_dropout( 0.0f ), m_adaOpt(adaOptNo),m_l2reg( 0.0f )
{
	Create( nDstCount, nSrcCount, nDepthwise, nBias, sampler, activation ) ;
}

NNPerceptron::NNPerceptron( const NNPerceptron& nnp )
	: m_behavior( 0 ),
		m_xInvalidLeft( 0 ), m_yInvalidTop( 0 ),
		m_xInvalidRight( 0 ), m_yInvalidBottom( 0 ),
		m_id( nnp.m_id ),
		m_matrix( nnp.m_matrix ),
		m_bias( nnp.m_bias ),
		m_depthwise( nnp.m_depthwise ),
		m_depthActiv( nnp.m_depthActiv ),
		m_deltaFactor( nnp.m_deltaFactor ),
		m_gradFactor( nnp.m_gradFactor ),
		m_adaOpt( nnp.m_adaOpt ),
		m_adaParam( nnp.m_adaParam ),
		m_l2reg( nnp.m_l2reg ),
		m_dropout( nnp.m_dropout ),
		m_sampler( nnp.m_sampler ),
		m_activation( nnp.m_activation ),
		m_connection( nnp.m_connection ),
		m_generator( nnp.m_generator )
{
	for ( int i = 0; i < sizeof(m_matAdaOpt)/sizeof(m_matAdaOpt[0]); i ++ )
	{
		m_matAdaOpt[i] = nnp.m_matAdaOpt[i] ;
	}
}

// 作成
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::Create
	( size_t nDstCount, size_t nSrcCount,
		size_t nDepthwise, size_t nBias,
		std::shared_ptr<NNSamplingFilter> sampler,
		std::shared_ptr<NNActivationFunction> activation )
{
	float	s = (float) sqrt( 2.0f / (float) __max(nSrcCount+nDstCount,1) ) ;
	m_matrix.Create( nDstCount, nSrcCount + nBias ) ;
	m_matrix.RandomizeNormalDist( 0.0f, s ) ;
	m_bias = nBias ;
	if ( nDepthwise != 0 )
	{
		m_depthwise = nDepthwise ;
		m_depthActiv = nDepthwise ;
	}

	Specialize( ) ;

	m_sampler = sampler ;
	m_activation = activation ;
}

// 入力情報追加
//////////////////////////////////////////////////////////////////////////////
NNPerceptron * NNPerceptron::AddConnection( const NNPerceptron::Connection& cn )
{
	m_connection.push_back( cn ) ;
	return	this ;
}

NNPerceptron * NNPerceptron::AddConnection
	( int iLayer, int iDelay,
		size_t iChannel, size_t nChCount, int xOffset, int yOffset )
{
	Connection	cn ;
	cn.iLayer = (int32_t) iLayer ;
	cn.iDelay = (int32_t) iDelay ;
	cn.iChannel = (uint32_t) iChannel ;
	cn.nChannels = (uint32_t) nChCount ;
	cn.xOffset = (int32_t) xOffset ;
	cn.yOffset = (int32_t) yOffset ;
	//
	return	AddConnection( cn ) ;
}

// 入力情報取得
//////////////////////////////////////////////////////////////////////////////
const std::vector<NNPerceptron::Connection>& NNPerceptron::GetConnection( void ) const
{
	return	m_connection ;
}

// 入力情報クリア
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::ClearAllConnection( void )
{
	m_connection.clear() ;
}

// 入力生成器
//////////////////////////////////////////////////////////////////////////////
NNPerceptron * NNPerceptron::SetGenerator( std::shared_ptr<NNGeneratorFunction> pGen )
{
	m_generator = pGen ;
	return	this ;
}

std::shared_ptr<NNGeneratorFunction> NNPerceptron::GetGenerator( void ) const
{
	return	m_generator ;
}

// 行列
//////////////////////////////////////////////////////////////////////////////
NNMatrix& NNPerceptron::Matrix( void )
{
	return	m_matrix ;
}

const NNMatrix& NNPerceptron::GetMatrix( void ) const
{
	return	m_matrix ;
}

// バイアス項
//////////////////////////////////////////////////////////////////////////////
size_t NNPerceptron::GetBias( void ) const
{
	return	m_bias ;
}

// 対角化単位
//////////////////////////////////////////////////////////////////////////////
size_t NNPerceptron::GetDepthwise( void ) const
{
	return	m_depthwise ;
}

// 活性化関数出力次数パラメータ
//////////////////////////////////////////////////////////////////////////////
size_t NNPerceptron::GetActivationDepthwise( void ) const
{
	return	m_depthActiv ;
}

NNPerceptron * NNPerceptron::SetActivationDepthwise( size_t nDepthwise )
{
	m_depthActiv = nDepthwise ;
	return	this ;
}

// 勾配更新最適化
//////////////////////////////////////////////////////////////////////////////
NNPerceptron::AdaptiveOptimization
	NNPerceptron::GetAdaptiveOptimization( void ) const
{
	return	m_adaOpt ;
}

const NNPerceptron::AdaptiveHyperparameter&
	NNPerceptron::GetAdaptiveHyperparameter( void ) const
{
	return	m_adaParam ;
}

NNPerceptron * NNPerceptron::SetAdaptiveOptimization
	( NNPerceptron::AdaptiveOptimization adaOpt,
			const NNPerceptron::AdaptiveHyperparameter& adaParam )
{
	m_adaOpt = adaOpt ;
	m_adaParam = adaParam ;
	return	this ;
}

// δ逆伝播係数（※レイヤー毎に調整したい場合）
//////////////////////////////////////////////////////////////////////////////
float NNPerceptron::GetDeltaFactor( void ) const
{
	return	m_deltaFactor ;
}

NNPerceptron * NNPerceptron::SetDeltaFactor( float delta )
{
	m_deltaFactor = delta ;
	return	this ;
}

// 学習速度係数（※レイヤー毎に調整したい場合）
//////////////////////////////////////////////////////////////////////////////
float NNPerceptron::GetGradientFactor( void ) const
{
	return	m_gradFactor ;
}

NNPerceptron * NNPerceptron::SetGradientFactor( float grad )
{
	m_gradFactor = grad ;
	return	this ;
}

// L2 正則化パラメータ
//////////////////////////////////////////////////////////////////////////////
float NNPerceptron::GetRidgeParameter( void ) const
{
	return	m_l2reg ;
}

NNPerceptron * NNPerceptron::SetRidgeParameter( float l2reg )
{
	m_l2reg = l2reg ;
	return	this ;
}

// ドロップアウト率
//////////////////////////////////////////////////////////////////////////////
float NNPerceptron::GetDropoutRate( void ) const
{
	return	m_dropout ;
}

NNPerceptron * NNPerceptron::SetDropoutRate( float dropout )
{
	m_dropout = dropout ;
	return	this ;
}

// 識別子
//////////////////////////////////////////////////////////////////////////////
const std::string& NNPerceptron::GetIdentity( void ) const
{
	return	m_id ;
}

NNPerceptron * NNPerceptron::SetIdentity( const char * pszId )
{
	m_id = pszId ;
	return	this ;
}

// 動作フラグ
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::SetBehaviorFlags( uint32_t flags )
{
	m_behavior = flags ;
}

uint32_t NNPerceptron::GetBehaviorFlags( void ) const
{
	return	m_behavior ;
}

bool NNPerceptron::IsMatrixFixed( void ) const
{
	return	(m_behavior & behaviorFixed) != 0 ;
}

bool NNPerceptron::IsDeltaCutOff( void ) const
{
	return	(m_behavior & behaviorCutOff) != 0 ;
}

bool NNPerceptron::IsNoDropout( void ) const
{
	return	(m_behavior & behaviorNoDropout) != 0 ;
}

void NNPerceptron::SetLossDeltaInvalidMargin
	( size_t xLeft, size_t yTop, size_t xRight, size_t yBottom )
{
	m_behavior |= behaviorInvalidMargin ;
	m_xInvalidLeft = xLeft ;
	m_yInvalidTop = yTop ;
	m_xInvalidRight = xRight ;
	m_yInvalidBottom = yBottom ;
}

// 正規化
//////////////////////////////////////////////////////////////////////////////
NNPerceptron * NNPerceptron::SetNormalization
		( std::shared_ptr<NNNormalizationFilter> pNorm )
{
	m_normalizer = pNorm ;
	if ( m_normalizer != nullptr )
	{
		m_normalizer->CreateFilter
			( m_sampler->CalcOutputChannels( m_matrix.GetLineCount() ) ) ;
	}
	return	this ;
}

std::shared_ptr<NNNormalizationFilter> NNPerceptron::GetNormalization( void ) const
{
	return	m_normalizer ;
}

// サンプラー
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNSamplingFilter> NNPerceptron::GetSampler( void ) const
{
	return	m_sampler ;
}

// 活性化関数
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNActivationFunction> NNPerceptron::GetActivation( void ) const
{
	return	m_activation ;
}

// 行列の特殊化
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::Specialize( void )
{
}

// シリアライズ
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::Serialize( NNSerializer& ser )
{
	ser.Descend( CHHDRID_BODY ) ;

	ser.Descend( CHHDRID_MATRIX ) ;
	uint32_t	line = (uint32_t) m_matrix.GetLineCount() ;
	uint32_t	col = (uint32_t) (m_matrix.GetColumnCount() - m_bias) ;
	uint32_t	bias = (uint32_t) m_bias ;
	uint32_t	depthwise = (uint32_t) m_depthwise ;
	ser.Write( &line, sizeof(line) ) ;
	ser.Write( &col, sizeof(col) ) ;
	ser.Write( &bias, sizeof(bias) ) ;
	ser.Write( &depthwise, sizeof(depthwise) ) ;
	ser.Write( m_matrix.GetConstArray(), m_matrix.GetLength() * sizeof(float) ) ;
	ser.Ascend() ;

	ser.Descend( CHHDRID_PARAM ) ;
	uint32_t	exFlags = extendInfoAdaptiveOptimization
							| extendInfoL2regularization
							| extendInfoDeltaFactor
							| extendInfoGradientFactor
							| extendInfoDropout
							| extendInfoActivationDepthwise
							| extendInfoIdentity ;
	if ( (m_matAdaOpt[0].GetLineCount() == m_matrix.GetLineCount())
		&& (m_matAdaOpt[0].GetColumnCount() == m_matrix.GetColumnCount()) )
	{
		exFlags |= extendAdaptiveMatrixGrad0 ;
	}
	if ( (m_matAdaOpt[1].GetLineCount() == m_matrix.GetLineCount())
		&& (m_matAdaOpt[1].GetColumnCount() == m_matrix.GetColumnCount()) )
	{
		exFlags |= extendAdaptiveMatrixGrad1 ;
	}
	uint32_t	adaOpt = (uint32_t) m_adaOpt ;
	float		l2reg = m_l2reg ;
	float		delta = m_deltaFactor ;
	float		grad = m_gradFactor ;
	float		dropout = m_dropout ;
	uint32_t	depthActiv = (uint32_t) m_depthActiv ;
	uint32_t	lenId = (uint32_t) m_id.length() ;
	uint32_t	sizeChar = sizeof(char) ;
	ser.Write( &exFlags, sizeof(uint32_t) ) ;
	ser.Write( &adaOpt, sizeof(adaOpt) ) ;
	ser.Write( &m_adaParam, sizeof(m_adaParam) ) ;
	if ( exFlags & extendAdaptiveMatrixGrad0 )
	{
		ser.Write( m_matAdaOpt[0].GetConstArray(),
					m_matrix.GetLength() * sizeof(float) ) ;
	}
	if ( exFlags & extendAdaptiveMatrixGrad1 )
	{
		ser.Write( m_matAdaOpt[1].GetConstArray(),
					m_matrix.GetLength() * sizeof(float) ) ;
	}
	ser.Write( &l2reg, sizeof(l2reg) ) ;
	ser.Write( &delta, sizeof(delta) ) ;
	ser.Write( &grad, sizeof(grad) ) ;
	ser.Write( &dropout, sizeof(dropout) ) ;
	ser.Write( &depthActiv, sizeof(depthActiv) ) ;
	ser.Write( &lenId, sizeof(lenId) ) ;
	ser.Write( &sizeChar, sizeof(sizeChar) ) ;
	if ( lenId > 0 )
	{
		ser.Write( m_id.c_str(), lenId * sizeof(char) ) ;
	}
	ser.Ascend() ;

	ser.Descend( CHHDRID_SAMPLER ) ;
	ser.WriteString( m_sampler->GetSamplerName() ) ;
	m_sampler->Serialize( ser ) ;
	ser.Ascend() ;

	ser.Descend( CHHDRID_ACTIVATION ) ;
	ser.WriteString( m_activation->GetFunctionName() ) ;
	m_activation->Serialize( ser ) ;
	ser.Ascend() ;

	if ( m_normalizer != nullptr )
	{
		ser.Descend( CHHDRID_NORM ) ;
		ser.WriteString( m_normalizer->GetFilterName() ) ;
		m_normalizer->Serialize( ser ) ;
		ser.Ascend() ;
	}

	if ( m_generator != nullptr )
	{
		ser.Descend( CHHDRID_GENERATOR ) ;
		ser.WriteString( m_generator->GetFunctionName() ) ;
		m_generator->Serialize( ser ) ;
		ser.Ascend() ;
	}

	ser.Descend( CHHDRID_CONNECTION ) ;
	uint32_t	lenCon = (uint32_t) m_connection.size() ;
	ser.Write( &lenCon, sizeof(lenCon) ) ;
	ser.Write( m_connection.data(), lenCon * sizeof(Connection) ) ;
	ser.Ascend() ;

	ser.Descend( CHHDRID_EXTEND ) ;
	SerializeExtendInfo( ser ) ;
	ser.Ascend() ;

	ser.Ascend() ;
}

void NNPerceptron::SerializeExtendInfo( NNSerializer& ser )
{
}

// デシリアライズ
//////////////////////////////////////////////////////////////////////////////
bool NNPerceptron::Deserialize( NNDeserializer & dsr )
{
	if ( dsr.Descend( CHHDRID_BODY ) != CHHDRID_BODY )
	{
		return	false ;
	}

	if ( dsr.Descend( CHHDRID_MATRIX ) != CHHDRID_MATRIX )
	{
		return	false ;
	}
	uint32_t	line = 0, col = 0, bias = 0, depthwise = 1 ;
	dsr.Read( &line, sizeof(line) ) ;
	dsr.Read( &col, sizeof(col) ) ;
	dsr.Read( &bias, sizeof(bias) ) ;
	dsr.Read( &depthwise, sizeof(depthwise) ) ;
	m_matrix.Create( line, col + bias ) ;
	m_bias = (size_t) bias ;
	m_depthwise = (size_t) depthwise ;
	dsr.Read( m_matrix.GetArray(), m_matrix.GetLength() * sizeof(float) ) ;
	dsr.Ascend() ;

	for ( ; ; )
	{
		uint32_t	idChunk = dsr.Descend() ;
		if ( idChunk == 0 )
		{
			break ;
		}
		if ( idChunk == CHHDRID_PARAM )
		{
			if ( dsr.GetChunkBytes() > sizeof(uint32_t) )
			{
				uint32_t	exFlags = 0 ;
				dsr.Read( &exFlags, sizeof(uint32_t) ) ;

				if ( exFlags & extendInfoAdaptiveOptimization )
				{
					uint32_t	adaOpt = adaOptNo ;
					dsr.Read( &adaOpt, sizeof(adaOpt) ) ;
					m_adaOpt = (AdaptiveOptimization) adaOpt ;
					dsr.Read( &m_adaParam, sizeof(m_adaParam) ) ;
				}
				if ( exFlags & extendAdaptiveMatrixGrad0 )
				{
					m_matAdaOpt[0].Create( line, col + bias ) ;
					dsr.Read( m_matAdaOpt[0].GetArray(),
								m_matrix.GetLength() * sizeof(float) ) ;
				}
				if ( exFlags & extendAdaptiveMatrixGrad1 )
				{
					m_matAdaOpt[1].Create( line, col + bias ) ;
					dsr.Read( m_matAdaOpt[1].GetArray(),
								m_matrix.GetLength() * sizeof(float) ) ;
				}
				if ( exFlags & extendInfoL2regularization )
				{
					float	l2reg = 0.0f ;
					dsr.Read( &l2reg, sizeof(l2reg) ) ;
					m_l2reg = l2reg ;
				}
				if ( exFlags & extendInfoDeltaFactor )
				{
					float	delta = 1.0f ;
					dsr.Read( &delta, sizeof(delta) ) ;
					m_deltaFactor = delta ;
				}
				if ( exFlags & extendInfoGradientFactor )
				{
					float	grad = 1.0f ;
					dsr.Read( &grad, sizeof(grad) ) ;
					m_gradFactor = grad ;
				}
				if ( exFlags & extendInfoDropout )
				{
					float	dropout = 0.0f ;
					dsr.Read( &dropout, sizeof(dropout) ) ;
					m_dropout = dropout ;
				}
				if ( exFlags & extendInfoActivationDepthwise )
				{
					uint32_t	depthActiv = 1 ;
					dsr.Read( &depthActiv, sizeof(depthActiv) ) ;
					m_depthActiv = (size_t) depthActiv ;
				}
				else
				{
					m_depthActiv = m_depthwise ;
				}
				if ( exFlags & extendInfoIdentity )
				{
					uint32_t	lenId = 0, sizeChar = sizeof(char) ;
					dsr.Read( &lenId, sizeof(lenId) ) ;
					dsr.Read( &sizeChar, sizeof(sizeChar) ) ;
					if ( sizeChar == sizeof(char) )
					{
						std::vector<char>	buf ;
						buf.resize( (size_t) lenId + 1 ) ;
						dsr.Read( buf.data(), lenId * sizeof(char) ) ;
						buf.data()[lenId] = 0 ;
						m_id = buf.data() ;
					}
					else
					{
						dsr.Skip( (size_t) (lenId * sizeChar) ) ;
					}
				}
			}
		}
		else if ( idChunk == CHHDRID_SAMPLER )
		{
			std::string	strSampler = dsr.ReadString() ;
			m_sampler = NNSamplingFilter::Make( strSampler.c_str() ) ;
			if ( m_sampler != nullptr )
			{
				m_sampler->Deserialize( dsr ) ;
			}
		}
		else if ( idChunk == CHHDRID_ACTIVATION )
		{
			std::string	strActivation = dsr.ReadString() ;
			m_activation = NNActivationFunction::Make( strActivation.c_str() ) ;
			if ( m_activation != nullptr )
			{
				m_activation->Deserialize( dsr ) ;
			}
		}
		else if ( idChunk == CHHDRID_NORM )
		{
			std::string	strNorm = dsr.ReadString() ;
			m_normalizer = NNNormalizationFilter::Make( strNorm.c_str() ) ;
			if ( m_normalizer != nullptr )
			{
				m_normalizer->Deserialize( dsr ) ;
			}
		}
		else if ( idChunk == CHHDRID_GENERATOR )
		{
			std::string	strGen = dsr.ReadString() ;
			m_generator = NNGeneratorFunction::Make( strGen.c_str() ) ;
			if ( m_generator != nullptr )
			{
				m_generator->Deserialize( dsr ) ;
			}
		}
		else if ( idChunk == CHHDRID_CONNECTION )
		{
			uint32_t	lenCon = 0 ;
			dsr.Read( &lenCon, sizeof(lenCon) ) ;
			m_connection.resize( (size_t) lenCon ) ;
			dsr.Read( m_connection.data(), lenCon * sizeof(Connection) ) ;
		}
		else if ( idChunk == CHHDRID_EXTEND )
		{
			if ( !DeserializeExtendInfo( dsr ) )
			{
				return	false ;
			}
		}
		dsr.Ascend() ;
	}

	dsr.Ascend() ;	// CHHDRID_BODY

	Specialize( ) ;

	if ( (m_sampler == nullptr) || (m_activation == nullptr) )
	{
		return	false ;
	}
	return	true ;
}

bool NNPerceptron::DeserializeExtendInfo( NNDeserializer & dsr )
{
	return	true ;
}

// 関数生成準備
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::InitMake( void )
{
	s_mapMakePerceptron.clear() ;
	Register<NNPerceptron>() ;
	Register<NNDepthwisePerceptron>() ;
	Register<NNFixedPerceptron>() ;
	Register<NNIdentityPerceptron>() ;
	Register<NNPointwiseAddPerceptron>() ;
	Register<NNPointwiseMulPerceptron>() ;
}

// 関数生成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNPerceptron> NNPerceptron::Make( const char * pszName )
{
	decltype(s_mapMakePerceptron)::iterator iter = s_mapMakePerceptron.find(pszName) ;
	assert( iter != s_mapMakePerceptron.end() ) ;
	if ( iter != s_mapMakePerceptron.end() )
	{
		return	(iter->second)() ;
	}
	return	nullptr ;
}

// 出力バッファサイズの計算
//////////////////////////////////////////////////////////////////////////////
size_t NNPerceptron::CalcOutputChannels( void ) const
{
	return	m_activation->CalcOutputChannels
				( m_sampler->CalcOutputChannels
					( m_matrix.GetLineCount() ), GetActivationDepthwise() ) ;
}

NNBufDim NNPerceptron::CalcOutputDim( const NNBufDim& dimSrc ) const
{
	NNBufDim	dimInner = CalcInnerDim( dimSrc ) ;
	return	NNBufDim( dimInner.x, dimInner.y,
					m_activation->CalcOutputChannels
						( dimInner.z, GetActivationDepthwise() ) ) ;
}

NNBufDim NNPerceptron::CalcInnerDim( const NNBufDim& dimSrc ) const
{
	NNBufDim	dimInner = m_sampler->CalcOutputDim( dimSrc ) ;
	return	NNBufDim( dimInner.x, dimInner.y,
						m_sampler->CalcOutputChannels( m_matrix.GetLineCount() ) ) ;
}

NNBufDim NNPerceptron::CalcMatrixPointDim( const NNBufDim& dimSrc ) const
{
	return	m_sampler->CalcMatrixPointDim( dimSrc ) ;
}

// 入力バッファサイズの計算
//////////////////////////////////////////////////////////////////////////////
NNBufDim NNPerceptron::CalcInputDim
	( const BufferArray& bufArray,
		size_t iThisLayer, const NNBufDim& dimSrc0 ) const
{
	return	CalcInputDim
		( iThisLayer, dimSrc0, bufArray.size(),
			[&]( size_t iLayer )
				{ return bufArray.at(iLayer)->bufOutput.GetSize() ; } ) ;
}

NNBufDim NNPerceptron::CalcInputDim
	( const std::vector<NNBufDim>& dimArray,
		size_t iThisLayer, const NNBufDim& dimSrc0 ) const
{
	return	CalcInputDim
		( iThisLayer, dimSrc0, dimArray.size(),
			[&]( size_t iLayer ) { return dimArray.at(iLayer) ; } ) ;
}

NNBufDim NNPerceptron::CalcInputDim
	( size_t iThisLayer, const NNBufDim& dimSrc0,
		size_t nLayerCount, std::function<NNBufDim(size_t)> funcGetDim ) const
{
	if ( (m_connection.size() == 0)
		|| ((m_connection.size() == 1)
			&& (m_connection.at(0).iDelay == 0)
			&& (m_connection.at(0).iChannel == 0)
			&& (m_connection.at(0).nChannels == 0)) )
	{
		const int	iLayer = (m_connection.size() == 1)
								? m_connection.at(0).iLayer : 1 ;
		if ( (iLayer <= (int) iThisLayer)
			&& ((size_t)(iThisLayer - iLayer) < nLayerCount) )
		{
			size_t	iRefLayer = iThisLayer - iLayer ;
			return	funcGetDim(iRefLayer) ;
		}
		else
		{
			return	dimSrc0 ;
		}
	}
	NNBufDim	dimInput( 0, 0, 0 ) ;
	for ( auto cn : m_connection )
	{
		if ( cn.iLayer == NNPerceptron::conLayerNull )
		{
			assert( cn.nChannels != 0 ) ;
			if ( dimInput.x * dimInput.y == 0 )
			{
				NNBufDim	dimRef = dimSrc0 ;
				if ( iThisLayer >= 1 )
				{
					dimRef = funcGetDim(iThisLayer - 1) ;
				}
				dimInput.x = dimRef.x ;
				dimInput.y = dimRef.y ;
				dimInput.n = dimInput.x * dimInput.y ;
			}
			dimInput.z += cn.nChannels ;
			continue ;
		}
		NNBufDim	dimRef ;
		if ( (cn.iLayer <= (int) iThisLayer)
			&& ((size_t)(iThisLayer - cn.iLayer) < nLayerCount) )
		{
			size_t	iRefLayer = iThisLayer - cn.iLayer ;
			dimRef = funcGetDim(iRefLayer) ;
		}
		else
		{
			dimRef = dimSrc0 ;
		}
		if ( dimInput.x * dimInput.y == 0 )
		{
			dimInput.x = dimRef.x ;
			dimInput.y = dimRef.y ;
			dimInput.n = dimInput.x * dimInput.y ;
		}
		else
		{
			assert( (dimRef.x == 0) || (dimInput.x + cn.xOffset*2 == dimRef.x) ) ;
			assert( (dimRef.y == 0) || (dimInput.y + cn.xOffset*2 == dimRef.y) ) ;
		}
		if ( cn.nChannels == 0 )
		{
			dimInput.z += dimRef.z ;
		}
		else
		{
			assert( dimRef.z >= cn.iChannel + cn.nChannels ) ;
			dimInput.z += cn.nChannels ;
		}
	}
	return	dimInput ;
}

// 中間バッファの準備
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::ResetBuffer( NNPerceptron::Buffer& bufThis, size_t iThisLayer )
{
	bufThis.forLearning = false ;
	bufThis.reqDelay = false ;
	bufThis.reqDelta2 = false ;
	bufThis.transMatrix = false ;
	bufThis.fixedMatrix = false ;
	bufThis.linearActivation = false ;
	bufThis.sharedInputBuf = false ;
	bufThis.iThisLayer = iThisLayer ;
	bufThis.nRefOut = 0 ;
	bufThis.nRefOut2 = 0 ;
	bufThis.iRefMostRearLayer = iThisLayer + 1 ;
	bufThis.bufMatrix.Free() ;
	bufThis.bufDropoutMask.Free() ;
	bufThis.bufInput.Free() ;
	bufThis.bufInAct.Free() ;
	bufThis.bufOutput.Free() ;
	bufThis.bufDelay.Free() ;
	bufThis.bufPrevDelta.Free() ;
	bufThis.bufInDelta.Free() ;
	bufThis.bufOutDelta.Free() ;
	bufThis.bufGradient.Free() ;
	bufThis.xGradientBlock = 0 ;
	bufThis.yGradientBlock = 0 ;
}

void NNPerceptron::PrepareBuffer
	( Buffer& bufThis, const NNBufDim& dimSrc,
		NNPerceptron::BufferArray& bufArray,
		const NNLoopStream& stream,
		size_t iThisLayer, uint32_t flagsBuffer, bool flagMemoryCommit ) const
{
	uint32_t	cudaMemFlags = NNBuffer::cudaNoMemory ;
	uint32_t	cudaDevFlags = NNBuffer::cudaNoMemory ;
	if ( stream.m_useCuda )
	{
		cudaMemFlags = NNBuffer::cudaAllocate ;
		cudaDevFlags = NNBuffer::cudaAllocate | NNBuffer::cudaDeviceOnly ;
	}
	if ( flagMemoryCommit )
	{
		cudaMemFlags |= NNBuffer::allocateWithCommit ;
		cudaDevFlags |= NNBuffer::allocateWithCommit ;
	}

	bool	flagInSrcOutput = false ;
	if ( m_connection.size() == 0 )
	{
		flagInSrcOutput = true ;
	}
	else if ( (m_connection.size() == 1)
			&& (m_connection.at(0).iLayer != conLayerNull)
			&& (m_connection.at(0).iDelay == 0)
			&& (m_connection.at(0).iChannel == 0) )
	{
		if ( m_connection.at(0).nChannels == 0 )
		{
			flagInSrcOutput = true ;
		}
		else if ( (m_connection.at(0).iLayer >= 1)
				&& (iThisLayer >= m_connection.at(0).iLayer) )
		{
			Buffer *	pBuf = bufArray.at(iThisLayer - m_connection.at(0).iLayer).get() ;
			if ( pBuf->bufOutput.GetSize().z == m_connection.at(0).nChannels )
			{
				flagInSrcOutput = true ;
			}
		}
	}
	if ( flagInSrcOutput )
	{
		bufThis.inSrc = inputSrcOutput ;
	}
	else
	{
		bufThis.inSrc = inputTemporary ;
		//
		SharedBufferMap::const_iterator iter = bufArray.m_mapInputBuf.find(dimSrc) ;
		if ( iter != bufArray.m_mapInputBuf.end() )
		{
			bufThis.sharedInputBuf = true ;
			bufThis.bufInput.DuplicateBuffer( *(iter->second) ) ;
		}
		else
		{
			std::shared_ptr<NNBuffer>	pSharedBuf = std::make_shared<NNBuffer>() ;
			pSharedBuf->Create( dimSrc.x, dimSrc.y, dimSrc.z, 0, cudaDevFlags ) ;
			bufArray.m_mapInputBuf.insert( std::make_pair(dimSrc,pSharedBuf) ) ;
			bufThis.bufInput.DuplicateBuffer( *pSharedBuf ) ;
		}
	}

	bufThis.forLearning = ((flagsBuffer & bufferForLearning) != 0) ;
	bufThis.fixedMatrix = IsMatrixFixed() ;
	bufThis.linearActivation = m_activation->IsLinearActivation() ;

	bufThis.bufMatrix.Allocate
		( m_matrix.GetColumnCount(),
			m_matrix.GetLineCount(),
			1, m_matrix.GetLength(), cudaMemFlags ) ;

	bufThis.bufOutDelta.Allocate( dimSrc.x, dimSrc.y, dimSrc.z, 0, cudaDevFlags ) ;

	uint32_t	flagsOutBuf = ((iThisLayer + 1 == bufArray.size())
										? cudaMemFlags : cudaDevFlags) ;
	NNBufDim	dimInner = CalcInnerDim( dimSrc ) ;
	bufThis.bufInAct.Allocate( dimInner.x, dimInner.y, dimInner.z, 0, flagsOutBuf ) ;

	NNBufDim	dimOut = CalcOutputDim( dimSrc ) ;
	if ( bufThis.linearActivation )
	{
		assert( dimInner == dimOut ) ;
		bufThis.bufInAct.Commit() ;
		bufThis.bufOutput.DuplicateBuffer( bufThis.bufInAct ) ;
	}
	else
	{
		bufThis.bufOutput.Allocate( dimOut.x, dimOut.y, dimOut.z, 0, flagsOutBuf ) ;
	}
	bufThis.bufDelay.Allocate( dimOut.x, dimOut.y, dimOut.z, 0, flagsOutBuf ) ;

	if ( m_normalizer != nullptr )
	{
		m_normalizer->PrepareWorkBuf
			( bufThis.normWorkBuf, dimInner, bufThis.forLearning, stream ) ;
	}

	if ( m_generator != nullptr )
	{
		bufThis.pGenWorkBuf = m_generator->MakeWorkBuffer( dimSrc, stream );
	}

	if ( flagsBuffer & bufferForLearning )
	{
		if ( !(m_behavior & behaviorNoDropout)
			&& (m_dropout != 0.0f) && !(flagsBuffer & bufferNoDropout) )
		{
			bufThis.bufDropoutMask.Create( 1, 1, dimOut.z, 0, cudaMemFlags ) ;
			bufThis.bufDropoutMask.Fill( 1.0f ) ;
		}

		bufThis.bufPrevDelta.Allocate( dimOut.x, dimOut.y, dimOut.z, 0, cudaDevFlags ) ;
		bufThis.bufPrevDelta2.Allocate( dimOut.x, dimOut.y, dimOut.z, 0, cudaDevFlags ) ;

		if ( bufThis.linearActivation )
		{
			assert( dimInner == dimOut ) ;
			bufThis.bufPrevDelta.Commit() ;
			bufThis.bufInDelta.DuplicateBuffer( bufThis.bufPrevDelta ) ;
		}
		else
		{
			bufThis.bufInDelta.Allocate
				( dimInner.x, dimInner.y, dimInner.z, 0, cudaDevFlags ) ;
		}

		if ( stream.m_useCuda )
		{
			NNBufDim	dimMatrix = CalcMatrixPointDim( dimSrc ) ;
			bufThis.xGradientBlock = nncuda_CalcMatrixGradientBlockSizeX( dimMatrix.x, dimMatrix.y ) ;
			bufThis.yGradientBlock = nncuda_CalcMatrixGradientBlockSizeY( dimMatrix.x, dimMatrix.y ) ;
			bufThis.bufGradient.Allocate
				( nncuda_CalcMatrixGradientBlockX( dimMatrix.x, dimMatrix.y ),
					nncuda_CalcMatrixGradientBlockY( dimMatrix.x, dimMatrix.y ),
					m_matrix.GetLength(), 0, cudaMemFlags ) ;
		}
	}

	// 入力元レイヤー逆リンク情報
	if ( m_connection.size() == 0 )
	{
		if ( iThisLayer >= 1 )
		{
			Buffer *	pRef = bufArray.at(iThisLayer - 1).get() ;
			pRef->nRefOut ++ ;
			pRef->iRefMostRearLayer =
					__max( pRef->iRefMostRearLayer, iThisLayer + 1 ) ;
		}
	}
	else
	{
		for ( auto cn : m_connection )
		{
			if ( ((int) iThisLayer >= cn.iLayer)
				&& (cn.iLayer != conLayerNull)
				&& ((size_t) (iThisLayer - cn.iLayer) < bufArray.size()) )
			{
				Buffer *	pRef = bufArray.at(iThisLayer - cn.iLayer).get() ;
				if ( cn.iDelay >= 1 )
				{
					pRef->reqDelay = true ;
				}
				if ( cn.iLayer <= 0 )
				{
					pRef->reqDelta2 = true ;
					pRef->nRefOut2 ++ ;
				}
				else
				{
					pRef->nRefOut ++ ;
				}
				pRef->iRefMostRearLayer =
						__max( pRef->iRefMostRearLayer, iThisLayer + 1 ) ;
			}
		}
	}
}

// 省メモリモードでの予測処理で不要なバッファの解放
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::LowMemoryBuffer
	( NNPerceptron::Buffer& bufThis, size_t iPredictedLayer )
{
	if ( bufThis.iThisLayer < iPredictedLayer )
	{
		bufThis.bufMatrix.Uncommit() ;
		bufThis.bufInput.Uncommit() ;
		bufThis.bufPrevDelta.Uncommit() ;
		bufThis.bufInDelta.Uncommit() ;
		bufThis.bufOutDelta.Uncommit() ;
	}
	if ( bufThis.iRefMostRearLayer < iPredictedLayer )
	{
		bufThis.bufInAct.Uncommit() ;
		bufThis.bufOutput.Uncommit() ;
		bufThis.bufDelay.Uncommit() ;
	}
}

// 作業バッファの準備
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::PrepareWorkArray
	( NNPerceptron::CPUWorkArray& bufWorks, size_t nCount ) const
{
	bufWorks.resize( nCount ) ;
	PrepareLossAndGradientBuf( bufWorks ) ;

	for ( CPUWorkBuf& work : bufWorks )
	{
		PrepareWorkBuf( work ) ;
	}

	ResetWorkArrayInBatch( bufWorks ) ;
}

void NNPerceptron::PrepareWorkBuf( NNPerceptron::CPUWorkBuf& bufWork ) const
{
	PrepareLossAndGradient( bufWork );
	bufWork.vecSrc.resize( m_matrix.GetColumnCount() ) ;
	bufWork.vecDst.resize( m_matrix.GetLineCount() ) ;
	bufWork.vecDiff.resize( m_matrix.GetLineCount() ) ;
	bufWork.vecOutDelta.resize( m_matrix.GetColumnCount() ) ;

	ResetWorkBufInBatch( bufWork ) ;
}

void NNPerceptron::PrepareLossAndGradientBuf( LossAndGradientBuf& lagb ) const
{
	PrepareLossAndGradient( lagb ) ;

	for ( int i = 0; i < 2; i ++ )
	{
		lagb.matAdaOpt[i].Create( m_matrix.GetLineCount(), m_matrix.GetColumnCount() ) ;
		lagb.matAdaOpt[i].InitDiagonal( 0.0f ) ;
		//
		if ( (m_matAdaOpt[i].GetLineCount() == m_matrix.GetLineCount())
			&& (m_matAdaOpt[i].GetColumnCount() == m_matrix.GetColumnCount()) )
		{
			lagb.matAdaOpt[i] = m_matAdaOpt[i] ;
		}
	}

	if ( m_normalizer != nullptr )
	{
		m_normalizer->PrepareGradBuf( lagb.normGrad ) ;
	}
}

void NNPerceptron::PrepareLossAndGradient( LossAndGradient& lag ) const
{
	lag.matGradient.Create( m_matrix.GetLineCount(), m_matrix.GetColumnCount() ) ;

}

// ミニバッチ毎の処理
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::PrepareForMiniBatch
	( NNPerceptron::Buffer& bufThis,
		uint32_t flagsBuffer,
		std::random_device::result_type rndSeed,
		NNLoopStream& stream ) const
{
	if ( bufThis.bufDropoutMask.IsCommitted() )
	{
		NNBufDim	dimMask = bufThis.bufDropoutMask.GetSize() ;

		if ( !(m_behavior & (behaviorNoDropout | behaviorFixed))
			&& (m_dropout > 0.0f)
			&& (flagsBuffer & bufferForLearning)
			&& !(flagsBuffer & bufferNoDropout) )
		{
			bufThis.bufDropoutMask.Fill( 1.0f / (1.0f - m_dropout) ) ;

			std::mt19937	engine( rndSeed ) ;
			float *			pMask = bufThis.bufDropoutMask.GetBuffer() ;
			for ( size_t i = 0; i < dimMask.z; i ++ )
			{
				if ( ((engine() % 1000000) / 1000000.0f) < m_dropout )
				{
					pMask[i] = 0.0f ;
				}
			}
		}
		else
		{
			bufThis.bufDropoutMask.Fill( 1.0f ) ;
		}
		if ( stream.m_useCuda )
		{
			bufThis.bufDropoutMask.CudaAsyncToDevice( stream.m_cudaStream ) ;
			stream.m_cudaStream.VerifySync() ;
		}
	}
}

// 勾配反映後のバッファ処理
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::ResetBufferInBatch( NNPerceptron::Buffer& bufThis ) const
{
	bufThis.transMatrix = false ;

	if ( m_normalizer != nullptr )
	{
		m_normalizer->ResetWorkBuf( bufThis.normWorkBuf ) ;
	}
}

// 行列勾配・損失合計値初期化
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::ResetWorkArrayInBatch( NNPerceptron::CPUWorkArray& bufWorks ) const
{
	for ( CPUWorkBuf& work : bufWorks )
	{
		ResetWorkBufInBatch( work ) ;
	}
	bufWorks.ResetGradient() ;
	bufWorks.ResetLoss() ;
}

void NNPerceptron::ResetWorkBufInBatch( NNPerceptron::CPUWorkBuf& bufWork ) const
{
	ResetLossAndGradient( bufWork ) ;
}

void NNPerceptron::ResetLossAndGradient( LossAndGradient& lag ) const
{
	lag.ResetGradient() ;
	lag.ResetLoss() ;
}

// エポック開始時処理
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::OnBeginEpoch( void )
{
	if ( m_normalizer != nullptr )
	{
		m_normalizer->OnBeginEpoch() ;
	}
}

// エポック終了時処理
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::OnEndEpoch
	( NNPerceptron::Buffer& bufThis, NNPerceptron::CPUWorkArray& bufWorks )
{
	if ( m_normalizer != nullptr )
	{
		m_normalizer->OnEndEpoch( bufThis.normWorkBuf ) ;
	}
}

// 損失・勾配加算
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::AddLossAndGradient
	( NNPerceptron::LossAndGradientBuf& lagDst,
		NNPerceptron::LossAndGradientBuf& lagSrc ) const
{
	lagDst.matGradient += lagSrc.matGradient ;
	lagDst.nGradient += lagSrc.nGradient ;
	lagDst.fpLoss += lagSrc.fpLoss ;
	lagDst.nLossSamples += lagSrc.nLossSamples ;

	if ( m_normalizer != nullptr )
	{
		lagDst.normGrad.AddGradient( lagSrc.normGrad ) ;
	}
}

// 入力バッファの準備
//////////////////////////////////////////////////////////////////////////////
NNPerceptron::InputBuffer NNPerceptron::PrepareInput
	( const NNPerceptron::BufferArray& bufArray,
		size_t iThisLayer, NNBuffer& bufInput0,
		size_t iFirstInputLayer, NNLoopStream& stream )
{
	InputBuffer	inBuf ;
	inBuf.pInput = nullptr ;
	inBuf.nMultiInput = 0 ;

	std::shared_ptr<Buffer>	pbufThis = bufArray.at( iThisLayer ) ;
	if ( pbufThis->inSrc == inputSrcOutput )
	{
		size_t	iOffsetLayer = 1 ;
		if ( m_connection.size() == 1 )
		{
			iOffsetLayer = m_connection.at(0).iLayer ;
		}
		if ( iOffsetLayer + iFirstInputLayer <= iThisLayer )
		{
			size_t	iRefLayer = iThisLayer - iOffsetLayer ;
			inBuf.pInput = &(bufArray.at(iRefLayer)->bufOutput) ;
		}
		else
		{
			inBuf.pInput = &bufInput0 ;
		}
	}
	else
	{
		typedef	std::function<void(NNBuffer&,size_t,const NNBuffer&,int,int,size_t,size_t)>	FuncShiftCopy ;
		FuncShiftCopy	cpuShiftCopy =
			[]( NNBuffer& nnDstBuf, size_t iDstChannel,
				const NNBuffer& nnSrcBuf, int xShiftSample, int yShiftSample,
				size_t iSrcChannel, size_t nSrcChCount )
			{
				nnDstBuf.ShiftCopyChannelFrom
					( iDstChannel, nnSrcBuf,
						xShiftSample, yShiftSample, iSrcChannel, nSrcChCount ) ;
			} ;
		FuncShiftCopy	cudaShiftCopy =
			[&]( NNBuffer& nnDstBuf, size_t iDstChannel,
				const NNBuffer& nnSrcBuf, int xShiftSample, int yShiftSample,
				size_t iSrcChannel, size_t nSrcChCount )
			{
				nnDstBuf.CudaCopyChannelFrom
					( 0, 0, iDstChannel, nnSrcBuf, xShiftSample, yShiftSample,
						iSrcChannel, nSrcChCount, 0, 0, stream.m_cudaStream ) ;
			} ;
		FuncShiftCopy&	ShiftCopy = stream.m_useCuda ? cudaShiftCopy : cpuShiftCopy ;
		//
		inBuf.pInput = &(pbufThis->bufInput) ;
		inBuf.pInput->Commit() ;
		//
		size_t	chNext = 0 ;
		for ( auto cn : m_connection )
		{
			if ( cn.iLayer == conLayerNull )
			{
				if ( m_generator != nullptr )
				{
					m_generator->Generate
						( pbufThis->bufInput,
							pbufThis->pGenWorkBuf.get(),
							chNext, cn.nChannels, stream ) ;
				}
				chNext += cn.nChannels ;
				continue ;
			}
			NNBuffer *	pInputBuf = &bufInput0 ;
			int			iDelay = 0 ;
			if ( (cn.iLayer + (int) iFirstInputLayer <= (int) iThisLayer)
				&& ((size_t) (iThisLayer - cn.iLayer) < bufArray.size()) )
			{
				size_t	iRefLayer = iThisLayer - cn.iLayer ;
				if ( (cn.iDelay >= 1) && bufArray.at(iRefLayer)->bufDelay.IsCommitted() )
				{
					pInputBuf = &(bufArray.at(iRefLayer)->bufDelay) ;
					iDelay = cn.iDelay ;
				}
				else
				{
					pInputBuf = &(bufArray.at(iRefLayer)->bufOutput) ;
				}
			}
			if ( stream.m_useCuda )
			{
				pInputBuf->CommitCuda() ;
			}
			else
			{
				pInputBuf->Commit() ;
			}
			NNBufDim	dimRef = pInputBuf->GetSize() ;
			if ( cn.nChannels == 0 )
			{
				assert( cn.iChannel == 0 ) ;
				ShiftCopy
					( *(inBuf.pInput), chNext,
						*pInputBuf, iDelay - cn.xOffset, - cn.yOffset, 0, dimRef.z ) ;
				chNext += dimRef.z ;
			}
			else
			{
				assert( dimRef.z >= cn.iChannel + cn.nChannels ) ;
				ShiftCopy
					( *(inBuf.pInput), chNext,
						*pInputBuf, iDelay - cn.xOffset, - cn.yOffset,
						cn.iChannel, cn.nChannels ) ;
				chNext += cn.nChannels ;
			}
		}
	}

	return	inBuf ;
}

// 予測処理
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::Prediction
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis,
		const NNPerceptron::InputBuffer bufInput,
		NNLoopStream& stream, size_t xLeftBounds )
{
	if ( stream.m_useCuda )
	{
		cudaPrediction( bufWorks, bufThis, bufInput, stream, xLeftBounds ) ;
	}
	else
	{
		cpuPrediction( bufWorks, bufThis, bufInput, stream, xLeftBounds ) ;
	}
}

void NNPerceptron::cpuPrediction
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis,
		const NNPerceptron::InputBuffer bufInput,
		NNLoopStream& stream, size_t xLeftBounds )
{
	const NNBufDim	dimInAct = bufThis.bufInAct.GetSize() ;
	const NNBufDim	dimOutput = bufThis.bufOutput.GetSize() ;
	const NNBufDim	dimInput = bufInput.pInput->GetSize() ;
	const NNBufDim	dimDropout = bufInput.pInput->GetSize() ;
	assert( dimInAct.x == dimOutput.x ) ;
	assert( dimInAct.y == dimOutput.y ) ;
	assert( dimInAct.n == dimOutput.n ) ;
	assert( m_sampler->CalcOutputChannels(m_matrix.GetLineCount()) == dimInAct.z ) ;
	assert( !bufThis.bufDropoutMask.IsCommitted() || (bufThis.bufDropoutMask.GetSize().z == dimOutput.z) ) ;

	bufThis.bufInAct.Commit() ;
	if ( bufThis.linearActivation )
	{
		assert( bufThis.bufInAct.IsEqualBuffer( bufThis.bufOutput ) ) ;
	}
	else
	{
		bufThis.bufOutput.Commit() ;
	}

	if ( m_behavior & behaviorDisabled )
	{
		if ( !bufThis.bufInAct.IsEqualBuffer( bufThis.bufOutput ) )
		{
			bufThis.bufInAct.Fill( 0.0f ) ;
		}
		bufThis.bufOutput.Fill( 0.0f ) ;
		return ;
	}

	const bool	flagDropout = !(m_behavior & (behaviorFixed | behaviorNoDropout))
								&& (m_dropout != 0.0f)
								&& bufThis.bufDropoutMask.IsCommitted() ;

	if ( m_normalizer != nullptr )
	{
		// 行列
		stream.m_ploop.Loop( 0, dimOutput.y, [&]( size_t iThread, size_t y )
		{
			const float *			pInputBuf = bufInput.pInput->GetConstBuffer() ;
			const size_t			nSrcChannels = m_matrix.GetColumnCount() ;
			float *					pInAct = bufThis.bufInAct.GetBufferAt( xLeftBounds, y ) ;
			float *					pSrcVec = bufWorks.at(iThread).vecSrc.data() ;
			const size_t			nDepthwise = GetDepthwise() ;
			const size_t			xMatrixBias = m_matrix.GetColumnCount() - m_bias ;
			NNSamplingFilter *		pSampler = m_sampler.get() ;

			assert( bufWorks.at(iThread).vecSrc.size() >= nSrcChannels ) ;

			for ( size_t x = xLeftBounds; x < dimOutput.x; x ++ )
			{
				pSampler->cpuMatrix
					( pInAct, (int) x, (int) y, m_matrix,
						pSrcVec, nSrcChannels,
						pInputBuf, dimInput, xMatrixBias, nDepthwise ) ;
				pInAct += dimInAct.z ;
			}
		} ) ;

		// 正規化
		if ( bufThis.forLearning )
		{
			m_normalizer->cpuAggregateSample
				( bufThis.normWorkBuf, bufThis.bufInAct, stream ) ;
		}
		m_normalizer->cpuNormalize
			( bufThis.bufInAct, bufThis.normWorkBuf, stream, xLeftBounds ) ;

		// 活性化関数とドロップアウト
		if ( !bufThis.linearActivation || flagDropout )
		{
			stream.m_ploop.Loop( 0, dimOutput.y, [&]( size_t iThread, size_t y )
			{
				const float *			pInputBuf = bufInput.pInput->GetConstBuffer() ;
				const float *			pDropout = flagDropout ? bufThis.bufDropoutMask.GetConstBuffer() : nullptr ;
				const size_t			nSrcChannels = m_matrix.GetColumnCount() ;
				float *					pInAct = bufThis.bufInAct.GetBufferAt( xLeftBounds, y ) ;
				float *					pOutput = bufThis.bufOutput.GetBufferAt( xLeftBounds, y ) ;
				const size_t			nDepthwise = GetActivationDepthwise() ;
				NNActivationFunction *	pActivation = m_activation.get() ;

				for ( size_t x = xLeftBounds; x < dimOutput.x; x ++ )
				{
					pActivation->cpuFunction( pOutput, pInAct, dimInAct.z, nDepthwise ) ;

					if ( pDropout != nullptr )
					{
						for ( size_t i = 0; i < dimOutput.z; i ++ )
						{
							pOutput[i] *= pDropout[i] ;
						}
					}
					pInAct += dimInAct.z ;
					pOutput += dimOutput.z ;
				}
			} ) ;
		}
	}
	else
	{
		// 正規化無し
		stream.m_ploop.Loop( 0, dimOutput.y, [&]( size_t iThread, size_t y )
		{
			const float *			pInputBuf = bufInput.pInput->GetConstBuffer() ;
			const float *			pDropout = flagDropout ? bufThis.bufDropoutMask.GetConstBuffer() : nullptr ;
			const size_t			nSrcChannels = m_matrix.GetColumnCount() ;
			float *					pInAct = bufThis.bufInAct.GetBufferAt( xLeftBounds, y ) ;
			float *					pOutput = bufThis.bufOutput.GetBufferAt( xLeftBounds, y ) ;
			float *					pSrcVec = bufWorks.at(iThread).vecSrc.data() ;
			const size_t			nDepthwise = GetDepthwise() ;
			const size_t			nActDepthwise = GetActivationDepthwise() ;
			const size_t			xMatrixBias = m_matrix.GetColumnCount() - m_bias ;
			NNSamplingFilter *		pSampler = m_sampler.get() ;
			NNActivationFunction *	pActivation = m_activation.get() ;

			assert( bufWorks.at(iThread).vecSrc.size() >= nSrcChannels ) ;

			for ( size_t x = xLeftBounds; x < dimOutput.x; x ++ )
			{
				pSampler->cpuMatrix
					( pInAct, (int) x, (int) y, m_matrix,
						pSrcVec, nSrcChannels,
						pInputBuf, dimInput, xMatrixBias, nDepthwise ) ;
				pActivation->cpuFunction( pOutput, pInAct, dimInAct.z, nActDepthwise ) ;

				if ( pDropout != nullptr )
				{
					for ( size_t i = 0; i < dimOutput.z; i ++ )
					{
						pOutput[i] *= pDropout[i] ;
					}
				}
				pInAct += dimInAct.z ;
				pOutput += dimOutput.z ;
			}
		} ) ;
	}

	bufThis.bufInAct.CheckOverun() ;
	bufThis.bufOutput.CheckOverun() ;
}

void NNPerceptron::cudaPrediction
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis,
		const NNPerceptron::InputBuffer bufInput,
		NNLoopStream& stream, size_t xLeftBounds )
{
	const NNBufDim	dimInAct = bufThis.bufInAct.GetSize() ;
	const NNBufDim	dimOutput = bufThis.bufOutput.GetSize() ;
	const NNBufDim	dimInput = bufInput.pInput->GetSize() ;
	assert( dimInAct.x == dimOutput.x ) ;
	assert( dimInAct.y == dimOutput.y ) ;
	assert( dimInAct.n == dimOutput.n ) ;
	assert( m_sampler->CalcOutputChannels(m_matrix.GetLineCount()) == dimInAct.z ) ;

	bufThis.bufMatrix.CommitCuda() ;
	bufInput.pInput->CommitCuda() ;
	bufThis.bufInAct.CommitCuda() ;
	if ( bufThis.linearActivation )
	{
		assert( bufThis.bufInAct.IsEqualBuffer( bufThis.bufOutput ) ) ;
	}
	else
	{
		bufThis.bufOutput.CommitCuda() ;
	}

	if ( m_behavior & behaviorDisabled )
	{
		if ( !bufThis.bufInAct.IsEqualBuffer( bufThis.bufOutput ) )
		{
			bufThis.bufInAct.CudaFill( 0.0f, stream.m_cudaStream ) ;
		}
		bufThis.bufOutput.CudaFill( 0.0f, stream.m_cudaStream ) ;
		return ;
	}

	if ( !bufThis.transMatrix )
	{
		bufThis.bufMatrix.CudaCopyAsyncFrom
			( m_matrix.GetConstArray(), m_matrix.GetLength(), stream.m_cudaStream ) ;
		stream.m_cudaStream.VerifySync() ;
		bufThis.transMatrix = true ;
	}

	m_sampler->cudaMatrix
		( bufThis.bufInAct.GetCudaPtr(), dimInAct,
			bufInput.pInput->GetCudaPtr(), dimInput,
			bufThis.bufMatrix.GetCudaPtr(),
			m_matrix.GetColumnCount(),
			m_matrix.GetLineCount(),
			m_matrix.GetColumnCount() - m_bias,
			xLeftBounds, (int) GetDepthwise(), stream.m_cudaStream ) ;
	stream.m_cudaStream.VerifySync() ;

	if ( m_normalizer != nullptr )
	{
		if ( bufThis.forLearning )
		{
			m_normalizer->cudaAggregateSample
				( bufThis.normWorkBuf, bufThis.bufInAct, stream ) ;
		}
		m_normalizer->cudaNormalize
			( bufThis.bufInAct, bufThis.normWorkBuf, stream, xLeftBounds ) ;
	}

	if ( !bufThis.linearActivation )
	{
		m_activation->cudaFunction
			( bufThis.bufOutput.GetCudaPtr(), dimOutput,
				bufThis.bufInAct.GetCudaPtr(), dimInAct,
				xLeftBounds, GetActivationDepthwise(), stream.m_cudaStream ) ;
		stream.m_cudaStream.VerifySync() ;
	}

	if ( !(m_behavior & (behaviorFixed | behaviorNoDropout))
		&& (m_dropout != 0.0f) && bufThis.bufDropoutMask.IsCommitted() )
	{
		nncuda_MaskPattern
			( bufThis.bufOutput.GetCudaPtr(), dimOutput,
				bufThis.bufDropoutMask.GetCudaPtr(), stream.m_cudaStream ) ;
		stream.m_cudaStream.VerifySync() ;
	}
}

// 損失計算
//////////////////////////////////////////////////////////////////////////////
double NNPerceptron::cpuCalcLoss
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis,
		const NNBuffer& bufTeaching,
		NNLoopStream& stream, NNLossFunction * pLossFunc )
{
	const NNBufDim	dimOutput = bufThis.bufOutput.GetSize() ;
	const NNBufDim	dimInAct = bufThis.bufInAct.GetSize() ;
	const NNBufDim	dimTeaching = bufTeaching.GetSize() ;

	// ※最終レイヤーの活性化関数はチャネル数を変換しないこと
	assert( dimOutput.x <= dimTeaching.x ) ;
	assert( dimOutput.y <= dimTeaching.y ) ;
	assert( dimOutput.x == dimInAct.x ) ;
	assert( dimOutput.y == dimInAct.y ) ;
	assert( dimOutput.z == m_activation->CalcOutputChannels(dimInAct.z,GetActivationDepthwise()) ) ;

	if ( pLossFunc == nullptr )
	{
		pLossFunc = m_activation.get() ;
	}
	assert( pLossFunc->IsValidTeachingChannels(dimInAct.z,GetActivationDepthwise(),dimTeaching.z) ) ;

	bufThis.bufInAct.Commit() ;

	stream.m_ploop.Loop( 0, dimOutput.y, [&]( size_t iThread, size_t y )
	{
		const float *	pOutput = bufThis.bufOutput.GetConstBufferAt( 0, y ) ;
		const float *	pInAct = bufThis.bufInAct.GetBufferAt( 0, y ) ;
		double	loss = 0.0 ;
		for ( size_t x = 0; x < dimOutput.x; x ++ )
		{
			loss += pLossFunc->cpuLoss
				( pInAct, pOutput,
					bufTeaching.GetConstBufferAt(x,y),
					dimInAct.z, GetActivationDepthwise() ) ;
			pInAct += dimInAct.z ;
			pOutput += dimOutput.z ;
		}
		CPUWorkBuf&	bufWork = bufWorks.at( iThread ) ;
		bufWork.fpLoss += loss ;
		bufWork.nLossSamples += dimOutput.x ;
	} ) ;

	for ( CPUWorkBuf& work : bufWorks )
	{
		bufWorks.fpLoss += work.fpLoss ;
		bufWorks.nLossSamples += work.nLossSamples ;
		work.fpLoss = 0 ;
		work.nLossSamples = 0 ;
	}
	return	bufWorks.fpLoss / (double) bufWorks.nLossSamples ;
}

// 出力を遅延バッファにコピー
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::CopyToDelayBuffer
	( NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	if ( stream.m_useCuda )
	{
		cudaCopyToDelayBuffer( bufThis, stream ) ;
	}
	else
	{
		cpuCopyToDelayBuffer( bufThis ) ;
	}
}

void NNPerceptron::cpuCopyToDelayBuffer( NNPerceptron::Buffer& bufThis )
{
	if ( bufThis.reqDelay )
	{
		bufThis.bufDelay.Commit() ;
		bufThis.bufDelay.CopyFrom( bufThis.bufOutput ) ;
	}
}

void NNPerceptron::cudaCopyToDelayBuffer
	( NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	if ( bufThis.reqDelay )
	{
		bufThis.bufDelay.CommitCuda() ;
		bufThis.bufDelay.CudaCopyFrom( bufThis.bufOutput, stream.m_cudaStream ) ;
	}
}

// 遅延バッファをシフト
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::ShiftBufferWithStreaming
	( NNPerceptron::Buffer& bufThis,
			size_t xShift, NNLoopStream& stream )
{
	if ( stream.m_useCuda )
	{
		cudaShiftBufferWithStreaming( bufThis, xShift, stream ) ;
	}
	else
	{
		cpuShiftBufferWithStreaming( bufThis, xShift ) ;
	}
}

void NNPerceptron::cpuShiftBufferWithStreaming
	( NNPerceptron::Buffer& bufThis, size_t xShift )
{
	if ( bufThis.reqDelay )
	{
		bufThis.bufDelay.Commit() ;
		bufThis.bufDelay.CopyFrom( bufThis.bufOutput, NNBufDim( xShift, 0, 0 ) ) ;
	}
	else if ( bufThis.bufOutput.IsCommitted() )
	{
		bufThis.bufOutput.ShiftCopyChannelFrom
				( 0, bufThis.bufOutput, - (int) xShift, 0 ) ;
	}
}

void NNPerceptron::cudaShiftBufferWithStreaming
	( NNPerceptron::Buffer& bufThis,
			size_t xShift, NNLoopStream& stream )
{
	if ( bufThis.reqDelay )
	{
		NNBufDim	dimOutput = bufThis.bufOutput.GetSize() ;
		bufThis.bufDelay.CommitCuda() ;
		bufThis.bufDelay.CudaCopyChannelFrom
			( 0, 0, 0, bufThis.bufOutput, - (int) xShift, 0,
				0, dimOutput.z, dimOutput.x, dimOutput.y, stream.m_cudaStream ) ;
	}
	else if ( bufThis.bufOutput.IsCommitted() )
	{
		assert( bufThis.bufOutput.IsCommittedCuda() ) ;
		const NNBufDim	dimOutput = bufThis.bufOutput.GetSize() ;
		bufThis.bufDelay.CommitCuda() ;
		bufThis.bufDelay.CudaCopyChannelFrom
			( 0, 0, 0, bufThis.bufOutput, - (int) xShift, 0,
				0, dimOutput.z, dimOutput.x, dimOutput.y, stream.m_cudaStream ) ;
		bufThis.bufOutput.SwapBuffer( bufThis.bufDelay ) ;
		//
		if ( bufThis.linearActivation )
		{
			assert( bufThis.bufDelay.IsEqualBuffer( bufThis.bufInAct ) ) ;
			bufThis.bufInAct.DuplicateBuffer( bufThis.bufOutput ) ;
		}
	}
}

// 遅延バッファに教師データをコピー
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::CopyTeachingDataToDelayBuffer
	( NNPerceptron::Buffer& bufThis,
		NNBuffer& bufTeaching, NNLoopStream& stream )
{
	if ( stream.m_useCuda )
	{
		cudaCopyTeachingDataToDelayBuffer( bufThis, bufTeaching, stream ) ;
	}
	else
	{
		cpuCopyTeachingDataToDelayBuffer( bufThis, bufTeaching ) ;
	}
}

void NNPerceptron::cpuCopyTeachingDataToDelayBuffer
	( NNPerceptron::Buffer& bufThis, const NNBuffer& bufTeaching )
{
	if ( bufThis.reqDelay )
	{
		bufThis.bufDelay.Commit() ;
		bufThis.bufDelay.CopyFrom( bufTeaching ) ;
	}
}

void NNPerceptron::cudaCopyTeachingDataToDelayBuffer
	( NNPerceptron::Buffer& bufThis,
			NNBuffer& bufTeaching, NNLoopStream& stream )
{
	if ( bufThis.reqDelay )
	{
		NNBufDim	dimTeaching = bufTeaching.GetSize() ;
		bufThis.bufDelay.CommitCuda() ;
		bufTeaching.CommitCuda() ;
		bufThis.bufDelay.CudaCopyChannelFrom
			( 0, 0, 0, bufTeaching, 0, 0,
				0, dimTeaching.z, dimTeaching.x, dimTeaching.y, stream.m_cudaStream ) ;
	}
}

// 逆伝播用バッファクリア
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::ClearDeltaBuffer
	( NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	if ( stream.m_useCuda )
	{
		cudaClearDeltaBuffer( bufThis, stream ) ;
	}
	else
	{
		cpuClearDeltaBuffer( bufThis ) ;
	}
}

void NNPerceptron::cpuClearDeltaBuffer( NNPerceptron::Buffer& bufThis )
{
	bufThis.bufPrevDelta.Commit() ;
	bufThis.bufPrevDelta.Fill( 0.0f ) ;

	if ( bufThis.reqDelta2 )
	{
		bufThis.bufPrevDelta2.Commit() ;
		bufThis.bufPrevDelta2.Fill( 0.0f ) ;
	}
}

void NNPerceptron::cudaClearDeltaBuffer
	( NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	bufThis.bufPrevDelta.CommitCuda() ;
	bufThis.bufPrevDelta.CudaFill( 0.0f, stream.m_cudaStream ) ;
	stream.m_cudaStream.VerifySync() ;

	if ( bufThis.reqDelta2 )
	{
		bufThis.bufPrevDelta2.CommitCuda() ;
		bufThis.bufPrevDelta2.CudaFill( 0.0f, stream.m_cudaStream ) ;
		stream.m_cudaStream.VerifySync() ;
	}
}

// 逆伝播用バッファを２パス用からコピー／又はクリア
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::SwitchDeltaSecondaryBuffer
	( NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	if ( bufThis.reqDelta2 )
	{
		if ( stream.m_useCuda )
		{
			cudaSwitchDeltaSecondaryBuffer( bufThis, stream ) ;
		}
		else
		{
			cpuSwitchDeltaSecondaryBuffer( bufThis ) ;
		}
	}
	else
	{
		ClearDeltaBuffer( bufThis, stream ) ;
	}
}

void NNPerceptron::cpuSwitchDeltaSecondaryBuffer( NNPerceptron::Buffer& bufThis )
{
	assert( bufThis.reqDelta2 ) ;
	bufThis.bufPrevDelta.CopyFrom( bufThis.bufPrevDelta2 ) ;
}

void NNPerceptron::cudaSwitchDeltaSecondaryBuffer
	( NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	assert( bufThis.reqDelta2 ) ;
	NNBufDim	dimDelta = bufThis.bufPrevDelta.GetSize() ;
	bufThis.bufPrevDelta.CudaCopyChannelFrom
		( 0, 0, 0, bufThis.bufPrevDelta2, 0, 0,
			0, dimDelta.z, dimDelta.x, dimDelta.y, stream.m_cudaStream ) ;
}

// 損失関数δ計算
//////////////////////////////////////////////////////////////////////////////
double NNPerceptron::LossDelta
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis,
		NNBuffer& bufTeaching,
		NNLoopStream& stream, NNLossFunction * pLossFunc )
{
	if ( stream.m_useCuda )
	{
		return	cudaLossDelta
					( bufWorks, bufThis, bufTeaching, stream, pLossFunc ) ;
	}
	else
	{
		return	cpuLossDelta
					( bufWorks, bufThis, bufTeaching, stream, pLossFunc ) ;
	}
}

double NNPerceptron::cpuLossDelta
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis,
		const NNBuffer& bufTeaching,
		NNLoopStream& stream, NNLossFunction * pLossFunc )
{
	const NNBufDim	dimOutput = bufThis.bufOutput.GetSize() ;
	const NNBufDim	dimInAct = bufThis.bufInAct.GetSize() ;
	const NNBufDim	dimTeaching = bufTeaching.GetSize() ;

	// ※最終レイヤーの活性化関数はチャネル数を変換しないこと
	// （pLossFunc==nullptr の時、bufPrevDelta ではなく bufInDelta へ直接出力する）
	assert( dimOutput.x <= dimTeaching.x ) ;
	assert( dimOutput.y <= dimTeaching.y ) ;
	assert( dimOutput.x == dimInAct.x ) ;
	assert( dimOutput.y == dimInAct.y ) ;
	assert( dimOutput.z == m_activation->CalcOutputChannels(dimInAct.z,GetActivationDepthwise()) ) ;
	assert( dimInAct == bufThis.bufInDelta.GetSize() ) ;

	bool	flagActDelta = false ;
	if ( pLossFunc == nullptr )
	{
		pLossFunc = m_activation.get() ;
	}
	else
	{
		assert( dimInAct == bufThis.bufPrevDelta.GetSize() ) ;
		flagActDelta = true ;
		bufThis.bufPrevDelta.Commit() ;
	}
	assert( pLossFunc->IsValidTeachingChannels(dimInAct.z,GetActivationDepthwise(),dimTeaching.z) ) ;

	bufThis.bufInDelta.Commit() ;

	size_t	xLeft = 0 ;
	size_t	xRight = dimOutput.x ;
	size_t	yTop = 0 ;
	size_t	yBottom = dimOutput.y ;
	if ( m_behavior & behaviorInvalidMargin )
	{
		xLeft = m_xInvalidLeft ;
		xRight -= __min( dimOutput.x, m_xInvalidRight ) ;
		yTop = m_yInvalidTop ;
		yBottom -= __min( dimOutput.y, m_yInvalidBottom ) ;
	}
	stream.m_ploop.Loop( yTop, yBottom, [&]( size_t iThread, size_t y )
	{
		const float *	pOutput = bufThis.bufOutput.GetConstBufferAt( xLeft, y ) ;
		const float *	pInAct = bufThis.bufInAct.GetConstBufferAt( xLeft, y ) ;
		float *			pDelta = flagActDelta
									? bufThis.bufPrevDelta.GetBufferAt( xLeft, y )
									: bufThis.bufInDelta.GetBufferAt( xLeft, y ) ;
		const float *	pTeaching = bufTeaching.GetConstBufferAt( xLeft, y ) ;
		const size_t	nDepthwise = GetActivationDepthwise() ;
		double			loss = 0.0 ;
		for ( size_t x = xLeft; x < xRight; x ++ )
		{
			pLossFunc->cpuLossDelta
				( pDelta, pInAct, pOutput, pTeaching, dimInAct.z, nDepthwise ) ;
			loss += pLossFunc->cpuLoss
				( pInAct, pOutput, pTeaching, dimInAct.z, nDepthwise ) ;
			pDelta += dimInAct.z ;
			pInAct += dimInAct.z ;
			pOutput += dimOutput.z ;
			pTeaching += dimTeaching.z ;
		}
		CPUWorkBuf&	bufWork = bufWorks.at( iThread ) ;
		bufWork.fpLoss += loss ;
		bufWork.nLossSamples += dimOutput.x ;
	} ) ;

	if ( flagActDelta )
	{
		bufThis.bufPrevDelta.CheckOverun() ;
		cpuActivationDeltaBack( bufWorks, bufThis, stream ) ;
	}
	else if ( m_normalizer != nullptr )
	{
		m_normalizer->cpuDeltaBack
			( bufThis.normWorkBuf, bufThis.bufInDelta, bufThis.bufInAct, stream ) ;
	}
	bufThis.bufInDelta.CheckOverun() ;

	for ( CPUWorkBuf& work : bufWorks )
	{
		bufWorks.fpLoss += work.fpLoss ;
		bufWorks.nLossSamples += work.nLossSamples ;
		work.fpLoss = 0 ;
		work.nLossSamples = 0 ;
	}
	return	bufWorks.fpLoss / (double) bufWorks.nLossSamples ;
}

double NNPerceptron::cudaLossDelta
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis,
		NNBuffer& bufTeaching,
		NNLoopStream& stream, NNLossFunction * pLossFunc )
{
	const NNBufDim	dimOutput = bufThis.bufOutput.GetSize() ;
	const NNBufDim	dimInAct = bufThis.bufInAct.GetSize() ;
	const NNBufDim	dimInDelta = bufThis.bufInDelta.GetSize() ;
	const NNBufDim	dimTeaching = bufTeaching.GetSize() ;

	// ※最終レイヤーの活性化関数はチャネル数を変換しないこと
	// （pLossFunc==nullptr の時、bufPrevDelta ではなく bufInDelta へ直接出力する）
	assert( dimOutput.x <= dimTeaching.x ) ;
	assert( dimOutput.y <= dimTeaching.y ) ;
	assert( dimOutput.x == dimInDelta.x ) ;
	assert( dimOutput.y == dimInDelta.y ) ;
	assert( dimOutput.z == m_activation->CalcOutputChannels(dimInDelta.z,GetActivationDepthwise()) ) ;
	assert( dimInDelta == dimInAct ) ;

	bool	flagActDelta = false ;
	if ( pLossFunc == nullptr )
	{
		pLossFunc = m_activation.get() ;
	}
	else
	{
		flagActDelta = true ;
		bufThis.bufPrevDelta.Commit() ;
	}
	assert( pLossFunc->IsValidTeachingChannels(dimInAct.z,GetActivationDepthwise(),dimTeaching.z) ) ;

	bufThis.bufInDelta.CommitCuda() ;
	bufTeaching.CommitCuda() ;

	NNBuffer&	bufDstDelta =
		(flagActDelta ? bufThis.bufPrevDelta : bufThis.bufInDelta) ;
	pLossFunc->cudaLossDelta
		( bufDstDelta.GetCudaPtr(), dimInDelta,
			bufThis.bufInAct.GetCudaPtr(), dimInAct,
			bufThis.bufOutput.GetCudaPtr(), dimOutput,
			bufTeaching.GetCudaPtr(), dimTeaching,
			GetActivationDepthwise(), stream.m_cudaStream ) ;
	stream.m_cudaStream.VerifySync() ;

	if ( m_behavior & behaviorInvalidMargin )
	{
		bufDstDelta.CudaFillExterior
			( m_xInvalidLeft, m_yInvalidTop,
				dimOutput.x - __min(dimOutput.x,m_xInvalidRight),
				dimOutput.y - __min(dimOutput.y,m_yInvalidBottom),
				0.0f, stream.m_cudaStream ) ;
		stream.m_cudaStream.VerifySync() ;
	}

	if ( flagActDelta )
	{
		cudaActivationDeltaBack( bufWorks, bufThis, stream ) ;
	}
	else if ( m_normalizer != nullptr )
	{
		m_normalizer->cudaDeltaBack
			( bufThis.normWorkBuf, bufThis.bufInDelta, bufThis.bufInAct, stream ) ;
	}

	return	cpuCalcLoss( bufWorks, bufThis, bufTeaching, stream, pLossFunc ) ;
}

// 活性化関数のδ逆伝播処理
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::ActivationDeltaBack
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	if ( stream.m_useCuda )
	{
		cudaActivationDeltaBack( bufWorks, bufThis, stream ) ;
	}
	else
	{
		cpuActivationDeltaBack( bufWorks, bufThis, stream ) ;
	}
}

void NNPerceptron::cpuActivationDeltaBack
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	const NNBufDim	dimPrevDelta = bufThis.bufPrevDelta.GetSize() ;
	const NNBufDim	dimInDelta = bufThis.bufInDelta.GetSize() ;
	const NNBufDim	dimInAct = bufThis.bufInAct.GetSize() ;
	const NNBufDim	dimOutput = bufThis.bufOutput.GetSize() ;
	assert( dimPrevDelta.x == dimInDelta.x ) ;
	assert( dimPrevDelta.y == dimInDelta.y ) ;
	assert( dimPrevDelta.n == dimInDelta.n ) ;
	assert( dimInAct == dimInDelta ) ;
	assert( dimOutput.x == dimInDelta.x ) ;
	assert( dimOutput.y == dimInDelta.y ) ;
	assert( dimOutput.z == m_activation->CalcOutputChannels(dimInAct.z,GetActivationDepthwise()) ) ;

	NNActivationFunction *	pActivation = m_activation.get() ;

	if ( !bufThis.linearActivation )
	{
		bufThis.bufInDelta.Commit() ;

		stream.m_ploop.Loop( 0, dimPrevDelta.y, [&]( size_t iThread, size_t y )
		{
			const float *	pSrcDelta = bufThis.bufPrevDelta.GetConstBufferAt( 0, y ) ;
			float *			pDstDelta = bufThis.bufInDelta.GetBufferAt( 0, y ) ;
			const float *	pInAct = bufThis.bufInAct.GetConstBufferAt( 0, y ) ;
			const float *	pOutput = bufThis.bufOutput.GetConstBufferAt( 0, y ) ;
			float *			pDiffAct = bufWorks.at(iThread).vecDiff.data() ;
			const size_t	nDepthwise = GetActivationDepthwise() ;

			assert( bufWorks.at(iThread).vecDiff.size() >= dimInAct.z ) ;

			for ( size_t x = 0; x < dimPrevDelta.x; x ++ )
			{
				pActivation->cpuDifferential( pDiffAct, pInAct, pOutput, dimInAct.z, nDepthwise ) ;
				pActivation->cpuDeltaBack( pDstDelta, pSrcDelta, pDiffAct, dimInAct.z, nDepthwise ) ;
				pSrcDelta += dimPrevDelta.z ;
				pDstDelta += dimInDelta.z ;
				pInAct += dimInAct.z ;
				pOutput += dimOutput.z ;
			}
		} ) ;

		bufThis.bufInDelta.CheckOverun() ;
	}
	else
	{
		assert( bufThis.bufInDelta.IsEqualBuffer(bufThis.bufPrevDelta) ) ;
	}

	if ( m_normalizer != nullptr )
	{
		m_normalizer->cpuDeltaBack
			( bufThis.normWorkBuf, bufThis.bufInDelta, bufThis.bufInAct, stream ) ;
	}
}

void NNPerceptron::cudaActivationDeltaBack
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	const NNBufDim	dimPrevDelta = bufThis.bufPrevDelta.GetSize() ;
	const NNBufDim	dimInDelta = bufThis.bufInDelta.GetSize() ;
	const NNBufDim	dimInAct = bufThis.bufInAct.GetSize() ;
	const NNBufDim	dimOutput = bufThis.bufOutput.GetSize() ;
	assert( dimPrevDelta.x == dimInDelta.x ) ;
	assert( dimPrevDelta.y == dimInDelta.y ) ;
	assert( dimPrevDelta.n == dimInDelta.n ) ;
	assert( dimInAct == dimInDelta ) ;
	assert( dimOutput.x == dimInDelta.x ) ;
	assert( dimOutput.y == dimInDelta.y ) ;
	assert( dimOutput.z == m_activation->CalcOutputChannels(dimInAct.z,GetActivationDepthwise()) ) ;

	if ( !bufThis.linearActivation )
	{
		bufThis.bufInDelta.CommitCuda() ;

		m_activation->cudaBackDelta
			( bufThis.bufInDelta.GetCudaPtr(), dimInDelta,
				bufThis.bufPrevDelta.GetCudaPtr(), dimPrevDelta,
				bufThis.bufInAct.GetCudaPtr(), dimInAct,
				bufThis.bufOutput.GetCudaPtr(), dimOutput,
				GetActivationDepthwise(), stream.m_cudaStream ) ;
		stream.m_cudaStream.VerifySync() ;
	}
	else
	{
		assert( bufThis.bufInDelta.IsEqualBuffer(bufThis.bufPrevDelta) ) ;
	}

	if ( m_normalizer != nullptr )
	{
		m_normalizer->cudaDeltaBack
			( bufThis.normWorkBuf, bufThis.bufInDelta, bufThis.bufInAct, stream ) ;
	}
}

// 行列更新用勾配計算
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::CalcMatrixGradient
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis,
		const NNPerceptron::InputBuffer bufInput,
		NNLoopStream& stream )
{
	if ( m_behavior & behaviorFixed )
	{
		return ;
	}
	if ( stream.m_useCuda )
	{
		cudaCalcMatrixGradient( bufWorks, bufThis, bufInput, stream ) ;
	}
	else
	{
		cpuCalcMatrixGradient( bufWorks, bufThis, bufInput, stream ) ;
	}
}

void NNPerceptron::cpuCalcMatrixGradient
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis,
		const NNPerceptron::InputBuffer bufInput,
		NNLoopStream& stream )
{
	const NNBufDim	dimDelta = bufThis.bufInDelta.GetSize() ;
	const NNBufDim	dimInput = bufInput.pInput->GetSize() ;
	const NNBufDim	dimInner = m_sampler->CalcOutputDim( dimInput ) ;
	const NNBufDim	dimMatPoint = m_sampler->CalcMatrixPointDim( dimInput ) ;
	assert( dimInner.x == dimDelta.x ) ;
	assert( dimInner.y == dimDelta.y ) ;

	stream.m_ploop.Loop( 0, dimMatPoint.y, [&]( size_t iThread, size_t y )
	{
		CPUWorkBuf&			bufWork = bufWorks.at( iThread ) ;
		assert( m_sampler->CalcOutputChannels( bufWork.matGradient.GetLineCount() ) == dimDelta.z ) ;

		const float *		pInputBuf = bufInput.pInput->GetConstBuffer() ;
		const int			nLines = (int) m_matrix.GetLineCount() ;
		const int			nColumns = (int) m_matrix.GetColumnCount() ;
		const size_t		nSrcCount = m_matrix.GetColumnCount() - m_bias ;
		NNSamplingFilter *	pSampler = m_sampler.get() ;
		const int			nScaleX = (int) pSampler->UpSamplingScaleX() ;
		const int			nScaleY = (int) pSampler->UpSamplingScaleY() ;
		const float *		pDelta = bufThis.bufInDelta.GetConstBuffer() ;
		float *				pSrcVec = bufWork.vecSrc.data() ;
		float *				pDeltaVec = bufWork.vecDst.data() ;

		assert( bufWork.vecSrc.size() >= nColumns ) ;
		assert( bufWork.vecDst.size() >= nLines ) ;

		for ( size_t x = 0; x < dimMatPoint.x; x ++ )
		{
			pSampler->cpuCalcMatrixGradient
				( bufWork.matGradient, (int) x, (int) y,
					pSrcVec, pInputBuf, dimInput, nSrcCount,
					pDeltaVec, pDelta, dimDelta ) ;
		}
		bufWork.nGradient += dimInput.x ;
	} ) ;
}

void NNPerceptron::cudaCalcMatrixGradient
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis,
		const NNPerceptron::InputBuffer bufInput,
		NNLoopStream& stream )
{
	const NNBufDim	dimGradient = bufThis.bufGradient.GetSize() ;
	const NNBufDim	dimDelta = bufThis.bufInDelta.GetSize() ;
	const NNBufDim	dimInput = bufInput.pInput->GetSize() ;
	const NNBufDim	dimInner = m_sampler->CalcOutputDim( dimInput ) ;
	assert( dimInner.x == dimDelta.x ) ;
	assert( dimInner.y == dimDelta.y ) ;

	bufThis.bufGradient.CommitCuda() ;

	m_sampler->cudaCalcMatrixGradient
		( bufThis.bufGradient.GetCudaPtr(), dimGradient,
			bufThis.xGradientBlock, bufThis.yGradientBlock,
			m_matrix.GetColumnCount(),
			m_matrix.GetLineCount(),
			m_matrix.GetColumnCount() - m_bias, (int) GetDepthwise(),
			bufThis.bufInDelta.GetCudaPtr(), dimDelta,
			bufInput.pInput->GetCudaPtr(), dimInput, stream.m_cudaStream ) ;
	stream.m_cudaStream.VerifySync() ;

	bufThis.bufGradient.CudaAsyncFromDevice( stream.m_cudaStream ) ;
	stream.m_cudaStream.VerifySync() ;

	const NNBufDim	dimMatPoint = m_sampler->CalcMatrixPointDim( dimInput ) ;
	bufWorks.nGradient += dimMatPoint.n ;
}

// 更新用行列勾配を統合する
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::IntegrateMatrixGradient
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	if ( m_behavior & behaviorFixed )
	{
		return ;
	}
	if ( stream.m_useCuda )
	{
		cudaIntegrateMatrixGradient( bufWorks, bufThis ) ;
	}
	else
	{
		cpuIntegrateMatrixGradient( bufWorks, bufThis ) ;
	}
}

void NNPerceptron::cpuIntegrateMatrixGradient
	( NNPerceptron::CPUWorkArray& bufWorks, NNPerceptron::Buffer& bufThis )
{
	for ( CPUWorkBuf& work : bufWorks )
	{
		bufWorks.matGradient += work.matGradient ;
		bufWorks.nGradient += work.nGradient ;
		work.matGradient.InitDiagonal( 0.0f ) ;
		work.nGradient = 0 ;
	}

	if ( m_normalizer != nullptr )
	{
		m_normalizer->cpuIntegrateGradient( bufWorks.normGrad, bufThis.normWorkBuf ) ;
	}
}

void NNPerceptron::cudaIntegrateMatrixGradient
	( NNPerceptron::CPUWorkArray& bufWorks, NNPerceptron::Buffer& bufThis )
{
	const NNBufDim	dimGradient = bufThis.bufGradient.GetSize() ;
	const NNBufDim	dimOutput = bufThis.bufOutput.GetSize() ;

	assert( m_matrix.GetLineCount() == bufWorks.matGradient.GetLineCount() ) ;
	assert( m_matrix.GetColumnCount() == bufWorks.matGradient.GetColumnCount() ) ;
	assert( bufWorks.matGradient.GetLength() == dimGradient.z ) ;

	float *			pGradientDst = bufWorks.matGradient.GetArray() ;
	const float *	pGradientSrc = bufThis.bufGradient.GetConstBuffer() ;
	for ( size_t i = 0; i < dimGradient.n; i ++ )
	{
		for ( size_t j = 0; j < dimGradient.z; j ++ )
		{
			pGradientDst[j] += pGradientSrc[j] ;
		}
		pGradientSrc += dimGradient.z ;
	}

	if ( m_normalizer != nullptr )
	{
		m_normalizer->cudaIntegrateGradient( bufWorks.normGrad, bufThis.normWorkBuf ) ;
	}
}

// 行列のδ逆伝播処理
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::MatrixDeltaBack
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	if ( m_behavior & behaviorCutOff )
	{
		return ;
	}
	if ( stream.m_useCuda )
	{
		cudaMatrixDeltaBack( bufWorks, bufThis, stream ) ;
	}
	else
	{
		cpuMatrixDeltaBack( bufWorks, bufThis, stream ) ;
	}
}

void NNPerceptron::cpuMatrixDeltaBack
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	const NNBufDim	dimInDelta = bufThis.bufInDelta.GetSize() ;
	const NNBufDim	dimOutDelta = bufThis.bufOutDelta.GetSize() ;
	const NNBufDim	dimInner = m_sampler->CalcOutputDim( dimOutDelta ) ;
	assert( dimInner.x == dimInDelta.x ) ;
	assert( dimInner.y == dimInDelta.y ) ;
	assert( m_sampler->CalcOutputChannels(m_matrix.GetLineCount()) == dimInDelta.z ) ;
	assert( m_matrix.GetColumnCount() >= dimOutDelta.z ) ;

	bufThis.bufOutDelta.Commit() ;

	stream.m_ploop.Loop( 0, dimOutDelta.y, [&]( size_t iThread, size_t yDst )
	{
		CPUWorkBuf&			bufWork = bufWorks.at(iThread) ;
		const float *		pInDelta = bufThis.bufInDelta.GetConstBuffer() ;
		float *				pOutDelta = bufThis.bufOutDelta.GetBufferAt( 0, yDst ) ;
		float *				pDeltaVec = bufWork.vecDst.data() ;
		const float *		pMatrix = m_matrix.GetConstArray() ;
		const size_t		nLines = m_matrix.GetLineCount() ;
		const size_t		nColumns = m_matrix.GetColumnCount() ;
		const size_t		nDepthwise = GetDepthwise() ;
		NNSamplingFilter *	pSampler = m_sampler.get() ;
		const size_t		nUpSamplingScaleX = pSampler->UpSamplingScaleX() ;
		const size_t		nUpSamplingScaleY = pSampler->UpSamplingScaleY() ;

		assert( bufWork.vecDst.size() >= nLines ) ;

		for ( int xDst = 0; (size_t) xDst < dimOutDelta.x; xDst ++ )
		{
			for ( size_t i = 0; i < dimOutDelta.z; i ++ )
			{
				pOutDelta[i] = 0.0f ;
			}
			for ( int yc = 0; (size_t) yc < pSampler->m_yConv; yc ++ )
			{
				const int	ySrc = ((int) yDst - yc * pSampler->m_yPitch
										- pSampler->m_yOffset) / pSampler->m_yStride ;
				if ( (ySrc < 0)
					|| ((size_t) ySrc * nUpSamplingScaleY >= dimInDelta.y)
					|| (ySrc * pSampler->m_yStride + pSampler->m_yOffset
										!= (int) yDst - yc * pSampler->m_yPitch) )
				{
					continue ;
				}
				for ( int xc = 0; xc < pSampler->m_xConv; xc ++ )
				{
					const int	xSrc = (xDst - xc * pSampler->m_xPitch
											- pSampler->m_xOffset) / pSampler->m_xStride ;
					if ( (xSrc < 0)
						|| ((size_t) xSrc * nUpSamplingScaleX >= dimInDelta.x)
						|| (xSrc * pSampler->m_xStride + pSampler->m_xOffset
											!= xDst - xc * pSampler->m_xPitch) )
					{
						continue ;
					}
					pSampler->cpuBackSample
						( pDeltaVec, nLines, pInDelta, dimInDelta, xSrc, ySrc ) ;
					pSampler->cpuAddMatrixDeltaBackAt
						( pOutDelta, dimOutDelta.z,
							pDeltaVec, nLines, nDepthwise,
							pMatrix, nColumns, xc, yc ) ;
				}
			}
			pOutDelta += dimOutDelta.z ;
		}
	} ) ;

	bufThis.bufOutDelta.CheckOverun() ;
}

void NNPerceptron::cudaMatrixDeltaBack
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	const NNBufDim	dimInDelta = bufThis.bufInDelta.GetSize() ;
	const NNBufDim	dimOutDelta = bufThis.bufOutDelta.GetSize() ;
	const NNBufDim	dimInner = m_sampler->CalcOutputDim( dimOutDelta ) ;
	assert( dimInner.x == dimInDelta.x ) ;
	assert( dimInner.y == dimInDelta.y ) ;
	assert( m_sampler->CalcOutputChannels(m_matrix.GetLineCount()) == dimInDelta.z ) ;
	assert( m_matrix.GetColumnCount() >= dimOutDelta.z ) ;

	bufThis.bufOutDelta.CommitCuda() ;

	m_sampler->cudaMatrixDeltaBack
		( bufThis.bufOutDelta.GetCudaPtr(), dimOutDelta,
			bufThis.bufInDelta.GetCudaPtr(), dimInDelta,
			bufThis.bufMatrix.GetCudaPtr(),
			(unsigned int) m_matrix.GetColumnCount(),
			(unsigned int) m_matrix.GetLineCount(),
			(int) GetDepthwise(), dimOutDelta.z, stream.m_cudaStream ) ;
	stream.m_cudaStream.VerifySync() ;
}

// 入力元レイヤーへδを逆伝播
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::LayerDeltaBack
	( const NNPerceptron::BufferArray& bufArray,
		const NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	if ( m_behavior & behaviorCutOff )
	{
		return ;
	}
	typedef	std::function<void(NNBuffer&,size_t,size_t,size_t,size_t,size_t,size_t,const NNBuffer&,int,size_t,float)>	FuncAddChannel ;
	FuncAddChannel	cpuAddChannelTo =
		[]( NNBuffer& nnDstBuf, size_t xDst, size_t yDst, size_t iDstChannel,
				size_t nWidth, size_t nHeight, size_t nChannelCount,
				const NNBuffer& nnSrcBuf, int xShift, size_t iSrcChannel, float scaleSrc )
		{
			nnDstBuf.AddChannelValue
				( xDst, yDst, iDstChannel, nnSrcBuf,
					xShift, iSrcChannel, nChannelCount,
					nWidth, nHeight, scaleSrc ) ;
		} ;
	FuncAddChannel	cudaAddChannelTo =
		[&]( NNBuffer& nnDstBuf, size_t xDst, size_t yDst, size_t iDstChannel,
				size_t nWidth, size_t nHeight, size_t nChannelCount,
				const NNBuffer& nnSrcBuf, int xShift, size_t iSrcChannel, float scaleSrc )
		{
			nnDstBuf.CudaAddChannelFrom
				( xDst, yDst, iDstChannel, nnSrcBuf,
					xShift, 0, iSrcChannel, nChannelCount,
					nWidth, nHeight, scaleSrc, stream.m_cudaStream ) ;
		} ;
	FuncAddChannel&	AddChannelTo = stream.m_useCuda ? cudaAddChannelTo : cpuAddChannelTo ;

	const float		scaleDelta = m_deltaFactor ;
	const size_t	iThisLayer = bufThis.iThisLayer ;
	if ( m_connection.size() == 0 )
	{
		if ( iThisLayer >= 1 )
		{
			std::shared_ptr<Buffer>	pBufRef = bufArray.at(iThisLayer - 1) ;
			NNBufDim	dimRef = pBufRef->bufPrevDelta.GetSize() ;
			AddChannelTo( pBufRef->bufPrevDelta, 0, 0, 0,
							dimRef.x, dimRef.y, dimRef.z,
							bufThis.bufOutDelta, 0, 0,
							scaleDelta /*/ (float) max(pBufRef->nRefOut,1)*/ ) ;
		}
	}
	else
	{
		size_t	chNext = 0 ;
		for ( size_t i = 0; i < m_connection.size(); i ++ )
		{
			if ( (behaviorCutOffCon0 << i) & (m_behavior & behaviorCutOffConMask) )
			{
				continue ;
			}
			const Connection&	cn = m_connection.at(i) ;
			if ( (iThisLayer < cn.iLayer) || (cn.iLayer == conLayerNull) )
			{
				continue ;
			}
			std::shared_ptr<Buffer>	pBufRef = bufArray.at(iThisLayer - cn.iLayer) ;
			NNBuffer&	bufDstDelta = (cn.iLayer > 0) ? pBufRef->bufPrevDelta
														: pBufRef->bufPrevDelta2 ;
			float	scaleRef = 1.0f ;
			/*
			if ( cn.iLayer > 0 )
			{
				scaleRef = 1.0f / (float) max( pBufRef->nRefOut, 1 ) ;
			}
			else
			{
				scaleRef = 1.0f / (float) max( pBufRef->nRefOut2, 1 ) ;
			}
			*/
			NNBufDim	dimRef = bufDstDelta.GetSize() ;
			NNBufDim	dimSrc = bufThis.bufOutDelta.GetSize() ;
			bufDstDelta.Commit() ;
			//
			assert( dimRef.x >= dimSrc.x + cn.xOffset ) ;
			assert( dimRef.y >= dimSrc.y + cn.yOffset ) ;
			if ( cn.nChannels == 0 )
			{
				assert( cn.iChannel == 0 ) ;
				AddChannelTo
					( bufDstDelta, cn.xOffset, cn.yOffset, 0,
						dimRef.x - cn.xOffset*2,
						dimRef.y - cn.yOffset*2, dimRef.z,
						bufThis.bufOutDelta, - cn.iDelay, chNext,
						scaleDelta * scaleRef ) ;
				chNext += dimRef.z ;
			}
			else
			{
				assert( dimRef.z >= cn.iChannel + cn.nChannels ) ;
				AddChannelTo
					( bufDstDelta, cn.xOffset, cn.yOffset, cn.iChannel,
						dimSrc.x, dimSrc.y, cn.nChannels,
						bufThis.bufOutDelta, - cn.iDelay, chNext,
						scaleDelta * scaleRef ) ;
				chNext += cn.nChannels ;
			}
		}
	}
}

// 入力元レイヤーへδを逆伝播（別の MLP 最終レイヤーへ損失関数δとして）
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::LayerDeltaBackTo
	( NNPerceptron::Buffer& bufDst,
		const NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
	NNBufDim	dimRef = bufThis.bufOutDelta.GetSize() ;
	assert( (m_connection.size() == 0) || (m_connection.size() == 1) ) ;
	size_t	iDstChannel = 0 ;
	if ( m_connection.size() == 1 )
	{
		iDstChannel = m_connection.at(0).iChannel ;
		if ( m_connection.at(0).nChannels != 0 )
		{
			assert( m_connection.at(0).nChannels <= dimRef.z ) ;
			dimRef.z = m_connection.at(0).nChannels ;
		}
	}
	const NNBufDim	dimDst = bufDst.bufPrevDelta.GetSize() ;
	assert( (dimRef.x == dimDst.x)
			&& (dimRef.y == dimDst.y)
			&& (dimRef.z + iDstChannel <= dimDst.z) ) ;
	if ( stream.m_useCuda )
	{
		bufDst.bufPrevDelta.CommitCuda() ;
		bufDst.bufPrevDelta.CudaAddChannelFrom
			( 0, 0, iDstChannel, bufThis.bufOutDelta, 0, 0, 0, dimRef.z,
				dimRef.x, dimRef.y, m_deltaFactor, stream.m_cudaStream ) ;
	}
	else
	{
		bufDst.bufPrevDelta.Commit() ;
		bufDst.bufPrevDelta.AddChannelValue
			( 0, 0, iDstChannel, bufThis.bufOutDelta, 0, 0, dimRef.z,
				dimRef.x, dimRef.y, m_deltaFactor ) ;
	}
}

// 勾配を行列に更新する デルタ率
//////////////////////////////////////////////////////////////////////////////
void NNPerceptron::AddMatrixGradient
	( NNPerceptron::LossAndGradientBuf& laGradient, float deltaRate, float scaleGradient )
{
	if ( (laGradient.nGradient == 0)
		|| (m_behavior & behaviorFixed) )
	{
		return ;
	}
	assert( m_matrix.GetLineCount() == laGradient.matGradient.GetLineCount() ) ;
	assert( m_matrix.GetColumnCount() == laGradient.matGradient.GetColumnCount() ) ;

	NNMatrix	matGradient = laGradient.matGradient ;
	matGradient *= scaleGradient / (float) laGradient.nGradient ;

	if ( m_adaOpt == adaOptMomentum )
	{
		// Momentum
		matGradient += laGradient.matAdaOpt[0] * m_adaParam.alpha ;
		laGradient.matAdaOpt[0] = matGradient ;
		m_matAdaOpt[0] = laGradient.matAdaOpt[0] ;
		matGradient *= m_adaParam.delta ;
	}
	else if ( m_adaOpt == adaOptRMSProp )
	{
		// RMSProp
		float *			pRMSProp = laGradient.matAdaOpt[0].GetArray() ;
		float *			pMatrix = matGradient.GetArray() ;
		const size_t	nCount = matGradient.GetLength() ;
		assert( laGradient.matAdaOpt[0].GetLength() == nCount ) ;
		for ( size_t i = 0; i < nCount; i++ )
		{
			pRMSProp[i] = m_adaParam.beta * pRMSProp[i]
						+ (1.0f - m_adaParam.beta) * pMatrix[i] * pMatrix[i] ;
			pMatrix[i] *= m_adaParam.delta / (sqrt(pRMSProp[i]) + 1.0e-7f) ;
		}
		m_matAdaOpt[0] = laGradient.matAdaOpt[0] ;
	}
	else if ( m_adaOpt == adaOptAdam )
	{
		// Adam
		float *			pAdam1 = laGradient.matAdaOpt[0].GetArray() ;
		float *			pAdam2 = laGradient.matAdaOpt[1].GetArray() ;
		float *			pMatrix = matGradient.GetArray() ;
		const size_t	nCount = matGradient.GetLength() ;
		assert( laGradient.matAdaOpt[0].GetLength() == nCount ) ;
		assert( laGradient.matAdaOpt[1].GetLength() == nCount ) ;
		for ( size_t i = 0; i < nCount; i++ )
		{
			pAdam1[i] = m_adaParam.alpha * pAdam1[i]
						+ (1.0f - m_adaParam.alpha) * pMatrix[i] ;
			pAdam2[i] = m_adaParam.beta * pAdam2[i]
						+ (1.0f - m_adaParam.beta) * pMatrix[i] * pMatrix[i] ;
			//
			const float	s = pAdam1[i] / (1.0f - m_adaParam.alpha) ;
			const float	r = pAdam2[i] / (1.0f - m_adaParam.beta) ;
			pMatrix[i] *= m_adaParam.delta * s / (sqrt(r) + 1.0e-7f) ;
		}
		m_matAdaOpt[0] = laGradient.matAdaOpt[0] ;
		m_matAdaOpt[1] = laGradient.matAdaOpt[1] ;
	}
	deltaRate *= m_gradFactor ;

	const float	l2regFactor = 2.0f * m_l2reg * deltaRate * scaleGradient ;
	m_matrix -= matGradient * deltaRate + m_matrix * l2regFactor ;

	Specialize( ) ;

	if ( m_normalizer != nullptr )
	{
		m_normalizer->AddGradient
			( laGradient.normGrad,
				deltaRate * scaleGradient, m_l2reg ) ;
	}
}



//////////////////////////////////////////////////////////////////////////////
// Depthwise 用パーセプトロン
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNDepthwisePerceptron::NNDepthwisePerceptron( void )
{
}

NNDepthwisePerceptron::NNDepthwisePerceptron
	( size_t nDstCount, size_t nSrcCount,
		size_t nDepthwise, size_t nBias,
		std::shared_ptr<NNSamplingFilter> sampler,
		std::shared_ptr<NNActivationFunction> activation )
	: NNPerceptron( nDstCount, nSrcCount, nDepthwise, nBias, sampler, activation )
{
}

NNDepthwisePerceptron::NNDepthwisePerceptron( const NNDepthwisePerceptron& dwp )
	: NNPerceptron( dwp )
{
}

// 行列の特殊化
//////////////////////////////////////////////////////////////////////////////
void NNDepthwisePerceptron::Specialize( void )
{
	const size_t	wMatrixCol = m_matrix.GetColumnCount() - GetBias() ;
	const size_t	hMatrixLine = m_matrix.GetLineCount() ;
	const size_t	nDepthwise = GetDepthwise() ;
	for ( size_t line = 0; line < hMatrixLine; line ++ )
	{
		size_t	lineOdd = line % nDepthwise ;
		for ( size_t col = 0; col < wMatrixCol; col ++ )
		{
			if ( (col % nDepthwise) != lineOdd )
			{
				m_matrix.At( line, col ) = 0.0f ;
			}
		}
	}
}



//////////////////////////////////////////////////////////////////////////////
// 固定値行列パーセプトロン基底
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNFixedPerceptron::NNFixedPerceptron( void )
{
}

NNFixedPerceptron::NNFixedPerceptron
	( size_t nDstCount, size_t nSrcCount,
		size_t nDepthwise, size_t nBias,
		std::shared_ptr<NNSamplingFilter> sampler,
		std::shared_ptr<NNActivationFunction> activation )
	: NNPerceptron( nDstCount, nSrcCount, nDepthwise, nBias, sampler, activation )
{
}

NNFixedPerceptron::NNFixedPerceptron( const NNFixedPerceptron& fxp )
	: NNPerceptron( fxp )
{
}

// 動作フラグ（固定値行列）
//////////////////////////////////////////////////////////////////////////////
bool NNFixedPerceptron::IsMatrixFixed( void ) const
{
	return	true ;
}

// 勾配反映後のバッファ処理
//////////////////////////////////////////////////////////////////////////////
void NNFixedPerceptron::ResetBufferInBatch( NNPerceptron::Buffer& bufThis ) const
{
}

// 行列更新用勾配計算
//////////////////////////////////////////////////////////////////////////////
void NNFixedPerceptron::CalcMatrixGradient
	( NNPerceptron::CPUWorkArray& bufWorks, NNPerceptron::Buffer& bufThis,
		const NNPerceptron::InputBuffer bufInput, NNLoopStream& stream )
{
}

void NNFixedPerceptron::cpuCalcMatrixGradient
	( NNPerceptron::CPUWorkArray& bufWorks, NNPerceptron::Buffer& bufThis,
		const NNPerceptron::InputBuffer bufInput, NNLoopStream& stream )
{
}

void NNFixedPerceptron::cudaCalcMatrixGradient
	( NNPerceptron::CPUWorkArray& bufWorks, NNPerceptron::Buffer& bufThis,
		const NNPerceptron::InputBuffer bufInput, NNLoopStream& stream )
{
}

// 更新用行列勾配を統合する
//////////////////////////////////////////////////////////////////////////////
void NNFixedPerceptron::IntegrateMatrixGradient
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis, NNLoopStream& stream )
{
}

void NNFixedPerceptron::cpuIntegrateMatrixGradient
	( NNPerceptron::CPUWorkArray& bufWorks, NNPerceptron::Buffer& bufThis )
{
}

void NNFixedPerceptron::cudaIntegrateMatrixGradient
	( NNPerceptron::CPUWorkArray& bufWorks, NNPerceptron::Buffer& bufThis )
{
}



//////////////////////////////////////////////////////////////////////////////
// 単位行列パーセプトロン
//////////////////////////////////////////////////////////////////////////////

// 構築関数
//////////////////////////////////////////////////////////////////////////////
NNIdentityPerceptron::NNIdentityPerceptron( void )
	: m_scaler( 1.0f ), m_unitDiagonal( 0 )
{
}

NNIdentityPerceptron::NNIdentityPerceptron
	( size_t nDstCount, size_t nSrcCount,
		float scaler, size_t nDepthwise,
		std::shared_ptr<NNSamplingFilter> sampler,
		std::shared_ptr<NNActivationFunction> activation,
		size_t nDiagonalUnit )
	: m_scaler( scaler ), m_unitDiagonal( nDiagonalUnit )
{
	Create( nDstCount, nSrcCount, nDepthwise, 0, sampler, activation ) ;
}

NNIdentityPerceptron::NNIdentityPerceptron( const NNIdentityPerceptron& idp )
	: NNFixedPerceptron( idp ),
		m_scaler( idp.m_scaler ), m_unitDiagonal( idp.m_unitDiagonal )
{
}

// 対角化単位
//////////////////////////////////////////////////////////////////////////////
size_t NNIdentityPerceptron::GetDepthwise( void ) const
{
	if ( m_unitDiagonal != 0 )
	{
		return	m_unitDiagonal ;
	}
	const size_t	wMatrixCol = m_matrix.GetColumnCount() - GetBias() ;
	const size_t	hMatrixLine = m_matrix.GetLineCount() ;
	return	__min( wMatrixCol, hMatrixLine ) ;
}

// 行列の特殊化
//////////////////////////////////////////////////////////////////////////////
void NNIdentityPerceptron::Specialize( void )
{
	const size_t	wMatrixCol = m_matrix.GetColumnCount() - GetBias() ;
	const size_t	hMatrixLine = m_matrix.GetLineCount() ;
	const size_t	nUnitSize = GetDepthwise() ;

	for ( size_t line = 0; line < hMatrixLine; line ++ )
	{
		const size_t	lineOdd = line % nUnitSize ;
		float *			pMatrixLine = m_matrix.GetLineArray( line ) ;

		for ( size_t col = 0; col < wMatrixCol; col ++ )
		{
			pMatrixLine[col] = 0.0f ;
		}
		for ( size_t col = lineOdd; col < wMatrixCol; col += nUnitSize )
		{
			pMatrixLine[col] = m_scaler ;
		}
	}
}

// シリアライズ
//////////////////////////////////////////////////////////////////////////////
void NNIdentityPerceptron::SerializeExtendInfo( NNSerializer& ser )
{
	uint32_t	flags = 1 ;
	uint32_t	unit = (uint32_t) m_unitDiagonal ;
	ser.Write( &flags, sizeof(flags) ) ;
	ser.Write( &m_scaler, sizeof(m_scaler) ) ;
	ser.Write( &unit, sizeof(unit) ) ;
}

// デシリアライズ
//////////////////////////////////////////////////////////////////////////////
bool NNIdentityPerceptron::DeserializeExtendInfo( NNDeserializer & dsr )
{
	if ( dsr.GetChunkBytes() >= sizeof(uint32_t) + sizeof(m_scaler) )
	{
		uint32_t	flags = 0 ;
		uint32_t	unit = (uint32_t) m_unitDiagonal ;
		dsr.Read( &flags, sizeof(flags) ) ;
		dsr.Read( &m_scaler, sizeof(m_scaler) ) ;
		if ( flags != 0 )
		{
			dsr.Read( &unit, sizeof(unit) ) ;
			m_unitDiagonal = (size_t) unit ;
		}
	}
	return	true ;
}



//////////////////////////////////////////////////////////////////////////////
// 単純計算パーセプトロン（ｎ入力１出力計算）
//////////////////////////////////////////////////////////////////////////////

// 入力バッファの準備
//////////////////////////////////////////////////////////////////////////////
NNPerceptron::InputBuffer NNPrimitivePerceptron::PrepareInput
	( const NNPerceptron::BufferArray& bufArray,
		size_t iThisLayer, NNBuffer& bufInput0,
		size_t iFirstInputLayer, NNLoopStream& stream )
{
	InputBuffer	inBuf ;
	if ( IsPrimitiveMultiInput
		( inBuf, bufArray, iThisLayer, bufInput0, iFirstInputLayer, stream ) )
	{
		return	inBuf ;
	}
	return	NNIdentityPerceptron::PrepareInput
			( bufArray, iThisLayer, bufInput0, iFirstInputLayer, stream ) ;
}

// 入力を複数バッファから直接行うか？
//////////////////////////////////////////////////////////////////////////////
bool NNPrimitivePerceptron::IsPrimitiveMultiInput
	( NNPerceptron::InputBuffer& inBuf,
		const NNPerceptron::BufferArray& bufArray,
		size_t iThisLayer, NNBuffer& bufInput0,
		size_t iFirstInputLayer, NNLoopStream& stream ) const
{
	if ( (m_connection.size() <= 1)
		|| (m_connection.size() > MaxMultiInput) )
	{
		// チャネル数が範囲外の時は無効
		return	false ;
	}
	std::shared_ptr<Buffer>	pbufThis = bufArray.at( iThisLayer ) ;
	if ( pbufThis->forLearning )
	{
		// 予測時のみ特殊化によって高速化／学習時には無効
		return	false ;
	}
	for ( size_t i = 0; i < m_connection.size(); i ++ )
	{
		const Connection&	cn = m_connection.at(i) ;
		if ( cn.iLayer == conLayerNull )
		{
			return	false ;
		}
		NNBuffer *	pInputBuf = &bufInput0 ;
		int			iDelay = 0 ;
		if ( (cn.iLayer + (int) iFirstInputLayer <= (int) iThisLayer)
			&& ((size_t) (iThisLayer - cn.iLayer) < bufArray.size()) )
		{
			size_t	iRefLayer = iThisLayer - cn.iLayer ;
			if ( (cn.iDelay >= 1) && bufArray.at(iRefLayer)->bufDelay.IsCommitted() )
			{
				pInputBuf = &(bufArray.at(iRefLayer)->bufDelay) ;
				iDelay = cn.iDelay ;
			}
			else
			{
				pInputBuf = &(bufArray.at(iRefLayer)->bufOutput) ;
			}
		}
		if ( stream.m_useCuda )
		{
			pInputBuf->CommitCuda() ;
		}
		else
		{
			pInputBuf->Commit() ;
		}
		NNBufDim	dimRef = pInputBuf->GetSize() ;
		inBuf.multi[i].pBuffer = pInputBuf ;
		inBuf.multi[i].iChannel = cn.iChannel ;
		inBuf.multi[i].nChannels = (cn.nChannels != 0) ? cn.nChannels : dimRef.z ;
		inBuf.multi[i].xShift = iDelay - cn.xOffset ;
		inBuf.multi[i].yShift = - cn.yOffset ;
	}
	inBuf.pInput = nullptr ;
	inBuf.nMultiInput = m_connection.size() ;
	return	true ;
}




//////////////////////////////////////////////////////////////////////////////
// 加算パーセプトロン
//////////////////////////////////////////////////////////////////////////////

// 入力を複数バッファから直接行うか？
//////////////////////////////////////////////////////////////////////////////
bool NNPointwiseAddPerceptron::IsPrimitiveMultiInput
	( NNPerceptron::InputBuffer& inBuf,
		const NNPerceptron::BufferArray& bufArray,
		size_t iThisLayer, NNBuffer& bufInput0,
		size_t iFirstInputLayer, NNLoopStream& stream ) const
{
	if ( !stream.m_useCuda
		|| (m_normalizer != nullptr)
		|| (m_behavior & behaviorDisabled) )
	{
		return	false ;
	}
	if ( NNPrimitivePerceptron::IsPrimitiveMultiInput
		( inBuf, bufArray, iThisLayer, bufInput0, iFirstInputLayer, stream ) )
	{
		return	(inBuf.nMultiInput == 2) ;
	}
	return	false ;
}

// 予測処理
//////////////////////////////////////////////////////////////////////////////
void NNPointwiseAddPerceptron::cudaPrediction
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis,
		const InputBuffer bufInput,
		NNLoopStream& stream, size_t xLeftBounds )
{
	if ( bufInput.nMultiInput == 2 )
	{
		const NNBufDim	dimInAct = bufThis.bufInAct.GetSize() ;
		const NNBufDim	dimOutput = bufThis.bufOutput.GetSize() ;
		const NNBufDim	dimInput0 = bufInput.multi[0].pBuffer->GetSize() ;
		const NNBufDim	dimInput1 = bufInput.multi[1].pBuffer->GetSize() ;
		bufThis.bufInAct.CommitCuda() ;
		if ( bufThis.linearActivation )
		{
			assert( bufThis.bufInAct.IsEqualBuffer( bufThis.bufOutput ) ) ;
		}
		else
		{
			bufThis.bufOutput.CommitCuda() ;
		}

		nncuda_Primitive_Add
			( bufThis.bufInAct.GetCudaPtr(), dimInAct,
				bufInput.multi[0].pBuffer->GetCudaPtr(), dimInput0,
				bufInput.multi[0].iChannel,
				bufInput.multi[0].xShift, bufInput.multi[0].yShift,
				bufInput.multi[1].pBuffer->GetCudaPtr(), dimInput1,
				bufInput.multi[1].iChannel,
				bufInput.multi[1].xShift, bufInput.multi[1].yShift,
				xLeftBounds, stream.m_cudaStream ) ;
		stream.m_cudaStream.VerifySync() ;

		if ( !bufThis.linearActivation )
		{
			m_activation->cudaFunction
				( bufThis.bufOutput.GetCudaPtr(), dimOutput,
					bufThis.bufInAct.GetCudaPtr(), dimInAct,
					xLeftBounds, GetActivationDepthwise(), stream.m_cudaStream ) ;
			stream.m_cudaStream.VerifySync() ;
		}
		return ;
	}
	NNPrimitivePerceptron::cudaPrediction
		( bufWorks, bufThis, bufInput, stream, xLeftBounds ) ;
}



//////////////////////////////////////////////////////////////////////////////
// 乗算パーセプトロン
//////////////////////////////////////////////////////////////////////////////

// 入力を複数バッファから直接行うか？
//////////////////////////////////////////////////////////////////////////////
bool NNPointwiseMulPerceptron::IsPrimitiveMultiInput
	( NNPerceptron::InputBuffer& inBuf,
		const NNPerceptron::BufferArray& bufArray,
		size_t iThisLayer, NNBuffer& bufInput0,
		size_t iFirstInputLayer, NNLoopStream& stream ) const
{
	if ( !stream.m_useCuda
		|| (m_normalizer != nullptr)
		|| (m_behavior & behaviorDisabled) )
	{
		return	false ;
	}
	if ( NNPrimitivePerceptron::IsPrimitiveMultiInput
		( inBuf, bufArray, iThisLayer, bufInput0, iFirstInputLayer, stream ) )
	{
		return	(inBuf.nMultiInput == 2) ;
	}
	return	false ;
}

// 予測処理
//////////////////////////////////////////////////////////////////////////////
void NNPointwiseMulPerceptron::cudaPrediction
	( NNPerceptron::CPUWorkArray& bufWorks,
		NNPerceptron::Buffer& bufThis,
		const NNPerceptron::InputBuffer bufInput,
		NNLoopStream& stream, size_t xLeftBounds )
{
	if ( bufInput.nMultiInput == 2 )
	{
		const NNBufDim	dimOutput = bufThis.bufOutput.GetSize() ;
		const NNBufDim	dimInput0 = bufInput.multi[0].pBuffer->GetSize() ;
		const NNBufDim	dimInput1 = bufInput.multi[1].pBuffer->GetSize() ;
		bufThis.bufOutput.CommitCuda() ;

		nncuda_Primitive_Multiply
			( bufThis.bufOutput.GetCudaPtr(), dimOutput,
				bufInput.multi[0].pBuffer->GetCudaPtr(), dimInput0,
				bufInput.multi[0].iChannel,
				bufInput.multi[0].xShift, bufInput.multi[0].yShift,
				bufInput.multi[1].pBuffer->GetCudaPtr(), dimInput1,
				bufInput.multi[1].iChannel,
				bufInput.multi[1].xShift, bufInput.multi[1].yShift,
				xLeftBounds, stream.m_cudaStream ) ;
		stream.m_cudaStream.VerifySync() ;
		return ;
	}
	NNPrimitivePerceptron::cudaPrediction
		( bufWorks, bufThis, bufInput, stream, xLeftBounds ) ;
}



