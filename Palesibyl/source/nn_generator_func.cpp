
#include "nn_perceptron.h"

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// 入力生成器
//////////////////////////////////////////////////////////////////////////////

std::map< std::string, std::function< std::shared_ptr<NNGeneratorFunction>() > >
	NNGeneratorFunction::s_mapMakeFunc ;

// 関数生成準備
//////////////////////////////////////////////////////////////////////////////
void NNGeneratorFunction::InitMake( void )
{
	s_mapMakeFunc.clear() ;
	Register<NNGaussianGenerator>() ;
}

// 関数生成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNGeneratorFunction>
	NNGeneratorFunction::Make( const char * pszName )
{
	decltype(s_mapMakeFunc)::iterator iter = s_mapMakeFunc.find(pszName) ;
	assert( iter != s_mapMakeFunc.end() ) ;
	if ( iter != s_mapMakeFunc.end() )
	{
		return	(iter->second)() ;
	}
	return	nullptr ;
}

// 作業バッファ生成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNGeneratorFunction::WorkBuf>
	NNGeneratorFunction::MakeWorkBuffer
		( const NNBufDim& dimInput, const NNLoopStream& stream )
{
	return	nullptr ;
}

// 生成
//////////////////////////////////////////////////////////////////////////////
void NNGeneratorFunction::Generate
	( NNBuffer& bufDst, WorkBuf * pWorkBuf,
		size_t iChannel, size_t nChannels, NNLoopStream& stream )
{
}

// シリアライズ
//////////////////////////////////////////////////////////////////////////////
void NNGeneratorFunction::Serialize( NNSerializer& ser )
{
}

// デシリアライズ
//////////////////////////////////////////////////////////////////////////////
void NNGeneratorFunction::Deserialize( NNDeserializer & dsr )
{
}



//////////////////////////////////////////////////////////////////////////////
// N(0,1) 正規分布乱数生成器
//////////////////////////////////////////////////////////////////////////////

// 生成器バッファ構築
//////////////////////////////////////////////////////////////////////////////
NNGaussianGenerator::RandGenWorkBuf::RandGenWorkBuf
	( const NNBufDim& dimInput, const NNLoopStream& stream )
		: m_engine( m_random() ), m_dist( 0.0f, 1.0f ),
			m_useCuda( stream.m_useCuda )
{
	if ( stream.m_useCuda )
	{
//		curandCreateGenerator( &m_curand, CURAND_RNG_PSEUDO_DEFAULT ) ;
//		curandSetPseudoRandomGeneratorSeed( m_curand, m_engine() ) ;
//		curandSetStream( m_curand, stream.m_cudaStream ) ;
		m_bufRand.Allocate( dimInput, NNBuffer::cudaAllocate ) ;
	}
}

NNGaussianGenerator::RandGenWorkBuf::~RandGenWorkBuf( void )
{
	if ( m_useCuda )
	{
//		curandDestroyGenerator( m_curand ) ;
	}
}

// 関数名
//////////////////////////////////////////////////////////////////////////////
const char * NNGaussianGenerator::GetFunctionName( void ) const
{
	return	FunctionName ;
}

// 作業バッファ生成
//////////////////////////////////////////////////////////////////////////////
std::shared_ptr<NNGeneratorFunction::WorkBuf>
	NNGaussianGenerator::MakeWorkBuffer
		( const NNBufDim& dimInput, const NNLoopStream& stream )
{
	return	std::make_shared<RandGenWorkBuf>( dimInput, stream ) ;
}

// 生成
//////////////////////////////////////////////////////////////////////////////
void NNGaussianGenerator::Generate
	( NNBuffer& bufDst, WorkBuf * pWorkBuf,
		size_t iChannel, size_t nChannels, NNLoopStream& stream )
{
	RandGenWorkBuf *	prgwb = dynamic_cast<RandGenWorkBuf*>( pWorkBuf ) ;
	assert( prgwb != nullptr ) ;

	if ( stream.m_useCuda )
	{
		assert( prgwb->m_useCuda ) ;
		bufDst.CommitCuda() ;

		const NNBufDim	dimDst = bufDst.GetSize() ;
		assert( iChannel + nChannels <= dimDst.z ) ;
		/*
		if ( (iChannel == 0) && (nChannels == dimDst.z) )
		{
			curandGenerateNormal
				( prgwb->m_curand,
					bufDst.GetCudaPtr(),
					dimDst.n * dimDst.z, 0.0f, 1.0f ) ;
		}
		else
		{
			const NNBufDim	dimRand = prgwb->m_bufRand.GetSize() ;
			assert( dimDst.n == dimRand.n ) ;
			assert( dimDst.z == dimRand.z ) ;
			prgwb->m_bufRand.CommitCuda() ;
			curandGenerateNormal
				( prgwb->m_curand,
					prgwb->m_bufRand.GetCudaPtr(),
					dimRand.n * dimRand.z, 0.0f, 1.0f ) ;
			bufDst.CudaCopyChannelFrom
				( 0, 0, iChannel,
					prgwb->m_bufRand, 0, 0, iChannel, nChannels,
					dimRand.x, dimRand.y, stream.m_cudaStream ) ;
		}
		*/
		if ( (iChannel == 0) && (nChannels == dimDst.z) )
		{
			bufDst.CommitCudaWithHost() ;
			cpuGenerateGaussian( bufDst, *prgwb, iChannel, nChannels ) ;
			bufDst.CudaAsyncToDevice( stream.m_cudaStream ) ;
		}
		else
		{
			const NNBufDim	dimRand = prgwb->m_bufRand.GetSize() ;
			assert( dimDst.n == dimRand.n ) ;
			assert( dimDst.z == dimRand.z ) ;
			prgwb->m_bufRand.CommitCuda() ;

			cpuGenerateGaussian( prgwb->m_bufRand, *prgwb, iChannel, nChannels ) ;
			prgwb->m_bufRand.CudaAsyncToDevice( stream.m_cudaStream ) ;

			bufDst.CudaCopyChannelFrom
				( 0, 0, iChannel,
					prgwb->m_bufRand, 0, 0, iChannel, nChannels,
					dimRand.x, dimRand.y, stream.m_cudaStream ) ;
		}
	}
	else
	{
		cpuGenerateGaussian( bufDst, *prgwb, iChannel, nChannels ) ;
	}
}

void NNGaussianGenerator::cpuGenerateGaussian
	( NNBuffer& bufDst,
		NNGaussianGenerator::RandGenWorkBuf& wgwb,
				size_t iChannel, size_t nChannels )
{
	const NNBufDim	dimDst = bufDst.GetSize() ;
	assert( iChannel + nChannels <= dimDst.z ) ;

	float *	pDst = bufDst.GetBuffer() ;
	for ( size_t i = 0; i < dimDst.n; i ++ )
	{
		for ( size_t z = 0; z < nChannels; z ++ )
		{
			pDst[iChannel + z] = wgwb.m_dist( wgwb.m_engine ) ;
		}
		pDst += dimDst.z ;
	}
}



