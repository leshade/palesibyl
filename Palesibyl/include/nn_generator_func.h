
#ifndef	__NN_GENERATOR_FUNC_H__
#define	__NN_GENERATOR_FUNC_H__

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// 入力生成器
//////////////////////////////////////////////////////////////////////////////

class	NNGeneratorFunction
{
protected:
	static std::map
		< std::string,
			std::function
				< std::shared_ptr
					<NNGeneratorFunction>() > >	s_mapMakeFunc ;

public:
	// 関数生成準備
	static void InitMake( void ) ;
	// 関数生成
	static std::shared_ptr<NNGeneratorFunction> Make( const char * pszName ) ;
	// 登録
	template <class T> static void Register( const char * pszName )
	{
		s_mapMakeFunc.insert
			( std::make_pair(std::string(pszName),
							[]() { return std::make_shared<T>(); } ) ) ;
	}
	template <class T> static void Register( void )
	{
		s_mapMakeFunc.insert
			( std::make_pair(std::string(T::FunctionName),
							[]() { return std::make_shared<T>(); } ) ) ;
	}

public:
	// 作業バッファ
	class	WorkBuf
	{
	public:
		virtual ~WorkBuf( void ) { }
	} ;

public:
	// 関数名
	virtual const char * GetFunctionName( void ) const = 0 ;
	// 作業バッファ生成
	std::shared_ptr<WorkBuf> MakeWorkBuffer
			( const NNBufDim& dimInput, const NNLoopStream& stream ) ;
	// 生成
	virtual void Generate
		( NNBuffer& bufDst, WorkBuf * pWorkBuf,
			size_t iChannel, size_t nChannels, NNLoopStream& stream ) ;
	// シリアライズ
	virtual void Serialize( NNSerializer& ser ) ;
	// デシリアライズ
	virtual void Deserialize( NNDeserializer & dsr ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// N(0,1) 正規分布乱数生成器
//////////////////////////////////////////////////////////////////////////////

class	NNGaussianGenerator	: public NNGeneratorFunction
{
protected:
	class	RandGenWorkBuf	: public WorkBuf
	{
	public:
		std::random_device				m_random ;
		std::mt19937					m_engine ;
		std::normal_distribution<float>	m_dist ;
		bool							m_useCuda ;
//		curandGenerator_t				m_curand ;
		NNBuffer						m_bufRand ;
	public:
		RandGenWorkBuf( const NNBufDim& dimInput, const NNLoopStream& stream ) ;
		virtual ~RandGenWorkBuf( void ) ;
	} ;

public:
	// 関数名
	constexpr static const char	FunctionName[] = "rand_gaussian" ;
	virtual const char * GetFunctionName( void ) const ;
	// 作業バッファ生成
	std::shared_ptr<WorkBuf> MakeWorkBuffer
			( const NNBufDim& dimInput, const NNLoopStream& stream ) ;
	// 生成
	virtual void Generate
		( NNBuffer& bufDst, WorkBuf * pWorkBuf,
			size_t iChannel, size_t nChannels, NNLoopStream& stream ) ;

} ;


}

#endif

