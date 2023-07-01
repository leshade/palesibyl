
#ifndef	__NN_SERIALIZER_H__
#define	__NN_SERIALIZER_H__

#include <fstream>
#include <iostream>

namespace	Palesibyl
{

//////////////////////////////////////////////////////////////////////////////
// チャンクID
//////////////////////////////////////////////////////////////////////////////

constexpr inline uint32_t NNCHUNKID
	( unsigned char a, unsigned char b, unsigned char c, unsigned char d )
{
	return	(uint32_t) a | (((uint32_t) b) << 8)
			| (((uint32_t) c) << 16) | (((uint32_t) d) << 24) ;
}

//////////////////////////////////////////////////////////////////////////////
// チャンク
//////////////////////////////////////////////////////////////////////////////

struct	NNSerializerChunkHeader
{
	uint32_t	id ;
	uint32_t	bytes ;

	NNSerializerChunkHeader( uint32_t cid = 0 ) : id(cid), bytes(0) { }
} ;

struct	NNSerializerChunk
{
	std::streampos			sposHeader ;
	std::streamoff			soffInBytes ;
	NNSerializerChunkHeader	chunkHeader ;
} ;



//////////////////////////////////////////////////////////////////////////////
// シリアライザ
//////////////////////////////////////////////////////////////////////////////

class	NNSerializer
{
private:
	std::ofstream&					m_ofs ;
	std::vector<NNSerializerChunk>	m_chunks ;

public:
	NNSerializer( std::ofstream& ofs )
		: m_ofs( ofs ) {}
	~NNSerializer( void )
	{
		assert( m_chunks.size() == 0 ) ;
	}
	void Descend( uint32_t chunkid ) ;
	void Ascend( void ) ;
	void Write( const void * buf, size_t bytes ) ;
	void WriteString( const char * str ) ;

} ;



//////////////////////////////////////////////////////////////////////////////
// デシリアライザ
//////////////////////////////////////////////////////////////////////////////

class	NNDeserializer
{
private:
	std::ifstream&					m_ifs ;
	std::vector<NNSerializerChunk>	m_chunks ;

public:
	NNDeserializer( std::ifstream& ifs )
		: m_ifs( ifs ) {}
	~NNDeserializer( void )
	{
		assert( m_chunks.size() == 0 ) ;
	}
	uint32_t Descend( uint32_t chunkid = 0 ) ;
	uint32_t GetChunkBytes( void ) const ;
	long long int LocalPosition( void ) const ;
	void Ascend( void ) ;
	void Skip( size_t bytes ) ;
	void Read( void * buf, size_t bytes ) ;
	std::string ReadString( void ) ;

} ;

}

#endif

