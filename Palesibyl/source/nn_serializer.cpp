
#include "nn_perceptron.h"

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// シリアライザ
//////////////////////////////////////////////////////////////////////////////

void NNSerializer::Descend( uint32_t chunkid )
{
	NNSerializerChunk	chunk ;
	chunk.sposHeader = m_ofs.tellp() ;
	chunk.soffInBytes = chunk.sposHeader ;
	chunk.soffInBytes += sizeof(NNSerializerChunkHeader) ;
	chunk.chunkHeader.id = chunkid ;
	m_chunks.push_back( chunk ) ;

	Write( &(chunk.chunkHeader), sizeof(chunk.chunkHeader) ) ;
}

void NNSerializer::Ascend( void )
{
	std::streampos	fposEnd = m_ofs.tellp() ;
	std::streamoff	offEnd = fposEnd ;
	//
	NNSerializerChunk	chunk = m_chunks.back() ;
	chunk.chunkHeader.bytes = (uint32_t) (offEnd - chunk.soffInBytes) ;
	//
	m_ofs.seekp( chunk.sposHeader ) ;
	Write( &(chunk.chunkHeader), sizeof(NNSerializerChunkHeader) ) ;
	m_ofs.seekp( fposEnd ) ;
	//
	m_chunks.pop_back() ;
}

void NNSerializer::Write( const void * buf, size_t bytes )
{
	m_ofs.write( (const char*) buf, bytes ) ;
}

void NNSerializer::WriteString( const char * str )
{
	uint32_t	len = (uint32_t) strlen( str ) ;
	Write( &len, sizeof(len) ) ;
	Write( str, len * sizeof(char) ) ;
}




//////////////////////////////////////////////////////////////////////////////
// デシリアライザ
//////////////////////////////////////////////////////////////////////////////

uint32_t NNDeserializer::Descend( uint32_t chunkid )
{
	NNSerializerChunk	chunk ;
	for ( ; ; )
	{
		if ( (m_chunks.size() >= 1)
			&& (LocalPosition() + sizeof(NNSerializerChunkHeader) > GetChunkBytes()) )
		{
			return	0 ;
		}
		chunk.sposHeader = m_ifs.tellg() ;
		Read( &chunk.chunkHeader, sizeof(chunk.chunkHeader) ) ;
		if ( !m_ifs.good() )
		{
			return	0 ;
		}
		if ( (chunkid == 0) || (chunkid == chunk.chunkHeader.id) )
		{
			break ;
		}
		m_ifs.seekg( (std::streampos) chunk.chunkHeader.bytes, std::ios_base::cur ) ;
		if ( m_ifs.eof() )
		{
			return	0 ;
		}
	}

	chunk.soffInBytes = chunk.sposHeader ;
	chunk.soffInBytes += sizeof(NNSerializerChunkHeader) ;
	m_chunks.push_back( chunk ) ;

	return	chunk.chunkHeader.id ;
}

uint32_t NNDeserializer::GetChunkBytes( void ) const
{
	assert( m_chunks.size() > 0 ) ;
	return	m_chunks.back().chunkHeader.bytes ;
}

long long int NNDeserializer::LocalPosition( void ) const
{
	std::streampos	spos = m_ifs.tellg() ;
	std::streamoff	soff = spos ;
	if ( m_chunks.size() > 0 )
	{
		return	soff - m_chunks.back().soffInBytes ;
	}
	return	soff ;
}

void NNDeserializer::Ascend( void )
{
	NNSerializerChunk	chunk = m_chunks.back() ;
	//
	m_ifs.seekg( chunk.sposHeader ) ;
	m_ifs.seekg
		( (std::streampos) (chunk.chunkHeader.bytes
							+ sizeof(NNSerializerChunkHeader)), std::ios_base::cur ) ;
	//
	m_chunks.pop_back() ;
}

void NNDeserializer::Skip( size_t bytes )
{
	if ( bytes > 0 )
	{
		m_ifs.seekg( (std::streampos) bytes, std::ios_base::cur ) ;
	}
}

void NNDeserializer::Read( void * buf, size_t bytes )
{
	m_ifs.read( (char*) buf, bytes ) ;
}

std::string NNDeserializer::ReadString( void )
{
	uint32_t	len = 0 ;
	Read( &len, sizeof(len) ) ;
	//
	std::vector<char>	buf ;
	buf.resize( (size_t) len ) ;
	Read( buf.data(), len * sizeof(char) ) ;
	//
	return	std::string( buf.data(), (size_t) len ) ;
}

