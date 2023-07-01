
#include "sample_app_stub.h"


// エントリポイント
//////////////////////////////////////////////////////////////////////////////
int main( int argc, char *argv[], char *envp[] )
{
	cudaInit() ;
	NNMLPShell::StaticInitialize() ;

	int	codeExit = 0 ;
	{
		PalesibylApp	app ;
		app.Initialize() ;

		codeExit = app.ParseArguments( argc, argv ) ;
		if ( codeExit == 0 )
		{
			codeExit = app.Run() ;
		}
	}

	NNMLPShell::StaticRelase() ;
	return	codeExit ;
}



//////////////////////////////////////////////////////////////////////////////
// アプリケーション
//////////////////////////////////////////////////////////////////////////////

// ヘルプ表示
//////////////////////////////////////////////////////////////////////////////
int PalesibylApp::RunHelp( void )
{
	PalesibylBasicApp::RunHelp() ;
	std::cout << s_pszSpecificDescription ;
	return	0 ;
}

