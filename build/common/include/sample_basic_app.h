
#include <palesibyl.h>

using namespace Palesibyl ;


//////////////////////////////////////////////////////////////////////////////
// アプリケーション
//////////////////////////////////////////////////////////////////////////////

class	PalesibylBasicApp
{
public:
	// NNMLPShell 固有の実装
	class	AppShell	: public NNMLPShell
	{
	private:
		std::filesystem::path	m_pathTrainImage ;
		std::filesystem::path	m_pathValidImage ;
		std::string				m_strLogFile ;
		bool					m_logOutput ;
		bool					m_logGradient ;

	public:
		AppShell( void ) : m_logOutput(false), m_logGradient(false) {}
		// 訓練データ出力を画像ファイルとして出力する
		void SetTrainingImageFile( const char * pszFilePath ) ;
		// 検証データ出力を画像ファイルとして出力する
		void SetValidationImageFile( const char * pszFilePath ) ;
		// 出力 CSV ファイルを開く
		bool MakeOutputCSV( const char * pszFilePath ) ;
		// 勾配をログ出力させるか
		void SetLogGradient( bool log ) ;
		// ログファイルを開く
		std::unique_ptr<std::ofstream> OpenLogFile( bool flagCreate ) const ;
		// 学習進捗
		virtual void OnLearningProgress
			( LearningEvent le, const LearningProgressInfo& lpi ) ;
	} ;

public:
	enum	ArgumentFlag
	{
		argumentHelp			= 0x0001,
		argumentLearn			= 0x0002,
		argumentPrediction		= 0x0004,
	} ;
	uint32_t						m_flagsArg ;		// 引数フラグ
	NNMLPShell::LearningParameter	m_param ;			// 学習パラメータ
	std::string						m_strModelFile ;	// モデルファイル名
	std::string						m_strLearnLogFile ;	// 学習ログ（CSV）ファイル名

	// モデル
	AppShell								m_shell ;
	NNMLPShell::ShellConfig					m_cfgShell ;
	NNMultiLayerPerceptron::BufferConfig	m_cfgBuf ;

public:
	// PalesibylBasicApp 構築
	PalesibylBasicApp( void ) ;
	// アプリ初期化
	virtual void Initialize( void ) ;
	// 引数解釈
	virtual int ParseArguments( int nArgs, char * pszArgs[] ) ;
	virtual bool ParseArgumentAt( int& iArg, int nArgs, char * pszArgs[] ) ;
	virtual bool CheckArgumentsAfterParse( void ) ;
	virtual bool IsValidNextArgument( int& iArg, int nArgs, char * pszArgs[] ) ;
	virtual int NextArgumentInt( bool& validResult, int& iArg, int nArgs, char * pszArgs[] ) ;
	virtual double NextArgumentFloat( bool& validResult, int& iArg, int nArgs, char * pszArgs[] ) ;
	// 実行
	virtual int Run( void ) ;
	// ヘルプ表示
	virtual int RunHelp( void ) ;
	// 学習実行
	virtual int RunLearning( void ) ;
	// 予測実行
	virtual int RunPrediction( void ) ;

	// モデルを作成
	virtual void BuildModel( NNMLPShell::Iterator * pIter ) = 0 ;
	// 学習実行前
	virtual void BeforeLearning( void ) = 0 ;
	// 予測実行前
	virtual void BeforePrediction( void ) = 0 ;
	// 学習用イテレーター作成
	virtual std::shared_ptr<NNMLPShell::Iterator> MakeLearningIter( void ) = 0 ;
	// 予測用イテレーター作成
	virtual std::shared_ptr<NNMLPShell::Iterator> MakePredictiveIter( void ) = 0 ;

} ;


