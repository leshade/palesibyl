
#include "sample_basic_app.h"

#include <palesibyl_lib.h>



//////////////////////////////////////////////////////////////////////////////
// アプリケーション
//////////////////////////////////////////////////////////////////////////////

class	PalesibylApp	: public PalesibylBasicApp
{
protected:
	NNMLPShell		m_classifier ;			// 分類器

	std::string		m_strClassModelFile ;	// 分類器モデルファイル名
	size_t			m_nGANLoopCount ;		// GAN ループ回数

	// アプリ固有の説明（ファイルの配置など）
	static const char *	s_pszSpecificDescription ;

public:
	// 構築関数
	PalesibylApp( void ) ;
	// アプリ初期化
	virtual void Initialize( void ) ;
	// 引数解釈
	virtual bool ParseArgumentAt( int& iArg, int nArgs, char * pszArgs[] ) ;
	// ヘルプ表示
	virtual int RunHelp( void ) ;
	// 学習実行
	virtual int RunLearning( void ) ;

	// モデルを作成
	virtual void BuildModel( NNMLPShell::Iterator * pIter ) ;
	virtual void BuildClassifier( NNMLPShell::Iterator * pIter ) ;

	// 学習実行前
	virtual void BeforeLearning( void ) ;
	// 予測実行前
	virtual void BeforePrediction( void ) ;

	// 学習用イテレーター作成
	virtual std::shared_ptr<NNMLPShell::Iterator> MakeLearningIter( void ) ;
	// 予測用イテレーター作成
	virtual std::shared_ptr<NNMLPShell::Iterator> MakePredictiveIter( void ) ;

} ;


