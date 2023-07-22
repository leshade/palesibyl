
#include <palesibyl.h>
#include <palesibyl_lib.h>
#include "sample_basic_app.h"


//////////////////////////////////////////////////////////////////////////////
// アプリケーション
//////////////////////////////////////////////////////////////////////////////

class	PalesibylApp	: public PalesibylBasicApp
{
protected:
	// アプリ固有の説明（ファイルの配置など）
	static const char *	s_pszSpecificDescription ;

public:
	// アプリ初期化
	virtual void Initialize( void ) ;
	// ヘルプ表示
	virtual int RunHelp( void ) ;
	// モデルを作成
	virtual void BuildModel( NNMLPShell::Iterator * pIter ) ;
	// 学習実行前
	virtual void BeforeLearning( void ) ;
	// 予測実行前
	virtual void BeforePrediction( void ) ;
	// 学習用イテレーター作成
	virtual std::shared_ptr<NNMLPShell::Iterator> MakeLearningIter( void ) ;
	// 予測用イテレーター作成
	virtual std::shared_ptr<NNMLPShell::Iterator> MakePredictiveIter( void ) ;

} ;


