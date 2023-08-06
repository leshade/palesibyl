
#include <palesibyl.h>
#include <palesibyl_lib.h>
#include "sample_basic_app.h"


//////////////////////////////////////////////////////////////////////////////
// アプリケーション
//////////////////////////////////////////////////////////////////////////////

class	PalesibylApp	: public PalesibylBasicApp,
							public NNMLPShell::ProgressListener
{
protected:
	// アプリ固有の説明（ファイルの配置など）
	static const char *	s_pszSpecificDescription ;

public:
	bool		m_flagInterpolate ;		// 補間予測
	std::string	m_strInterpolateSrc1 ;
	std::string	m_strInterpolateSrc2 ;

	bool		m_flagMeanVariance ;	// エンコード値の平均と分散を計算
	bool		m_flagEncodedCsvFile ;	// エンコード値を CSV へ出力
	std::string	m_strEncodedCsvFile ;	// エンコード値出力 CSV ファイル名
	std::unique_ptr<std::ofstream>
				m_ofsEncodedCsvFile ;	// エンコード値出力 CSV

	NNNormalizationFilter::Aggregation
				m_aggrEncoded ;			// エンコード値の平均と分散を計算用

public:
	// 構築関数
	PalesibylApp( void ) ;
	// アプリ初期化
	virtual void Initialize( void ) ;
	// 引数解釈
	virtual bool ParseArgumentAt( int& iArg, int nArgs, char * pszArgs[] ) ;
	// 実行
	virtual int Run( void ) ;
	// ヘルプ表示
	virtual int RunHelp( void ) ;
	// 予測実行
	virtual int RunPrediction( void ) ;
	// 補間予測実行
	virtual int RunInterpolation( void ) ;
	void EncodeForInterpolation
		( NNBuffer& bufEncoded, NNBuffer& bufSource,
			NNMultiLayerPerceptron::BufferArrays& bufArrays ) ;
	NNBuffer * DecodeForInterpolation
		( NNBuffer& bufInter,
			NNMultiLayerPerceptron::BufferArrays& bufArrays ) ;

	// 予測実行完了
	virtual void OnProcessedPrediction
		( NNMLPShell& shell,
			const char * pszSourcePath,
			NNBuffer * pSource, NNBuffer * pOutput,
			const NNMultiLayerPerceptron::BufferArrays& bufArrays ) ;

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

	// 出力値の集計
	void OutputAggregation
		( NNNormalizationFilter::Aggregation& aggr, const NNBuffer& bufOutput ) ;
} ;


