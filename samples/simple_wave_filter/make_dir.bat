@echo off
@echo ディレクトリ作成
@echo learn\source - 学習用ソース
@echo learn\teacher - 学習用教師データ
@echo predict\src - 予測ソース
@echo predict\out - 予測出力

if not exist learn md learn
if not exist learn\source md learn\source
if not exist learn\teacher md learn\teacher
if not exist predict md predict
if not exist predict\src md predict\src
if not exist predict\out md predict\out

pause
