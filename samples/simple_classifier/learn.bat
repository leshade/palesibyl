@echo off

if not "%PYTHON_BIN_PATH%" == "" (
	start "" /B "%PYTHON_BIN_PATH%" plotlog.py
)

@echo 学習途中で中断してモデルファイルを保存するには ESC キーを押してください

simple_classifier /loop 1000 /batch 100 /batch_thread 8 /delta 0.1 /l simple_classifier.mlp /lgrd /log trained_log.csv
pause
