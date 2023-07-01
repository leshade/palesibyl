@echo off

if not "%PYTHON_BIN_PATH%" == "" (
	start "" /B "%PYTHON_BIN_PATH%" plotlog.py
)

@echo 学習途中で中断してモデルファイルを保存するには ESC キーを押してください

simple_gan /ganloop 50 /loop 50 /nlfb /batch 100 /batch_thread 8 /delta 0.1 /clsf classifier.mlp /l simple_gan.mlp /log trained_log.csv
pause
