@echo off

if not "%PYTHON_BIN_PATH%" == "" (
	start "" /B "%PYTHON_BIN_PATH%" plotlog.py
)

@echo 学習途中で中断してモデルファイルを保存するには ESC キーを押してください

sample_resnet /cuda /loop 500 /delta 0.03,0.01 /l sample_resnet.mlp /lgrd /log trained_log.csv /vio predict\out\valid_image.bmp
pause
