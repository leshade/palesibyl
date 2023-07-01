@echo off

if not "%PYTHON_BIN_PATH%" == "" (
	start "" /B "%PYTHON_BIN_PATH%" plotlog.py
)

@echo 学習途中で中断してモデルファイルを保存するには ESC キーを押してください

sample_unet /cuda /loop 1500 /delta 0.03,0.01 /l sample_unet.mlp /lgrd /log trained_log.csv /vio predict\out\valid_image.bmp /tio predict\out\train_image.bmp
pause
