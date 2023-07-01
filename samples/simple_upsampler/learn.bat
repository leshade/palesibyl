@echo off

if not "%PYTHON_BIN_PATH%" == "" (
	start "" /B "%PYTHON_BIN_PATH%" plotlog.py
)

@echo 学習途中で中断してモデルファイルを保存するには ESC キーを押してください

simple_upsampler /cuda /loop 500 /delta 0.1,0.05 /l simple_upsampler.mlp /lgrd /log trained_log.csv /vio predict\out\valid_image.bmp
pause
