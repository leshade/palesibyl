@echo off

if not "%PYTHON_BIN_PATH%" == "" (
	start "" /B "%PYTHON_BIN_PATH%" plotlog.py
)

@echo 学習途中で中断してモデルファイルを保存するには ESC キーを押してください

simple_wave_filter /cuda /loop 500 /subloop 10 /delta 0.0001 /l simple_wave_filter.mlp /lgrd /log trained_log.csv
pause
