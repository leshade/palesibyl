@echo off

if not "%PYTHON_BIN_PATH%" == "" (
	start "" /B "%PYTHON_BIN_PATH%" plotlog.py
)

@echo 学習途中で中断してモデルファイルを保存するには ESC キーを押してください

simple_color_filter /loop 150 /subloop 10 /delta 0.1 /l simple_color_filter.mlp /log trained_log.csv
pause
