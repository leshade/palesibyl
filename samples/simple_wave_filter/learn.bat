@echo off

if not "%PYTHON_BIN_PATH%" == "" (
	start "" /B "%PYTHON_BIN_PATH%" plotlog.py
)

@echo �w�K�r���Œ��f���ă��f���t�@�C����ۑ�����ɂ� ESC �L�[�������Ă�������

simple_wave_filter /cuda /loop 500 /subloop 10 /delta 0.0001 /l simple_wave_filter.mlp /lgrd /log trained_log.csv
pause
