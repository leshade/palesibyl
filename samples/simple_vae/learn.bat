@echo off

if not "%PYTHON_BIN_PATH%" == "" (
	start "" /B "%PYTHON_BIN_PATH%" plotlog.py
)

@echo �w�K�r���Œ��f���ă��f���t�@�C����ۑ�����ɂ� ESC �L�[�������Ă�������

simple_vae /loop 2000 /batch 100 /subloop 10 /batch_thread 8 /delta 0.1,0.02 /l simple_vae.mlp /lgrd /log trained_log.csv

pause
