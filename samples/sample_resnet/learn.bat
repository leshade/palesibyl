@echo off

if not "%PYTHON_BIN_PATH%" == "" (
	start "" /B "%PYTHON_BIN_PATH%" plotlog.py
)

@echo �w�K�r���Œ��f���ă��f���t�@�C����ۑ�����ɂ� ESC �L�[�������Ă�������

sample_resnet /cuda /loop 500 /delta 0.03,0.01 /l sample_resnet.mlp /lgrd /log trained_log.csv /vio predict\out\valid_image.bmp
pause
