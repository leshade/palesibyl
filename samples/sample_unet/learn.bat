@echo off

if not "%PYTHON_BIN_PATH%" == "" (
	start "" /B "%PYTHON_BIN_PATH%" plotlog.py
)

@echo �w�K�r���Œ��f���ă��f���t�@�C����ۑ�����ɂ� ESC �L�[�������Ă�������

sample_unet /cuda /loop 1500 /delta 0.03,0.01 /l sample_unet.mlp /lgrd /log trained_log.csv /vio predict\out\valid_image.bmp /tio predict\out\train_image.bmp
pause
