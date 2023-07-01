@echo off
@echo make_learn\src\*.png �t�@�C������
@echo �w�K�p�f�[�^�ilearn\source\*.bmp, learn\teacher\*.bmp�j�𐶐����܂�
@echo .

if not exist learn md learn
if not exist learn\source md learn\source
if not exist learn\teacher md learn\teacher
if not exist predict md predict
if not exist predict\src md predict\src
if not exist predict\out md predict\out

cd make_learn

if not "%PYTHON_BIN_PATH%" == "" (
	"%PYTHON_BIN_PATH%" grayscale_image.py
) else (
	python grayscale_image.py
)
pause
