
python3 plotlog.py &

./sample_unet /cuda /loop 1500 /delta 0.03,0.01 /l sample_unet.mlp /lgrd /log trained_log.csv /vio predict/out/valid_image.bmp /tio predict/out/train_image.bmp

