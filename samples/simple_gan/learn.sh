
python3 plotlog.py &

./simple_gan /ganloop 50 /loop 50 /nlfb /batch 100 /batch_thread 8 /delta 0.1 /clsf classifier.mlp /l simple_gan.mlp /log trained_log.csv

