set -ex
python train.py --dataroot ./../datasets/eo_sar_256 --name eo_sar_cyclegan --model cycle_gan --pool_size 50 --no_dropout
