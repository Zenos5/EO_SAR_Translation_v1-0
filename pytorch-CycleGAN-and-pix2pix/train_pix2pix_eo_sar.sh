set -ex
python train.py --dataroot /home/atrc/angel/sar_eo_translation/datasets/eo_sar_256 --name eo_sar_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
