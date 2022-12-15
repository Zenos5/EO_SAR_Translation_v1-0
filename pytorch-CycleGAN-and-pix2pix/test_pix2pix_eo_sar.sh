set -ex
python test.py --dataroot /home/atrc/angel/sar_eo_translation/datasets/eo_sar_256 --name eo_sar_pix2pix --model pix2pix --netG unet_256 --direction BtoA --dataset_mode aligned --norm batch
