CUDA_VISIBLE_DEVICES=1 python calc_ind_acc.py \
--arch resnet50 \
--batch_size_per_gpu 256 \
--data_path /home/et21-lijl/Datasets/Imagenet100/train \
--pretrained_weights /home/et21-lijl/Documents/dino/model_saving/imagenet100/checkpoint0399.pth