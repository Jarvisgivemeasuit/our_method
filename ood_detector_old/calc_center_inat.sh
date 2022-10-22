CUDA_VISIBLE_DEVICES=1 python calc_center.py \
--arch resnet50 \
--batch_size_per_gpu 256 \
--k 2 \
--data_path /home/ljl/Datasets/inat \
--num_labels 100 \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/inat_surr_2/checkpoint0400.pth \
--output_dir /home/ljl/Documents/our_method/model_saving/inat_surr_2