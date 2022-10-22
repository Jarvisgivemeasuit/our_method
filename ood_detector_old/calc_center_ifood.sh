CUDA_VISIBLE_DEVICES=1 python calc_center.py \
--arch resnet50 \
--batch_size_per_gpu 256 \
--k 1 \
--data_path /home/ljl/Datasets/ifood \
--num_labels 50 \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/ifood_surr_1/checkpoint0400.pth \
--output_dir /home/ljl/Documents/our_method/model_saving/ifood_surr_1