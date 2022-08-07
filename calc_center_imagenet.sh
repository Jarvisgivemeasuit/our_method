OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 torchrun --master_port 29503 calc_center.py \
--arch resnet50 \
--batch_size_per_gpu 1024 \
--k 2 \
--data_path /home/ljl/Datasets/ImageNet \
--num_labels 100 \
--out_dim 1024 \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/imagenet_surr_2/checkpoint0400.pth \
--output_dir /home/ljl/Documents/our_method/model_saving/imagenet_surr_2