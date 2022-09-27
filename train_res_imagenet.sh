# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 main.py \
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python main.py \
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port 29500 main.py \
--arch resnet50 \
--num_workers 20 \
--optimizer sgd \
--lr 0.03 \
--out_dim 8192 \
--k 0 \
--num_labels 100 \
--epochs 300 \
--batch_size_per_gpu 96 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.05 0.14 \
--data_path /home/ljl/Datasets/ImageNet/ \
--output_dir /home/ljl/Documents/our_method/model_saving/imagenet_js_1
