# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python eval_linear.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 29500 eval_linear.py \
--arch resnet50 \
--lr 0.01 \
--batch_size_per_gpu 256 \
--data_path /home/ljl/Datasets/ImageNet \
--k 2 \
--num_labels 150 \
--output_dir /home/ljl/Documents/our_method/model_saving/imagenet_classifier_2 \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/imagenet_2/checkpoint0400.pth