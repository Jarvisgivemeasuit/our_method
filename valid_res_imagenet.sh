# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python eval_linear.py \
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port 29501 eval_linear.py \
--arch resnet50 \
--lr 0.01 \
--batch_size_per_gpu 256 \
--data_path /home/ljl/Datasets/ImageNet \
--k 0 \
--num_labels 100 \
--output_dir /home/ljl/Documents/our_method/model_saving/imagenet_classifier_gmm_0.0001 \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/imagenet_gmm_0.0001/checkpoint.pth