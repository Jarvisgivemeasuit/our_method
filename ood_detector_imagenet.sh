OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port 29501 ood_detector.py \
--arch resnet50 \
--num_labels 100 \
--out_dim 1024 \
--lr 0.001 \
--batch_size_per_gpu 256 \
--k 0 \
--data_path /home/ljl/Datasets/ImageNet/ \
--output_dir /home/ljl/Documents/our_method/model_saving/imagenet_ood_nonorm_classifier_0 \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/imagenet_surr_0/