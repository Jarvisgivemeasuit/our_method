CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port 29501 calc_ind_acc.py \
--arch resnet50 \
--num_labels 100 \
--batch_size_per_gpu 128 \
--data_path /home/ljl/Datasets/ImageNet/ \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/imagenet_js_6/checkpoint.pth