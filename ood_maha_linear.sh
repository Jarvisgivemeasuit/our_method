OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port 29499 ood_detector_maha_linear.py \
--arch resnet50 \
--num_labels 100 \
--lr 1e-4 \
--epochs 100 \
--batch_size_per_gpu 128 \
--in_data_path /home/ljl/Datasets/ImageNet/ \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/imagenet_js_4/checkpoint.pth \
--output_dir /home/ljl/Documents/our_method/model_saving/classifier_01