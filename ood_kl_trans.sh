OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 29498 ood_detector_kl_trans.py \
--arch resnet50 \
--num_labels 100 \
--lr 1e-2 \
--epochs 100 \
--batch_size_per_gpu 128 \
--in_data_path /home/ljl/Datasets/ImageNet/ \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/imagenet_js_4/checkpoint.pth \
--output_dir /home/ljl/Documents/our_method/model_saving/maha_trans_01