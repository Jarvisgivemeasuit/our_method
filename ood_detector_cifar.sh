# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port 29501 origin_ood_detector.py \
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4 python origin_ood_detector.py \
--arch wideresnet \
--num_labels 100 \
--out_dim 1024 \
--lr 0.001 \
--batch_size_per_gpu 256 \
--k 0 \
--data_path_in /home/ljl/Datasets/cifar100/ \
--data_path_out /home/ljl/Datasets/tiny-imagenet-200 \
--output_dir /home/ljl/Documents/our_method/model_saving/cifar_ood_classifier \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/cifar100_wrn/cifar100_wrn_pretrained_epoch_99.pt