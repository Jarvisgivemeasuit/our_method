CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 29500 eval_linear.py \
--arch resnet50 \
--lr 0.01 \
--batch_size_per_gpu 256 \
--data_path /home/ljl/Datasets/ifood \
--k 0 \
--num_labels 50 \
--output_dir /home/ljl/Documents/our_method/model_saving/ifood_classifier_surr_0 \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/ifood_surr_0/checkpoint0400.pth