OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ood_detector.py \
--arch resnet50 \
--num_labels 100 \
--lr 0.001 \
--batch_size_per_gpu 256 \
--k 0 \
--data_path /home/ljl/Datasets/inat/ \
--output_dir /home/ljl/Documents/our_method/model_saving/inat_ood_classifier_0 \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/inat_surr_0/