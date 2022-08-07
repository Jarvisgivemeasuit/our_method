OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ood_detector.py \
--arch resnet50 \
--num_labels 50 \
--lr 0.001 \
--batch_size_per_gpu 256 \
--k 2 \
--data_path /home/ljl/Datasets/ifood/ \
--output_dir /home/ljl/Documents/our_method/model_saving/ifood_ood_classifier_2 \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/ifood_surr_2/