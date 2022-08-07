OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python ood_test.py \
--arch wideresnet \
--num_labels 100 \
--out_dim 1024 \
--batch_size_per_gpu 256 \
--data_path_in /home/ljl/Datasets/cifar100/ \
--output_dir /home/ljl/Documents/our_method/model_saving/cifar_ood_classifier \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/cifar100_wrn/cifar100_wrn_pretrained_epoch_99.pt \
--classifier_weights /home/ljl/Documents/our_method/model_saving/cifar_ood_classifier/checkpoint.pth.tar