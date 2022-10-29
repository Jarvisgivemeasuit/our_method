OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python ood_test_kl.py \
--arch resnet50 \
--num_labels 100 \
--batch_size_per_gpu 128 \
--in_data_path /home/ljl/Datasets/ImageNet/ \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/imagenet_js_4/checkpoint.pth \
--output_dir /home/ljl/Documents/our_method/model_saving/kl_div_trans_02