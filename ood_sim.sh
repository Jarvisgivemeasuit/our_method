OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python ood_detector_sim.py \
--arch resnet50 \
--num_labels 100 \
--threshold 0.6 \
--batch_size_per_gpu 32 \
--in_data_path /home/ljl/Datasets/ImageNet/ \
--pretrained_weights /home/ljl/Documents/our_method/model_saving/imagenet_js_4/checkpoint.pth 