cd dfme_movinet;
#--ckpt checkpoint/teacher/cifar10-resnet34_8x.pt
export CUDA_VISIBLE_DEVICES=1; python3 train.py --num_classes 600 --model_id "blackbox" --lambd 0  --grad_m 1 --query_budget 20 --log_dir save_results/cifar10  --lr_G 4e-4 --student_model resnet50 --loss l1 --nz 559 --batch_size 1 --epoch_itrs 30 ;
