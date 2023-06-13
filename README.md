# IDSNN
# Change the path to your datasets in conf/global_settings.py

# baseline training
python -u -m torch.distributed.launch --master_port=1234 --nproc_per_node=4 train_baseline.py -net Sresnet18 -dataset cifar100 -b 64 -lr 0.05

# downsample initialization training
python -u -m torch.distributed.launch --master_port=1234 --nproc_per_node=4 train_ds.py -net Sresnet18 -dataset cifar100 -b 64 -lr 0.05 

# kd training
python -u -m torch.distributed.launch --master_port=1234 --nproc_per_node=4 train_kd.py -net Sresnet18 -dataset cifar100 -teacher_net resnet34 -b 64 -lr 0.05 

# # IDSNN training
python -u -m torch.distributed.launch --master_port=1234 --nproc_per_node=4 train_kd_ds.py -net Sresnet18 -teacher_net resnet34 -dataset cifar100 -b 64 -lr 0.05

The batchsize (-b) can be adjusted linearly to your GPU memory, and the learning rate should be adjusted accordingly.
