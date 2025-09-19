# python glow.py --train \
#                --dataset=mnist \
#                --data_dir="./dataset/" \
#                --n_levels=3 \
#                --depth=32 \
#                --width=512 \
#                --batch_size=16

# nohup python -m torch.distributed.launch --nproc_per_node=1 \
#        glow.py --train \
#                --distributed \
#                --dataset=mnist \
#                --data_dir="./dataset/" \
#                --n_levels=3 \
#                --depth=16 \
#                --width=512 \
#                --batch_size=32 > out_no_aug.log 2>&1 &


# nohup python -m torch.distributed.launch --nproc_per_node=1 \
#        glow.py --train \
#                --dataset=cifar10 \
#                --data_dir="./dataset/" \
#                --n_levels=3 \
#                --depth=32 \
#                --width=512 \
#                --batch_size=64 > cifar10.log 2>&1 &


# python -m torch.distributed.launch --nproc_per_node=1 \
#        glow.py --train \
#                --dataset=cifar10 \
#                --data_dir="./dataset/" \
#                --n_levels=3 \
#                --depth=32 \
#                --width=512 \
#                --batch_size=64

# nohup python -m torch.distributed.launch \
#        --nproc_per_node=1 \
#        glow.py --train \
#                --distributed \
#                --dataset=cifar10 \
#                --data_dir="./dataset/" \
#                --n_levels=3 \
#                --depth=32 \
#                --width=512 \
#                --batch_size=128 > cifar10_no_dequantisation_bs_128.log 2>&1 &

# python glow.py --evaluate \
#                --restore_file="./results/glow/2025-08-07_12-22-23/checkpoint.pt" \
#                --dataset=cifar10 \
#                --data_dir="./dataset" \
#                --width=512\
#                --depth=32\
#                --batch_size=32\
#                --n_levels=3
#         #        --[options of the saved model: n_levels, depth, width, batch_size]


python glow.py --generate \
               --restore_file="./results/glow/2025-08-07_12-22-23/checkpoint.pt" \
               --dataset=cifar10 \
               --data_dir="./dataset" \
               --width=512\
               --depth=32\
               --batch_size=32\
               --n_levels=3\
               --z_std=1.0