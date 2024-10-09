CUDA_VISIBLE_DEVICES=0 python3.12 train.py --which_example=darcy --which_model=PTFNO --random_seed=0
CUDA_VISIBLE_DEVICES=0 python3.12 train.py --which_example=darcy --which_model=FNO --random_seed=0
CUDA_VISIBLE_DEVICES=0 python3.12 train.py --which_example=darcy --which_model=GFNO --random_seed=0
CUDA_VISIBLE_DEVICES=0 python3.12 train.py --which_example=darcy --which_model=RFNO --random_seed=0

