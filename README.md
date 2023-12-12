# Setup

```
conda create --name multi-gpu python=3.10
conda activate multi-gpu
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia  # Remove 'pytorch-cuda=11.8' if installing on a Mac
```

# Commands

## SGD

The losses/grads are still slightly different between CPU/GPU/DP/DDP likely due to numerical precision differences. The batch size is per process. For DDP, LR has to be scaled in this case.

```
python main.py --no_shuffle --epochs 3 --opt sgd --batch_size 10 --lr 1  # CPU
python main.py --no_shuffle --epochs 3 --opt sgd --batch_size 10 --lr 1  --gpus 0  # GPU
python main.py --no_shuffle --epochs 3 --opt sgd --batch_size 10 --lr 1  --gpus 0,1  # DP
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --no_shuffle --epochs 3 --opt sgd --batch_size 5 --lr 2 --gpus 0,1  # DDP
```

## Scale-Invariant Updates

For DDP, LR should not be scaled in this case. If eps!=0, the DDP update will not be the same, but still LR shouldn't be scaled.

```
python main.py --no_shuffle --epochs 3 --opt adam --batch_size 10 --lr 1  # CPU
python main.py --no_shuffle --epochs 3 --opt adam --batch_size 10 --lr 1  --gpus 0  # GPU
python main.py --no_shuffle --epochs 3 --opt adam --batch_size 10 --lr 1  --gpus 0,1  # DP
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --no_shuffle --epochs 3 --opt adam --batch_size 5 --lr 1 --eps 0 --gpus 0,1  # DDP (using eps > 0 like 1e-5 will make the updates slightly different since they're no longer scale invariant)
```

### How DDP distributes data to the ranks

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --num_examples 7 --dim 1 --batch_size 3 --epochs 1 --no_shuffle --gpus 0,1
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --num_examples 7 --dim 1 --batch_size 3 --epochs 1 --no_shuffle --gpus 0,1 --drop_last
```
