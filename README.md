# Setup

```
conda create --name multi-gpu python=3.10
conda activate multi-gpu
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia  # Remove 'pytorch-cuda=11.8' if installing on a Mac
```

# Commands

```
# Single-process single-device (CPU)
python main.py --no_shuffle --batch_size 10

# Single-process single-device (GPU)
python main.py --no_shuffle --batch_size 10 --gpus 0

# Single-process multi-device (DP)
python main.py --no_shuffle --batch_size 10 --gpus 0,1

# Multi-process multi-device (DDP)
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --no_shuffle --gpus 0,1 --batch_size 5
```
