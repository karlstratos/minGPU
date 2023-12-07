# Setup

```
conda create --name multi-gpu python=3.10
conda activate multi-gpu
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia  # Remove 'pytorch-cuda=11.8' if installing on a Mac
```
