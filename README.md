# Setup

Inside the directory, create a virtual environment and install PyTorch.

```
python -m venv myenv
. myenv/bin/activate
pip install -r requirements.txt
```

If using CUDA 11, you will have to (as of Jan 3 2022) explicitly install a compatible version by following with

```
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
