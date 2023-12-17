This repository contains minimal code that demonstrates the basics of multi-GPU computing in PyTorch.

# Setup

```
conda create --name minGPU python=3.10
conda activate minGPU
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia  
```

# Exercise: Getting the Same Updates
We'll train a toy model with vanilla SGD by
 * **Non-distributed:** Single-process single-device, where the device is either CPU or GPU
 * **DP (Data Parallel)**: Single-process multi-GPU 
 * **DDP (Distributed Data Parallel)**: Multi-process (we'll use $K=2$ processes), each process is single-GPU 
```
python main.py --no_shuffle --epochs 3 --opt sgd --batch_size 10 --lr 1  # CPU
python main.py --no_shuffle --epochs 3 --opt sgd --batch_size 10 --lr 1  --gpus 0  # GPU
python main.py --no_shuffle --epochs 3 --opt sgd --batch_size 10 --lr 1  --gpus 0,1  # DP
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --no_shuffle --epochs 3 --opt sgd --batch_size 5 --lr 2 --gpus 0,1  # DDP
```
Note that DDP must use the batch size $5$ to achieve the effective batch size $10$ with $2$ processes.
Also, DDP must use the learning rate $2 \eta$ since it all-reduces and *averages* the gradients across processes in backward:
$w' = w - \eta g = w - (2 \eta) (g/2)$.

## Scale-Invariant Updates
But if the update is invariant to gradient scaling (e.g., any AdaGrad-style update, including Adam),
the learning rate should *not* be scaled since $w' = w - \eta \text{Update}(g) = w - \eta \text{Update}(g/2)$.
Such an update is usually not exactly scaling-invariant because of the $\epsilon$ smoothing, but is still approximately invariant so $\eta$ shouldn't be scaled.
We'll use $\epsilon=0$ here for demonstration.
```
python main.py --no_shuffle --epochs 3 --opt adam --batch_size 10 --lr 1 --eps 0 # CPU
python main.py --no_shuffle --epochs 3 --opt adam --batch_size 10 --lr 1 --eps 0 --gpus 0  # GPU
python main.py --no_shuffle --epochs 3 --opt adam --batch_size 10 --lr 1 --eps 0 --gpus 0,1  # DP
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --no_shuffle --epochs 3 --opt adam --batch_size 5 --lr 1 --eps 0 --gpus 0,1  # DDP
```

## Data Sharding

DDP shards $N$ examples so that each of the $K$ processes receives a batch of $B$ examples in each iteration.
That means an epoch consists of $\lfloor N/(KB) \rfloor$ iterations, after which we have $M < KB$ examples left.
If $M \neq 0$, DDP shards them so that each process receives a batch of $\lceil M/K \rceil$ examples
where the batch of the last process is padded with the first examples.
For instance,

<div align="center">
<table>
<tr><th>Examples</th><th>Iteration 1</th><th>Iteration 2</th></tr>
<tr><td>

x1 x2 x3 x4 x5 x6 x7 x8 x9

</td><td>

|||
|-----------|----------|
| Process 1   | x1 x3 x5 |
| Process 2   | x2 x4 x6 |

</td><td>

|||
|-----------|----------|
| Process 1   | x7 x9 |
| Process 2   | x8 **x1** |

</td></tr>
</table>
</div>

Alternatively, we can drop $T = M \mod K$ trailing examples (`drop_last`). The above example becomes

<div align="center">
<table>
<tr><th>Examples</th><th>Iteration 1</th><th>Iteration 2</th></tr>
<tr><td>

x1 x2 x3 x4 x5 x6 x7 x8 x9

</td><td>

|||
|-----------|----------|
| Process 1   | x1 x3 x5 |
| Process 2   | x2 x4 x6 |

</td><td>

|||
|-----------|----------|
| Process 1   | x7 |
| Process 2   | x8 |

</td></tr>
</table>
</div>

where x9 is discarded. The following code demonstrates the example.

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --num_examples 9 --dim 1 --batch_size 3 --epochs 1 --no_shuffle --gpus 0,1
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --num_examples 9 --dim 1 --batch_size 3 --epochs 1 --no_shuffle --gpus 0,1 --drop_last
```
