This repository contains minimal code that demonstrates how distributed computing actually works in PyTorch.

# Setup

```
conda create --name multi-gpu python=3.10
conda activate multi-gpu
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia  # Remove 'pytorch-cuda=11.8' if installing on a Mac
```

# CPU, GPU, DP vs DDP 

We first demonstrate that we obtain the same updates whether we do 
single-process single-device (CPU/GPU), single-process multi-device (DP), or multi-process multi-device (DDP).
(There are very slight differences in losses/gradients due to numerical precision issues.)
The batch size $B$ is per process, so we need $B = B_{\mathrm{eff}}/K$ where $B_{\mathrm{eff}}$ is the effective batch size and $K$ is the number of processes.
Furthermore, DDP all-reduces and *averages* the gradients across processes in backward, so the learning rate $\eta$ has to be $K$ times bigger with simple updates like SGD
since $w' = w - \eta g = w - (K \eta) (g/K)$.
The following code produces the same results.
```
python main.py --no_shuffle --epochs 3 --opt sgd --batch_size 10 --lr 1  # CPU
python main.py --no_shuffle --epochs 3 --opt sgd --batch_size 10 --lr 1  --gpus 0  # GPU
python main.py --no_shuffle --epochs 3 --opt sgd --batch_size 10 --lr 1  --gpus 0,1  # DP
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --no_shuffle --epochs 3 --opt sgd --batch_size 5 --lr 2 --gpus 0,1  # DDP
```
However, when we use an update that is invariant to gradient scaling (e.g., any AdaGrad-style update including Adam), 
the learning rate should *not* be scaled since $w' = w - \eta \text{Update}(g) = w - \eta \text{Update}(g/K)$. 
In this case, multiplying $\eta$ by $K$ will make the update $K$ times bigger.
(Such an update is usually not exactly scaling-invariant because of $\epsilon > 0$ smoothing, but is still approximately invariant so $\eta$ shouldn't be scaled.)
The following code produces the same results.
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
