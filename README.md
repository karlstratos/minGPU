This repository contains minimal code that demonstrates the basics of multi-GPU computing in PyTorch.

# Setup

```
conda create --name minGPU python=3.10
conda activate minGPU
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

# Exercise: Get the Same Updates

We'll train a toy model with vanilla SGD using
 * **Non-distributed:** Single-process single-device, where the device is either CPU or GPU
 * **DP (Data Parallel)**: Single-process multi-GPU
 * **DDP (Distributed Data Parallel)**: Multi-process (we'll use $K=2$ processes), each process is single-GPU
```
python main.py --no_shuffle --epochs 3 --opt sgd --batch_size 10 --lr 1  # CPU
python main.py --no_shuffle --epochs 3 --opt sgd --batch_size 10 --lr 1  --gpus 0  # GPU
python main.py --no_shuffle --epochs 3 --opt sgd --batch_size 10 --lr 1  --gpus 0,1  # DP
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --no_shuffle --epochs 3 --opt sgd --batch_size 5 --lr 2 --gpus 0,1  # DDP
```
Note that with $2$ processes, DDP must
 1. Use the (per-device) batch size $5$ to achieve the effective batch size $10$.
 2. Use the learning rate $2 \eta$, since it all-reduces and *averages* the gradients across processes in backward: $w' = w - \eta g = w - (2 \eta) (g/2)$.

But if the update is invariant to gradient scaling (e.g., any AdaGrad-style update, including Adam), the learning rate should *not* be scaled since $w' = w - \eta \text{Update}(g) = w - \eta \text{Update}(g/2)$.
Such an update is usually not exactly scaling-invariant because of the $\epsilon$ smoothing, but is still approximately invariant so $\eta$ shouldn't be scaled.
We'll use $\epsilon=0$ here for demonstration.
```
python main.py --no_shuffle --epochs 3 --opt adam --batch_size 10 --lr 1 --eps 0 # CPU
python main.py --no_shuffle --epochs 3 --opt adam --batch_size 10 --lr 1 --eps 0 --gpus 0  # GPU
python main.py --no_shuffle --epochs 3 --opt adam --batch_size 10 --lr 1 --eps 0 --gpus 0,1  # DP
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --no_shuffle --epochs 3 --opt adam --batch_size 5 --lr 1 --eps 0 --gpus 0,1  # DDP
```

# Distributed Sampler

PyTorch's [DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler) shards $N$ examples so that each of the $K$ processes receives a batch of $B$ examples in each iteration. That means
 1. One epoch consists of $\lfloor N/(KB) \rfloor$ iterations.
 2. After those iterations, we have $M < KB$ examples left ($M=0$ iff $KB$ divides $N$).

If $M \neq 0$, the sampler shards the $M$ remaining examples so that each process receives a batch of $\lceil M/K \rceil$ examples.
The last process receives a batch padded with the first examples. For instance,

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

Alternatively, we can drop $T = M$ % $K$ trailing examples by setting `drop_last=True`. In the above example, $T = 3$ % $2 = 1$ and we have

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

(i.e., x9 is discarded). Run the following to verify.

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --num_examples 9 --dim 1 --batch_size 3 --epochs 1 --no_shuffle --gpus 0,1
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --num_examples 9 --dim 1 --batch_size 3 --epochs 1 --no_shuffle --gpus 0,1 --drop_last
```

Note: when shuffling, the distributed sampler must call `set_epoch(epoch)` at the beginning of each epoch before creating the `DataLoader` iterator that uses the sampler. Otherwise, the same shuffling is used for all epochs.

# Fully Sharded Data Parallel (FSDP)

The only difference between FSDP and DDP is that FSDP shards the model parameters, gradients, and optimizer states across $K$ processes &mdash; in addition to data sharding (again using the distributed sampler).
We can turn FSDP into special cases simply by changing [`sharding_strategy`](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy), for instance
 - `NO_SHARD`: Each process just holds the entire model/grads/opt states. This is DDP.
 - `FULL_SHARD`: Model/grads/opt states are fully sharded. This is essentially [ZeRO Stage 3](https://www.deepspeed.ai/tutorials/zero/).

How do we get away with model/opt sharding? It uses the observation that not all params need to be loaded for the forward or backward pass (e.g., consider a multi-layer feedforward). So we can just [scatter/gather](https://pytorch.org/tutorials/intermediate/dist_tuto.html#collective-communication) as needed at some **communication cost** between processes. The idea is very similar to [activation checkpointing](https://pytorch.org/docs/stable/checkpoint.html), which is about intermediate hidden states rather than model parameters. FSDP and activation checkpointing can be used [together](https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/#activation-checkpointing).

## FSDP Wrapping
A central question how to shard parameters. PyTorch implements sharding by nested module wrapping.
Each FSDP wrap shards the parameters that are not sharded already by a descendent wrap by flattening them (`FlatParameter`) and evenly dividing them across $K$ processes, padding with zeros.

### Example

We use a toy $3$-layer feedforward with $21$ parameters for illustration: $u = C\\;\text{ReLU}(B\\;\text{ReLU}(A\\;x + a) + b) + c$ where

$$
A = \begin{bmatrix} \theta_1 & \theta_2 \\\ \theta_3 & \theta_4 \end{bmatrix} \quad a = \begin{bmatrix} \theta_5 \\\ \theta_6 \end{bmatrix} \qquad
B = \begin{bmatrix} \theta_7 & \theta_8 \\\ \theta_9 & \theta_{10} \end{bmatrix} \quad b = \begin{bmatrix} \theta_{11} \\\ \theta_{12} \end{bmatrix} \qquad
C = \begin{bmatrix} \theta_{13} & \theta_{14} \\\ \theta_{15} & \theta_{16} \\\ \theta_{17} & \theta_{18} \end{bmatrix} \quad c = \begin{bmatrix} \theta_{19} \\\ \theta_{20} \\\ \theta_{21}\end{bmatrix}
$$

If we just naively wrap the top module and use $K=2$ processes, the params will be sharded as

$$\begin{aligned}
\text{rank0} &= (\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6, \theta_7, \theta_8, \theta_9, \theta_{10}, \theta_{11})  \\\
\text{rank1} &= (\theta_{12}, \theta_{13}, \theta_{14}, \theta_{15}, \theta_{16}, \theta_{17}, \theta_{18}, \theta_{19}, \theta_{20}, \theta_{21}, \textcolor{red}{0})
\end{aligned}$$

Doing this will give no memory saving since at every layer one rank needs to get the params from the other, resulting in the full model in memory.
Only training will be slower due to communication cost.
```
python main_fsdp.py --batch_size 10  --gpus 0  # GPU
torchrun --standalone --nnodes=1 --nproc_per_node=2 main_fsdp.py --gpus 0,1 --batch_size 5 --dist ddp  # DDP
torchrun --standalone --nnodes=1 --nproc_per_node=2 main_fsdp.py --gpus 0,1 --batch_size 5 --dist fsdp --wrap naive  # FSDP
```
Instead, we can do layer-wise wrapping. In this case we will have the sharding:

$$\begin{aligned}
\text{rank0} &= (\theta_1, \theta_2, \theta_3),  (\theta_7, \theta_8, \theta_9), (\theta_{13}, \theta_{14}, \theta_{15}, \theta_{16}, \theta_{17})  \\\
\text{rank1} &= (\theta_4, \theta_5, \theta_6), (\theta_{10}, \theta_{11}, \theta_{12}), (\theta_{18}, \theta_{19}, \theta_{20}, \theta_{21}, \textcolor{red}{0})
\end{aligned}$$

Here, at no point do we need to store the entire model, since each FSDP wrap only needs to gather/scatter its own flattened params for the layer-wise computation. Thus the max number of parameters in memory during training is 10 (i.e., that of layer 3 with padding).
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 main_fsdp.py --gpus 0,1 --batch_size 5 --dist fsdp --wrap layer
```
Instead of having to wrap submodules manually, PyTorch provides an auto wrapper. For instance, the size-based auto wrapper with `min_num_params=7` will wrap a submodule only if the number of parameters is at least 7. So for the example we will only shard the layer 3 module and the top module, yielding

$$\begin{aligned}
\text{rank0} &= (\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6),  (\theta_{13}, \theta_{14}, \theta_{15}, \theta_{16}, \theta_{17})  \\\
\text{rank1} &= (\theta_7, \theta_8, \theta_9, \theta_{10}, \theta_{11}, \theta_{12}), (\theta_{18}, \theta_{19}, \theta_{20}, \theta_{21}, \textcolor{red}{0})
\end{aligned}$$

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 main_fsdp.py --gpus 0,1 --batch_size 5 --dist fsdp --wrap size
```

Layer-wise sharding seems natural for any multi-layer feedforward-style archiecture, including transformers. PyTorch provides an [auto wrap](https://github.com/pytorch/pytorch/blob/34fe850d0083688abf0a27f3e864723f0858aab1/torch/distributed/fsdp/wrap.py#L305C26-L305C26) for wrapping the specified transformer blocks.

### Computation-Communication Overlap

It is in our interest to maximize the overlap between (1) the time spent in doing the computation for the current FSDP unit (computation cost) and (2) the time spent in fetching the parameters of the next FSDP unit (communication cost), since they are independent and can be done simultaneously. Note however that the overlap can also increase the peak memory. The current best practice is to prefetch params before the current unit's grad computation ([`BACKWARD_PRE`](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.BackwardPrefetch)), which improves speed by up to 13% with <1% increased peak memory according to [this video](https://www.youtube.com/watch?v=sDM56HOziE4&list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT&index=8).
