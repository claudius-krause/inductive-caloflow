# Inductive CaloFlow
## by Matthew Buckley, Claudius Krause, Ian Pang, and David Shih

This repository contains the source code for reproducing the results of

_"Inductive CaloFlow"_ by Matthew Buckley, Claudius Krause, Ian Pang, and David Shih, [arxiv: 2305.11934](https://arxiv.org/abs/2305.11934)

If you use the code, please cite:
```
@article{Buckley:2023rez,
    author = "Buckley, Matthew R. and Krause, Claudius and Pang, Ian and Shih, David",
    title = "{Inductive CaloFlow}",
    eprint = "2305.11934",
    archivePrefix = "arXiv",
    primaryClass = "physics.ins-det",
    month = "5",
    year = "2023"
}
```

Generation of calorimeter showers for datasets 2 and 3 of the [Calorimeter Challenge 2022](https://calochallenge.github.io/homepage/).

## Idea

Split the problem into 3 steps and train one normalizing flow for each.
- Flow-1: learns how the energy is deposited across the 45 layers: $p_1(E_i | E_\text{inc})$
- Flow-2: learns how a normalized shower in layer 0 looks like, conditioned on $E_0$ and $E_\text{inc}$: $p_2(I_0 | E_0, E_\text{inc})$
- Flow-3: learns how a normalized shower in layer $n$ looks like, conditioned on $E_n$, $E_{n-1}$, $E_\text{inc}$ and $I_{n-1}$: $p_3(I_n | I_{n-1}, E_n, E_{n-1}, E_\text{inc})$

Generation of showers is done sequentially. For given $E_\text{inc}$, on first samples $E_i$ using Flow-1. The obtained $E_0$ together with $E_\text{inc}$ are then used to sample the showers of layer 0 using flow 2. These showers are renormalized to have total energy $E_0$. Given these showers and the $E_i$, Flow-3 then iteratively samples the shower, calolayer by calolayer.

## Usage

To train all 3 flows, one after another, run the code with
```python run.py --which_ds 2 --which_flow 7 --train --data_dir path/to/hdf5_dataset_folder```

Here, `--which_ds` specifies which dataset is used (2 or 3); `--which_flow` specifies which subset of the 3 flows is worked with; and `--data_dir` points to the folder in which the `dataset_X_1.hdf5` files are located. The subset of flows to be worked with are encoded in a binary sum: Flow-1 is contributes 1, Flow-2 contributes 2, and Flow-3 contributes 4. Training all 3 flows means `--which_flow` is 1+2+4=7. Only training Flow 1 and 3 means `--which_flow` is 1+4=5, etc.

To evaluate the flows, i.e. getting the LL of the test set, run

```python run.py --which_ds 2 --which_flow 7 --evaluate --data_dir path/to/hdf5_dataset_folder```

The LLs should be (close to) the ones you saw in training when the best model was saved.

To generate full showers, you only need to generate from flow 3, as that requires sampling from flows 1 and 2 and is done automatically:

```python run.py --which_ds 2 --which_flow 4 --generate --data_dir path/to/hdf5_dataset_folder```

Final samples as well as model checkpoints are stored at `results/`. This folder can be changed by adding the flag `--output_dir new_location/` when running the code. Output that is printed on screen while training/evaluating/generating is also saved at `results/results.txt`.