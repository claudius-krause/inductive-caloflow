# inductive-caloflow

Generation of calorimeter showers for datasets 2 and 3 of the [Calorimeter Challenge 2022](https://calochallenge.github.io/homepage/).

So far only tested for dataset 2.

## Idea

Split the problem into 3 steps and train one normalizing flow for each.
- flow 1: learns how the energy is deposited across the 45 layers: $p_1(E_i | E_\text{inc})$
- flow 2: learns how a normalized shower in layer 0 looks like, conditioned on $E_0$ and $E_\text{inc}$: $p_2(\mathcal{I}_0 | E_0, E_\text{inc})$
- flow 3: learns how a normalized shower in layer n looks like, conditioned on $E_n$, $E_{n-1}$, $E_\text{inc}$ and $\mathcal{I}_{n-1}$: $p_3(\mathcal{I}_n | \mathcal{I}_{n-1}, E_n, E_{n-1}, E_\text{inc})$


## Usage

