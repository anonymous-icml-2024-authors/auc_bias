## AUC Biases

### Setting Up

Run the following commands to clone this repo and create the Conda environment:

```
git clone {this repo}
cd auc_bias/real_world
conda env create -f environment.yml
conda activate auc_bias
```


### Training a Single Model
To train a single model, call `train.py` with the appropriate arguments, for example:

```
python -m auc_biases.train \ 
    --output_dir /output/dir \
    --dataset adult \
    --algorithm xgb \
    --post_hoc_calibrate \
    --balance_groups \
    --attribute 0 \
    --enforce_prevalence_ratio \
    --prevalence_ratio 3
```

Note that we are unable to provide the `mimic` dataset at this time due to the PhysioNet data usage agreement. 


### Training a Grid of Models

To reproduce the experiments in the paper which involve training a grid of models using different hyperparameters, use `sweep.py` as follows:

```
python sweep.py launch \
    --experiment {experiment_name} \
    --output_dir {output_root} \
    --command_launcher {launcher} 
```

where:
- `experiment_name` corresponds to experiments defined as classes in `experiments.py`
- `output_root` is a directory where experimental results will be stored.
- `launcher` is a string corresponding to a launcher defined in `launchers.py` (i.e. `slurm` or `local`).

The experiment `vary_group_weight_with_seeds` corresponds to Figure 3 (updated with 20 repeats).

### Aggregating Results

After an experiment has finished running, to create Figures 3, 7, and 8, run `notebooks/agg_results.ipynb`
