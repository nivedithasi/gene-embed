# gene-embed

## Setup
1. Clone the data repository: `git lfs clone https://github.com/jump-cellpainting/pilot-cpjump1-data`. If you do not have git lfs, refer to [installation instructions](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#getting-started).
2. Clone this repository: `git clone https://github.com/nivedithasi/gene-embed`
3. Symlink the CPJUMP data in gene-embed: `cd gene-embed` and `ln -s ../pilot-cpjump1-data/ .`
4. Create a micromamba environment: `micromamba env create -n gene-embed -f env.yaml`. If you do not have micromamba set up, refer to [this guide](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html).
5. Activate the environment `micromamba activate gene-embed` and install mkl: `pip install mkl==2022.1.0`

## Running experiments
0. `cd gene-embed/code` and activate the environment `micromamba activate gene-embed`.
1. Create a grid search by editing `grid.py` (edit the search space and `output_dir` folder path).
2. Create config files by running `python grid.py`. `output_dir` should contain sub-folders for each config now.
3. Launch the experiments: `bash run.sh output_dir`.
4. All experiment results and the trained model weights will be stored under `output_dir` (see test_scores.json, model.pt etc.)
5. Set Find the best model on the validation set by running `python find_best_val.py`. This will display the test set performance of the best val model under each experiment. Edit `source_folders` in this file to obtain results on additional experiments.