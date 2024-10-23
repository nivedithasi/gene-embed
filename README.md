# gene-embed

## Setup
1. Clone the data repository: `git lfs clone https://github.com/jump-cellpainting/pilot-cpjump1-data`. If you do not have git lfs, refer to [installation instructions](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#getting-started).
2. Clone this repository: `git clone https://github.com/nivedithasi/gene-embed`
3. Symlink the CPJUMP data in gene-embed: `cd gene-embed` and `ln -s ../pilot-cpjump1-data/ .`
4. Create a micromamba environment: `micromamba env create -n gene-embed -f env.yaml`. If you do not have micromamba set up, refer to [this guide](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html).
5. Activate the environment `micromamba activate gene-embed` and install mkl: `pip install mkl==2022.1.0`

## Download existing experiments
0. `cd gene-embed/code`
1. Download gene_embed_experiment_runs.zip from Zenodo ([link here](https://zenodo.org/records/13984543?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjcwYTE4MTJhLWI5MmMtNGY1NC04NDRmLWVmMTE4N2QwMWQ5MSIsImRhdGEiOnt9LCJyYW5kb20iOiJhOTJlNTE4OWRiODkzMWU4YTQwM2E1OTRhZmVlY2Q1NyJ9.1Hnpvme5b_5w648fX8VHMvZnkopmqhnvXPWhuTcmEPGYhIIGRxI7FpS9tNkMM020Imqo-7KTxO9L-znI2G10nQ)).
2. Unzip the folder to obtain `gene_embed_experiment_runs`.
3. Move the sub-folders into `gene-embed/code`: `mv gene_embed_experiment_runs/* gene-embed/code`

## Running experiments
0. `cd gene-embed/code` and activate the environment `micromamba activate gene-embed`.
1. Create a grid search by editing `grid.py` (edit the search space and `output_dir` folder path).
2. Create config files by running `python grid.py`. `output_dir` should contain sub-folders for each config now.
3. Launch the experiments: `bash run.sh output_dir`.
4. All experiment results and the trained model weights will be stored under `output_dir` (see test_scores.json, model.pt etc.)
5. Set Find the best model on the validation set by running `python find_best_val.py`. This will display the test set performance of the best val model under each experiment. Edit `source_folders` in this file to obtain results on additional experiments.
