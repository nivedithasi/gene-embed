import subprocess
import os

split = 'compound'
features = 'crispr_orf'

layers = [4, 6]
heads = [4, 5, 10]
hidden = [128, 512, 1024]
lr = [0.001, 0.0001]
dropout = [0.1, 0.2]

permute = True

if permute:
    output_dir = f"grid_search_{split}_permute"
else:
    output_dir = f"grid_search_{split}"

split_folder = os.path.join(os.getcwd(), "../split_metadata")

for layer in layers:
    for head in heads:
        for hid in hidden:
            for l in lr:
                for drop in dropout:
                    if permute:
                        # for permutation baseline
                        subprocess.run(["python", "config.py", split_folder, split, features, str(head), str(hid), str(layer), str(l), str(drop), output_dir, "--permute"])
                    else:
                        # for no permutation
                        subprocess.run(["python", "config.py", split_folder, split, features, str(head), str(hid), str(layer), str(l), str(drop), output_dir])
