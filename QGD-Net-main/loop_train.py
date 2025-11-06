import subprocess
from tqdm import tqdm
import os
from tqdm import tqdm

models = [
    "qgd_net",
    "unet",
]
datasets = [
    "isic2016",
    "isic2017",
    "chasedb1",
]
groups = [16, 8, 4, 0]

output_dir = "./outputs/"


for model in models:
    for dataset in datasets:
        for group in tqdm(groups, desc=f"Model {model}, Dataset {dataset}"):
            command = [
                "python",
                "-u",
                "train.py",
                "-m",
                f"{model}",
                "-d",
                f"{dataset}",
                "-g",
                f"{str(group)}",
                "-e",
                "20",
            ]

            output_file = f"{model}_{dataset}_group{group}.txt"
            os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(output_dir, output_file), "w") as f:
                subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
