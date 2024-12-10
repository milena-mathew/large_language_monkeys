from llmonk.utils import load_yaml, PlotSamplesConfig
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.special import comb
import numpy as np
import pydra

@pydra.main(PlotSamplesConfig)
def main(
    config: PlotSamplesConfig
):
    sample_files = list(Path(config.samples_dir).glob("*"))
    is_corrects = {}
    for i, file in tqdm(enumerate(sample_files), desc="loading eval files"):
        solution_data = load_yaml(file)
        solution_data["name"] = file.stem
        is_corrects[i] = solution_data['is_corrects']

    correct_counts = np.array([sum(is_corrects[k]) for k in is_corrects.keys()])
    num_samples_per = np.array([len(is_corrects[k]) for k in is_corrects.keys()])
    passes = []
    for k in range(1, 50):
        total_combos = comb(num_samples_per, k)
        incorrect_combos = comb(num_samples_per - correct_counts, k)
        pass_probs = np.clip(1 - incorrect_combos/total_combos, 0, 1)
        passes.append(np.mean(pass_probs))
    
    plt.plot(range(1, 50), passes)
    plt.xlabel("Number of Samples (#)")
    plt.ylabel("Coverage (pass @ k)")
    plt.savefig(config.output_file)

if __name__ == "__main__":
    main()
    