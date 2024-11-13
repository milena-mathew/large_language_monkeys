from llmonk.utils import load_yaml, PlotSamplesConfig
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import pydra

@pydra.main(PlotSamplesConfig)
def main(
    config: PlotSamplesConfig
):
    sample_files = list(Path(config.samples_dir).glob("*"))
    is_corrects = defaultdict(list)
    for file in tqdm(sample_files, desc="loading eval files"):
        solution_data = load_yaml(file)
        solution_data["name"] = file.stem
        for i in range(1, len(solution_data['is_corrects'])):
            is_corrects[i].append(any(solution_data['is_corrects'][:i]))

    num_samples = list(is_corrects.keys())
    percentages = [sum(is_corrects[num])/len(is_corrects[num]) for num in num_samples]
    plt.plot(num_samples, percentages)
    plt.xlabel("Number of Samples (#)")
    plt.ylabel("Coverage (pass @ k)")
    plt.savefig('test.png')

if __name__ == "__main__":
    main()
    