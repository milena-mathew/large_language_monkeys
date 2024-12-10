from pathlib import Path
from multiprocessing import Pool
import threading
from tqdm import tqdm
import yaml
import time
import concurrent.futures
import pydra
import re

from llmonk.evaluate.code_contests_utils import execution_server_client
from llmonk.utils import load_yaml, extract_first_code, EvaluateScriptConfig

MAX_CONCURRENT_REQUESTS = 512
semaphore = threading.Semaphore(value=MAX_CONCURRENT_REQUESTS)
NUM_RETRIES = 3
RETRY_BACKOFF = 3


def is_valid_python(snippet):
    try:
        compile(snippet, "<string>", "exec")
        return True
    except SyntaxError:
        return False
           
def extract_first_code(output_string: str):
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # sometimes the block of code is ```python ... ``` instead of ``` ... ```
        # in this case strip the python out

        if code.startswith("python"):
            code = code[len("python") :].strip()

        return code

    if is_valid_python(trimmed):
        return trimmed

    return None

def solution_is_correct(
    code,
    problem: dict,
    client: execution_server_client.ExecutionServerClient,
):
    if code is None:
        return False, None

    assert len(problem["test_cases"]["input"]) == len(problem["test_cases"]["output"])

    input_expected_output_pairs = list(
        zip(problem["test_cases"]["input"], problem["test_cases"]["output"])
    )

    with semaphore:
        for i in range(NUM_RETRIES):
            try:
                extracted_code = extract_first_code(code)
                if extracted_code is None:
                    is_correct = False
                    generated_output = None
                else:
                    is_correct, generated_output = client.execute_code(
                        extracted_code,
                        input_expected_output_pairs,
                        timeout=problem["timeout"] + 10,  # buffer for 10
                        memory_limit_bytes=2_000_000_000_000,  # double max limit
                    )
                break
            except:
                if i == NUM_RETRIES - 1:
                    raise
                time.sleep(RETRY_BACKOFF**i)

    return is_correct, generated_output


def grade_problems(
    solutions_data: dict,
    output_dir: Path,
    client: execution_server_client.ExecutionServerClient,
):
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_REQUESTS // 2
    ) as executor:
        is_corrects_futures = [
            executor.submit(
                solution_is_correct,
                code=code,
                problem=solutions_data,
                client=client,
            )
            for code in solutions_data["solutions"]
        ]
        is_corrects = []
        generated_outputs = []

        for i, future in enumerate(is_corrects_futures):
            if i % 100 == 0:
                print("Progress being made...")
            try:
                is_correct, generated_output = future.result()
            except:
                import pdb; pdb.set_trace()
            is_corrects.append(is_correct)
            generated_outputs.append(generated_output)

    solutions_data["is_corrects"] = is_corrects
    solutions_data["generated_outputs"] = generated_outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / solutions_data["name"], "w") as f:
        yaml.dump(
            solutions_data,
            f,
            sort_keys=True,
        )


def load_data_from_yaml_file(input_path):
    """Loads results data from a yaml file."""
    data = load_yaml(input_path)
    try:
        solutions = []
        for sample in data["samples"]:
            if isinstance(sample, str):
                solutions.append(extract_first_code(sample))
            elif isinstance(sample, list):
                if len(sample) != 1:
                    import pdb; pdb.set_trace()
                else:
                    solutions.append(extract_first_code(sample[-1]))
            else:
                import pdb; pdb.set_trace()
        data["solutions"] = solutions
    except:
        import pdb; pdb.set_trace()
    data["name"] = input_path.stem
    return data


@pydra.main(EvaluateScriptConfig)
def main(config):
    sample_files = list(Path(config.samples_dir).glob("*.yaml"))
    already_evaled = {p.stem for p in Path(config.save_dir).glob("*.yaml")}

    to_eval_files = [f for f in sample_files if f.stem not in already_evaled]
    print(
        f"Num to eval: {len(to_eval_files)}",
        f"Num already evaled: {len(already_evaled)}",
    )

    solutions_data = [load_data_from_yaml_file(file) for file in tqdm(to_eval_files, desc="loading yaml files")]

    print("Done loading yaml files.")
    import pdb; pdb.set_trace()
    # multiprocessing pool is used to load data
    with execution_server_client.ExecutionServerClient() as client:
        # threads are used to run code in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.num_workers
        ) as executor:
            futures = [
                executor.submit(
                    grade_problems,
                    solutions_data=solution_data,
                    output_dir=config.save_dir,
                    client=client,
                )
                for solution_data in solutions_data
            ]

            for future in tqdm(futures, desc="Running tests on problem"):
                future.result()


if __name__ == "__main__":
    main()
