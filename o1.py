import os
import json
import typing
import logging
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from IPython import embed
import numpy as np
import concurrent.futures
import statistics
from PIL import Image
from helpers.plot_helpers import plot_majority_vote_graph, plot_just_ask_nicely_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

O1_MODEL = "o1-mini"

OPENAI_CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
RESPONSE_CACHE_FILENAME = 'helpers/response_cache.json'
PROMPT = """You are a math problem solver. I will give you a problem from the American Invitational Mathematics Examination (AIME). At the end, provide the final answer as a single integer.

Important: You should try your best to use around {token_limit} tokens in your reasoning steps.
If you feel like you are finished early, spend the extra tokens trying to double check your work until you are absolutely sure that you have the correct answer.
Here's the problem:

{problem}

Solve this problem, use around {token_limit} tokens in your reasoning, and provide the final answer as a single integer.
"""


def load_2024_dataset() -> list[dict]:
    """
    Load the dataset of problems.

    Returns:
        list[dict]: The dataset of problems.
    """
    dataset_original = load_dataset("AI-MO/aimo-validation-aime")

    # Filter out problems that are not from 2024
    dataset = dataset_original["train"].filter(lambda example: "2024" in example["url"])

    logging.debug(f"Filtered dataset size: {len(dataset)}.")
    assert len(dataset) == 30, f"Expected 30 problems after filtering by 2024, but found {len(dataset)}"
    return dataset


def get_or_create_cache(filename: str) -> dict[str, typing.Any]:
    """
    Get the cache if it exists, otherwise create it.

    Args:
        filename (str): The filename of the cache to get or create.

    Returns:
        dict: The cache.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache, filename):
    with open(filename, 'w') as f:
        json.dump(cache, f)


def get_response(problem: str, token_limit: int, cache: dict, idx: int = 0) -> dict:
    """
    Get a response from the model.

    Args:
        problem (str): The problem to process.
        token_limit (int): The token limit for the model.
        cache (dict): The cache to use for storing responses.
        idx (int, optional): The index of the response to process. Defaults to 0.

    Returns:
        dict: The response from the model.
    """

    if idx > 0:
        cache_key = f"{O1_MODEL}_{PROMPT}_{problem}_{token_limit}_{idx}"
    else:
        cache_key = f"{O1_MODEL}_{PROMPT}_{problem}_{token_limit}"
    if cache_key in cache:
        logging.debug(f"Cache hit for problem: {problem[:20]}. idx: {idx}. Requested tokens: {token_limit}.")
        return cache[cache_key]
    
    formatted_prompt = PROMPT.format(problem=problem, token_limit=token_limit)
    logging.debug(f"Requesting {token_limit} tokens for problem starting with: {problem[:20]} running {idx} of {N} times.")
    response = OPENAI_CLIENT.chat.completions.create(
        model=O1_MODEL,
        messages=[{"role": "user", "content": formatted_prompt}]
    )
    result = {
        'content': response.choices[0].message.content,
        'tokens': response.usage.completion_tokens
    }
    cache[cache_key] = result
    logging.debug(f"Received {result['tokens']} tokens for problem starting with: {problem[:20]}. Requested tokens: {token_limit}.")
    return result


def extract_answer(response_content: str, cache: dict) -> int:
    """
    Extract the final integer answer from the response content.

    Args:
        response_content (str): The response content to extract the answer from.
        cache (dict): The cache to use for storing responses.

    Returns:
        int: The final integer answer.
    """
    cache_key = f"extract_answer_{response_content}"
    if cache_key in cache:
        return cache[cache_key]

    extraction_prompt = f"""
    Extract the final integer answer from the following problem solution. 
    Return only the integer, nothing else.

    Solution:
    {response_content}

    Final answer (integer only):
    """
    
    extraction_response = OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": extraction_prompt}]
    )
    
    extracted_answer = extraction_response.choices[0].message.content.strip()
    try:
        result = int(extracted_answer)
    except ValueError:
        result = None
    
    cache[cache_key] = result
    return result


def generate_single_response(example: dict, token_limit: int, cache: dict, idx: int) -> tuple[int, int]:
    """
    Get a single response for a problem.

    Args:
        example (dict): The problem to process.
        token_limit (int): The token limit for the model.
        cache (dict): The cache to use for storing responses.
        idx (int): The index of the response to process.

    Returns:
        tuple[int, int]: A tuple containing the answer and the number of tokens used.
    """
    response = get_response(example['problem'], token_limit, cache, idx=idx)
    answer = extract_answer(response['content'], cache)
    assert answer is not None, f"Answer is None for problem: {example['problem']}"
    return answer, response['tokens']


def process_single_example(example: dict, token_limit: int, cache: dict, N: int) -> tuple[bool, int]:
    """
    Process a single example by running the model N times and then taking the majority vote.

    Args:
        example (dict): The problem to process.
        token_limit (int): The token limit for the model.
        cache (dict): The cache to use for storing responses.
        N (int): The number of times to run the model.

    Returns:
        tuple[bool, int]: A tuple containing the majority vote result and the total number 
        of tokens used.
    """
    answers = []
    total_tokens = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(generate_single_response, example, token_limit, cache, idx) for idx in range(N)]

        for future in concurrent.futures.as_completed(futures):
            try:
                answer, tokens = future.result()
            except Exception as e:
                logging.exception(f"Error processing result: {e}.")
                answer, tokens = 0, 0

            answers.append(answer)
            total_tokens += tokens
    
    logging.debug(f"Obtained answers for problem starting with: {example['problem'][:20]}.\n"
                  f"Correct answer: {example['answer']}.\n"
                  f"Obtained answers: {sorted(answers)}.")

    # Compute majority vote
    majority_answer = statistics.mode(answers)
    is_correct = majority_answer == int(example['answer'])
    return is_correct, total_tokens


def run_experiments(dataset: list[dict], cache: dict[str, typing.Any], token_limit: int, N: int) -> tuple[float, float]:
    """
    Run experiments given the token limit and return results.

    Args:
        dataset (list[dict]): The dataset of problems to run the experiments on.
        cache (dict[str, typing.Any]): The cache to use for storing responses.
        token_limit (int): The token limit for the model.
        N (int): The number of times to run the model.

    Returns:
        tuple[float, float]: A tuple containing the accuracy and average tokens used.
    """
    correct_count = 0
    actual_tokens_used = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:

        futures = [executor.submit(process_single_example, example, token_limit, cache, N) for example in dataset]

        for future in concurrent.futures.as_completed(futures):
            is_correct, tokens = future.result()
            if is_correct:
                correct_count += 1
            actual_tokens_used.append(tokens)
        
        save_cache(cache, RESPONSE_CACHE_FILENAME)
    
    accuracy = correct_count / len(dataset)
    avg_tokens_used = np.mean(actual_tokens_used)
    logging.debug(f"Requested token limit: {token_limit}. Accuracy: {accuracy}. Average tokens used: {avg_tokens_used}.")
    return accuracy, avg_tokens_used


def run_majority_vote_inference_experiments(dataset: list[dict], cache: dict[str, typing.Any], shade_regions: bool = False) -> None:
    """
    Run experiments and create graphs that include majority vote extending past 2^14 tokens 
    for reasoning. We observe that models stop using more tokens even when asked to around 2^11.
    We solve this by doing repeated sampling and then taking the mode of the answers for all 
    queries above 2^11. This is not perfect, but still seems to help a bit.

    Args:
        dataset (list[dict]): The dataset of problems to run the experiments on.
        cache (dict[str, typing.Any]): The cache to use for storing responses.
        shade_regions (bool, optional): determines whether we include the plot with shaded 
        regions describing the different strategies. If False, it generates the headline 
        reconstruction plot of the o1 inference-time scaling laws.
    """
    logging.debug(f"Start running majority vote experiments.")

    if shade_regions:
        token_limits = [2**i for i in range(4, 19)]
    else:
        token_limits = [2**i for i in range(4, 15)]

    results = []

    for token_limit in tqdm(token_limits):
        actual_token_limit = min(2**11, token_limit)
        # We run the experiment N times for each token limit
        N = token_limit // actual_token_limit
        accuracy, avg_tokens_used = run_experiments(dataset, cache, actual_token_limit, N)
        result = {
            'token_limit': token_limit,
            'accuracy': accuracy,
            'avg_tokens_used': avg_tokens_used
        }
        results.append(result)

    plot_majority_vote_graph(results, shade_regions)


def run_just_ask_nicely_experiments(dataset: list[dict], cache: dict[str, typing.Any], run_full_range: bool = False) -> None:
    """
    Run experiments where we ask the model to use more tokens by asking it to use more tokens nicely.

    Args:
        dataset (list[dict]): The dataset of problems to run the experiments on.
        cache (dict[str, typing.Any]): The cache to use for storing responses.
    """
    logging.debug(f"Start running just ask nicely experiments.")
    token_limits = [2**i for i in range(4, 12)]
    if run_full_range:
        token_limits = [2**i for i in range(20)]
    results = []
    for token_limit in tqdm(token_limits):
        accuracy, avg_tokens_used = run_experiments(dataset, cache, token_limit, 1)
        result = {
            'token_limit': token_limit,
            'accuracy': accuracy,
            'avg_tokens_used': avg_tokens_used
        }
        results.append(result)
    plot_just_ask_nicely_graph(results, run_full_range)


dataset = load_2024_dataset()
cache = get_or_create_cache(RESPONSE_CACHE_FILENAME)
run_majority_vote_inference_experiments(dataset, cache)
run_just_ask_nicely_experiments(dataset, cache)
