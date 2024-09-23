import os
import json
import matplotlib.pyplot as plt
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from IPython import embed
import numpy as np
import concurrent.futures
import statistics
from PIL import Image

model = "o1-mini"
SHADE_REGIONS = False
dataset = load_dataset("AI-MO/aimo-validation-aime")
dataset = dataset["train"].filter(lambda example: "2024" in example["url"])

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

assert len(dataset) == 30, f"Expected 30 problems, but found {len(dataset)}"

PROMPT = """You are a math problem solver. I will give you a problem from the American Invitational Mathematics Examination (AIME). At the end, provide the final answer as a single integer.

Important: You should try your best to use around {token_limit} tokens in your reasoning steps.
If you feel like you are finished early, spend the extra tokens trying to double check your work until you are absolutely sure that you have the correct answer.
Here's the problem:

{problem}

Solve this problem, use around {token_limit} tokens in your reasoning, and provide the final answer as a single integer.
"""

def get_or_create_cache(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache, filename):
    with open(filename, 'w') as f:
        json.dump(cache, f)

def get_response(problem, token_limit, cache, idx=0):

    if idx > 0:
        cache_key = f"{model}_{PROMPT}_{problem}_{token_limit}_{idx}"
    else:
        cache_key = f"{model}_{PROMPT}_{problem}_{token_limit}"
    if cache_key in cache:
        return cache[cache_key]
    
    formatted_prompt = PROMPT.format(problem=problem, token_limit=token_limit)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": formatted_prompt}]
    )
    result = {
        'content': response.choices[0].message.content,
        'tokens': response.usage.completion_tokens
    }
    cache[cache_key] = result
    return result

cache = get_or_create_cache('response_cache.json')

def extract_answer(response_content, cache):
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
    
    extraction_response = client.chat.completions.create(
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

def process_single_response(example, token_limit, cache, idx):
    response = get_response(example['problem'], token_limit, cache, idx=idx)
    answer = extract_answer(response['content'], cache)
    assert answer is not None, f"Answer is None for problem: {example['problem']}"
    return answer, response['tokens']

def process_example(example, token_limit, cache, N):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_single_response, example, token_limit, cache, idx) for idx in range(N)]
        
        answers = []
        total_tokens = 0
        
        for future in concurrent.futures.as_completed(futures):
            try:
                answer, tokens = future.result()
            except Exception as e:
                print(f"Error processing result: {e}")
                answer, tokens = 0, 0

            answers.append(answer)
            total_tokens += tokens

    majority_answer = statistics.mode(answers)
    is_correct = majority_answer == int(example['answer'])
    return is_correct, total_tokens

# Graphs that include majority vote extending past 2^14 tokens for reasoning
def majority_vote_graphs():

    # SHADE_REGIONS determines whether we include the plot with shaded regions describing the different strategies
    # If False, it generates the headline reconstruction plot of the o1 inference-time scaling laws
    if SHADE_REGIONS:
        token_limits = [2**i for i in range(4, 19)]
    else:
        token_limits = [2**i for i in range(4, 15)]

    results = []
    for token_limit in tqdm(token_limits):
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:

            # We observe that models stop using more tokens even when asked to around 2^11
            # We solve this by doing repeated sampling and then taking the mode of the answers
            # for all queries above 2^11. This is not perfect, but still seems to help a bit.
            token_limit_cache = min(2**11, token_limit)
            N = token_limit // token_limit_cache

            futures = [executor.submit(process_example, example, token_limit_cache, cache, N) for example in dataset]
            
            correct_count = 0
            actual_tokens = []
            
            for future in concurrent.futures.as_completed(futures):
                is_correct, tokens = future.result()
                if is_correct:
                    correct_count += 1
                actual_tokens.append(tokens)
            
            save_cache(cache, 'response_cache.json')
        
        accuracy = correct_count / len(dataset)
        avg_tokens = np.mean(actual_tokens)
        results.append({
            'token_limit': token_limit,
            'accuracy': accuracy,
            'avg_tokens': avg_tokens
        })

    # Create the plot for the right subfigure
    plt.figure(figsize=(5, 6))
    plt.scatter([r['avg_tokens'] for r in results], [100*r['accuracy'] for r in results], marker='o')
    plt.xscale('log', base=2)
    plt.xlabel('tokens used at test-time (log scale)', fontsize=13)
    plt.ylabel('pass@1 accuracy', fontsize=13)
    plt.ylim(0, 100)
    plt.title('o1 mini AIME accuracy\nat test time (reconstructed)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)

    if SHADE_REGIONS:
        plt.axvline(x=2**14, color='black', linestyle='--')

        plt.axvspan(min([r['avg_tokens'] for r in results]) // 2, 2**14, facecolor='lightgreen', alpha=0.3)
        plt.axvspan(2**14, 2**17, facecolor='lightblue', alpha=0.3)

        plt.text(2**11, 85, "just ask o1-mini \nto think longer", fontsize=12, ha='center', va='center', color='green')
        plt.text(2**15*1.4, 85, 'majority\nvote', fontsize=12, ha='center', va='center', color='blue')

        plt.axvline(x=2**17, color='black', linestyle='--')
        plt.text(2**19, 85, 'no gains', fontsize=12, ha='center', va='center', color='red')
        plt.axvspan(2**17, 2**21, facecolor='lightcoral', alpha=0.3)


    plt.tight_layout()
    plt.savefig('accuracy_vs_tokens_{}.png'.format("shade_regions" if SHADE_REGIONS else "no_shade_regions"), dpi=300, facecolor='white', edgecolor='none')
    plt.close()

    print("Plot saved as accuracy_vs_tokens.png")

    plt.figure(figsize=(11, 10))
    plt.scatter([r['token_limit'] for r in results], [r['avg_tokens'] for r in results], marker='o')
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('Token Limit')
    plt.ylabel('Actual Tokens Used')
    plt.title('Token Limit vs. Actual Tokens Used')
    plt.tight_layout()
    plt.savefig('token_limit_vs_actual.png')

    with open('results_log.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("plots saved to accuracy_vs_tokens.png and token_limit_vs_actual.png")

# Try to get the model to use more tokens by asking it to use more tokens
def just_ask_nicely():
    # token_limits = [2**i for i in range(20)]
    token_limits = [2**i for i in range(4, 12)]
    results = []
    for token_limit in tqdm(token_limits):
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:

            futures = [executor.submit(process_example, example, token_limit, cache, 1) for example in dataset]
            
            correct_count = 0
            actual_tokens = []
            
            for future in concurrent.futures.as_completed(futures):
                is_correct, tokens = future.result()
                if is_correct:
                    correct_count += 1
                actual_tokens.append(tokens)
            
            save_cache(cache, 'response_cache.json')
        
        accuracy = correct_count / len(dataset)
        avg_tokens = np.mean(actual_tokens)
        results.append({
            'token_limit': token_limit,
            'accuracy': accuracy,
            'avg_tokens': avg_tokens
        })

    plt.figure(figsize=(6, 6))
    plt.scatter([r['token_limit'] for r in results], [r['avg_tokens'] for r in results], marker='o')
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('Token Limit')
    plt.ylabel('Actual Tokens Used')
    plt.title('Token Limit vs. Actual Tokens Used')
    plt.tight_layout()
    plt.savefig('just_ask_nicely_tokens.png')

    with open('results_log.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("plots saved to just_ask_nicely_tokens.png")

majority_vote_graphs()
just_ask_nicely()