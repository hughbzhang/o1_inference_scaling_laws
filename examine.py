import json
import random
from IPython import embed

with open('response_cache.json', 'r') as file:
    data = json.load(file)

# Filter out samples that don't have 2^11 in the key name
token_limit = str(2**11)

filtered_data = {
    k: v for k, v in data.items() if token_limit in k
}

print(random.choice(list(filtered_data.values()))['content'])