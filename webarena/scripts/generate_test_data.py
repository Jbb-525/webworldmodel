"""Replace the website placeholders with website domains from env_config
Generate the test data"""
import json
import os
import random

from browser_env.env_config import *


def main() -> None:
    DATASET = os.environ["DATASET"]
    if DATASET == "webarena":
        print("DATASET: webarena")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        
        inp_paths = ["config_files/wa/test_webarena.raw.json"]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
        }
    else:
        raise ValueError(f"Dataset not implemented: {DATASET}")
        
    for inp_path in inp_paths:
        output_dir = inp_path.replace('.raw.json', '')
        os.makedirs(output_dir, exist_ok=True)
        
        with open(inp_path, "r") as f:
            raw = f.read()
        
        for k, v in replace_map.items():
            raw = raw.replace(k, v)

        # with open(inp_path.replace(".raw", ""), "w") as f:
        #     f.write(raw)
        
        data = json.loads(raw)
        

        print(f"\n{'='*60}")
        print(f"Processing: {inp_path}")
        print(f"Total tasks before filtering: {len(data)}")
        

        shopping_tasks = [
            item for item in data 
            if "shopping" in item.get("sites", [])
        ]
        reddit_tasks = [
            item for item in data 
            if "reddit" in item.get("sites", [])
        ]
        
        print(f"Shopping tasks found: {len(shopping_tasks)}")
        print(f"Reddit tasks found: {len(reddit_tasks)}")
        
        
        if len(shopping_tasks) >= 50:
            selected_shopping = random.sample(shopping_tasks, 50)
            print(f"Sampled 50 shopping tasks")
        else:
            selected_shopping = shopping_tasks
            print(f"Using all {len(shopping_tasks)} shopping tasks (less than 50)")
        
        if len(reddit_tasks) >= 50:
            selected_reddit = random.sample(reddit_tasks, 50)
            print(f"Sampled 50 reddit tasks")
        else:
            selected_reddit = reddit_tasks
            print(f"Using all {len(reddit_tasks)} reddit tasks (less than 50)")
        
        selected_data = selected_shopping + selected_reddit
        
        print(f"\nFinal selection:")
        print(f"  - Shopping: {len(selected_shopping)} tasks")
        print(f"  - Reddit: {len(selected_reddit)} tasks")
        print(f"  - Total: {len(selected_data)} tasks")
        print(f"{'='*60}\n")

        for idx, item in enumerate(selected_data):
            with open(os.path.join(output_dir, f"{idx}.json"), "w") as f:
                json.dump(item, f, indent=2)
        
        print(f"✅ Saved {len(selected_data)} tasks to {output_dir}/\n")


if __name__ == "__main__":
    main()