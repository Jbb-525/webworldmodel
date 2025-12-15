import sys
import os

from verl.model_merger.fsdp_model_merger import FSDPModelMerger
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
class Config:
    def __init__(self, local_dir, target_dir):
        self.local_dir = local_dir  
        self.target_dir = target_dir 
        self.operation = "merge"  
        self.backend = "fsdp"
        self.hf_upload = False   
        self.test_hf_dir = None
        self.trust_remote_code = True
        self.hf_model_config_path = BASE_MODEL

CKPT_DIR = "/scratch/bj2414/WebAgent/model_saved/run_2/global_step_2200"


SAVE_DIR = os.path.join(CKPT_DIR, "merged_official")


def main():
    print(f"use verl official code to merge...")
    print(f"Input: {CKPT_DIR}")
    print(f"Output: {SAVE_DIR}")
    
    # Check if the path exists
    if not os.path.exists(CKPT_DIR):
        print(f"Error: Path not found {CKPT_DIR}")
        return

    # Initialize config
    config = Config(CKPT_DIR, SAVE_DIR)
    
    # Initialize merger
    merger = FSDPModelMerger(config)
    
    # Execute merge
    merger.merge_and_save()
    print("successful! Please test the model in the merged_official folder.")

if __name__ == "__main__":
    main()