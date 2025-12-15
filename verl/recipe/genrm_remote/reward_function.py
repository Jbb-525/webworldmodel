# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import requests
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

# ================= Configs =================
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.openai.com/v1" 
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_WORKERS = 32
MODEL_NAME = "gpt-4o-mini"

# ================= Prompt =================

GENRM_PROMPT_TEMPLATE = """You are evaluating a model's world modeling capability for web interactions.

PREDICTED NEXT STATE:
{predicted_next_state}

ACTUAL NEXT STATE:
{actual_next_state}

Evaluate the prediction accuracy based on:
1. Completeness (40%): Did it identify all significant changes in the accessibility tree?
2. Accuracy (40%): Are the predicted changes factually correct?
3. Causal Reasoning (20%): Does it understand why the action causes these changes?

Scoring Guidelines:
- 0.9-1.0: Excellent - captures all changes accurately with clear reasoning
- 0.7-0.89: Good - most changes captured correctly, minor gaps
- 0.5-0.69: Adequate - key changes identified but notable errors
- 0.3-0.49: Poor - significant missing or incorrect predictions
- 0.0-0.29: Very Poor - fails to model state changes correctly

Please put your final score(number only) in \\boxed{{}}.
""".strip()


def get_response(solution_str, ground_truth):
    prompt = GENRM_PROMPT_TEMPLATE.format(
        predicted_next_state=solution_str, 
        actual_next_state=ground_truth
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    # 构建 Header (必须包含 Authorization)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # URL 拼接
    chat_url = f"{BASE_URL}/chat/completions"
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(chat_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result_json = response.json()
            content = result_json["choices"][0]["message"]["content"]
            return content
            
        except Exception as e:
            print(f"[RewardFn] Request failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                sleep(BASE_DELAY * (2 ** attempt))
            else:
                print(f"[RewardFn] All retries failed for prompt start: {prompt[:100]}...")
                
    return None


def compute_reward(response_str):

    reward_score = 0.0
    try:
        boxed_result = last_boxed_only_string(response_str)
        if boxed_result is not None:
            result_text = remove_boxed(boxed_result)
            reward_score = float(result_text)
        else:
            print(f"[RewardFn] No boxed score found. Response: {response_str[:100]}...")
            
    except Exception as e:
        print(f"[RewardFn] Reward parsing failed. Error: {e}. Response: {response_str[:100]}...")
        reward_score = 0.3
        
    return reward_score


def compute_score(data_source, solution_str, ground_truth, extra_info):

    split = extra_info.get("split", "train")
    
    if split == "test":
        from verl.utils.reward_score import default_compute_score
        return default_compute_score(data_source, solution_str, ground_truth, extra_info)

    judge_response = get_response(solution_str, ground_truth)
    
    if judge_response is not None:
        reward = compute_reward(judge_response)
    else:
        reward = 0.0 

    return reward


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for src, sol, gt, info in zip(data_sources, solution_strs, ground_truths, extra_infos):

            future = executor.submit(compute_score, src, sol, gt, info)
            futures.append(future)

        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                print(f"[RewardFn] Batch execution error: {e}")
                results.append(0.0)

    return results