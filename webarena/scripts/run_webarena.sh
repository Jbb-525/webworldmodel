#!/bin/bash
export DATASET=webarena

export HOST_NAME="<your_host_name>" 
# export OPENAI_API_KEY="<your_key>"
export OPENAI_API_BASE="https://api.openai.com/v1"

echo "OPENAI_API_KEY: ${OPENAI_API_KEY:0:20}..."

export SHOPPING="http://${HOST_NAME}:7770"
export SHOPPING_ADMIN="http://${HOST_NAME}:7780/admin"
export REDDIT="http://${HOST_NAME}:9999"
export GITLAB="http://${HOST_NAME}:8023"

# change <your-world_model_name> to the real world model name.
# You can get it by running `curl http://<your-server-ip>:8000/v1/models`
# To do this, you should start the vllm service first by running
# python -m vllm.entrypoints.openai.api_server --model binnnnnid/Webworldmodel --served-model-name binnnnnid/Webworldmodel --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --trust-remote-code --dtype auto --max-model-len 8192


world_model_name="binnnnnid/Webworldmodel"
world_model_url="http://localhost:8000/v1"
# world_model_url="http://<your-server-ip>:8000/v1"

# value_model_name=""
# value_model_url=""

model="gpt-4o-mini"
value_function="gpt-4o-mini"


# Policy LLM
policy_temperature=0.7
policy_top_p=1.0

# World Model
world_model_temperature=0
world_model_top_p=0.9

# Value Function LLM
value_temperature=0.3
value_top_p=0.95

max_depth=2  # max_depth=4 means 5 step lookahead
max_steps=5
branching_factor=5
vf_budget=20
agent="world_model"
world_model_training=True
value_model_training=False
my_world_model=True
next_state_format="description_with_tao"
result_dir="log"
instruction_path="agent/prompts/jsons/p_cot_id_actree_2s_no_na.json"
state_prediction_prompt_path="agent/prompts/jsons/state_prediction/sft_world_model_prompt.json"
value_function_prompt_path="agent/prompts/jsons/value_function/text_only_value_function_likert.json"

mkdir "${result_dir}"
mkdir "${result_dir}/logs"

### Code to run the experiments
function run_job() {
    local start_idx=$1
    local end_idx=$2
    local job_num=$3

    if [ -f logs/wma_${next_state_format}_format_job_${job_num}.log ]; then
        echo "----------------------------------------" >> logs/wma_${next_state_format}_format_job_${job_num}.log
        echo "New log entry started at $(date)" >> logs/wma_${next_state_format}_format_job_${job_num}.log
        echo "----------------------------------------" >> logs/wma_${next_state_format}_format_job_${job_num}.log
    else
        touch ${result_dir}/logs/wma_${next_state_format}_format_job_${job_num}.log
    fi


    nohup env \
        OPENAI_API_KEY="$OPENAI_API_KEY" \
        OPENAI_API_BASE="$OPENAI_API_BASE" \
        python run_w_world_model.py \
    --instruction_path $instruction_path \
    --test_start_idx $start_idx \
    --test_end_idx $end_idx \
    --model $model \
    --agent_type $agent   --max_depth $max_depth  --branching_factor $branching_factor  --vf_budget $vf_budget   \
    --result_dir $result_dir \
    --test_config_base_dir=config_files/wa/test_webarena \
    --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
    --top_p 0.95   --temperature 1.0  --max_steps $max_steps --value_function $value_function\
    --state_prediction_prompt_path $state_prediction_prompt_path --value_function_prompt_path $value_function_prompt_path --total_indices $total_indices\
    --next_state_format $next_state_format \
    --policy_temperature $policy_temperature --policy_top_p $policy_top_p \
    --world_model_temperature $world_model_temperature --world_model_top_p $world_model_top_p \
    --value_temperature $value_temperature --value_top_p $value_top_p \
    $( [ "$world_model_training" = True ] && echo "--world_model_training" ) \
    $( [ "$world_model_training" = True ] && echo "--world_model_name $world_model_name" ) \
    $( [ "$world_model_training" = True ] && echo "--world_model_url $world_model_url" ) \
    $( [ "$value_model_training" = True ] && echo "--value_model_training" ) \
    $( [ "$value_model_training" = True ] && echo "--value_model_name $value_model_name" ) \
    $( [ "$value_model_training" = True ] && echo "--value_model_url $value_model_url" ) \
    >> ${result_dir}/logs/wma_${next_state_format}_format_job_${job_num}.log 2>&1 &
}

total_indices=100
indices_per_thread=1
batch_size=1
batch_end=0
current_start=0
job_count=0

echo "start evaluation..."

while [ "$current_start" -lt "$total_indices" ]; do
  batch_end=$((batch_end + batch_size))
  if [ "$batch_end" -gt "$total_indices" ]; then
      batch_end=$total_indices
  fi
  echo "a new round has been started...\n"
  while [ "$current_start" -lt "$batch_end" ]; do
      current_end=$((current_start + indices_per_thread))
      if [ "$current_end" -gt "$total_indices" ]; then
          current_end=$total_indices
      fi

      # Run the job
      run_job $current_start $current_end $job_count

      ((job_count++))

      # Increment start index for next job
      current_start=$current_end
  done
  ### Wait for all jobs to complete
  wait
done
