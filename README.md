## Installation

```bash
# Python 3.10 or 3.11 recommended
conda create myenv python=3.10
cd verl
pip install -e .
```

```bash
# Python 3.10 or 3.11 recommended
conda create -n webagent python=3.10
cd webarena
pip install vllm
pip install -r requirements.txt
playwright install
pip install -e .
```

## Data Preparation

All data has been uploaded to data folder

## Supervised Fine-Tuning

```bash
bash verl/examples/sft/web_agent/ run_web_sft.sh 2
```

To merge the LoRA weights into the base model, run the following command:

```bash
python model_saved/merge.py
```
## End-to-end Evaluation on Webarena
1. Setup the WA environments.
Please check out [this page](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md) for details. We recommend using the AWS for enviroment setup.

2. Configurate the urls for each website.
First, export the `DATASET` to be `webarena`:
```bash
export DATASET=webarena
```
Then, set the URL for the websites

Please change <your-server-hostname> to the real host name of your AWS machine.

```bash
export SHOPPING="http://<your-server-hostname>:7770"
export SHOPPING_ADMIN="http://<your-server-hostname>:7780/admin"
export REDDIT="http://<your-server-hostname>:9999"
```
3. Generate config files for each test example:
```bash
python webarena/scripts/generate_test_data.py
```
4. Launch the evaluation.

```bash
bash webarena/scripts/run_webarena.sh
```

