## Installation

```bash
# Python 3.10 or 3.11 recommended
conda create -n myenv python=3.10
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

All data has been uploaded to [data](./data) folder

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
python scripts/generate_test_data.py
```
You will see `*.json` files generated in the [config_files](webarena/config_files) folder. Each file contains the configuration for one test example.

4. Obtain and save the auto-login cookies for all websites:
```
bash prepare.sh
```

5. Set up API keys.
If using OpenAI models, set a valid OpenAI API key (starting with `sk-`) as the environment variable:
```
export OPENAI_API_KEY=your_key
```

6. create .env files for environment variables
Here is an example of the `.env` file:

```
DATASET=webarena
SHOPPING="http://ec2-3-140-250-97.us-east-2.compute.amazonaws.com:7770"
SHOPPING_ADMIN="http://ec2-3-140-250-97.us-east-2.compute.amazonaws.com:7780/admin"
REDDIT="http://ec2-3-140-250-97.us-east-2.compute.amazonaws.com:9999"
```

7. Launch the evaluation. You can run the script provided:

Check the `run_webarena.sh` script and set up environment variables and vllm as instructed.

Then, run:

```bash
bash scripts/run_webarena.sh
```

