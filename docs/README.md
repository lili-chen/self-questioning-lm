<h1 style="text-align: center;">Self-Questioning Language Models</h1>

Self-Questioning Language Models (SQLM): an asymmetric self-play framework where a proposer is given the topic and generates a question for a solver, who tries to answer it.

SQLM is built on top of the **[verl](https://github.com/volcengine/verl)** library.

## Installation:

Please refer to the existing verl quickstart for installation : [verl Installation](https://verl.readthedocs.io/en/latest/start/install.html)

## Example: Run SQLM on Qwen2.5-3B-Instruct on the Arithmetic task

**Prepare data:**
All the train and test datasets can be found in the `selfplay_data` folder. The train data for self-play is simply the prompt duplicated many times, and you can generate it with the following command:

```
python ./examples/data_preprocess/selfplay.py --prompt_version arithmetic_v1
```

**Run Training:**

*Adjust the configuration in `ppo_trainer.yaml` to match your desired training configuration (number of gpus, batch size, etc.). To override this config somewhere else, see ["Creating Custom Configurations"](#Creating-Custom-Configurations)*

```
python -m verl.trainer.main_ppo exps="[grpo,multiply_selfplay,smallbs,majority]" trainer.experiment_name=debug
```


## Creating Custom Configurations

We use an extensible config setup, allowing you to override default configurations for specific tasks/jobs.

To define a custom configuration, create a new yaml file in `verl/trainer/config/exps`. **NOTE**: you MUST include `# @package _global_` at the beginning of the file in order to override other configs.

To use different configuration files, simply add them to the `exps="[...]"` argument to `verl.trainer.main_ppo`. Note: configurations are applied from left-to-right order, so configs to the right will override configs to the left!