#!/bin/bash

# 设置国内镜像
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0

# 模型下载
## HuggingFace 模型下载
hf download Qwen/Qwen2.5-0.5B  --local-dir ~/hfhub/models/Qwen/Qwen2.5-0.5B
hf download Qwen/Qwen3-0.6B --local-dir ~/hfhub/models/Qwen/Qwen3-0.6B
hf download facebook/opt-125m  --local-dir ~/hfhub/models/facebook/opt-125m

# 数据集下载
# hf download --repo-type dataset openai/gsm8k --local-dir ~/hfhub/datasets/openai/gsm8k
# hf download --repo-type dataset tatsu-lab/alpaca --local-dir ~/hfhub/datasets/tatsu-lab/alpaca
