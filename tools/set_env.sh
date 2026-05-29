#!/bin/bash

source ~/.bashrc

# 加载 CANN 环境变量（路径需根据实际安装位置调整）
# install_path=/usr/local/Ascend
install_path=/home/jianzhnie/llmtuner/Ascend/CANN8.3.RC1
source $install_path/ascend-toolkit/set_env.sh
source $install_path/nnal/atb/set_env.sh

# 关闭有问题的 NPU 专用 kernel
export MINIVLLM_USE_NPU_ROPE=0
export MINIVLLM_USE_NPU_FA=1

# 常规 NPU 环境
export ASCEND_RT_VISIBLE_DEVICES=0,1

# 模型
export MINIVLLM_MODEL="/Users/robin/hfhub/models/Qwen/Qwen3-0.6B"

# conda activate vllm091
source /home/jianzhnie/llmtuner/software/miniconda3/bin/activate vllm091
