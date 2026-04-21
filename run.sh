#!/bin/bash

# ============================================================
# DuQuant++ : MXFP4 W4A4 Quantization
# ============================================================
# Supported models: LLaMA-3-8B, LLaMA-3.2-3B, LLaMA-3-8B-Instruct, LLaMA-3.1-8B-Instruct
# QA tasks (Table 1): arc_easy, arc_challenge, winogrande, hellaswag, openbookqa, lambada_openai, piqa

# --- DuQuant++ (without GPTQ) ---
python main.py \
    --block_size 32 \
    --max_rotation_step 128 \
    --wbits 4 \
    --abits 4 \
    --model meta-llama/Llama-3-8B \
    --alpha 0.6 \
    --smooth \
    --eval_ppl \
    --batch_size 32 \
    --tasks arc_easy,arc_challenge,winogrande,hellaswag,openbookqa,lambada_openai,piqa


# --- DuQuant++* (with GPTQ) ---
# python main.py \
#     --block_size 32 \
#     --max_rotation_step 128 \
#     --wbits 4 \
#     --abits 4 \
#     --model meta-llama/Llama-3-8B \
#     --alpha 0.6 \
#     --gptq \
#     --smooth \
#     --eval_ppl \
#     --batch_size 32 \
#     --tasks arc_easy,arc_challenge,winogrande,hellaswag,openbookqa,lambada_openai,piqa
