# DuQuant++: Fine-grained Rotation Enhances Microscaling FP4 Quantization

<h5 align="center">

[![arXiv](https://img.shields.io/badge/DuQuant++-2604.17789-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2604.17789)
[![License](https://img.shields.io/badge/⚖️%20Code%20License-MIT-yellow)](LICENSE)
 <br>

</h5>

Welcome to the official code repository for "**[DuQuant++: Fine-grained Rotation Enhances Microscaling FP4 Quantization](https://arxiv.org/abs/2604.17789)**".

## 📰 News
* [2026/04/21] 🚀 Our DuQuant++ code is released!
* [2026/04/21] 🚀 Our DuQuant++ paper is available on arXiv!
* [2024/09/26] 🌟 Our [DuQuant](https://arxiv.org/abs/2406.01721) paper has been accepted for an Oral presentation at NeurIPS 2024!


## 👀 Introduction

DuQuant++ extends DuQuant to the **MXFP4 (Microscaling FP4)** quantization format, achieving state-of-the-art W4A4 quantization performance for LLMs with fine-grained rotation transformations.

Key features:
- **MXFP4 W4A4 quantization** with block_size=32 aligned to MXFP4 group size
- **Fine-grained rotation transformation** for outlier distribution
- **Optional GPTQ compensation** for further accuracy improvement
- Support for **LLaMA-3** model families


## 🔧 Installation
```bash
conda create -n duquant python=3.10 -y
conda activate duquant
pip install --upgrade pip 
pip install -r requirements.txt
```

## ⚙️ Usage
### 1. Preprocessing
```bash
# Generate rotation matrices (run once for all models)
python get_rot.py

# Generate activation scales and shifts (run once per model)
python generate_act_scale_shift.py --model meta-llama/Llama-3-8B
```

### 2. Quantization & Evaluation
The bash script for DuQuant++ can be found in `run.sh`.

```bash
# DuQuant++ (without GPTQ)
python main.py \
    --block_size 32 \
    --max_rotation_step 256 \
    --wbits 4 \
    --abits 4 \
    --model meta-llama/Llama-3-8B \
    --alpha 0.6 \
    --smooth \
    --eval_ppl \
    --bath_size 64 \
    --tasks arc_easy,arc_challenge,winogrande,hellaswag,openbookqa,lambada_openai,piqa

# DuQuant++* (with GPTQ)
python main.py \
    --block_size 32 \
    --max_rotation_step 256 \
    --wbits 4 \
    --abits 4 \
    --model meta-llama/Llama-3-8B \
    --alpha 0.6 \
    --gptq \
    --smooth \
    --eval_ppl \
    --bath_size 64 \
    --tasks arc_easy,arc_challenge,winogrande,hellaswag,openbookqa,lambada_openai,piqa
```

#### Explanation of arguments:
- `--model`: the local model path or HuggingFace model name.
- `--wbits`: weight quantization bits.
- `--abits`: activation quantization bits.
- `--block_size`: the block size of rotation matrices (32 for MXFP4).
- `--max_rotation_step`: the max greedy search steps of rotation transformation.
- `--gptq`: enable GPTQ for weight error compensation.
- `--resume`: loading pre-trained DuQuant parameters.
- `--multigpu`: to inference larger network on multiple GPUs.
- `--save_dir`: saving the quantization model for further exploration.
- `--eval_ppl`: evaluating the perplexity of quantized models.
- `--tasks`: evaluating on zero-shot QA tasks (comma-separated).

### 3. Model Zoo

Currently, we support the following model families:

| Models      | Supported |
| ----------- | --------- |
| LLaMA-3     | ✅         |
| LLaMA-3.1   | ✅         |
| LLaMA-3.2   | ✅         |


## 📂 Contact
For immediate queries or further information, please open an issue or contact <haokunlin2-c@my.cityu.edu.hk>.

## 🙏 Acknowledgement
This repo is built upon the following projects:

* [DuQuant](https://github.com/Hsu1023/DuQuant)
* [OmniQuant](https://github.com/OpenGVLab/OmniQuant)

We thank the authors for their code.

## 📝 Citation
We kindly request that you cite our work if you utilize the code or reference our findings in your research:
```
@article{lin2026duquant++,
  title={DuQuant++: Fine-grained Rotation Enhances Microscaling FP4 Quantization},
  author={Lin, Haokun and Jia, Xinle and Xu, Haobo and Yao, Bingchen and Guo, Xianglong and Wu, Yichen and Lu, Zhichao and Wei, Ying and Zhang, Qingfu and Sun, Zhenan},
  journal={arXiv preprint arXiv:2604.17789},
  year={2026}
}

@article{lin2024duquant,
  title={DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs},
  author={Lin, Haokun and Xu, Haobo and Wu, Yichen and Cui, Jingzhi and Zhang, Yingtao and Mou, Linzhan and Song, Linqi and Sun, Zhenan and Wei, Ying},
  journal={arXiv preprint arXiv:2406.01721},
  year={2024}
}
```
