import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.duquant import duquant
from tqdm import tqdm
import utils
from pathlib import Path
import transformers
from quantize.gptq import gptq

torch.backends.cudnn.benchmark = True

net_choices = [
    "llama-7b", "llama-13b", "llama-30b", "llama-65b", "Llama-2-7b",
    "Llama-2-13b", "Llama-2-70b", "Llama-3-8b", "Llama-3-70b", "Vicuna-1.5-7b",
    "Vicuna-1.5-13b",
]


@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}
    if args.multigpu:
        map_layers_to_multi_gpus(lm.model.model.layers)
        input_device = lm.model.model.layers[0].device
        output_device = lm.model.model.layers[-1].device
        assert input_device == output_device
        lm._device = input_device
        lm.model.model.embed_tokens.to(input_device)
        lm.model.model.norm.to(output_device)
        lm.model.lm_head.to(output_device)
    else:
        lm.model = lm.model.to(lm.device)

    if args.eval_ppl:
        for dataset in ["wikitext2", "c4-new"]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader, weights_only=False)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen):((i + 1) * lm.seqlen)].to(
                    lm.device)
                outputs = lm.model.model(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (
                    i * lm.seqlen):((i + 1) * lm.seqlen)][:, 1:].to(
                        lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()

    if args.tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
        hflm = HFLM(pretrained=lm.model, tokenizer=tokenizer, batch_size=args.batch_size)

        task_manager = lm_eval.tasks.TaskManager()

        task_names = args.tasks.split(',')

        QAresults = {}
        for task_name in task_names:
            logger.info(f"Evaluating {task_name}...")
            result = lm_eval.simple_evaluate(hflm, tasks=[task_name], batch_size=args.batch_size, task_manager=task_manager)['results']
            result = result[task_name]
            acc = round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
            QAresults[task_name] = acc
            logger.info(f"acc: {acc}%")
        metric_vals = {task: result for task, result in QAresults.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 2)
        logger.info(metric_vals)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir",
                        default="./cache",
                        type=str,
                        help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir",
                        default="./log/",
                        type=str,
                        help="direction of logging file")
    parser.add_argument("--save_dir",
                        default=None,
                        type=str,
                        help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix", "pile"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples",
                        type=int,
                        default=128,
                        help="Number of calibration data samples.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="batch size.")
    parser.add_argument("--seed",
                        type=int,
                        default=2,
                        help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_alpha", type=float, default=0.8)
    parser.add_argument("--act_group_size", type=int, default=None)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--smooth_lr", type=float, default=1e-4)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--smooth_epochs", type=int, default=0)
    parser.add_argument("--smooth", default=False, action="store_true")
    parser.add_argument("--let",
                        default=False,
                        action="store_true",
                        help="activate learnable equivalent transformation")
    parser.add_argument("--lwc",
                        default=False,
                        action="store_true",
                        help="activate learnable weight clipping")
    parser.add_argument("--aug_loss",
                        default=False,
                        action="store_true",
                        help="calculate additional loss with same input")
    parser.add_argument("--symmetric",
                        default=False,
                        action="store_true",
                        help="symmetric quantization")
    parser.add_argument("--a_dynamic_method",
                        type=str,
                        default="per_token",
                        choices=["per_token"])
    parser.add_argument("--w_dynamic_method",
                        type=str,
                        default="per_channel",
                        choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu",
                        action="store_true",
                        help="at eval, map model to multiple gpus")
    parser.add_argument("--deactive_amp",
                        action="store_true",
                        help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        required=False,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)
    parser.add_argument("--gptq", default=False, action="store_true", help="use gptq for further compensation")

    # DuQuant
    parser.add_argument("--max_rotation_step",
                        type=int,
                        default=256,
                        help="max steps for rotation transformation")
    parser.add_argument("--permutation_times",
                        type=int,
                        default=0,
                        help="times of permutation transformation")
    parser.add_argument("--lac",
                        type=float,
                        default=None,
                        help="activation clipping ratio")
    parser.add_argument("--swc",
                        type=float,
                        default=None,
                        help="weight clipping ratio, enable without lwc")
    parser.add_argument("--block_size",
                        type=int,
                        default=128,
                        help="block size for rotation matrices")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.epochs > 0:
        assert args.lwc or args.let

    if (args.wbits < 16 and args.wbits >= 8) or (args.abits < 16
                                                 and args.abits >= 8):
        args.deactive_amp = True

    args.quant_method = "duquant"

    # init logger
    args.output_dir = os.path.join(
        args.output_dir,
        f"{args.model.split('/')[-1]}_w{args.wbits}a{args.abits}")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)

    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    args.model_family = args.net.split('-')[0]
    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc": args.lwc,
        "swc": args.swc,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "lac": args.lac,
        "act_group_size": args.act_group_size,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "lac": args.lac,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "lac": args.lac,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
    }
    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'

    # quantization
    if args.wbits < 16 or args.abits < 16:
        logger.info("=== start quantization ===")
        tick = time.time()
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)
        act_scales = None
        act_shifts = None
        if args.smooth:
            act_scales = torch.load(args.act_scales)
            act_shifts = torch.load(args.act_shifts)
        duquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(time.time() - tick)
        if args.gptq:
            tick = time.time()
            with torch.no_grad():
                gptq(lm, args, dataloader, logger)
            logger.info(time.time() - tick)
    evaluate(lm, args, logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
