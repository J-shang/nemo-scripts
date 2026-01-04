import argparse
import torch
import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.collections.common.tokenizers import AutoTokenizer
from megatron.core.optimizer import OptimizerConfig
from megatron.core.fusions.fused_bias_geglu import quick_gelu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes", type=int)
    parser.add_argument("--device-number", type=int)
    args = parser.parse_args()
    nnodes = int(args.nnodes)
    device_number = int(args.device_number)

    seq_length = 2048
    global_batch_size = 512

    ## setup the dataset
    data_root = "/data/nishang/data/Nemotron-CC-MFormat/quality=high/kind=actual/kind2=actual"
    shard_ids = ["CC-MAIN-2013-20", "CC-MAIN-2013-48", "CC-MAIN-2014-10", "CC-MAIN-2014-15", "CC-MAIN-2014-23", "CC-MAIN-2014-35", "CC-MAIN-2014-41", "CC-MAIN-2014-42", "CC-MAIN-2014-49", "CC-MAIN-2014-52", "CC-MAIN-2015-06", "CC-MAIN-2015-11", "CC-MAIN-2015-14", "CC-MAIN-2015-18", "CC-MAIN-2015-22", "CC-MAIN-2015-27", "CC-MAIN-2015-32", "CC-MAIN-2015-35", "CC-MAIN-2015-40", "CC-MAIN-2015-48", "CC-MAIN-2016-07", "CC-MAIN-2016-18", "CC-MAIN-2016-22", "CC-MAIN-2016-26", "CC-MAIN-2016-30", "CC-MAIN-2016-36", "CC-MAIN-2016-40", "CC-MAIN-2016-44", "CC-MAIN-2016-50", "CC-MAIN-2017-04", "CC-MAIN-2017-09", "CC-MAIN-2017-13", "CC-MAIN-2017-17", "CC-MAIN-2017-22", "CC-MAIN-2017-26", "CC-MAIN-2017-30", "CC-MAIN-2017-34", "CC-MAIN-2017-39", "CC-MAIN-2017-43", "CC-MAIN-2017-47", "CC-MAIN-2017-51", "CC-MAIN-2018-05", "CC-MAIN-2018-09", "CC-MAIN-2018-13", "CC-MAIN-2018-17", "CC-MAIN-2018-22", "CC-MAIN-2018-26", "CC-MAIN-2018-30", "CC-MAIN-2018-34", "CC-MAIN-2018-39", "CC-MAIN-2018-43", "CC-MAIN-2018-47", "CC-MAIN-2018-51", "CC-MAIN-2019-04", "CC-MAIN-2019-09", "CC-MAIN-2019-13", "CC-MAIN-2019-18", "CC-MAIN-2019-22", "CC-MAIN-2019-26", "CC-MAIN-2019-30", "CC-MAIN-2019-35", "CC-MAIN-2019-39", "CC-MAIN-2019-43", "CC-MAIN-2019-47", "CC-MAIN-2019-51", "CC-MAIN-2020-05", "CC-MAIN-2020-10", "CC-MAIN-2020-16", "CC-MAIN-2020-24", "CC-MAIN-2020-29", "CC-MAIN-2020-34", "CC-MAIN-2020-40", "CC-MAIN-2020-45", "CC-MAIN-2020-50", "CC-MAIN-2021-04", "CC-MAIN-2021-10", "CC-MAIN-2021-17", "CC-MAIN-2021-21", "CC-MAIN-2021-25", "CC-MAIN-2021-31", "CC-MAIN-2021-39", "CC-MAIN-2021-43", "CC-MAIN-2021-49", "CC-MAIN-2022-05", "CC-MAIN-2022-21", "CC-MAIN-2022-27", "CC-MAIN-2022-33", "CC-MAIN-2022-40", "CC-MAIN-2022-49", "CC-MAIN-2023-06", "CC-MAIN-2023-14", "CC-MAIN-2023-23", "CC-MAIN-2023-40", "CC-MAIN-2023-50", "CC-MAIN-2024-10", "CC-MAIN-2024-18", "CC-MAIN-2024-22", "CC-MAIN-2024-26", "CC-MAIN-2024-30"]
    train_paths = [f"{data_root}/{sid}_text_document" for sid in shard_ids[-3:]]
    tokenizer = AutoTokenizer("openai/gpt-oss-20b")

    data_config = {
        "paths": train_paths,
        "tokenizer": tokenizer,
        "seq_length": 2048,
        "micro_batch_size": 4,
        "global_batch_size": 128,
        "num_workers": 8,
        "reset_position_ids": False,
        "reset_attention_mask": False,
        "eod_mask_loss": False,
        "seed": 1234,
        "split": "900,50,50",
    }
    data = llm.PreTrainingDataModule(**data_config)

    ## initialize a small GPT-OSS model
    gpt_config = llm.GPTOSSConfig20B(
        moe_aux_loss_coeff=0.9,
        activation_func=quick_gelu,
        bf16=True,
        fp16=False,
        params_dtype=torch.bfloat16,
    )
    model = llm.GPTOSSModel(gpt_config, tokenizer=data.tokenizer)

    ## initialize the strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        expert_model_parallel_size=8,
    )

    ## setup the optimizer
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=6e-4,
        bf16=True,
        fp16=False,
        use_distributed_optimizer=True,
    )
    opt = nl.MegatronOptimizerModule(config=opt_config)

    trainer = nl.Trainer(
        devices=device_number, ## you can change the number of devices to suit your setup
        num_nodes=nnodes,
        max_steps=100,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
        ),
        limit_val_batches=0,
    )

    nemo_logger = nl.NeMoLogger(
        log_dir="/root/nemo_test_logdir", ## logs and checkpoints will be written here
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer='data',
        optim=opt,
    )
