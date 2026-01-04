import argparse
import torch
import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes", type=int)
    parser.add_argument("--device-number", type=int)
    args = parser.parse_args()
    nnodes = int(args.nnodes)
    device_number = int(args.device_number)

    seq_length = 2048
    global_batch_size = 128

    ## setup the dummy dataset
    data = llm.MockDataModule(seq_length=seq_length, global_batch_size=global_batch_size)

    ## initialize a small GPT model
    gpt_config = llm.GPTConfig(
        num_layers=6,
        hidden_size=384,
        ffn_hidden_size=1536,
        num_attention_heads=6,
        seq_length=seq_length,
        init_method_std=0.023,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
    )
    model = llm.GPTModel(gpt_config, tokenizer=data.tokenizer)

    ## initialize the strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
    )

    ## setup the optimizer
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=6e-4,
        bf16=True,
    )
    opt = nl.MegatronOptimizerModule(config=opt_config)

    trainer = nl.Trainer(
        devices=device_number, ## you can change the number of devices to suit your setup
        num_nodes=nnodes,
        max_steps=50,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
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
