torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=$NODE_RANK \
  --master_addr=10.45.175.69 \
  --master_port=12345 \
  mock_train.py \
  --nnodes=2 \
  --device-number=8 2>&1 >log.txt

torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=10.45.175.69 \
  --master_port=12345 \
  pretrain.py \
  --nnodes=2 \
  --device-number=8 2>&1 >log.txt

torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=10.45.175.69 \
  --master_port=12345 \
  pretrain.py \
  --nnodes=2 \
  --device-number=8 2>&1 >log.txt
