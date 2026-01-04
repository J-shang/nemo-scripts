#!/bin/bash
# QUALITY=high KIND=actual KIND2=actual bash process_data.sh
# QUALITY=high KIND=synthetic KIND2=distill bash process_data.sh
# QUALITY=high KIND=synthetic KIND2=extract_knowledge bash process_data.sh
# QUALITY=high KIND=synthetic KIND2=wrap_medium bash process_data.sh
# QUALITY=high KIND=synthetic KIND2=diverse_qa_pairs bash process_data.sh
# QUALITY=high KIND=synthetic KIND2=knowledge_list bash process_data.sh

apt-get update
apt-get install -y zstd

ROOT_DIR=${ROOT_DIR:-"/data/nishang/data/Nemotron-CC"}
QUALITY=${QUALITY:-"high"}
KIND=${KIND:-"actual"}
KIND2=${KIND2:-"actual"}
RAW_DATA_DIR=${ROOT_DIR}/quality=${QUALITY}/kind=${KIND}/kind2=${KIND2}

OUTPUT_PREFIX=${OUTPUT_PREFIX:-"/data/nishang/data/Nemotron-CC-MergeShard"}
MFORMAT_PREFIX=${MFORMAT_PREFIX:-"/data/nishang/data/Nemotron-CC-MFormat"}

OUTPUT_DIR=${OUTPUT_PREFIX}/quality=${QUALITY}/kind=${KIND}/kind2=${KIND2}
MEGATRON_PREFIX=${MFORMAT_PREFIX}/quality=${QUALITY}/kind=${KIND}/kind2=${KIND2}
mkdir -p $OUTPUT_DIR

# 1. 提取所有唯一的标识符 (例如 CC-MAIN-2013-20)
# 我们利用 sed 正则表达式提取 "CC-MAIN-数字-数字" 的部分，然后去重
echo "正在扫描唯一的年份ID..."
ALL_IDS=$(find $RAW_DATA_DIR -name "*CC-MAIN-*-part-*.jsonl.zstd" | sed -E 's/.*(CC-MAIN-[0-9]{4}-[0-9]{2}).*/\1/' | sort | uniq)

# 2. 打印总长度
# echo "$变量" | wc -l 用于计算行数
TOTAL_COUNT=$(echo "$ALL_IDS" | wc -l)
echo "----------------------------------------"
echo "共发现 $TOTAL_COUNT 个唯一的年份ID"

# 3. 取前三个 ID
# 使用 head -n 3 截取
# TARGET_IDS=$(echo "$ALL_IDS" | head -n 3)
TARGET_IDS=$ALL_IDS

# 4. 遍历每个 ID 进行合并
for id in $TARGET_IDS; do
    output_file="${OUTPUT_DIR}/${id}_merged.jsonl"
    
    echo "----------------------------------------"
    echo "正在处理分组: $id"
    echo "输出文件: $output_file"
    
    # 查找文件名中包含这个 ID 的所有文件，并进行解压合并
    # 注意：这里我们匹配 "*$id-part*" 确保只选中该年份的文件
    find $RAW_DATA_DIR -name "*${id}-part-*.jsonl.zstd" -print0 | xargs -0 zstd -dc > "$output_file"

    python /opt/megatron-lm/tools/preprocess_data.py \
       --input $output_file \
       --output-prefix $MEGATRON_PREFIX/${id} \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model openai/gpt-oss-120b \
       --append-eod \
       --workers 64

    echo "完成: $id"
done

echo "所有年份合并完成。"
