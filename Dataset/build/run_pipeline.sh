#!/bin/bash

file_name=$1
pretrained_model=$2
# sberbank-ai/sbert_large_nlu_ru
# DeepPavlov/rubert-base-cased
# bert-base-multilingual-cased
# bert-base-multilingual-uncased

if [ -z $file_name ]; then
    echo "Not specified FILE_NAME"
    echo "example: $0 FILE_NAME PRETRAINED_MODEL"
    exit 1
fi
if [ -z $pretrained_model ]; then
    echo "Not specified PRETRAINED_MODEL"
    echo "example: $0 $file_name PRETRAINED_MODEL"
    exit 2
fi
if [ ! -f $file_name ]; then
    echo "File $file_name not found!"
    exit 3
fi
if [ ! -s $file_name ]; then
    echo "File $file_name is empty!"
    exit 4
fi

file_name_array=(`echo $file_name | tr '.' ' '`)
topic=${file_name_array[0]}

words=(`cat ${file_name} | grep "^\S*$" | wc --words`)
echo "$words single words found"

cat ${file_name} | grep "^\S*$" | ../../XMorphy/build/xmorphy -c -m --format JSONEachSentence > ${topic}_xmorphy_output.txt \
&& python3 1_stage_parse_xmorphy.py --input-file ${topic}_xmorphy_output.txt --output-file ${topic}_xmorphy_parsed.json \
&& python3 2_stage_bert_tokenizer.py --input-file ${topic}_xmorphy_parsed.json --output-file ${topic}_xmorphy_bert_parsed.json --pretrained-model ${pretrained_model} \
&& echo "Success!"