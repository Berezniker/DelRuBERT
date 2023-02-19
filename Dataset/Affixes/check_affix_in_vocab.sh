#!/bin/bash

affixes_paths=`ls ./affixes/*.txt`
vocab_paths=`ls ../../BERT/*/vocab.txt`

for vocab_path in $vocab_paths
do
    for affixes_path in $affixes_paths
    do
        (
            set -o xtrace
            python3.9 check_affix_in_vocab.py --affixes-path ${affixes_path} --vocab-path ${vocab_path}
        )
        echo
    done
    echo
done
