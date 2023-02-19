from transformers import AutoTokenizer
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('i', "--input-file", required=True, type=str)
    parser.add_argument('o', "--output-file", required=False, type=str)
    parser.add_argument('m', "--pretrained-model", required=True, type=str)
    args = parser.parse_args()

    with open(args.input_file, mode='r') as f:
        words = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    for key in words:
        words[key][args.pretrained_model] = {
            "tokens": tokenizer.tokenize(key)
        }

    if args.output_file:
        with open(args.output_file, mode='w') as f:
            json.dump(words, f, ensure_ascii=False, indent=4)
    else:
        print(json.dumps(words, ensure_ascii=False, indent=4))
