import argparse
import json


def get_tokens(phem_info):
    if phem_info is None:
        return None

    # phem_info example: "аддит:ROOT/ивн:SUFF/ост:SUFF/ь:END"
    tokens = list(map(lambda morph: morph.split(":")[0], phem_info.split("/")))
    for i in range(1, len(tokens)):
        tokens[i] = "##" + tokens[i]

    return tokens


def parse_xmorphy(file_name):
    with open(file_name, mode='r') as f:
        words = f.read().strip().split('\n\n')
    result = dict()

    for word in words:
        word = json.loads(word)
        key = list(word.keys())[0].split('_')[1]
        value = list(word.values())[0][0]
        result[key] = {
            "lemma": value.get("lemma"),
            "xmorphy": {
                "phem_info": value.get("phem_info"),
                "tokens": get_tokens(value.get("phem_info")),
            }
        }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('i', "--input-file", required=True, type=str)
    parser.add_argument('o', "--output-file", required=False, type=str)
    args = parser.parse_args()

    result = parse_xmorphy(args.input_file)

    if args.output_file:
        with open(args.output_file, mode='w') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=4))
