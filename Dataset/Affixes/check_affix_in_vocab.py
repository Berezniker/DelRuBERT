import argparse
from prettytable import PrettyTable
from typing import List


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--affixes-path', required=True, type=str)
    parser.add_argument('-v', '--vocab-path', required=True, type=str)
    parser.add_argument('-p', '--print-table', action='store_true')
    args = parser.parse_args()
    return args


def readlines_from_file(path_to_file: str) -> List[str]:
    return list(map(str.strip, open(path_to_file, mode='r').readlines()))


def main():
    args = argparser()
    vocab = readlines_from_file(args.vocab_path)
    affixes = readlines_from_file(args.affixes_path)

    pt = PrettyTable()
    pt.field_names = ["affix", "has in vocab?"]

    total = 0
    for affix in affixes:
        is_in_vocab = "YES" if affix in vocab else "NO"
        pt.add_row([affix, is_in_vocab])
        total += int(is_in_vocab == "YES")

    print(f'{total} аффиксов из {len(affixes)} ({100. * total / len(affixes):.2f}%) есть в словаре.')
    if args.print_table:
        print()
        print(pt)


if __name__ == "__main__":
    main()