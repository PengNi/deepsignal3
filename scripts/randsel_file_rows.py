import random
import argparse
import gzip


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_reader(filename):
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt', encoding='utf-8')
    return open(filename, 'r', encoding='utf-8')


def random_select_file_rows(ori_file, w_file, maxrownum, header=True):
    print(f"Pass 1: counting lines in {ori_file} ...")

    total = 0
    with get_reader(ori_file) as f:
        if header:
            f.readline()
        for _ in f:
            total += 1

    print(f"  Total lines: {total}")

    k = min(maxrownum, total)
    # set of selected line indices — memory: ~28 bytes × k
    selected = set(random.sample(range(total), k))

    print(f"Pass 2: writing {k} selected lines ...")

    with get_reader(ori_file) as f, \
            open(w_file, 'w', buffering=1024 * 1024 * 32) as wf:
        if header:
            wf.write(f.readline())
        for i, line in enumerate(f):
            if i in selected:
                wf.write(line)

    print(f"Done. Wrote {k} lines to {w_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Two-pass uniform sampling for large gzip/plain files.')
    parser.add_argument('--ori_filepath', type=str, required=True)
    parser.add_argument('--write_filepath', type=str, required=True)
    parser.add_argument('--num_lines', type=int, required=True)
    parser.add_argument('--header', type=str, default='true')
    args = parser.parse_args()

    random_select_file_rows(
        args.ori_filepath,
        args.write_filepath,
        args.num_lines,
        str2bool(args.header)
    )


if __name__ == '__main__':
    main()
