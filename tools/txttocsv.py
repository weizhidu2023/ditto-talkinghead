#!/usr/bin/env python3
"""
Combine multiple 63-line exp TXT files into a single CSV.

Behavior:
 - Each input TXT becomes one column in the output CSV.
 - The first column header is `dim`, with rows 1..N (N default 63).
 - Subsequent column headers are the base filenames (without extension).
 - The script skips lines starting with '#' when reading the TXT files.

Usage:
  python tools/txttocsv.py --out out/combined.csv out/image_face_exp.txt out/image1_face_exp.txt

Or provide a directory (will pick all *.txt files inside):
  python tools/txttocsv.py --dir out --out out/combined.csv
"""
import argparse
import os
import sys
import csv

def read_txt_vals(path):
    vals = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            # parse float
            try:
                v = float(line)
            except Exception:
                # try split if headerless single row
                parts = line.split()
                if len(parts) > 1:
                    for p in parts:
                        vals.append(float(p))
                    continue
                raise
            vals.append(v)
    return vals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='Directory containing txt files to combine')
    parser.add_argument('--out', required=True, help='Output CSV path')
    parser.add_argument('txts', nargs='*', help='Explicit list of txt files to include (overrides --dir)')
    parser.add_argument('--dims', type=int, default=63, help='Number of dimensions/rows (default 63)')
    args = parser.parse_args()

    files = []
    if args.txts:
        files = args.txts
    elif args.dir:
        files = [os.path.join(args.dir, p) for p in sorted(os.listdir(args.dir)) if p.lower().endswith('.txt')]
    else:
        parser.error('Either provide txt files as positional args or use --dir')

    if not files:
        print('No txt files found to combine', file=sys.stderr)
        sys.exit(1)

    cols = []
    headers = []
    for p in files:
        if not os.path.isfile(p):
            print('Warning: skipping not-found file', p, file=sys.stderr)
            continue
        vals = read_txt_vals(p)
        if len(vals) != args.dims:
            print(f'Warning: file {p} has {len(vals)} values (expected {args.dims})', file=sys.stderr)
        cols.append(vals)
        headers.append(os.path.splitext(os.path.basename(p))[0])

    # Determine max rows (use args.dims or max read)
    max_rows = max(len(c) for c in cols) if cols else 0
    # Build CSV rows: first column is dim indices 1..max_rows
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as cf:
        writer = csv.writer(cf)
        # header row
        hdr = ['dim'] + headers
        writer.writerow(hdr)
        for i in range(max_rows):
            row = [i+1]
            for col in cols:
                if i < len(col):
                    row.append(col[i])
                else:
                    row.append('')
            writer.writerow(row)

    print('Saved combined CSV to', args.out)


if __name__ == '__main__':
    main()
