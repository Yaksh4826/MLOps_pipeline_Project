#!/usr/bin/env python3
"""Merge transformed eval TFRecord shards into one file for the What-If Tool file picker.

The TensorBoard WIT dialog expects a path to TFRecord data; one merged file is easier
to select than many gzip shards.

Example:
  export WIT_OUT=/tmp/wit_eval.tfrecord.gz
  python scripts/prep_wit_eval_tfrecord.py \\
    --pattern '/path/to/.../Transform/transformed_examples/42/Split-eval/*.gz' \\
    --max-examples 500 \\
    --out "$WIT_OUT"
"""

from __future__ import annotations

import argparse

import tensorflow as tf


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pattern",
        required=True,
        help="Glob for eval TFRecord shards (e.g. .../Split-eval/*.gz).",
    )
    p.add_argument("--out", required=True, help="Output path (.tfrecord or .tfrecord.gz).")
    p.add_argument("--max-examples", type=int, default=500, help="Cap rows loaded.")
    args = p.parse_args()

    paths = sorted(tf.io.gfile.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched: {args.pattern}")

    comp = "GZIP" if args.out.endswith(".gz") else ""
    opts = tf.io.TFRecordOptions(compression_type=comp) if comp else tf.io.TFRecordOptions()
    n = 0
    with tf.io.TFRecordWriter(args.out, options=opts) as w:
        ds = tf.data.TFRecordDataset(paths, compression_type="GZIP")
        for raw in ds.take(args.max_examples):
            w.write(raw.numpy())
            n += 1
    print("Wrote", n, "examples to", args.out)


if __name__ == "__main__":
    main()
