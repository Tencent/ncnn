"""Export HunyuanOCR tokenizer to vocab.txt + merges.txt for ncnn inference.

The HF tokenizer uses byte-level BPE (BBPE) with 120000 base tokens + 818 added
special tokens. vocab.txt is indexed by token id so that line N is token N.

Usage:
    python export_tokenizer.py --model ./HunyuanOCR-hf --output ./hunyuan_ocr
"""
import argparse
import os
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HunyuanOCR HF model directory")
    ap.add_argument("--output", default="./hunyuan_ocr")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    tj_path = os.path.join(args.model, "tokenizer.json")
    tc_path = os.path.join(args.model, "tokenizer_config.json")

    with open(tj_path, encoding="utf-8") as f:
        tj = json.load(f)
    with open(tc_path, encoding="utf-8") as f:
        tc = json.load(f)

    model = tj.get("model", {})
    vocab = model.get("vocab", {})        # token -> id
    merges = model.get("merges", [])
    added = tc.get("added_tokens_decoder", {})  # str(id) -> {content, ...}

    max_id = max(vocab.values()) if vocab else -1
    for k in added:
        max_id = max(max_id, int(k))
    size = max_id + 1

    id2tok = [""] * size
    for tok, idx in vocab.items():
        id2tok[idx] = tok
    for k, v in added.items():
        id2tok[int(k)] = v.get("content", "")

    vocab_path = os.path.join(args.output, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for tok in id2tok:
            f.write(tok + "\n")
    print(f"[vocab] {size} tokens -> {vocab_path}")

    merges_path = os.path.join(args.output, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        for m in merges:
            if isinstance(m, list) and len(m) == 2:
                f.write(f"{m[0]} {m[1]}\n")
            elif isinstance(m, str):
                f.write(m + "\n")
    print(f"[merges] {len(merges)} rules -> {merges_path}")


if __name__ == "__main__":
    main()
