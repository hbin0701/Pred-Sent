#!/usr/bin/env python
import argparse
import json
import sys
from typing import Any, Dict

from transformers import AutoTokenizer


def check_newline_token(tokenizer, token_str: str = "<|new_line|>") -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    token_id = tokenizer.convert_tokens_to_ids(token_str)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    exists = token_id not in (None, -1) and (unk_id is None or token_id != unk_id)

    report["token_str"] = token_str
    report["token_id"] = token_id
    report["unk_token_id"] = unk_id
    report["exists"] = bool(exists)

    # Encode the token string as-is (no added specials)
    enc_ids = tokenizer.encode(token_str, add_special_tokens=False)
    report["encode_ids"] = enc_ids
    report["encode_is_single"] = len(enc_ids) == 1
    report["encode_matches_id"] = len(enc_ids) == 1 and exists and enc_ids[0] == token_id

    # Decode back the id to token string (via token -> token)
    decoded_token = None
    if exists:
        try:
            decoded_token = tokenizer.convert_ids_to_tokens([token_id])[0]
        except Exception:
            decoded_token = None
    report["decoded_token"] = decoded_token
    report["decoded_matches"] = (decoded_token == token_str)

    # Mixed string should contain the standalone token id
    mixed_ids = tokenizer.encode(f"foo{token_str}bar", add_special_tokens=False)
    report["mixed_contains_id"] = bool(exists and token_id in mixed_ids)

    # Overall pass condition
    report["ok"] = bool(
        report["exists"]
        and report["encode_is_single"]
        and report["encode_matches_id"]
        and report["decoded_matches"]
        and report["mixed_contains_id"]
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Tokenizer/model directory or HF repo ID")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    report = check_newline_token(tokenizer, token_str="<|new_line|>")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Token string: {report['token_str']}")
        print(f"Token id: {report['token_id']}, UNK id: {report['unk_token_id']}, exists: {report['exists']}")
        print(f"encode_ids: {report['encode_ids']}, encode_is_single: {report['encode_is_single']}, encode_matches_id: {report['encode_matches_id']}")
        print(f"decoded_token: {report['decoded_token']}, decoded_matches: {report['decoded_matches']}")
        print(f"mixed_contains_id: {report['mixed_contains_id']}")
        print(f"OK: {report['ok']}")

    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()



