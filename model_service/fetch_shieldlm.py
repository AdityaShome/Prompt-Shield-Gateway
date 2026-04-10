#!/usr/bin/env python3
import json
from pathlib import Path

from datasets import load_dataset


KEEP_COLUMNS = {"text", "label_binary"}


def main() -> None:
    dataset = load_dataset("dmilush/shieldlm-prompt-injection")
    output_dir = Path("model_service/data/shieldlm")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split in dataset.items():
        trimmed = split.remove_columns(
            [column for column in split.column_names if column not in KEEP_COLUMNS]
        )
        output_path = output_dir / f"{split_name}.jsonl"

        with output_path.open("w", encoding="utf-8") as handle:
            for row in trimmed:
                item = {
                    "text": row["text"],
                    "label": "unsafe" if int(row["label_binary"]) == 1 else "safe",
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"wrote {len(trimmed)} rows to {output_path}")


if __name__ == "__main__":
    main()
