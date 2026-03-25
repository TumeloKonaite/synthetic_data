from synthetic_consults.io.jsonl_writer import write_jsonl
from synthetic_consults.pipelines.generate_text_dataset import build_record


def main():
    records = []

    for i in range(5):  # start small
        conversation_id = f"consult_{i:06d}"
        record = build_record(conversation_id)
        records.append(record)

    write_jsonl("data/processed/train.jsonl", records)
    print(f"Generated {len(records)} records.")


if __name__ == "__main__":
    main()