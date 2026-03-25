from synthetic_consults.io.jsonl_writer import write_jsonl
from synthetic_consults.pipelines.generate_text_dataset import build_record


def main() -> None:
    record = build_record("consult_000001")
    write_jsonl("data/processed/full_dataset.jsonl", [record])
    print("Wrote 1 record to data/processed/full_dataset.jsonl")


if __name__ == "__main__":
    main()