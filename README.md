# Synthetic Patient DR Data

Synthetic doctor-patient consultation dataset generation pipeline with structured clinical outputs and optional multi-speaker audio synthesis.

This repository currently supports two main stages:

1. Text dataset generation: create synthetic consultation records with scenario metadata, dialogue, clinical extraction, quality labels, transcript references, and TTS-ready turns.
2. Audio dataset generation: convert processed records into turn-level WAV files, stitched full consultations, transcript references, and audio manifests using OpenAI TTS.

## What The Project Produces

Each consultation record is stored as JSONL and includes:

- Scenario metadata such as specialty, urgency, patient persona, and audio profile
- A doctor-patient conversation with turn IDs and intents
- Clinical outputs including summaries, differentials, next steps, and patient-facing artifacts
- Quality labels from a critic step
- A TTS script for downstream audio generation
- Audio manifest placeholders or generated audio artifact references

When the audio pipeline runs successfully, it writes:

- `tts_script.json`
- `transcript_reference.json`
- `audio_manifest.json`
- Turn-level WAV files in `turns/`
- A stitched consultation file such as `full.wav`
- An `audio_generation_summary.json` file at the output root

## Current Repository State

- `data/processed/train.jsonl` currently contains 50 synthetic consultation records
- `artifacts/audio/` already contains generated audio outputs for a subset of those records
- The committed audio summary shows 37 successful audio generations and 1 failure caused by OpenAI quota exhaustion
- Some export-related scripts are present as placeholders and are not yet implemented end to end

## Project Layout

```text
configs/                     YAML config files
data/processed/              Generated structured consultation records
artifacts/audio/             Generated audio outputs and summary files
prompts/                     Prompt templates for generation steps
schemas/                     JSON schema and examples
scripts/                     Top-level helper scripts
src/synthetic_consults/
  generators/                Scenario, conversation, critic, revision, extraction
  pipelines/                 Text and audio generation entry points
  models/                    Pydantic data models
  validators/                Dialogue, schema, clinical, and audio validation
  audio/                     TTS script building, synthesis, stitching, manifests
tests/                       Unit tests for core pipeline pieces
```

## Requirements

- Python 3.12 or newer
- An OpenAI API key in the environment as `OPENAI_API_KEY`
- Network access for LLM and TTS generation

The code currently uses:

- `gpt-5.4-mini` for scenario, conversation, critique, revision, and extraction
- `gpt-4o-mini-tts` for audio synthesis by default

## Installation

### Using `uv`

```bash
uv sync
```

### Using `pip`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Set your API key before running any generation step:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

## Quick Start

### 1. Generate structured consultation records

This script creates 50 records and writes them to `data/processed/train.jsonl`.

```bash
uv run python scripts/generate_dataset.py
```

The text generation pipeline builds each record in this order:

1. Scenario generation
2. Conversation generation
3. Dialogue validation
4. Quality critique
5. Optional revision if quality checks fail
6. Clinical extraction
7. TTS script construction
8. Final `ConsultationRecord` assembly

### 2. Generate audio from processed records

The repository exposes a console script:

```bash
uv run generate-audio-dataset --record data/processed/train.jsonl --tts-config configs/tts.yaml --output-dir artifacts/audio --limit 5 --verbose
```

You can also process a directory of JSON or JSONL files:

```bash
uv run generate-audio-dataset --input-dir data/processed --pattern *.jsonl --tts-config configs/tts.yaml --output-dir artifacts/audio --full-only
```

Important flags:

- `--record`: process a single JSON or JSONL file
- `--input-dir`: recursively process a directory of records
- `--pattern`: file glob when using `--input-dir`
- `--limit`: cap the number of records processed
- `--skip-existing`: skip consultations that already have `full.wav` and `audio_manifest.json`
- `--fail-fast`: stop on first failure
- `--verbose`: print per-record progress
- `--full-only`: delete intermediate turn WAV files after stitching

## Configuration

### `configs/tts.yaml`

Current defaults:

- Provider: OpenAI
- Model: `gpt-4o-mini-tts`
- Locale: `en-ZA`
- Sample rate: `16000`
- Output format: `wav`
- Default voices:
  - doctor: `alloy`
  - patient: `verse`

### Other config files

- `configs/generation.yaml`
- `configs/dataset_split.yaml`
- `configs/validation.yaml`

These are present for future extension, but the current text generation script mainly relies on code-defined defaults and prompt files.

## Data Model Summary

The core record type is `ConsultationRecord`, which includes:

- `conversation_id`
- `generator` metadata
- `locale` information
- `scenario`
- `conversation`
- `tts_script`
- `audio_manifest`
- `transcript_reference`
- `clinical_outputs`
- `quality_labels`
- `tags`

This makes the dataset usable for:

- Clinical dialogue research
- Structured extraction benchmarking
- Speech synthesis benchmarking
- Multi-modal synthetic patient workflows

## Testing

Run the test suite with:

```bash
$env:PYTHONPATH="src"
pytest
```

If your system temp directory causes pytest permission issues on Windows, use a repo-local base temp:

```bash
$env:PYTHONPATH="src"
pytest --basetemp .pytest_tmp_run
```

Relevant test coverage already exists for:

- Pydantic models
- Validators
- TTS script building
- Audio stitching
- Export-related scaffolding
- End-to-end audio dataset generation behavior

## Linting

Run Ruff in check mode:

```bash
uv run ruff check .
```

Auto-fix what Ruff can safely rewrite:

```bash
uv run ruff check . --fix
```

## Example Outputs

Example structured record:

- `data/processed/train.jsonl`

Example generated audio artifacts:

- `artifacts/audio/consult_000000/audio_manifest.json`
- `artifacts/audio/consult_000000/transcript_reference.json`
- `artifacts/audio/consult_000000/full.wav`

## Known Gaps

- `scripts/export_to_hf.py` is currently empty
- `src/synthetic_consults/pipelines/build_hf_export.py` is currently empty
- The current audio generation summary shows quota-related failures when OpenAI usage limits are hit

## Notes

- The current dataset targets South African English consultation audio and dialogue
- Text normalization in the audio pipeline includes cleanup for mojibake found in generated records
- Audio synthesis validates records before generating artifacts and writes a summary of successes and failures at the end of the run

## License

No license file is currently included in this repository. Add one before external distribution if that matters for your use case.
