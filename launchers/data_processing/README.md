# RedPajama Data Processing

Scripts for converting RedPajama raw dataset into Megatron binary format (.bin/.idx).

## Scripts

| Script | Description | Use Case |
|--------|-------------|----------|
| `preprocess_redpajama.sh` | Full dataset (all 10 sources) | Production training |
| `preprocess_redpajama_small.sh` | 2 sources (Wikipedia + StackExchange) | Development/testing |
| `preprocess_redpajama_tiny.sh` | 10k samples from Wikipedia | Quick debugging |

## Usage

```bash
# Full dataset preprocessing
bash preprocess_redpajama.sh

# Small test dataset
bash preprocess_redpajama_small.sh

# Tiny dataset (configurable sample count)
NUM_SAMPLES=10000 bash preprocess_redpajama_tiny.sh
```

## Output

- `.bin` - Binary tokenized data
- `.idx` - Index file with document offsets
- `data_blend.txt` - Weight configuration for multi-source training

## Dependencies

These scripts call `tools/preprocess_data.py` (core Megatron utility) with HuggingFace tokenizer (default: Qwen/Qwen3-8B).
