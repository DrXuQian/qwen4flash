# Descriptor fixtures

These are metadata-only inventories from Hugging Face revision
`38bb39ee97821de2c9009abb7e93950eec396e66`, obtained using bounded HTTP ranges.
They contain no tensor payloads and cannot run a model.

The target is the four `UD-Q4_K_XL` shards. The sidecar is
`MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (32 tensors).
Tokenizer arrays and unrelated descriptive metadata are omitted. The tokenizer
entry count is retained as `qwen4exp.vocab_size` for config validation.
Tensor descriptors, source file sizes and source data offsets are retained.

Receipt hashes identify the **original fetched header bytes**, before trimming,
not these JSON files and not the full weight payloads. The loader revalidates
shard identity, tensor extents and architecture/shape contracts when reading JSON.
The CLI deliberately separates `inspect-index` / `prepare --metadata-only` from
loading actual GGUF files.
