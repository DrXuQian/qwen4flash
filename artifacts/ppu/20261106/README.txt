PPU operator wheels for spark-local-inference
============================================

This artifacts/ppu-20261106 branch hosts the previously built wheels in Git LFS:

  actlize_la-0.2.1-1ppu10version221184ee6-py3-none-linux_x86_64.whl
  flashinfer_python-0.6.18+ppu.flare1-py3-none-any.whl

GitHub refused new LFS uploads to the public spark-local-inference fork. This
previously authorized qwen4flash repository supplies the LFS storage instead;
the framework can use the same objects through its explicit .lfsconfig.
The original Rust implementation on main is unchanged.

actlizeLA 0.2.1 contains the symbolic serving-handle fake binding fix (2e107e2)
with the unchanged ppu10 / SDK2.2.1-184ee6 native payload from 4ea342e. This
binary is not a ppu15 artifact.

FlashInfer 0.6.18+ppu.flare1 contains the prepared sampling interfaces and
request-local chain RNG port (7a7989e), based on t-head/flashinfer-for-sail
ppu/0.6.18 e502dcf. It packages PPU JIT sources. First preparation compiles the
sampling module against the box's SDK and TVM-FFI, then reuses the cached module.
Installing the wheel does not rebuild FlashInfer; serving does not compile.

In this repository, retrieve the files with:

  git lfs install --local
  git lfs pull --include="artifacts/ppu/20261106/wheels/*.whl" --exclude=""

In the configured PPU Python environment, install with --no-deps to retain
platform Torch, Triton, TVM-FFI and other platform libraries:

  python -m pip install --no-deps --force-reinstall artifacts/ppu/20261106/wheels/*.whl

Use spark-local-inference feat/qwen38-gguf at 2a5dc34 or newer for the QSA and
sampling component/catalog gates. These wheels have host-test and PPU SDK
compile evidence, but their PPU numerical/graph gates, full target/MTP,
PPU export/Rust replay and no-bubble measurements remain pending.
