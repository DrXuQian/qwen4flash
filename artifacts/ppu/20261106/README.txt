PPU operator wheels for spark-local-inference
============================================

This artifacts/ppu-20261106 branch hosts the current wheels in Git LFS:

  actlize_la-0.2.1-1ppu10version221184ee6-py3-none-linux_x86_64.whl
  flashinfer_python-0.6.18+ppu.flare2-py3-none-any.whl
  flare_heliopause-0.1.0-cp312-cp312-linux_x86_64.whl

GitHub refused new LFS uploads to the public spark-local-inference fork. This
previously authorized qwen4flash repository supplies the LFS storage instead;
the framework can use the same objects through its explicit .lfsconfig.
The original Rust implementation on main is unchanged.

actlizeLA 0.2.1 contains the symbolic serving-handle fake binding fix (2e107e2)
with the unchanged ppu10 / SDK2.2.1-184ee6 native payload from 4ea342e. This
binary is not a ppu15 artifact.

FlashInfer 0.6.18+ppu.flare2 contains the prepared sampling interfaces and
request-local chain RNG port plus the sampling stub-linkage fix (762b396),
based on t-head/flashinfer-for-sail
ppu/0.6.18 e502dcf. It packages PPU JIT sources. First preparation compiles the
sampling module against the box's SDK and TVM-FFI, then reuses the cached module.
Installing the wheel does not rebuild FlashInfer; serving does not compile.
The new version omits unsupported-API stub definitions from sampling's staged
compatibility header, avoiding conflicts with implemented SDK APIs. Eighteen
host checks and the real SDK2.2.1 sampling compile/link pass. Device replay on
the box remains pending. flare1 is retained for history; do not install both.

Heliopause is the framework's separate Rust/Python extension. Its CPython3.12 /
Linux x86_64 wheel contains the generic PCI identity handling introduced in
framework 2275803. It needs glibc >=2.34, OpenSSL3 and libstdc++; it does not ship
Torch or GPU operator libraries. Installation and isolated import pass in a
fresh environment, and fourteen native package publication/conversion cases
pass. PPU module recording/export/replay still require device qualification.
The refreshed wheel also checks all required driver callback names before
subscribing, reports missing/mismatched names together, and enables the previously
omitted FatBinary module-load callback. Fifteen recorder unit tests and thirteen
native/Python ABI checks pass on the host; one device test is skipped. SDK2.1.1
rejects callback 742; local SDK2.2.1 resolves all 47 names. The deployment box
reports LAZY loading and still needs compatible HGPTI event delivery. Reinstall
this refreshed wheel with --force-reinstall; its package version remains 0.1.0.
See the framework's docs/ppu-hgpti-callback-prompt.txt for the SDK task and gates.

In this repository, retrieve the files with:

  git lfs install --local
  git lfs pull --include="artifacts/ppu/20261106/wheels/*.whl" --exclude=""

In the configured PPU Python environment, install with --no-deps to retain
platform Torch, Triton, TVM-FFI and other platform libraries:

  python -m pip install --no-deps --force-reinstall \
    artifacts/ppu/20261106/wheels/actlize_la-0.2.1-1ppu10version221184ee6-py3-none-linux_x86_64.whl \
    artifacts/ppu/20261106/wheels/flashinfer_python-0.6.18+ppu.flare2-py3-none-any.whl \
    artifacts/ppu/20261106/wheels/flare_heliopause-0.1.0-cp312-cp312-linux_x86_64.whl

Check native installation with the interpreter used for model preparation:

  python -c "import sys, flare_heliopause; print(sys.executable); print(flare_heliopause.__file__)"

Use spark-local-inference feat/qwen38-gguf at 2a5dc34 or newer for the QSA and
sampling component/catalog gates. These wheels have host-test and PPU SDK
compile evidence, but their PPU numerical/graph gates, full target/MTP,
PPU export/Rust replay and no-bubble measurements remain pending.
