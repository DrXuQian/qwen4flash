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
omitted FatBinary module-load callback. The latest wheel also accepts the explicit
module_resource_abi="hgpti-bare" workaround for SDK2.2.1's bare module-resource
payloads. It recovers context from same-thread nested driver callbacks and keeps
image ownership and unload/context teardown checks. Standard CUPTI remains the
default. Twenty recorder unit tests and seventeen native/Python ABI checks pass
on the host; one device test is skipped. Actual SDK2.2.1 synthetic driver/module
events also reach this native workaround without a crash and preserve context.
SDK2.1.1 rejects callback 742; SDK2.2.1 passes activation/allocation checks on the
box, but complete device export/replay still need qualification. Reinstall
this refreshed wheel with --force-reinstall; its package version remains 0.1.0.
See the framework's docs/ppu-hgpti-callback-prompt.txt for the SDK task and gates.

The current native wheel includes framework aa3bbbc.
It reads ABI-4 PPU ELF argument records in export and native loading, because
SDK2.2.1 FuncGetParamInfo/FuncLoad are stubs. Hidden arguments are excluded;
counts and widths remain validated. Packed-buffer offsets and unqualified
sub-word register parameters fail explicitly. Both HGCC ppu10/15 compiler-only
cases pass, along with 23 Rust capture/ELF and 62 Python/native checks. Actual
PPU export and Rust replay still need the box. Rebuild the Rust executable as
well as reinstalling this wheel; no SDK/Torch/Triton patch is installed.
It also audits HGPTI graph attribute defaults before handling SDK2.2.1's
INVALID_VALUE replies. The graph getter/setter accepts only IDs 1/2/4, although
LaunchKernelEx can set policy 5. Nineteen additional mandatory callbacks prove
fresh single-stream capture defaults with an exact launch inventory. Explicit,
copied, inherited or unobserved state and other errors remain fatal. Supported
getter values and NVIDIA behavior are unchanged; serving adds no work.
Forty-nine focused Rust checks pass (three CUDA ignored), along with 64
Python/native checks (one device skipped). Real SDK callback names and ABI
layouts match; synthetic delivery reaches the audit. These host checks do not
qualify device callback activation or replay. A broader export test hit an
existing inode-mutation test failure, also reproduced in the pre-change binary.
Reinstall this wheel and rerun the tiny PPU export/replay gate. Nondefault and
more complex graph histories need SDK getters/setters; see the framework's
docs/ppu-graph-attributes-prompt.txt for that task.
Wheel SHA256: 14829b55c8a3d18634d6054ebe1a1b974605378f77e5bd5c0a6a20c4e2d7c364
Wheel bytes: 7554375

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
