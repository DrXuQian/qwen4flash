# CUDA proxy：在 NVIDIA 上验证框架和格式

这条后端用于 PPU 设备暂不可用时验证框架改动。Rust 模型和 MTP 事务保持同一份实现，
矩阵乘、attention 加权计算、GDN 递推调用 CUDA C ABI。CPU 仍明确执行归一化、卷积、
RoPE、路由、QSA 选择、PLE 查表及小型逐元素计算，并持有状态。

2026-11-04 在 NVIDIA RTX PRO 6000 Blackwell Server Edition（SM120、97887 MiB、
驱动 580.82.09、CUDA 12.8.93）验证通过。它不是旧款 Turing RTX 6000。
完整模型端到端 logits、语言生成、性能与 PPU kernel 尚未验证。

## 构建与运行 Rust 图

运行时只需 Rust 二进制、显式指定的共享库和 CUDA/cuBLAS；不加载 Python 或 Torch。

```sh
cargo build --release --locked
python3 tools/build_cuda_proxy.py --arch sm_120 --out native/build/cuda
target/release/q4e demo --tokens 1,3,2,4,7 --generate 20 \
  --cuda-proxy-lib native/build/cuda/libq4e_cuda_proxy.so
Q4E_CUDA_PROXY_LIB="$PWD/native/build/cuda/libq4e_cuda_proxy.so" \
  cargo test -p q4e-cuda --test proxy --locked -- --ignored --test-threads=1
```

`reference` 命令也接受 `--cuda-proxy-lib`、`--device` 和 `--weight-budget-gib`，
默认权重预算 88 GiB。首次矩阵调用上传其原始量化张量，后续保持 GPU 驻留；
权重名称和不可变 storage identity 必须一致。权重预算不包含 scratch/KV/主机状态。
格式、维度、ABI 或 CUDA 执行不支持时明确失败，不自动切回标量矩阵/attention/GDN。

3 项 GPU 测试检查：原始 GGUF 的专家步长及 M=1/4/17；实际 Hk=16/Hv=48/D=128 的 GDN
及 24Q/2KV/D=256 的 attention；同一缩小模型的 CPU/CUDA logits 与 CUDA 普通解码/MTP
输出及完整目标状态一致性。MTP dense attention 的历史超过 QSA 选择预算仍不启用 indexer。

## QUactlize：保持字节合同

固定 QUactlize `4fb7ce037f36768bf900f1c33d606f7eeaf87f80`，NVIDIA CUTLASS
`f94ec46f4f63f96003d6cfdf2014731e7672c281`。参数路径指向包含 `quactlize/` 的仓库根目录，
以及包含 `include/cutlass/` 的 NVIDIA CUTLASS 根目录。

```sh
python3 tools/build_cuda_proxy.py --arch sm_120 --out native/build/cuda \
  --quactlize /path/to/QUactlize --cutlass /path/to/NVIDIA-cutlass
```

构建使用原始 `packing/sizes.cpp`、`packing/ppu_pack.cu` 和 portable format traits。
`compat/hggc_runtime.h` 只桥接 runtime 名称，不伪造 SM90、PPU 架构或解除 target 检查。
该库暴露可选 packing 校验 API；Rust 图当前使用 raw reader，并未调用 PPU 优化 dense kernel。

| 数据 | 合同 |
|---|---|
| GGUF | 保留每个张量的实际类型、block 大小、位序、scales/min、专家步长 |
| gate/up | 每个专家原始 N=640，先 gate 后 up 拼接为 N=1280；不解量化或 padding |
| Q4_K canonical | 原始 arrangement、layout=1、mapping_id=0x51344b5034540001 |
| Q5_K/Q6_K canonical | 原始 arrangement、layout=2、mapping_id=0x514b504b54000001 |
| Q8_0 canonical | 原始 arrangement、layout=4、mapping_id=0x51384b5032540001 |
| Q8_0 down K=640 | 此版本 canonical producer 不支持，继续使用 raw reader |
| Q5_1 down | 保留 Q5_1 的 min 项和原始块，不能改叫 Q5_K |
| IQ4_NL PLE | 使用非线性码表，不能按 Q4_0 解码 |

packed producer 先查询公共 arrangement/size，GPU 输出与原始 CPU packer逐字节对比，
再由 portable packed reader 解码，与独立 gguf-py 比较。映射 ID 错误和 payload 缺字节必须拒绝。
F32 CUDA consumer 验证格式和算式；PPU dense F16/BF16 路径需要单独的数值容差与实机测试。

## 独立算子对照

先准备匹配设备的 PyTorch。实测使用机器已有的 `torch 2.8.0+cu128`，没有替换全局环境。

```sh
python3 -m venv --system-site-packages artifacts/cuda-venv
artifacts/cuda-venv/bin/pip install -r tools/requirements-cuda-validation.txt
python3 tools/fetch_proxy_samples.py --out artifacts/proxy-samples
artifacts/cuda-venv/bin/python tools/check_cuda_proxy.py \
  --library native/build/cuda/libq4e_cuda_proxy.so \
  --samples artifacts/proxy-samples --out artifacts/cuda-proxy-report.json
```

切片来自固定 HF revision，核对 HTTP 206、完整 Content-Range、字节数与 SHA256。
覆盖 Q4_K/Q5_K 专家、Q5_1/Q8_0 down、shared-Q8 MTP、IQ4_NL PLE 和 Q8_0 output head。
所取专家包含 0/17/511；另外使用 512-expert 合成张量检查原始步长。
它们是明确标记的切片，不是完整 checkpoint。可用 `--endpoint https://hf-mirror.com`；
本次所有 12 组镜像切片与官方 HF 端点独立读取的 SHA256 相同。

| 对照 | 本次实际运行 |
|---|---|
| GGUF 格式 | gguf-py 0.19.0 独立解量化 |
| 矩阵乘 | NumPy 与 cuBLAS pedantic F32，禁用 TF32，M=1/4/17/65 |
| GDN | 独立 PyTorch FP32 递推，9 步，head 映射 `vh % hk` |
| Attention | PyTorch F32 与 flash-attn-4 4.0.0b37，24Q/2KV/D=256，1/4/17/513 个选中位置 |
| DeepGEMM | 未运行：上游要求 SM90/SM100 和 CUDA >=12.9，本机 SM120/CUDA 12.8 不满足 |

FlashAttention 输入/输出为 BF16；proxy 使用相同、已舍入到 BF16 的输入转回 F32 计算，
分别记录 F32 对照和 BF16 输出误差，不宣称逐位一致。已经 gather 的合法 QSA KV 不应再应用
连续下标的 causal mask。固定 b37 使用 `_flash_attn_fwd(..., tile_mn=(64,64), pack_gqa=False)`；
默认 HD256 tile 会要求 131072 字节 shared memory，超过本卡 101376 字节限制。
这是固定版本内部校验 API，不是推理运行时依赖，升级后须重新核对。

上游依据：[FlashAttention](https://github.com/Dao-AILab/flash-attention)、
[DeepGEMM requirements](https://github.com/deepseek-ai/DeepGEMM#requirements)。
完整报告见 [实测证据](../evidence/rtx6000-sm120/README.md)。

## 当前限制与下一阶段

- 同步 host-buffer ABI，每次算子会搬运激活/状态；没有 device-resident 图或吞吐承诺。
- MTP 已执行 learned draft 和精确 greedy 提交/回滚，但 target verify 仍逐 token，不能获得批量 MTP 加速。
- proxy attention 一次最多 65535 个选中位置；dense MTP 因而不能在此后端验证 64K 以上上下文。
- proxy 权重映射与主机状态不实现 64 GiB RAM 的最终部署合同；容量规划针对未来 PPU 后端。
- 量化切片和小模型图通过不能证明整个 111 GB 模型正确，更不能证明 NVIDIA/PPU 性能相当。

完整制品可断点续传并逐文件检查 SHA256：

```sh
python3 tools/download_model.py --out /path/with/space/models --jobs 2
```

本次完整下载因链路缓慢未完成，已停止传输并保留 `.part`，没有用残缺文件尝试推理。
下一阶段应建立显存常驻状态与批量 verify，接入经过准入的 PPU 优化算子，再完成全模型 logits、
MTP 接受率、峰值内存和稳态 PCIe/SSD 流量测量。
