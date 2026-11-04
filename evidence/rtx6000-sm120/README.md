# RTX PRO 6000 CUDA proxy 验证记录

日期：2026-11-04 UTC。设备：NVIDIA RTX PRO 6000 Blackwell Server Edition，SM120，
97887 MiB，驱动 580.82.09，NVCC 12.8.93。独立 venv 使用已有 Torch 2.8.0+cu128。
这是格式和框架正确性证据，未验证完整 checkpoint 的输出或吞吐。

| 检查 | 结果 |
|---|---|
| Rust CPU | 23 passed |
| Rust 实卡框架 | 3 passed；同一 Rust 图，CPU/CUDA logits 容差，CUDA 普通/MTP token 与目标状态一致 |
| 权重 | 11 合成 + 12 真实切片 + 3 gate/up 拼接 = 26 组，292 条比较记录 |
| canonical packing | 8 组 GPU/CPU 原始 producer 输出逐字节相同；解量化/矩阵乘独立对照通过 |
| GDN 9 步 FP32 | 输出最大绝对误差 2.33e-9，状态 2.98e-8 |
| attention vs PyTorch FP32 | 最大绝对误差 2.38e-7 |
| attention vs FA4 BF16 输出 | 最大绝对误差 0.00546；使用明确 BF16 舍入输入和独立容差 |
| DeepGEMM | SM120/CUDA 12.8 不满足上游准入；没有运行，使用 cuBLAS pedantic F32 对照 |
| 官方 HF / 镜像 | 固定 revision 的全部 12 组切片 SHA256 一致 |

文件：

- [operator-report.json](operator-report.json)：各类型、专家、M、精度的误差与 packed plane SHA256。
- [samples.json](samples.json)：固定 revision 的原始分片、精确 byte range 与切片 SHA256；不包含权重。
- [framework-tests.txt](framework-tests.txt)：实际 GPU 测试输出。
- [cuda-demo.json](cuda-demo.json)、[cpu-demo.json](cpu-demo.json)：相同缩小网络的 20 token 输出；
  两者 token 相同。CUDA 有 5700 次矩阵、150 次 attention、75 次 GDN、65 次权重上传。
- [native-build.json](native-build.json)：NVCC 命令、二进制与 QUactlize 源文件哈希，机器路径已替换为 `<work>`。
- [provenance.json](provenance.json)：上游 commit、被测源代码与 Rust 二进制哈希、验证范围。

demo 使用随机确定性权重，1/51 的 draft 接受数没有真实模型性能意义。
FA4 使用固定 4.0.0b37 内部接口和 64×64 tile，避免 HD256 默认配置超出 SM120 shared-memory 上限。
Q8 down K=640 不满足当前 QUactlize canonical producer 条件，report 中明确标为 raw 路径。
PPU 优化 dense/GDN kernel、device-resident 状态、MTP 批量验证、64 GiB 稳态 RAM 和全模型 logits
均不在本报告的通过范围。复现见 [CUDA 文档](../../docs/cuda-proxy.md)。
