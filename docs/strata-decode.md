# Strata 为什么能有高 decode 吞吐

核对版本：`Niko1221/Strata@ffa80d73133ea5c548aefe57889a16e8085a938f`。
以下是对公开实现与已记录结果的分析，不是我们在 RTX 上复现的 Strata benchmark。
目前没有定位到用户提到的“96 GB、100 tok/s”的确切原始运行配置。

## 显存驻留和 CPU/SSD offload 是不同路径

小显存时，Strata 把常用专家放在 GPU，其余专家可由 CPU 在 RAM 原地计算，与 GPU 并行。
`--pcie-frac` 决定其中一部分缺失专家交给 GPU，另外还有自适应专家缓存。
因此“专家不在 GPU”不等于“每个 token 先将该专家权重复制进 GPU”。
参见 [内存分工](https://github.com/Niko1221/Strata/blob/ffa80d73133ea5c548aefe57889a16e8085a938f/docs/HOW_IT_WORKS.md)。

专家全在显存时，`verify.cpp` 的 `all_resident_` 检查会选择设备端专家调度，
去掉逐层 CPU doorbell、等待与空 PCIe launch，结果直接进入 combine。
参见 [准入检查](https://github.com/Niko1221/Strata/blob/ffa80d73133ea5c548aefe57889a16e8085a938f/src/core/verify.cpp#L1254)
与 [执行分支](https://github.com/Niko1221/Strata/blob/ffa80d73133ea5c548aefe57889a16e8085a938f/src/core/verify.cpp#L2380)。

本项目指定 Q4 主模型减去 PLE 表，再加 shared-Q8 MTP，原始权重约 79.44 GiB。
96 GiB 设备在受控上下文、workspace 与打包开销下有望全部驻留计算权重。
PLE 的 26.82 GiB 留在 RAM，只取当前 token 所需的少量行；不需要逐 token 传输整表。
这是来自 GGUF 元数据的容量推算，仍需实测打包后峰值，详见 [内存预算](memory.md)。

## MTP 必须批量验证才能获得这类加速

Strata 将当前 token 和若干 draft 一起通过 48 层，并捕获为一个 GPU 图。
这样 dense 权重可在窗口内复用，专家也按窗口内 token 分组，CPU miss 只处理所需专家集合。
文档报告典型平均每轮产出约 2.4–3.2 token；这是其测得的工作负载，不是本项目的接受率。
参见 [窗口语义](https://github.com/Niko1221/Strata/blob/ffa80d73133ea5c548aefe57889a16e8085a938f/include/strata/core/verify.hpp#L1)
与 [MTP 说明](https://github.com/Niko1221/Strata/blob/ffa80d73133ea5c548aefe57889a16e8085a938f/docs/HOW_IT_WORKS.md#guess-then-check)。

例如平均每轮接受 3 token、整轮 draft/verify/commit 共 30 ms，就对应 100 tok/s。
这只是计时示例：100 tok/s 不要求 100 次完整单 token forward/s。
模型装载和 prompt prefill 不应混入 decode 计时，也应另报 TTFT、上下文长度、接受率和端到端延迟。

## SSD 带宽成本仍然存在

同款 `UD-Q4_K_XL` 的历史报告：RTX 5070 12 GB、64 GB RAM、40 GiB 专家 RAM 预算，
每个 verify 窗口从 SSD 读约 1.1 GB，输出约 7–8.5 tok/s；24 GiB 预算时更慢。
这支持“大量实时权重读取不能免费达到高吞吐”的判断。
参见 [Q4 实测及计时拆分](https://github.com/Niko1221/Strata/blob/ffa80d73133ea5c548aefe57889a16e8085a938f/docs/UNSLOTH_Q4.md#ram-budget-speed-on-a-12-gb-card)。

另一条已公开历史结果是 RTX PRO 5000 Blackwell 72 GB、IQ3_S、全部专家驻留、`--spec 4`，
不同文本 decode 约 192–260 tok/s。它证明该实现具有完全驻留的高速路径，但模型量化、MTP 制品、
上下文和软件均不能直接套到我们的 Q4/shared-Q8 配置。
参见 [标明硬件与参数的结果表](https://github.com/Niko1221/Strata/blob/ffa80d73133ea5c548aefe57889a16e8085a938f/docs/DETAILS.md#history--older-measurements)。

## 本框架应优先实现的路径

1. 原始/packed 权重、KV、GDN 状态常驻设备，PLE 留主存；decode 不搬专家权重。
2. 小 M 的 grouped MoE 和投影、批量 MTP target verify、设备端接受与正确回滚。
3. 预热后捕获稳定图，减少逐层 host 同步；保留实际 PCIe/SSD 字节量计数。
4. 完成同权重、同上下文、同采样的无 MTP/MTP 对照，再考虑部分专家 offload。

CUDA proxy 目前验证了格式与状态语义；其同步逐算子调用和串行 verify 不能用于预测这些优化的收益。
