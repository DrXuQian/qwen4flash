# RAM 与显存预算

目标是单卡或少量 PPU，计算权重主要放在显存。SSD 负责保存制品和启动时加载。
以下数字来自固定 revision 的 GGUF 张量描述，不是实机 RSS 或显存测量。

| 内容 | 精确字节数 | GiB | 计划位置 |
|---|---:|---:|---|
| 主模型专家 | 77,017,907,200 | 71.73 | PPU |
| 主模型其他计算权重 | 5,505,584,640 | 5.13 | PPU |
| PLE 表（IQ4_NL） | 28,800,138,240 | 26.82 | RAM，保持量化格式 |
| shared-Q8 MTP 张量 | 2,775,621,632 | 2.59 | PPU |

MTP 文件为 2,786,568,256 字节，文件大小与张量大小的差别主要是元数据。
它没有另一份 embedding/output，不能重复预算这两项，也不能套用 Strata Q2 MTP 的约 0.9 GB 数字。
这是用户指定的 [shared-Q8 制品](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf)。

## 为什么 RAM 目标是 64 GiB

固定开销以 PLE 的 26.82 GiB 为主。默认规划另外包含 512 MiB 加载缓冲以及
12 GiB 系统/进程/CPU 临时空间预留，总计约 39.32 GiB。64 GiB 留出了约 24.68 GiB 余量。
32 GiB 会很紧；48 GiB 可能满足受控工作负载；64 GiB 是当前建议的部署目标。
128 GiB 适合同时开发、较大 CPU 缓存或未来 offload，但不是当前设计的稳态性能前提。

这个结论依赖以下已明确的实现约束：

1. 专家、主干计算权重和 MTP 载入 PPU 后不保留常驻 CPU 副本。
2. 原始权重和打包权重使用固定大小的搬运窗口，不在 CPU 同时展开全模型。
3. 完整 PLE 预热驻留，不依赖 SSD 随机读取性能。`ResidentPle` 使用只读文件映射和
   `mlock`，锁定失败明确报错；不会把失败当作等价的高性能模式。
4. KV、GDN 状态和 MTP 回滚缓冲主要放在 PPU。长上下文或并发增加时重新做显存规划。

`stream_tensor` 已实现量化块对齐、至多 512 MiB 的同步加载窗口和完成源页的缓存回收提示。
调用者必须等待 DMA 完成再复用窗口。异步双缓冲、native PPU allocator 和完整加载流程仍待接线。
`mmap` 映射 100 多 GiB 文件本身不代表 100 多 GiB RAM 常驻；实际触页和驻留策略才决定物理内存。

## 设备容量的边界

默认 32K、单会话、3 个 draft token、F16 KV：

- 原始主模型计算权重 + MTP：约 79.44 GiB。
- 目标 KV：0.75 GiB；MTP KV：0.0625 GiB。
- 索引器、GDN/卷积/PLE 历史与回滚快照另计。
- 加上默认 4 GiB workspace 预算，规划合计约 84.85 GiB。

这些是按原始量化权重推算的容量，不是原生执行准入。QUactlize 的实际打包大小、
加载期间源/目标同时存在、GDN prefill 临时空间、graph pool 和 allocator 对齐需要实测。
不能为了接入 N=640 的专家就把所有专家补齐到 N=768；应优先按专家拼接 gate/up 为 N=1280，
并单独实现原始 Q5_1 down 的路径。

性能验收需要记录峰值 RSS、锁页量、稳态 major faults、SSD 读取量、PCIe 流量和 tokens/s。
只有完成这些测量，才能确认指定工作负载下“64 GiB 不因内存不足降速”。
