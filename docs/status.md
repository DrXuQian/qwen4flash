# 开发状态

这是独立 Rust 工程，采用准备/执行分离。没有把 llama.cpp、Strata 或 Flare 作为推理运行时依赖。
参考其公开模型定义、存储格式和算子接口，但不复制框架的通用调度或 helper 层。

| 部分 | 当前结果 |
|---|---|
| GGUF v2/v3、分片 | 已实现元数据读取、量化字节数、范围/重叠/分片检查 |
| 固定目标与 shared-Q8 MTP | 已核对远端 1,224 + 32 张量；共享目标 embedding/output，匹配几何后加载 |
| 量化参考读取 | F32/F16/BF16、Q4_0/Q4_1、Q5_0/Q5_1、Q8_0、Q4_K/Q5_K/Q6_K、IQ4_NL；另有 Q2_0 标量读取器 |
| 模型参考执行 | HC、GDN、QSA、PLE、routed/shared MoE、learned MTP；完整大模型尚未做 logits 对齐 |
| MTP 事务 | greedy；目标模型验证；拒绝后恢复所有继续生成状态；提交失败不污染已提交状态 |
| PLE/加载 RAM 原语 | 保留量化数据、预热锁页、有限加载窗口；尚未接入完整设备加载器 |
| QUactlize | 公共 packing/dense C ABI 绑定、类型尺寸检查、prepare/run 生命周期合同 |
| actlizeLA | 原生 GDN WY prefill C ABI 绑定及 workspace 计算 |
| PPU 整模型执行 | 未实现完整 dispatch/allocator/计算图；禁止宣称已准入 |
| 多 PPU / 文本 API / HTTP | 待实现 |

## MTP 状态与一致性

残差位置 i 与 token i+1 配对，MTP 预测 token i+2。目标模型的当前 logits 给出第一个确定 token，
随后使用 learned draft block 生成候选，再由目标模型验证。只提交接受的前缀。

回滚不只是截断 KV：还包括 GDN 矩阵、GDN 卷积、PLE token/卷积历史、QSA pooled keys/tail、
位置及隐藏状态。参考后端使用独立深拷贝检查正确性；设备后端应使用不可变 KV 页/尾部长度和
有限 recurrent 快照，不能把浅拷贝可变 device pointer 当作独立快照。
草稿已提交状态用实际目标残差重新补齐，避免拒绝路径遗留 speculative hidden。

当前只实现 greedy 接受规则。没有把 greedy 的一致性测试扩展宣称为随机采样的分布一致性。
参考 `verify` 逐 token 执行，没有 MTP 加速；原生后端需要一次窗口验证和设备端接受前缀处理。

## 原生后端下一步

1. 在目标 PPU/SDK 上建立设备内存、stream/event、有限搬运窗口和 packed artifact 生命周期。
   使用真实 QUactlize query 检查每个 shape 与字节布局，并记录实际峰值。
2. 接入 QUactlize dense/grouped 路径；实现 N=640 专家的 gate/up 原始字节拼接。
   完成 Q5_1 down 原始 reader，不能把它按 Q5_K 解释，不能无提示回落到 CPU。
3. 接 GDN prefill/decode，验证 `value_head % key_heads` 映射、FP32 状态、sigmoid 输出门。
   如果 provider 使用连续分组，先做明确排列或扩展，再与参考对齐。
4. 补齐 HC、PLE、512 专家 top-10、QSA 池化/选择/尾部、RoPE/门控和输出 head 原生算子。
5. 接 learned MTP 的设备执行与批量 target verify；按接受前缀提交状态，测接受率及净收益。
6. 先完成单卡，再做少量 PPU 的层切分或张量并行以及 PCCL 通信。对整个过程做数值和内存验收。

## 已有证据与限制

本地 Rust 单元与真实描述回归测试可重现；QUactlize C++ public header 与 Rust ABI 尺寸检查通过。
远端文件只读取有限前缀，完整模型权重未下载；描述 receipt 的 SHA256 不是完整权重验证结果。
本机 SDK 在 `cudaGetDeviceCount` 中报告缺少 `libhggcrt1.so`，没有 PPU kernel/吞吐测试结果。

## 参考

- [Flare / spark-local-inference](https://github.com/NVIDIA/spark-local-inference)：准备与执行的分工。
- [llama.cpp qwen4exp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen4exp.cpp)：GGUF 模型结构和状态语义。
- [Strata UD-Q4 文档](https://github.com/Niko1221/Strata/blob/ffa80d73133ea5c548aefe57889a16e8085a938f/docs/UNSLOTH_Q4.md)：量化组合与 PLE 放置。
- [Unsloth MTP 文档](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/README.md)：共享预测层制品。
