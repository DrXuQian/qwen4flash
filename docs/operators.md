# 算子接入

Rust 持有模型语义、制品和执行状态。原生库负责计算；通过窄 C ABI 调用，不在稳态引入 Torch/Python。
QUactlize 的准备阶段可能使用其 JIT 工具；这属于启动阶段，不能混入捕获后的 enqueue。

| 来源 | 当前依据 | 接入位置与边界 |
|---|---|---|
| [QUactlize](https://github.com/DrXuQian/QUactlize) | 本地 develop `4fb7ce037f36768bf900f1c33d606f7eeaf87f80` 的公共 API | packing + dense；不是旧 main 的接口假设 |
| [actlizeLA](https://github.com/DrXuQian/actlizeLA) | `66ec165aae1fa2ae1e780e413c3618a0914c6ad2` | `gdn_wy_forward_residual_full_chunk` 原生入口；不加载 Torch extension |
| [t-head/pla](https://github.com/t-head/pla) | GDN/KDA decode、MTP 相关实现清单 | 候选 decode 路径；KDA 支持不等于已验证 Qwen GDN |
| [t-head/flash-attention-for-sail](https://github.com/t-head/flash-attention-for-sail) | 注意力候选 | QSA 的 pooled selection 和 causal tail 仍需模型专用适配 |
| [t-head/pccl](https://github.com/t-head/pccl) | 多设备通信 | 少量 PPU 执行阶段接入 |

## QUactlize 合同

- 外部激活/输出为 F32，计算精度在创建 context 时显式选 F16/BF16，不由指针猜测。
- 通过 `kpack_canonical_arrangement_v1` 和 `kpack_sizes_for_arrangement_v1` 获取布局与大小。
- `dense_supported_v1` 的 M=0 用于整个合法正 M 域的权重准入，实际 plan 必须 M>0。
- `dense_prepare_v1` 在图捕获前完成；`dense_run_v1` 仅 enqueue。
- context 拥有 plan/module，权重必须保持不可变，buffer、stream 和 library 必须覆盖异步使用期。
- Q5_1 不属于现有 K-quant dense 合同，需要独立原生 reader。
- 专家 FFN=640。gate/up 按每个专家拼接成 1280 行；现有 pair pack API 要求原始 N 自身对齐，
  不能直接以 N=640 调用并假定成功。拼接 adapter 尚待实现。
- Router 为 512 选 10，不能继承旧 helper 的 256 专家或固定少量 routed rows 限制。

## actlizeLA 合同

Q/K/V/beta 为 BF16，gate 的 BF16/F32 由参数指定，gate 为自然对数衰减；状态为 F32 K×V。
固定维度 128，chunk=64，按实际 token 数分配尾块。Rust 绑定计算 W、state snapshots、VN 和
gate cache 的 workspace 字节数。Qwen 的 head 复用是 `vh % Hk`，接入连续分组 provider 时需排列。
这份 ABI 本身不完成模型的输入投影、卷积、归一化或 head 排列。

## t-head 其余公开仓库的用途审查

已浏览组织清单，按所需算子选择接入点，不把整个组织作为构建依赖：

- 计算与编译候选：`DeepGEMM-for-sail`、`flash-linear-attention-for-sail`、`FlashMLA-for-sail`、
  `flashinfer-for-sail`、`MSA-for-sail`、`actlize`、`triton-for-sail`、`tilelang-for-sail`、`xformers-for-sail`。
- 通信/测量/示例：`gloo-for-sail`、`pccl-tests`、`sailbandwidth`、`hggc-samples`、`cmake-hgcc`。
- 整体框架参考：`sglang-for-sail`、`vllm-sail`、`pytorch-for-sail`；当前工程不依赖这些运行时。
- 工具/基础设施：`sailify`、`sail-ir-formatter`、`criu-for-sail`、`ppu-scheduler-action`、`ci-infra`、`.github`。

组织目录或上游 README 中列出某个 CUDA 算子，不代表它在目标 PPU/SDK/shape 上已经可用。
每个最终选择都需原生编译、数值对齐、调用轨迹和实际性能证据。第三方库保留各自许可证。
