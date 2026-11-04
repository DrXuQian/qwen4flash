# qwen4flash

面向 PPU 的轻量 Rust 推理框架，专门支持 Qwen3.8-Flash-Next 的
`qwen4exp` 架构、GGUF 权重和 learned MTP。参考 Flare 的准备阶段与运行阶段分离，
计算通过 QUactlize、actlizeLA 及适用的 t-head 算子接口接入。

**当前是可运行的开发原型，尚不能在 PPU 上运行完整大模型。**
已有数值参考执行器、共享 MTP 加载、完整状态回滚、真实模型的张量与容量校验，
以及原生算子 C ABI 绑定。原生计算图、若干模型专用算子、多卡执行和真实设备验证仍在开发中。
没有静默的 CPU 算子回退，也没有未经测量的 PPU 吞吐数字。

## 固定模型组合

同一仓库和 revision：

- 仓库：[`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66)。
- Revision：`38bb39ee97821de2c9009abb7e93950eec396e66`。
- 主模型：`UD-Q4_K_XL`，4 个分片，共 111,334,654,784 字节，1,224 个张量。
- MTP：`MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`，2,786,568,256 字节，32 个张量。
- MTP 复用主模型 `token_embd.weight` 和 `output.weight`，其余权重来自独立文件。
  校验 shared 标志、层数、宽度、词表、注意力/专家参数及每个张量形状，不重复装载共享权重。
- 本项目使用指定的 shared-Q8 MTP；Strata 的 Q2 专家 MTP 是另一套制品，不能混用容量估算。

权重不随代码分发。模型的许可证由原模型仓库提供。

## 运行

使用 Linux、Rust stable。参考执行不需要 Python、Torch、PPU SDK 或 llama.cpp。
本次本地验证使用 rustc 1.99.1；没有验证最低 Rust 版本。

```sh
cargo test --workspace --locked
cargo run --locked -- demo --tokens 1,3,2,4,7 --generate 20
```

`demo` 执行缩小的 GDN、QSA、PLE、MoE 和 MTP 网络。它使用确定性合成权重和 token ID，
用于检查数值和状态流转，不是语言模型生成能力或吞吐演示。

不下载大权重即可复现容量规划：

```sh
cargo run --locked -- inspect-index fixtures/target-index.json
cargo run --locked -- prepare fixtures/target-index.json \
  --metadata-only --mtp fixtures/mtp-index.json \
  --context 32768 --ram-gib 64 --vram-gib 96 --out artifacts/plan.json
```

`fixtures` 是从指定远端文件读取的张量描述，不含模型权重；tokenizer 内容已去除。
规划文件始终标明 `device_execution_admitted: false`，列出未完成的设备工作。
现有 CLI 参数决定运行配置；`configs/unsloth-ud-q4-k-xl.json` 是记录制品和部署目标的配置清单。

检查本地真实文件，或启动**非常慢的 CPU 数值参考**：

```sh
cargo run --release -- inspect /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
cargo run --release -- reference \
  /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp /models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --tokens 1,2,3 --generate 4 --context 128
```

`reference` 的输入输出是 token ID；目前没有文本 tokenizer、聊天模板、HTTP 服务或多模态接口。
它逐行解码量化权重，默认逐 token 验证 MTP，复制状态以检验回滚，不能用于性能评估。
没有在完整权重上完成 logits 对齐验证。

更新远端描述，使用 Python 标准库和 curl；只请求有上限的文件前缀，不下载完整权重：

```sh
python3 tools/inspect_remote_gguf.py --out artifacts/target.index.json
python3 tools/inspect_remote_gguf.py --directory MTP \
  --filename mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --out artifacts/mtp.index.json
```

## PPU 接入

`q4e-ppu` 动态加载显式指定的原生库，检查版本化结构布局，分别提供准备和 enqueue 接口。
QUactlize 的 dtype、canonical arrangement、实际打包大小和 workspace 由其公共 API 决定。
调用者负责设备内存、流、事件、生命周期和图捕获前预热。

```sh
cargo run -- probe --runtime /path/to/libcudart.so \
  --quactlize /path/to/libquactlize_ppu.so

g++ -std=c++17 -fsyntax-only native/abi_check.cpp \
  -I /path/to/QUactlize -I /path/to/QUactlize/quactlize/include
```

本地核对了 PPU SDK 2.1.1 和现有 PPU1.0 算子接口；实际部署代际仍需确认。
本机设备查询因缺少 `libhggcrt1.so` 失败，因此当前没有原生 kernel 或整模型性能验证结果。
PPU1.5/1.7 支持不能从 ABI 编译通过推断。

## 内存目标

主模型计算权重和 MTP 驻留 PPU、完整量化 PLE 表驻留 RAM：**主机 RAM 目标为 64 GiB**。
PLE 精确大小为 28,800,138,240 字节，约 26.82 GiB；按分块加载设计，不保留 CPU 专家镜像。
32K、单会话、3 个 draft token 的当前规划为约 39.32 GiB 主机预算和约 84.85 GiB 设备预算。
后者包含 4 GiB 预留 workspace，但尚未计入实际 QUactlize 打包变化和完整计算图峰值，不能作为装载成功的证明。

这套预算适用于正在实现的 PPU 后端。CPU `reference` 会访问专家权重和复制状态，
不满足这个稳态 RAM/性能目标。128 GiB 可增加开发和缓存余量，当前没有证据证明它会提高显存驻留方案的稳态吞吐。
详见 [容量假设](docs/memory.md)、[实现状态与下一步](docs/status.md)、[算子接入](docs/operators.md)。

## 验证

```sh
cargo fmt --all --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

测试覆盖错误/截断 GGUF、真实分片与 MTP 描述、量化格式关键位序、分块加载边界、
MTP 全接受/拒绝/EOS/失败事务，以及小模型的普通解码与 MTP 解码及后续状态一致性。
这些测试不能替代完整模型对齐、PPU 数值测试和性能测试。
