# LeanGraphSearch 简明指南

本匿名副本只保留最终论文涉及的实验：

| 任务 | 实例数 | 模型 |
|---|---:|---|
| MathlibQR | 810 条查询，171 个声明 | Qwen3 嵌入与重排 |
| MathlibMPR | 69 | Gemini 3.1 Pro / Claude Sonnet 5 |
| FATE-H | 100 | Gemini 3.1 Pro |
| MathlibMPR-Prop | 50 | Gemini 3.1 Pro |
| Lean-IMO-Bench | Basic 30 + Advanced 30 | Gemini 3.1 Pro |

MathlibMPR 的 69 是测试实例数；每个子查询检索及过滤的上限是 50。
论文参数与原工作目录的差异详见 [PAPER_SETTINGS.md](PAPER_SETTINGS.md)。

从解压目录运行 [README.md](README.md) 中的 CPU 检查，再按
[REPRODUCING.md](REPRODUCING.md) 安装完整依赖、下载公开资产、构建图、
启动统一检索服务，最后运行三个实验入口：

- `scripts/reproduce_mathlibqr.py`：MathlibQR 标准/图检索。
- `scripts/reproduce_premise_costed.py`：MathlibMPR 两种模型的标准/图检索。
- `scripts/reproduce_proving.py`：其余基准的五种证明条件与失败记忆开关。

证明统一使用 32 次编译、31 次检索动作、63 次模型调用；初始推理检索单独
计预算。旧的 32/8 协议、MiniF2F 框架、独立 Qwen 证明器、多路线实验、
优化搜索、历史分析报告和重复运行脚本已删除。

为了尽量自包含，随包保留小型基准输入、必要测试与第三方许可证。大型模型、
cuVS 索引和 Lean 工具链需另行准备；模型服务需要使用者自己的凭证。
原始实验结果、日志、生成证明、个人配置、Git 历史和论文文件不在包内。
新运行会产生自己的输出，不应直接混入此匿名补充材料。

完整逐文件清单见 [CONTENTS.md](CONTENTS.md)。
