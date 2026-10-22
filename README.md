# ETS Agent Demo

面向预计应答时间分桶预测的自动化实验 Agent。这个仓库是脱敏后的合成数据演示版本，用于展示研究编排与实验工程能力。

**这里不包含生产数据、真实订单、内部数据接口、公司网络地址、历史实验成绩、训练权重、原始 Notebook、聊天记录或运行环境。所有示例区域、数据和参考阈值均为人工构造，不代表实际业务效果。**

## 功能

- 数据画像、误差切片、特征重要性和冗余诊断。
- 大模型提出结构化特征配方，执行训练、评估、反馈和下一轮规划。
- 受限特征 DSL：表达式、历史/滚动聚合、缺失处理、编码和消融。
- XGBoost/MLP 对照、时间验证、多种子复验和概率融合。
- 实验预算、方案去重、候选筛选、特征谱系与研究记忆。
- Flask WebUI、流式任务进度、任务停止和实验详情。
- 冻结输入、源代码和配置；离线重训与预测一致性校验。
- 可选网页/论文研究与隔离代码工作区。

## 快速运行

使用 Python 3.12：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/run_feature_mining.sh --mock --smoke --no-agent --model-family xgb
bash scripts/run_feature_mining.sh --mock --smoke --no-agent --model-family mlp
bash run_webui.sh
```

浏览器访问 `http://127.0.0.1:5055`。界面默认使用合成数据。无 API 密钥时使用“仅跑 baseline”模式；命令行 `--smoke` 用更小规模验证流程。

固定人工配方可验证完整候选实验链路，无需外部 API：

```bash
bash scripts/run_feature_mining.sh --mock --smoke --rounds 1 --max-proposals-per-round 2 --proposal-file tests/smoke_proposals.json
```

## 可选大模型配置

复制 `.env.example` 为 `.env` 并设置文件权限为 `600`，填入自己的 API Key；运行 `scripts/configure_llm.py --help` 查看接口配置方法。仓库里的 `api.example.com` 和模型名均为无效占位符，必须自行配置。支持 Chat Completions 与 Responses 兼容接口。

```bash
cp .env.example .env
chmod 600 .env
python scripts/configure_llm.py --help
```

普通文本、流式响应与原生工具调用的支持程度取决于提供方。联网研究和 LLM 实验会调用自己配置的服务。

## 目录

| 目录 | 用途 |
| --- | --- |
| `agent/` | 实验编排、研究控制、提示词与记录 |
| `skills/` | 合成数据、特征、训练、评估、资料研究和复现 |
| `webui/` | Flask API 与原生 HTML/CSS/JavaScript 界面 |
| `configs/` | 人工示例配置和 API 占位模板 |
| `tests/` | 合成夹具和回归测试 |
| `scripts/` | 启动和配置工具 |

## 边界与评估

这是单用户研究演示服务。生产数据库适配器和解密实现已经移除；线上读取会明确拒绝。部分底层离线快照接口仅保留扩展结构，仓库不提供任何快照。

六桶边界示例为 1、5、10、20、30 分钟，评估包括准确桶、高估/低估及 H3 网格保序。普通特征搜索会复用测试窗口筛选候选，不能将该结果当作独立泛化效果；模型研究提供时间验证及多种子复验。所有合成数据成绩只证明演示流程能够执行。

PDF 正文提取需 Linux 与 Poppler（`pdftotext`）；macOS 保留原文件并提示不可提取，网页正文不受影响。

隔离代码执行依赖 Linux 和相应沙箱配置；缺少隔离能力时拒绝执行。运行产生的模型、实验、日志、研究材料和本地数据均由 `.gitignore` 排除。复现仅覆盖已冻结的特征输入及后续训练/评估，不会重新取数或重新调用 LLM。

## 测试

```bash
python -m unittest discover -s tests -p 'test_*.py'
node --test tests/*.js
```

发布副本的具体脱敏范围见 [SANITIZATION.md](SANITIZATION.md)。
