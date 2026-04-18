# 基于 AI 的电磁兼容研究

> 这是一个给纯新手看的中文 README。
> 你不用有编程基础，也不用懂很多英文。
> 这份文档的目标只有两个：
> 1. 让你能把项目跑起来
> 2. 让你能向老师说明”我现在在做什么”

---

## 1. 先看最重要的结论

这个项目是一个**基于 AI 的电磁兼容 (EMC) 信号故障诊断实验项目**。

核心研究内容是：**将三相电力系统的电气信号转化为自然语言文本，输入大语言模型 (Qwen2.5-7B) 进行故障分类**。

它做的事情是：

1. 读取三相电力系统的电流/电压信号数据（7861 个样本，6 类故障）
2. 通过物理驱动的特征工程将 6 维原始信号扩展至 25 维
3. 将数值特征序列化为自然语言文本
4. 使用 QLoRA 微调 Qwen2.5-7B-Instruct 进行 6 分类故障识别
5. 评估模型效果并生成图表和报告

**当前最佳实验结果：**

| 指标 | 数值 |
|------|------|
| Macro-F1 | 0.771 |
| Accuracy | 0.806 |
| 最佳验证 F1（v6 实验） | 0.825 |

如果你只是为了给老师看项目结果、写论文，那么你最应该先看的是：

- `docs/final_report.md` — 完整毕业论文报告（含实验数据、图表、分析）
- `docs/figures/` — 所有实验可视化图片

---

## 2. 你可以怎样向老师介绍这个项目

如果老师问你”这个项目是干什么的”，你可以直接这样说：

> 这个项目研究的是基于大语言模型的三相电力系统故障诊断。
> 我把三相电流和电压信号提取物理特征后，转化为自然语言文本，再用 Qwen2.5-7B 大模型通过 QLoRA 微调来识别 6 种不同的故障类型。
> 核心创新点是提出了”电气信号 → 语义文本 → LLM 分类”的新型诊断流程。

如果老师继续问”你具体做了什么”，你可以继续这样说：

> 我的流程是：数据预处理（去除标签泄漏列）→ 物理驱动特征工程（6 维扩展到 25 维）→ Fisher 判别比排序筛选 → 文本序列化 → Qwen2.5-7B QLoRA 微调 → 评估（Macro-F1 指标）→ 生成可视化报告。
> 最终在测试集上取得 Macro-F1 = 0.771，验证集最佳 F1 达到 0.825。

如果老师问“你现在跑的这一步是什么”，你可以按下面这张表回答：

| 你正在执行的命令 | 你可以怎么解释 |
| --- | --- |
| `prepare` | 我在整理原始数据，让模型后面能读懂 |
| `extract-features` | 我在从原始数据里提取有用信息，帮助模型判断 |
| `train` | 我在训练模型，让模型学会分类 |
| `evaluate` | 我在检查模型好不好，计算准确率和 F1 等指标 |
| `visualize` | 我在生成图表，方便老师看实验结果 |
| `export-report` | 我在导出实验报告，方便论文写作和提交 |

---

## 3. 这个项目现在的核心结论

当前仓库已经整理成适合交付和写论文的状态。

目前最重要的实验结论是：

- 论文主线数据集：**三相电力系统故障检测数据集**（7861 样本，6 类故障）
- 论文主任务：**fault_code 6 分类**（正常 + 5 种故障类型）
- 模型：**Qwen2.5-7B-Instruct + 4bit QLoRA 微调**
- 当前最佳测试集结果：**Macro-F1 = 0.771, Accuracy = 0.806**
- 验证集最佳结果（v6 实验，feature_limit=20）：**val_f1 = 0.825**
- 4 类故障识别优秀（平均 F1 = 0.933），2 类接地/非接地故障存在混淆（已定位根因为零序特征截断）
- 详细实验报告和可视化图表见 `docs/final_report.md`

简单说：

> 这个项目验证了大语言模型在结构化电气信号故障诊断中的可行性，提出了”电气信号 → 语义文本 → LLM 分类”的创新技术路线。

---

## 4. 仓库里每个重要目录是干什么的

你不需要一开始就看全部代码。先认识这些目录就够了。

- `src/emc_diag/`
  - 核心程序代码
  - 你运行的命令，最后都会调用这里面的逻辑

- `configs/`
  - 配置文件目录
  - 你可以把它理解成“实验说明书”
  - 不同的 `.yaml` 文件代表不同实验方案

- `data/`
  - 数据目录
  - 当前论文主数据已经在这里：
  - `data/Cognitive Radio Spectrum Sensing Dataset.csv`

- `artifacts/`
  - 实验运行后生成的结果目录
  - 包括中间结果、图表、报告等

- `artifacts/reports/paper-handoff-package/`
  - 最终交付包
  - 最适合直接给老师看

- `docs/`
  - 补充说明文档
  - 包括运行说明、数据集说明等

- `tests/`
  - 测试代码
  - 用来检查项目是不是还能正常工作

---

## 5. 如果你只想快速交作业，应该先看哪里

最简单的做法：

1. 打开 `docs/final_report.md` — 完整毕业论文报告
2. 查看 `docs/figures/` — 所有实验可视化图片：
   - `dataset_summary.png` — 数据集类别分布
   - `feature_importance.png` — 特征重要性排序
   - `confusion_matrix.png` — 混淆矩阵
   - `metrics_overview.png` — 整体性能指标
   - `per_class_metrics.png` — 逐类性能分析
3. 如果需要原始结果数据，看：
   - `artifacts/reports/paper-handoff-package/`

这一步你可以这样解释：

> 我先看项目已经生成好的最终交付材料，确认整体实验结果、图表和报告内容，再决定是否重新跑实验。

---

## 6. 运行前你要知道的几件事

### 6.1 这个项目需要什么环境

这个项目要求：

- `Python 3.12`
- `uv`

请注意：

- **不要依赖系统里乱七八糟的 Python 版本**
- 这个项目明确要求 `Python 3.12`
- 如果你的电脑里有 `Python 3.14`，也不要拿它直接跑这个项目

### 6.2 `uv` 是什么

你可以把 `uv` 理解成一个“帮你管理 Python 环境和依赖的工具”。

它的作用是：

- 帮你安装项目需要的包
- 帮你自动创建虚拟环境
- 帮你用正确的 Python 版本运行命令

对新手来说，最重要的一点是：

> 以后尽量用 `uv run python ...`，不要直接乱敲 `python ...`

因为有些电脑里直接输入 `python` 可能会报错，或者调用到错误版本。

### 6.3 第一次运行时看到 `.venv` 不要慌

第一次运行 `uv` 相关命令时，它可能会自动创建一个 `.venv` 文件夹。

这不是报错，这是正常现象。

你可以把 `.venv` 理解成：

> 这个项目专属的小环境，目的是避免和电脑里别的 Python 项目互相打架。

---

## 7. 最推荐的新手使用方式

### 方式 A：只查看现成结果

适合你现在就要交材料，或者只想先讲清楚项目。

你什么都不用跑，直接看：

- `artifacts/reports/paper-handoff-package/`

### 方式 B：自己重新跑一次完整流程

适合老师要求你：

- 亲自演示命令
- 说明实验流程
- 展示模型是怎么跑出来的

如果你要这样做，请看下面的详细步骤。

### 方式 C：直接使用一键命令

如果你不想自己手动记 `run-id`，也不想在很多目录之间来回找结果，直接运行：

```bash
uv run python -m emc_diag quickstart
```

如果你想强制使用 CPU，运行：

```bash
uv run python -m emc_diag quickstart --device cpu
```

这条命令会自动完成：

- 数据准备
- 特征提取
- 模型训练
- 评估与图表生成
- 报告整理

并把最适合新手查看的结果固定放到：

- `artifacts/latest_prepared/`
- `artifacts/latest_run/`

推荐你优先打开：

- `artifacts/latest_run/summary.md`
- `artifacts/latest_run/figures/`

---

## 8. 从零开始运行一次完整流程

下面默认你已经打开终端，并且当前路径在这个项目根目录。

也就是说，你打开终端后应该进入这个目录：

```bash
/Users/orange/Desktop/electronic-diagnoise
```

如果你不确定自己是不是在这个目录，可以运行：

```bash
pwd
```

如果输出不是项目目录，你就先切换过来。

---

## 9. 第一步：安装依赖

运行：

```bash
uv sync --dev
```

### 这一步在做什么

这一步是在安装项目需要的所有依赖包。

包括但不限于：

- `PyTorch`
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `pytest`

### 为什么要做这一步

因为没有这些包，后面的训练、评估、出图都跑不了。

### 你可以怎么向老师解释

> 我先同步项目依赖，确保实验环境一致，这样后面的结果才有可重复性。

### 这一步正常时可能看到什么

你可能看到类似下面的英文：

```text
Creating virtual environment at: .venv
Installed ...
```

这表示：

- 项目环境被创建了
- 依赖包已经在安装

不是报错。

### 更推荐的新手最短命令

如果你只是想直接跑通并拿到结果，不想手动一步一步执行，可以在完成 `uv sync --dev` 之后直接运行：

```bash
uv run python -m emc_diag quickstart
```

---

## 10. 第二步：准备数据

运行：

```bash
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_cnn.yaml
```

### 这一步在做什么

这一步是在读取原始数据，并把它整理成后续步骤能直接使用的格式。

你可以把它理解成：

> 先把“原始材料”收拾整齐，再交给模型使用。

### 为什么要做这一步

因为原始数据通常不能直接拿来训练，需要先整理、检查、转换。

### 这一步和论文主线有什么关系

这一步使用的是主配置文件：

- `configs/cognitive_radio_presence_cnn.yaml`

这个配置对应的是：

- 主数据集：`Cognitive Radio Spectrum Sensing Dataset`
- 主任务：`PU_Presence`

### 你可以怎么向老师解释

> 我现在在做数据准备。也就是把原始信号数据整理成统一格式，为后面的特征提取和模型训练做准备。

---

## 11. 第三步：提取特征

运行：

```bash
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_cnn.yaml
```

### 这一步在做什么

这一步是在从原始数据中提取更有代表性的特征。

“特征”你可以简单理解成：

> 数据里对判断结果有帮助的信息。

例如：

- 某些统计量
- 某些变化趋势
- 某些结构化信号特征

### 为什么要做这一步

因为模型不是人，它看不懂“意义”，只能看数字。  
特征提取的目的，就是把原始数据变成更适合模型学习的形式。

### 你可以怎么向老师解释

> 我现在在做特征提取，把原始信号转换成更适合模型学习的信息表示。

---

## 12. 第四步：训练模型

运行：

```bash
uv run python -m emc_diag train --config configs/cognitive_radio_presence_cnn.yaml --device auto
```

### 这一步在做什么

这一步是在真正训练模型。

模型会根据前面处理好的数据，学习如何判断：

- `PU_Presence`

也就是：

> 主用户是否存在。

### `--device auto` 是什么意思

它表示自动选择计算设备：

- 优先用 `CUDA`（NVIDIA 显卡）
- 其次用 `MPS`（苹果芯片）
- 都没有就用 `CPU`

简单理解：

> 有显卡就尽量用显卡，没有就用电脑普通处理器。

### 这一步可能会花多久

看你的电脑性能。

- 有显卡：通常更快
- 没显卡：可能明显更慢

### 训练结束后你最应该记住什么

训练结束时，终端会输出一个 `run_dir`。

类似：

```text
[train] completed run_dir=/.../artifacts/runs/run-时间戳-实验名
```

这个目录非常重要，因为后面的评估、画图、导出报告都要用它。

### 你可以怎么向老师解释

> 我现在在训练模型，让模型根据数据学习分类规则。训练完成后会生成一个运行目录，后续评估和图表都会从这个目录读取结果。

### 12.1 如果老师要求必须体现“开源大模型专项训练”

你可以直接运行下面这条主线：

```bash
uv sync --dev --group llm
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_qwen_qlora.yaml
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_qwen_qlora.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_presence_qwen_qlora.yaml --device cuda
```

这条配置现在默认会走：

- 开源模型：`Qwen/Qwen2.5-7B-Instruct`
- 训练方式：`4bit QLoRA`
- 主任务：`PU_Presence`
- 文本提示：`hybrid` 特征工程会附带认知无线电物理统计量（如能量检测 trace、MME/RLE 等）；`tabular_matrix_to_texts` 会把 `phys_mme`、`cross_su_cov_*`、`*_mme_x_*` 等列名转成英文可读名称，便于大模型理解谱感知语义

如果你想换成 `DeepSeek`，使用：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_deepseek7b_qlora.yaml
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_deepseek7b_qlora.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_presence_deepseek7b_qlora.yaml --device cuda
```

如果你的服务器只能连国内网络，就保留这两个环境变量再运行训练命令。
这里使用的是 `hf-mirror.com`。

你可以这样向老师解释：

> 现在这个项目不只是传统机器学习。  
> 我把电磁/无线信号特征转换成结构化文本，再用开源大模型 `Qwen` 或 `DeepSeek` 做针对 `PU_Presence` 的专项微调。  
> 这样更能体现这是“基于 AI 的电磁兼容研究”，而且训练产物会保存成 LoRA adapter，方便复现和对比。

这里要注意：

- `4090D 24GB` 不适合做 `7B` 全量微调
- 现在采用的是更现实的 `7B + 4bit QLoRA`
- 这是当前服务器最稳妥、最容易跑通的方案

---

## 13. 第五步：评估模型效果

把上一步输出的真实 `run_dir` 填进去，运行：

```bash
uv run python -m emc_diag evaluate --run-dir artifacts/runs/<run-id>
```

请把 `<run-id>` 替换成你真实的运行目录名。

例如：

```bash
uv run python -m emc_diag evaluate --run-dir artifacts/runs/run-20260401-xxxx
```

### 这一步在做什么

这一步是在计算模型表现。

通常会涉及：

- `accuracy`：准确率
- `f1`：综合评价指标

### 为什么要做这一步

因为训练完不等于真的好。  
必须评估，才能知道模型到底表现如何。

### 你可以怎么向老师解释

> 我现在在做模型评估，计算准确率和 F1 等指标，用来判断这个模型有没有达到实验要求。

---

## 14. 第六步：生成图表

运行：

```bash
uv run python -m emc_diag visualize --run-dir artifacts/runs/<run-id> --theme paper-bar
```

### 这一步在做什么

这一步是在生成实验图表。

例如可能包括：

- 指标图
- 混淆矩阵
- 特征重要性图
- 训练曲线

### 为什么要做这一步

因为老师和论文通常不会只看一堆数字，还需要图来说明问题。

### 你可以怎么向老师解释

> 我现在在把模型结果可视化，这样可以更直观地展示分类效果和实验现象。

---

## 15. 第七步：导出报告

运行：

```bash
uv run python -m emc_diag export-report --run-dir artifacts/runs/<run-id> --format md
```

### 这一步在做什么

这一步是在把本次运行结果整理成报告。

`--format md` 表示导出成 Markdown 文档格式。

Markdown 你可以简单理解成：

> 一种适合写技术文档和报告的轻量文本格式。

### 为什么要做这一步

因为最后要交材料、写论文、汇总结果，就不能只停留在终端输出上。

### 你可以怎么向老师解释

> 我现在在导出实验报告，把本次运行的结果、图表和指标整理成可阅读的文档。

---

## 16. 第八步：检查结果文件是否真的生成了

一个正常的运行目录里，至少应该重点检查这些文件：

- `metrics.json`
- `predictions.csv`
- `summary.md`
- `figures/metrics_overview.png`
- `figures/confusion_matrix.png`

这些文件通常位于：

- `artifacts/runs/<run-id>/`

### 这一步在做什么

这一步是在确认：

- 模型真的跑完了
- 结果真的保存了
- 图和报告真的生成了

### 你可以怎么向老师解释

> 我不只是看终端输出，我还会检查运行目录里是否生成了指标文件、预测文件、总结文件和图表，确保实验结果真正落盘保存。

---

## 17. 一套可以直接复制的完整命令

如果你只想按顺序跑一遍，可以直接照着执行。

```bash
uv sync --dev
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_cnn.yaml
uv run python -m emc_diag extract-features --config configs/cognitive_radio_presence_cnn.yaml
uv run python -m emc_diag train --config configs/cognitive_radio_presence_cnn.yaml --device auto
uv run python -m emc_diag evaluate --run-dir artifacts/runs/<run-id>
uv run python -m emc_diag visualize --run-dir artifacts/runs/<run-id> --theme paper-bar
uv run python -m emc_diag export-report --run-dir artifacts/runs/<run-id> --format md
```

注意：

- `train` 跑完以后，你要先记下真正生成的 `run-id`
- 后面 3 条命令都要把它替换进去

---

## 18. 这个项目里“推理”到底指什么

很多老师或者同学会把“把模型跑起来”统称为“推理”。

但严格来说，这个项目更完整的流程是：

1. 数据准备
2. 特征提取
3. 模型训练
4. 结果评估
5. 可视化
6. 报告导出

所以如果老师说“你跑一下推理”，你可以礼貌地解释：

> 这个仓库不只是单步预测，它更像一个完整实验流水线。  
> 我现在跑的是从数据准备到模型评估和报告导出的完整流程。

---

## 19. 最常用的配置文件说明

下面这些配置文件最常用：

### 电气故障诊断主线（论文核心）

- `configs/electrical_fault_fault_code_qwen_local.yaml`
  - **论文主线配置**
  - 三相电力系统 6 分类故障诊断
  - Qwen2.5-7B-Instruct + 4bit QLoRA
  - 需要 GPU（推荐 RTX 4090D 24GB）

### 认知无线电辅助实验

- `configs/cognitive_radio_presence_cnn.yaml`
  - 论文主任务 `PU_Presence`
  - 最适合新手先跑这个

- `configs/cognitive_radio_presence_qwen_qlora.yaml`
  - 开源大模型主线配置
  - 默认使用 `Qwen2.5-7B-Instruct + 4bit QLoRA`
  - 适合向老师展示“已经在做开源模型专项训练”

- `configs/cognitive_radio_presence_qwen7b_qlora.yaml`
  - 显式的 `Qwen 7B` 配置
  - 适合单独保留 Qwen 实验记录

- `configs/cognitive_radio_presence_deepseek7b_qlora.yaml`
  - 显式的 `DeepSeek 7B` 配置
  - 适合做开源模型对比

- `configs/cognitive_radio_presence_cnn_cv.yaml`
  - 更适合做交叉验证和消融分析

- `configs/cognitive_radio_burst_duration.yaml`
  - 一个表现比较高的辅助任务

- `configs/cognitive_radio_frequency_band.yaml`
  - 高分辅助任务，但不建议作为论文主创新点

- `configs/cognitive_radio_drift_type.yaml`
  - 中等难度任务，深度学习模型表现更能体现价值

如果你完全不知道先跑哪个：

> 先看 `docs/final_report.md` 了解完整实验结果

如果老师要求现场演示 LLM 训练：

> 跑 `configs/electrical_fault_fault_code_qwen_local.yaml`（需要 GPU 服务器）

如果只想在本地 CPU 快速演示：

> 先跑 `configs/cognitive_radio_presence_cnn.yaml`

---

## 20. Windows 用户最简单的做法

如果你是 Windows 用户，并且想尽量少输命令，可以尝试 PowerShell 方式：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1 -ProjectRoot "." -DatasetSource cognitive -RunPipeline
```

### 这条命令会做什么

它会尽量帮你自动完成：

- 检查并安装 `uv`
- 同步依赖
- 下载 Cognitive Radio 数据集
- 运行 `prepare`
- 运行 `extract-features`
- 运行 `train`

### 适合什么人

适合：

- Windows 新手
- 想先把主流程跑起来的人

### 但你要知道

这个仓库当前已经自带主数据文件，所以如果你本地已经有：

- `data/Cognitive Radio Spectrum Sensing Dataset.csv`

那么你不一定非要重新下载。

---

## 21. 常见英文词，直接看人话解释

- `dataset`
  - 数据集
  - 就是一整份实验数据

- `config`
  - 配置文件
  - 就是告诉程序“这次实验怎么跑”

- `feature`
  - 特征
  - 就是从原始数据里提取出来、对判断有帮助的信息

- `train`
  - 训练
  - 就是让模型学习

- `evaluate`
  - 评估
  - 就是检查模型学得怎么样

- `visualize`
  - 可视化
  - 就是把结果画成图

- `report`
  - 报告
  - 就是把结果整理成文档

- `run_dir`
  - 一次实验的结果目录
  - 就是“本次运行的所有输出文件都放在哪”

- `artifacts`
  - 产物目录
  - 就是程序运行后生成出来的结果文件

---

## 22. 常见问题

### 问题 1：终端里提示 `python: command not found`

原因：

- 你的系统里没有正确配置 `python`
- 或者当前命令没走项目环境

解决方法：

- 不要直接写 `python ...`
- 改用：

```bash
uv run python ...
```

### 问题 2：第一次运行时自动创建 `.venv`

这是正常现象，不是坏了。

### 问题 3：训练特别慢

原因可能是：

- 没有显卡
- 当前走的是 CPU

这是允许的，只是会慢一些。

### 问题 4：我找不到 `<run-id>`

你需要看 `train` 命令结束后的终端输出。  
训练完成后会打印真实的 `run_dir`。

### 问题 5：我是不是一定要重跑全部实验

不一定。

如果你的目的是：

- 写论文
- 给老师展示
- 看最终结果

那么优先看：

- `artifacts/reports/paper-handoff-package/`

### 问题 6：Windows 上出现 `OpenBLAS error: Memory allocation still failed`

这类报错通常不是数据集太大，而是 Windows 上底层数值库在启动时开了太多线程，结果在一开始分配内存时失败。

现在仓库里的 CLI 已经做了默认保护。  
如果你拉的是最新代码，先重新执行：

```bash
uv sync --dev
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_cnn.yaml
```

如果还是报同样的错，可以在 PowerShell 里先执行下面这些命令，再重新运行项目命令：

```powershell
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"
$env:BLIS_NUM_THREADS="1"
```

然后再运行：

```powershell
uv run python -m emc_diag prepare --config configs/cognitive_radio_presence_cnn.yaml
```

你可以这样向老师解释：

> 这是 Windows 底层数值库的线程初始化问题，不是实验逻辑本身出错。  
> 我通过限制数值线程数，避免库在启动阶段申请过多资源，然后项目就能继续正常执行。

---

## 23. 如果你要重建最终交付包

只有在你已经更新过最终报告材料，并且知道自己在做什么时，才建议做这一步。

运行：

```bash
PYTHONPATH=src uv run python scripts/build_paper_handoff.py
```

### 这一步在做什么

这一步是在重新生成最终交付包目录，也就是：

- `artifacts/reports/paper-handoff-package/`

如果你后面需要压缩包，可以在目录生成完成后，再手动把这个文件夹压缩成 `.zip`。

### 你要特别注意

这一步依赖更上游的论文素材目录。  
如果缺少类似下面这些源文件，它就会失败：

- `artifacts/reports/thesis-assets-final/...`

所以对新手来说，最稳妥的建议是：

> 如果仓库里已经有 `paper-handoff-package`，优先直接使用现成结果，不要一开始就尝试重建。

---

## 24. 这个仓库现在是什么状态

这个仓库已经被整理成**适合交付和论文写作**的状态。

当前特点是：

- 旧的无关运行目录已经清理过
- 本地主要保留了主数据集
- 最终交付材料已经保留在 `artifacts/reports/`

这意味着：

> 它现在更像一个“可交付实验仓库”，而不是一个随便乱试的新项目。

---

## 25. 老师检查时，你可以按这个顺序展示

推荐展示顺序：

1. 先说项目目标
   - 这是一个基于大语言模型的三相电力系统故障诊断研究项目

2. 再说主数据和主任务
   - 主数据集是三相电力系统故障检测数据集（7861 样本，6 类故障）
   - 主任务是 `fault_code` 6 分类（正常 + 5 种故障类型）

3. 再说你的方法
   - 物理驱动特征工程（6 维 → 25 维）
   - Fisher 判别比排序筛选
   - 文本序列化为自然语言
   - Qwen2.5-7B QLoRA 微调

4. 再展示结果
   - `docs/final_report.md` — 完整论文报告
   - `docs/figures/` — 实验可视化图表

5. 如果老师要求现场演示
   - 跑 `prepare`
   - 跑 `extract-features`
   - 跑 `train`
   - 说明后续会基于 `run_dir` 做评估、出图和导出报告

---

## 26. 最后给新手的直接建议

如果你现在很紧张，不知道从哪里开始，就照下面做：

1. 先打开 `artifacts/reports/paper-handoff-package/`
2. 先看已经生成好的报告和指标
3. 再运行 `uv sync --dev`
4. 然后按顺序跑：
   - `prepare`
   - `extract-features`
   - `train`
5. 记住训练产生的 `run_dir`
6. 再跑：
   - `evaluate`
   - `visualize`
   - `export-report`

如果你能把上面这几步说清楚，就已经足够向老师说明：

> 你不是在乱点命令，而是在完成一个完整的 AI 实验流程。
