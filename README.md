# Etude LLM项目(v0.1)

## 项目介绍

"Etude"一词源自法语，原意为"研习曲"或"练习曲"，在音乐领域特指为提高演奏技巧而创作的短小精悍的乐曲。在本项目中，"Etude LLM"这一名称寓意着这是一个用于学习、练习和探索语言模型技术的平台，就像音乐练习曲那样，通过精心设计的结构和渐进式的学习过程，帮助使用者掌握语言模型的核心技术和实现方法。

Etude LLM是一个基于Transformer架构的轻量级语言模型项目，旨在提供一个简单、可扩展的框架，用于训练和微调小型语言模型。该项目实现了从基础模型训练到监督微调（SFT）的完整流程，并提供了推理接口，使用户能够与模型进行交互式对话。

## 项目特点

- **轻量级架构**：基于Transformer架构实现的小型语言模型，参数量适中，便于在普通GPU设备上训练和部署
- **完整训练流程**：支持基础预训练和监督微调（SFT）两个阶段
- **交互式推理**：提供简洁的对话接口，支持用户与模型进行自然对话
- **模块化设计**：代码结构清晰，便于扩展和修改

## 项目结构

```
Etude LLM/
├── UI/                   # 用户界面相关文件
├── big_json/             # 大型训练数据集
├── full_sft.py           # 监督微调实现
├── inference.py          # 模型推理接口
├── json/                 # 基础训练数据
├── jsonl_sft/            # 监督微调数据
├── model_train.py        # 模型架构和基础训练实现
├── tool/                 # 工具脚本
├── weight/               # 模型权重保存目录
└── 一键炼丹模式.py        # 快速训练启动脚本
```

## 核心组件

### 模型架构 (model_train.py)

该项目实现了一个基于Transformer的语言模型，主要组件包括：

- **多头自注意力机制**：实现了多头注意力机制，支持并行处理不同位置的注意力信息
- **前馈神经网络**：用于特征转换和非线性映射
- **位置编码**：通过位置嵌入表示序列中的位置信息
- **层归一化**：用于稳定训练过程

### 监督微调 (full_sft.py)

实现了基于对话数据的监督微调（SFT）流程：

- **对话数据处理**：支持加载和处理对话格式的训练数据
- **渐进式微调**：支持多阶段的模型微调
- **参数优化**：使用较小的学习率和适当的优化器配置，保证微调效果

### 推理接口 (inference.py)

提供了用于模型推理和交互式对话的接口：

- **模型加载**：加载训练好的模型权重
- **文本生成**：实现基于概率采样的文本生成
- **交互式对话**：提供简单的命令行交互界面

## 快速开始

### 环境配置

```bash
# 安装依赖
pip install torch tiktoken tqdm
```

### 基础训练

```bash
python model_train.py
```

### 监督微调

```bash
python full_sft.py
```

### 模型推理

```bash
python inference.py
```

## 模型配置

模型的主要配置参数在`GPTConfig`类中定义：

- **block_size**: 文本的最大长度 (512)
- **batch_size**: 批处理大小 (8)
- **n_layer**: Transformer层数 (6)
- **n_head**: 注意力头数 (12)
- **n_embd**: 嵌入维度 (768)
- **vocab_size**: 词表大小 (50257)

## 推荐训练集

-语义学习:https://www.modelscope.cn/datasets/modelscope/SkyPile-150B
-对话微调（文件名称带有sft）：https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files

## 未来工作

- 将在v0.2引入Mixture of Experts (MoE)机制，提高模型容量和推理效率
- 升级现有架构，优化模型性能和训练稳定性
- 在v0.2或v0.3加入LoRA (Low-Rank Adaptation)微调技术，实现更高效的参数高效微调
- 将会在v0.5引入提供Web界面支持
- 支持多语言和更大规模的预训练数据集

## 开源代码引用

本项目在开发过程中参考或使用了以下开源项目的代码：

## nanoGPT https://github.com/karpathy/nanoGPT

## LLMs-Zero-to-Hero https://github.com/bbruceyuan/LLMs-Zero-to-Hero

## MiniMind https://github.com/jingyaogong/minimind


根据开源许可要求，我们保留了原始代码中的版权声明，并在此明确致谢这些项目的贡献。

## 致谢

感谢所有开源语言模型社区的贡献，本项目从中获得了许多灵感和参考。
因为我个人水平和精力原因，部分代码会使用开源项目中的代码，代码写的烂见谅。
