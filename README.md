# TCM-AI-RAG

本项目是一个基于 LlamaIndex + 本地 Qwen 模型的中医领域 RAG 问答系统，支持多轮对话。

## ✨ 特性

- 📚 加载本地向量索引进行 Retrieval-Augmented Generation（RAG）
- 🧠 使用 ChatMemoryBuffer 实现上下文记忆（支持多轮对话）
- 💡 本地运行，无需联网，无需 HuggingFace Token

## 🚀 快速开始

```bash
git clone https://github.com/yourname/tcm-ai-rag.git
cd tcm-ai-rag
pip install -r requirements.txt
python src/cli_chat.py
