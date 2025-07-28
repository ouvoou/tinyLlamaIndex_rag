import logging
import sys
import torch, os
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage, StorageContext, QueryBundle
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer

from llama_index.core.base.llms.types import ChatMessage
from collections import deque

##定义日志配置
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

##定义system prompt
SYSTEM_PROMPT = """你是一个优秀的中医AI助手.请回答我的专业性问题。"""
query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)

#调用本地LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path = "/root/autodl-tmp/LlamaIndexrag/LlamaIndex/tcm-ai-rag/tcm-ai-rag/models/Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto",
)

Settings.llm = HuggingFaceLLM(
    context_window = 4096,
    max_new_tokens = 512,
    generate_kwargs = {"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt = query_wrapper_prompt,
    tokenizer = tokenizer,
    model = model,
    # model_kwargs = {
    #     "trust_remote_code":True,
    #     "quantization_config": quantization_config,
    #     "device_map": "auto",
    # }
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="/root/autodl-tmp/LlamaIndexrag/LlamaIndex/tcm-ai-rag/tcm-ai-rag/models/BAAI/bge-base-zh-v1.5"
)

##从本地向量库中获取index
storage_context = StorageContext.from_defaults(persist_dir="/root/autodl-tmp/LlamaIndexrag/LlamaIndex/tcm-ai-rag/tcm-ai-rag/doc_emb")
index = load_index_from_storage(storage_context)

##构建查询引擎
memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
query_engine = index.as_chat_engine(chat_mode="context", memory=memory, streaming=True)


print("\n💬 输入你的中医问题，输入 'exit' 或 'quit' 退出：")

while True:
    user_input = input("用户：")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("程序已退出。")
        break
    response = query_engine.chat(user_input)
    print("LLM:", str(response))
