from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage, StorageContext, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def load_embed(model_path):
    Settings.embed_model = HuggingFaceEmbedding(
    model_name="/root/autodl-tmp/LlamaIndexrag/LlamaIndex/tcm-ai-rag/tcm-ai-rag/models/BAAI/bge-base-zh-v1.5"
    )

    return Settings.embed_model