from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage, StorageContext, QueryBundle
from llama_index.core.node_parser import SentenceSplitter

#使用Llamaindex内置构建本地向量库
documents = SimpleDirectoryReader("/root/autodl-tmp/LlamaIndexrag/LlamaIndex/tcm-ai-rag/tcm-ai-rag/data/filtered", required_exts=[".txt"]).load_data()

index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=256)])

index.storage_context.persist(persist_dir='./doc_emb')
