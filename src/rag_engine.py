from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage, StorageContext, QueryBundle
from llama_index.core.memory import ChatMemoryBuffer
def build_rag_engine(index_path, llm, embed_model, token_limit=2000):
    Settings.llm = llm
    Settings.embed_model = embed_model

    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)

    ##构建查询引擎
    memory = ChatMemoryBuffer.from_defaults(token_limit=token_limit)
    query_engine = index.as_chat_engine(chat_mode="context", memory=memory, streaming=True)

    return query_engine