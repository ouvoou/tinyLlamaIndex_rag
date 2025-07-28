from modelscope import snapshot_download

# model_id 模型的id
# cache_dir 缓存到本地的路径
model_dir = snapshot_download(model_id="BAAI/bge-base-zh-v1.5", cache_dir="/root/autodl-tmp/LlamaIndexrag/LlamaIndex/tcm-ai-rag/tcm-ai-rag/models")
model_dir = snapshot_download(model_id='Qwen/Qwen1.5-1.8B-Chat', cache_dir="/root/autodl-tmp/LlamaIndexrag/LlamaIndex/tcm-ai-rag/tcm-ai-rag/models")