from src.llm_loader import load_llm
from src.embed_loader import load_embed
from src.rag_engine import build_rag_engine
def main():
    llm = load_llm("/root/autodl-tmp/LlamaIndexrag/LlamaIndex/tcm-ai-rag/tcm-ai-rag/models/Qwen/Qwen1.5-1.8B-Chat")
    embed_model = load_embed("/root/autodl-tmp/LlamaIndexrag/LlamaIndex/tcm-ai-rag/tcm-ai-rag/models/BAAI/bge-base-zh-v1.5")
    query_engine = build_rag_engine(index_path="/root/autodl-tmp/LlamaIndexrag/LlamaIndex/tcm-ai-rag/tcm-ai-rag/doc_emb", llm=llm, embed_model=embed_model)

    print("\n💬 输入你的中医问题，输入 'exit' 或 'quit' 退出：")

    while True:
        user_input = input("用户：")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("程序已退出。")
            break
        response = query_engine.chat(user_input)
        print("LLM:", str(response))

if __name__ == '__main__':
    main()