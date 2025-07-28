from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage, StorageContext, QueryBundle
from llama_index.llms.huggingface import HuggingFaceLLM
def load_llm(model_path):

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

    SYSTEM_PROMPT = """你是一个优秀的中医AI助手.请回答我的专业性问题。"""
    query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
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

    return Settings.llm