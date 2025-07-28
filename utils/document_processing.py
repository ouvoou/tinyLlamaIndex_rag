import re

def remove_english(input_file, output_file):
    """
    去除文件中所有英文字符并生成新文件
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            content = f_in.read()

        # 使用正则表达式移除所有英文字母
        filtered_content = re.sub('[A-Za-z/]', '', content)

        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(filtered_content)
            
        print(f"处理完成，已生成新文件：{output_file}")
        
    except Exception as e:
        print(f"处理出错：{str(e)}")

remove_english('/root/autodl-tmp/day05-第4章 LlamaIndex知识管理与信息检索/day05-第4章 LlamaIndex知识管理与信息检索/tcm-ai-rag/tcm-ai-rag/data/zhongyi2.txt', 
               '/root/autodl-tmp/day05-第4章 LlamaIndex知识管理与信息检索/day05-第4章 LlamaIndex知识管理与信息检索/tcm-ai-rag/tcm-ai-rag/data/zhongyi2_filtered.txt')