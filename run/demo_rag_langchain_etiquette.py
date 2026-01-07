import os
import gradio as gr
from dotenv import load_dotenv
from tianji.knowledges.langchain_onlinellm.models import ZhipuAIEmbeddings, ZhipuLLM
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from tianji import TIANJI_PATH
from huggingface_hub import snapshot_download
import argparse

# 加载环境变量
load_dotenv()

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Launch Gradio RAG application')
parser.add_argument('--local-only', action='store_true', help='Only allow localhost access (127.0.0.1)')
parser.add_argument('--port', type=int, default=7860, help='The port the server should listen on (default: 7860)')
parser.add_argument('--root_path', type=str, default=None, help='The root path of the server')
args = parser.parse_args()

# 使用 Hugging Face 的 huggingface_hub 下载数据集
destination_folder = os.path.join(TIANJI_PATH, "temp", "tianji-chinese")
if not os.path.exists(destination_folder):
    snapshot_download(
        repo_id="sanbu/tianji-chinese",
        local_dir=destination_folder,
        repo_type="dataset",
    )


def create_vectordb(
    data_path: str,
    persist_directory: str,
    embedding_func,
    chunk_size: int,
    force: bool = False,
):
    if os.path.exists(persist_directory) and not force:
        return Chroma(
            persist_directory=persist_directory, embedding_function=embedding_func
        )

    if force and os.path.exists(persist_directory):
        if os.path.isdir(persist_directory):
            import shutil

            shutil.rmtree(persist_directory)
        else:
            os.remove(persist_directory)

    # 使用 UTF-8 编码加载文件，避免 Windows 系统上的编码问题
    loader = DirectoryLoader(
        data_path, 
        glob="*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(loader.load())
    if len(split_docs) == 0:
        raise gr.Error("当前知识数据无效,处理数据后为空")

    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_func,
        persist_directory=persist_directory,
    )
    return vector_db


def initialize_chain(chunk_size: int, persist_directory: str, data_path: str):
    print("初始化数据库开始")
    embeddings = ZhipuAIEmbeddings()
    vectordb = create_vectordb(data_path, persist_directory, embeddings, chunk_size)
    retriever = vectordb.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    prompt.messages[
        0
    ].prompt.template = """
    您是一名用于问答任务的助手。使用检索到的上下文来回答问题。如果您不知道答案，就直接说不知道。\
    1.根据我的提问,总结检索到的上下文中与提问最接近的部分,将相关部分浓缩为一段话返回;
    2.根据语料结合我的问题,给出建议和解释。\
    \n问题：{question} \n上下文：{context} \n答案：
    """
    llm = ZhipuLLM()
    print("初始化数据库结束")
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def handle_question(chain, question: str, chat_history):
    if not question:
        return "", chat_history
    if chat_history is None:
        chat_history = []
    try:
        result = chain.invoke(question)
        # 使用字典格式兼容新版本 Gradio
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": result})
        return "", chat_history
    except Exception as e:
        error_msg = str(e)
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": f"错误: {error_msg}"})
        return "", chat_history


# 确保数据存在
data_path = os.path.join(TIANJI_PATH, "temp", "tianji-chinese", "RAG", "1-etiquette")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"数据路径不存在: {data_path}")

# 初始化数据库
chunk_size = 1024
persist_directory = os.path.join(TIANJI_PATH, "temp", "chromadb_1-etiquette")
model_chain = initialize_chain(chunk_size, persist_directory, data_path)

# 创建Gradio界面

TITLE = """
# Tianji 人情世故大模型系统-敬酒版(基于知识库实现) 欢迎star！\n
## 🤖开源项目地址：https://github.com/SocialAI-tianji/Tianji
## 使用方法：输入提示,或点击Example自动填充
## 如果觉得回答不满意,可以重复多次询问
### 我们的愿景是构建一个从数据收集开始的大模型全栈垂直领域开源实践.
"""

with gr.Blocks() as demo:
    gr.Markdown(TITLE)

    init_status = gr.Textbox(label="初始化状态", value="数据库已初始化", interactive=False)
    chatbot = gr.Chatbot(height=450, value=[])
    msg = gr.Textbox(label="输入你的疑问")

    examples = gr.Examples(
        label="快速示例",
        examples=[
            "喝酒座位怎么排",
            "喝酒的完整流程是什么",
            "推荐的敬酒词怎么说",
            "宴会怎么点菜",
            "喝酒容易醉怎么办",
            "喝酒的规矩是什么",
        ],
        inputs=[msg],
    )

    with gr.Row():
        chat_button = gr.Button("聊天")
        clear_button = gr.ClearButton(components=[chatbot], value="清除聊天记录")

    # Define a function to invoke the chain
    def invoke_chain(question, chat_history):
        return handle_question(model_chain, question, chat_history)

    chat_button.click(
        invoke_chain,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
    )

# 启动Gradio应用
if __name__ == "__main__":
    # 默认监听所有网络接口，允许局域网内其他设备访问
    if args.local_only:
        server_name = '127.0.0.1'  # 仅本地访问
    else:
        server_name = '0.0.0.0'  # 允许外部访问（默认）
    
    server_port = args.port
    
    print(f"🚀 启动服务器: http://{server_name}:{server_port}")
    if server_name == '0.0.0.0':
        print("📱 局域网内其他设备可通过以下地址访问:")
        print(f"   http://<你的IP地址>:{server_port}")
        print("   提示: 在命令行运行 'ipconfig' (Windows) 或 'ifconfig' (Linux/Mac) 查看你的IP地址")
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        root_path=args.root_path
    )
