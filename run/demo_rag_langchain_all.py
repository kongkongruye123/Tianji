import os
import gradio as gr
from dotenv import load_dotenv
from tianji.knowledges.langchain_onlinellm.models import SiliconFlowEmbeddings, SiliconFlowLLM
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from tianji import TIANJI_PATH
import argparse
from huggingface_hub import snapshot_download
import requests
import loguru
load_dotenv()

parser = argparse.ArgumentParser(description='Launch Gradio application')
parser.add_argument('--listen', action='store_true', help='Specify to listen on 0.0.0.0')
parser.add_argument('--port', type=int, default=None, help='The port the server should listen on')
parser.add_argument('--root_path', type=str, default=None, help='The root path of the server')
parser.add_argument('--force', action='store_true', help='Force recreate the database')
parser.add_argument('--chunk_size', type=int, default=896, help='Chunk size for text splitting')
args = parser.parse_args()

# 开始前检查功能是否正常
try:
    llm = SiliconFlowLLM()
    test_response = llm._call("你好")
    loguru.logger.info("SiliconFlow聊天功能测试成功")
except Exception as e:
    loguru.logger.error("SiliconFlow聊天功能测试失败: {}", str(e))
    raise e
try:
    embeddings = SiliconFlowEmbeddings()
    test_text = "测试文本"
    test_embedding = embeddings.embed_query(test_text)
    if len(test_embedding) > 0:
        loguru.logger.info("SiliconFlow嵌入功能测试成功")
    else:
        raise ValueError("嵌入向量长度为0")
except Exception as e:
    loguru.logger.error("SiliconFlow嵌入功能测试失败: {}", str(e))
    raise e

# 正式代码
def check_internet_connection(url='http://www.google.com/', timeout=5):
    try:
        _ = requests.head(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        return False
    
destination_folder = os.path.join(TIANJI_PATH, "temp", "tianji-chinese")
if not os.path.exists(destination_folder):
    if not check_internet_connection():
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    for _ in range(5):
        try:
            snapshot_download(
                repo_id="sanbu/tianji-chinese",
                local_dir=destination_folder,
                repo_type="dataset",
                local_dir_use_symlinks=False,
                endpoint=os.environ.get('HF_ENDPOINT', None),
            )
            break
        except Exception as e:
            loguru.logger.error("Download failed, retrying... Error message: {}", str(e))
    else:
        loguru.logger.error("Download failed, maximum retry count reached.")


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
        loguru.logger.error("Invalid knowledge data, processing data results in empty, check if data download failed, can be downloaded manually")
        raise gr.Error("Invalid knowledge data, processing data results in empty, check if data download failed, can be downloaded manually")
    
    # 处理 API 速率限制：分批处理文档并添加延迟
    loguru.logger.info("开始创建向量数据库，文档数量: {}", len(split_docs))
    try:
        # 分批处理文档以避免速率限制
        batch_size = 10  # 每批处理10个文档
        import time
        
        # 如果文档数量较多，分批添加
        if len(split_docs) > batch_size:
            loguru.logger.info("文档数量较多，将分批处理以避免速率限制...")
            # 先创建空的向量数据库
            vector_db = Chroma(
                embedding_function=embedding_func,
                persist_directory=persist_directory,
            )
            
            # 分批添加文档
            for i in range(0, len(split_docs), batch_size):
                batch = split_docs[i:i + batch_size]
                loguru.logger.info("处理批次 {}/{} (共 {} 个文档)", 
                                  i // batch_size + 1, 
                                  (len(split_docs) + batch_size - 1) // batch_size,
                                  len(batch))
                try:
                    vector_db.add_documents(batch)
                    # 添加延迟以避免速率限制（每分钟请求数限制）
                    if i + batch_size < len(split_docs):
                        time.sleep(1)  # 每批之间延迟1秒
                except Exception as batch_error:
                    error_msg = str(batch_error)
                    if "RPM limit exceeded" in error_msg or "403" in error_msg:
                        loguru.logger.warning("遇到速率限制，等待60秒后重试...")
                        time.sleep(60)  # 等待60秒
                        vector_db.add_documents(batch)  # 重试
                    else:
                        raise batch_error
        else:
            # 文档数量少，直接创建
            vector_db = Chroma.from_documents(
                documents=split_docs,
                embedding=embedding_func,
                persist_directory=persist_directory,
            )
        loguru.logger.info("向量数据库创建成功")
    except Exception as e:
        error_msg = str(e)
        loguru.logger.error("创建数据库失败: {}", error_msg)
        if "RPM limit exceeded" in error_msg or "403" in error_msg:
            loguru.logger.error("API 速率限制已超出。请:")
            loguru.logger.error("1. 等待一段时间后重试")
            loguru.logger.error("2. 完成 SiliconFlow 身份验证以解除限制")
            loguru.logger.error("3. 使用 --force 参数时，可以分批运行或使用本地嵌入模型")
            raise gr.Error(
                "API 速率限制已超出。\n"
                "解决方案：\n"
                "1. 等待1-2分钟后重试\n"
                "2. 完成 SiliconFlow 身份验证以解除限制\n"
                "3. 如果已有数据库，不使用 --force 参数直接运行"
            )
        raise e
    return vector_db


def initialize_chain(chunk_size: int, persist_directory: str, data_path: str, force=False):
    loguru.logger.info("初始化数据库开始，当前数据路径为：{}", data_path)
    vectordb = create_vectordb(data_path, persist_directory, embeddings, chunk_size, force)
    retriever = vectordb.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    prompt.messages[
        0
    ].prompt.template = """
    您是一名用于问答任务的助手。使用检索到的上下文来回答问题。如果没有高度相关上下文 你就自由回答。\
    根据检索到的上下文，结合我的问题,直接给出最后的回答，要只紧扣问题围绕着回答，尽量根据涉及几个关键点用完整非常详细的几段话回复。。\
    \n问题：{question} \n上下文：{context} \n回答：
    """
    loguru.logger.info("初始化数据库结束")
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
        loguru.logger.error("处理问题时发生错误: {}", str(e))
        error_msg = str(e)
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": f"错误: {error_msg}"})
        return "", chat_history


# Define scenarios
scenarios = {
    "敬酒礼仪文化": "1-etiquette",
    "请客礼仪文化": "2-hospitality",
    "送礼礼仪文化": "3-gifting",
    "如何说对话": "5-communication",
    "化解尴尬场合": "6-awkwardness",
    "矛盾&冲突应对": "7-conflict",
}

# Initialize chains for all scenarios
chains = {}
loguru.logger.info("=" * 60)
loguru.logger.info("开始初始化所有场景的知识库...")
loguru.logger.info("提示：如果遇到速率限制错误，可以：")
loguru.logger.info("1. 等待1-2分钟后重试（不使用 --force）")
loguru.logger.info("2. 完成 SiliconFlow 身份验证以解除限制")
loguru.logger.info("3. 如果数据库已存在，直接运行（不使用 --force）")
loguru.logger.info("=" * 60)

for scenario_name, scenario_folder in scenarios.items():
    data_path = os.path.join(
        TIANJI_PATH, "temp", "tianji-chinese", "RAG", scenario_folder
    )
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    persist_directory = os.path.join(TIANJI_PATH, "temp", f"chromadb_{scenario_folder}")
    loguru.logger.info("正在初始化场景: {} ({})", scenario_name, scenario_folder)
    try:
        chains[scenario_name] = initialize_chain(args.chunk_size, persist_directory, data_path, args.force)
        loguru.logger.info("场景 {} 初始化成功", scenario_name)
    except Exception as e:
        error_msg = str(e)
        if "RPM limit exceeded" in error_msg or "403" in error_msg:
            loguru.logger.error("场景 {} 初始化失败：API 速率限制", scenario_name)
            loguru.logger.error("请等待后重试，或完成身份验证")
            # 如果其他场景已初始化，继续运行；否则退出
            if len(chains) == 0:
                raise e
        else:
            raise e

loguru.logger.info("所有场景初始化完成，共 {} 个场景", len(chains))

# Create Gradio interface
TITLE = """
# Tianji 人情世故大模型系统完整版(基于知识库实现) 欢迎star！\n
## 💫开源项目地址：https://github.com/SocialAI-tianji/Tianji
## 使用方法：选择你想提问的场景，输入提示，或点击Example自动填充
## 如果觉得回答不满意,可补充更多信息重复提问。
### 我们的愿景是构建一个从数据收集开始的大模型全栈垂直领域开源实践.
"""


def get_examples_for_scenario(scenario):
    # Define examples for each scenario
    examples_dict = {
        "敬酒礼仪文化": [
            "喝酒座位怎么排",
            "喝酒的先后顺序流程是什么",
            "喝酒需要注意什么",
            "推荐的敬酒词怎么说",
            "宴会怎么点菜",
            "喝酒容易醉怎么办",
            "喝酒的规矩是什么",
        ],
        "请客礼仪文化": ["请客有那些规矩", "如何选择合适的餐厅", "怎么请别人吃饭"],
        "送礼礼仪文化": ["送什么礼物给长辈好", "怎么送礼", "回礼的礼节是什么"],
        "如何说对话": [
            "怎么和导师沟通",
            "怎么提高情商",
            "如何读懂潜台词",
            "怎么安慰别人",
            "怎么和孩子沟通",
            "如何与男生聊天",
            "如何与女生聊天",
            "职场高情商回应技巧",
        ],
        "化解尴尬场合": ["怎么回应赞美", "怎么拒绝借钱", "如何高效沟通", "怎么和对象沟通", "聊天技巧", "怎么拒绝别人", "职场怎么沟通"],
        "矛盾&冲突应对": [
            "怎么控制情绪",
            "怎么向别人道歉",
            "和别人吵架了怎么办",
            "如何化解尴尬",
            "孩子有情绪怎么办",
            "夫妻吵架怎么办",
            "情侣冷战怎么办",
        ],
    }
    return examples_dict.get(scenario, [])


with gr.Blocks() as demo:
    gr.Markdown(TITLE)

    init_status = gr.Textbox(label="初始化状态", value="数据库已初始化", interactive=False)

    with gr.Tabs() as tabs:
        for scenario_name in scenarios.keys():
            with gr.Tab(scenario_name):
                chatbot = gr.Chatbot(height=450)
                msg = gr.Textbox(label="输入你的疑问")

                examples = gr.Examples(
                    label="快速示例",
                    examples=get_examples_for_scenario(scenario_name),
                    inputs=[msg],
                )

                with gr.Row():
                    chat_button = gr.Button("聊天")
                    clear_button = gr.ClearButton(components=[chatbot], value="清除聊天记录")

                # Define a function to invoke the chain for the current scenario
                def invoke_chain(question, chat_history, scenario=scenario_name):
                    loguru.logger.info(question)
                    return handle_question(chains[scenario], question, chat_history)

                chat_button.click(
                    invoke_chain,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot],
                )


if __name__ == "__main__":
    server_name = '0.0.0.0' if args.listen else None
    server_port = args.port
    demo.launch(server_name=server_name, server_port=server_port, root_path=args.root_path)
