from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

from typing import List
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
import requests
from langchain_core.embeddings import Embeddings
# 会话历史存储
store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 直接使用requests的嵌入函数
def siliconflow_embed_texts(texts, api_key, model="Qwen/Qwen3-Embedding-0.6B"):
    """直接使用requests调用SiliconFlow API"""
    url = "https://api.siliconflow.cn/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    embeddings = []
    for text in texts:
        payload = {
            "model": model,
            "input": text,
            "encoding_format": "float"
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            embeddings.append(result["data"][0]["embedding"])
        else:
            raise Exception(f"SiliconFlow API错误: {response.status_code} - {response.text}")

    return embeddings

# 简化嵌入类
class DirectSiliconFlowEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return siliconflow_embed_texts(texts, self.api_key, self.model)

    def embed_query(self, text: str) -> List[float]:
        return siliconflow_embed_texts([text], self.api_key, self.model)[0]


def qa_agent(deepseek_api_key,session_id,pdf_path,question):
    model = ChatOpenAI(model="deepseek-chat",
                       openai_api_key=deepseek_api_key,
                       base_url="https://api.deepseek.com/v1"
                       )
    embeddings_model = DirectSiliconFlowEmbeddings(
        api_key="sk-msuexzerxhvbbbzqafayeqokabvapubvzknaytlnaohasjck",  # 这里传入SiliconFlow API密钥
        model="Qwen/Qwen3-Embedding-0.6B"
    )
    #先将用户上传的文件本地化，再传给加载器
    file_content = pdf_path.read()#从上传的文件对象中读取所有内容并加载到内存中；返回bytes，内容的二进制数据
    temp_file_path = "temp.pdf"#指定临时保存文件的路径和名称
    with open(temp_file_path, "wb") as temp_file:#将内容写入临时文件，文件名为temp_file；open(路径，模式）as 文件名
        temp_file.write(file_content)#将内存中的二进制数据写入磁盘文件
    loader = PyPDFLoader(temp_file_path)#创建PDF加载器
    docs = loader.load()#加载PDF文档，即：将PDF内容转换为可处理的文本数据
    # loader = PyPDFLoader(pdf_path)
    # docs = loader.load()

    #分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)

    #文档内容向量化&储存进向量数据库
    db = FAISS.from_documents(texts, embeddings_model)#参数是分割好的文档列表和嵌入模型

    #创建检索器
    retriever = db.as_retriever()

    # 构建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的AI助手，根据上下文和对话历史回答问题。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "上下文:\n{context}\n\n问题: {question}\n\n回答:")
    ])

    # 创建基础 RAG 链
    rag_chain = (
            {
                "context": itemgetter("question") | retriever | format_docs,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history")
            }
            | prompt
            | model
            | StrOutputParser()
    )

    # 使用 RunnableWithMessageHistory 包装 RAG 链
    chain_with_history = RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )
    response = chain_with_history.invoke(# 调用链
        {"question": question},
        config={"configurable": {"session_id": session_id}}
     )
    return response

