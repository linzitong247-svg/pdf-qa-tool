import streamlit as st

from langchain_classic.memory import ConversationBufferMemory
from utils import qa_agent


st.title("📑RIRINA-AI智能PDF问答工具")

with st.sidebar:
    deepseek_api_key = st.text_input("请输入Deepseek API密钥：", type="password")
    st.markdown("[获取Deepseek API密钥](https://platform.deepseek.com/usage)")

# 初始化session_state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 使用固定的session_id来保持对话历史
session_id = "pdf_qa_session"

uploaded_file = st.file_uploader("上传你的PDF文件：", type="pdf")

col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input("对PDF的内容进行提问", disabled=not uploaded_file)
with col2:
    # 添加一些垂直间距来对齐
    st.write("")  # 空行
    submit_button = st.button("提交问题",
                              disabled=not (uploaded_file and question and deepseek_api_key))


if uploaded_file and question and not deepseek_api_key:
    st.info("请输入你的Deepseek API密钥")

# 只有当点击提交按钮时才执行问答
if submit_button and uploaded_file and question and deepseek_api_key:
    with st.spinner("AI正在思考中，请稍等..."):
        response = qa_agent(deepseek_api_key, session_id, uploaded_file, question)

    st.write("### 答案")
    st.write(response)
    st.session_state["chat_history"].append(("用户", question))
    st.session_state["chat_history"].append(("AI", response))

if st.session_state["chat_history"]:
    with st.expander("历史消息", expanded=False):
        for role, message in st.session_state["chat_history"]:
            st.markdown(f"**{role}**: {message}")