import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
from openai import OpenAI

# 1. 基础配置
st.set_page_config(page_title="AI 首席数据分析师", layout="wide")
st.title("👨‍💻 AI 首席数据分析师 (带手动微调功能)")

# --- 2. AI 客户端配置 ---
# --- 1. 连接 AI (安全模式) ---
try:
    # 无论是在本地还是云端，都只从 secrets 查找
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    # 如果找不到 Key，直接报错提醒，而不是暴露备选 Key
    st.error("❌ 未检测到 API Key。请在侧边栏或 Secrets 中配置 DEEPSEEK_API_KEY")
    st.stop() # 停止运行，防止后续报错

client = OpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com"
)

# --- 3. 侧边栏：业务记忆管理 ---
if "business_memory" not in st.session_state:
    st.session_state.business_memory = ""

with st.sidebar:
    st.header("🧠 业务知识库")
    context_input = st.text_area(
        "输入业务背景与字段定义：",
        value=st.session_state.business_memory,
        placeholder="例如：filesize是字节，usetime是毫秒...",
        height=200
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 保存背景"):
            st.session_state.business_memory = context_input
            st.session_state.needs_feedback = True
    with col2:
        if st.button("🗑️ 清除记忆"):
            st.session_state.business_memory = ""
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    uploaded_file = st.file_uploader("📂 上传 CSV 日志文件", type="csv")

# --- 4. 数据入库 ---
if uploaded_file:
    # 使用缓存读取数据，避免每次交互都重新读取大文件
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)
    
    df = load_data(uploaded_file)
    con = duckdb.connect(database=':memory:')
    con.register('data_table', df)
    column_info = "\n".join([f"- {col} ({df[col].dtype})" for col in df.columns])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # AI 背景同步反馈
    if st.session_state.get("needs_feedback", False) and st.session_state.business_memory:
        with st.status("AI 正在同步业务背景...", expanded=True) as status:
            check_res = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": "总结你对用户背景的理解。"}, {"role": "user", "content": st.session_state.business_memory}]
            )
            status.update(label="✅ 业务背景同步完成！", state="complete", expanded=False)
            st.info(f"✅ **AI 记忆同步成功：**\n\n{check_res.choices[0].message.content}")
            st.session_state.needs_feedback = False

    # 渲染历史
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- 5. 核心对话与微调逻辑 ---
    if prompt := st.chat_input("基于业务背景提问..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # 只要提了新问题，就清空旧的 SQL 暂存，让 AI 重新生成
        if "temp_sql" in st.session_state:
            del st.session_state.temp_sql
        st.rerun()

    # 处理最新的对话
    if st.session_state.chat_history:
        last_msg = st.session_state.chat_history[-1]
        
        if last_msg["role"] == "user":
            with st.chat_message("assistant"):
                
                # --- 第一步：仅在没有暂存 SQL 时请求 AI 生成 ---
                if "temp_sql" not in st.session_state:
                    with st.status("🧠 AI 正在根据您的背景编写指令...", expanded=True):
                        sys_msg = f"你是一个分析师。背景：{st.session_state.business_memory}。表名 data_table，结构：\n{column_info}。只返回 SQL 块。"
                        res = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": last_msg["content"]}]
                        )
                        ai_raw = res.choices[0].message.content
                        try:
                            initial_sql = ai_raw.split("```sql")[1].split("```")[0].strip()
                        except:
                            initial_sql = ai_raw.strip()
                        # 将 AI 生成的初稿锁存在 session_state 中
                        st.session_state.temp_sql = initial_sql

                # --- 第二步：微调区（现在它不会被覆盖了） ---
                st.markdown("### 🛠️ SQL 指令微调")
                # 这里我们不直接用 initial_sql，而是用 session_state 里的值
                edited_sql = st.text_area(
                    "您可以直接在此修改代码（修改后点击下方执行）：", 
                    value=st.session_state.temp_sql, 
                    height=200,
                    key="sql_editor_area" # 给它一个唯一的 key
                )
                
                # 同步更新暂存，防止刷新丢失
                st.session_state.temp_sql = edited_sql

                if st.button("🚀 执行当前框内的指令"):
                    with st.spinner("⏳ 正在计算数据..."):
                        try:
                            query_result = con.execute(st.session_state.temp_sql).df()
                            
                            if not query_result.empty:
                                # AI 解读结果
                                interpretation = client.chat.completions.create(
                                    model="deepseek-chat",
                                    messages=[
                                        {"role": "system", "content": f"背景：{st.session_state.business_memory}。"},
                                        {"role": "user", "content": f"结果：\n{query_result.to_string()}"}
                                    ]
                                )
                                final_answer = interpretation.choices[0].message.content
                                
                                st.markdown("### 📝 分析结论")
                                st.success(final_answer)
                                st.dataframe(query_result, use_container_width=True)
                                
                                if len(query_result.columns) >= 2:
                                    fig = px.bar(query_result, x=query_result.columns[0], y=query_result.columns[1], title="分析图表")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
                            else:
                                st.warning("查询成功，但数据为空。")
                        except Exception as e:
                            st.error(f"❌ SQL 执行报错：{e}")
                else:
                    st.info("💡 请检查上面的 SQL 语句，确认无误后点击执行。")

else:
    st.info("👋 欢迎！请上传 CSV 文件并设置背景。")