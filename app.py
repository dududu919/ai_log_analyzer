import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
from openai import OpenAI

# 1. 基础配置
st.set_page_config(page_title="AI 首席数据分析师", layout="wide")
st.title("👨‍💻 AI 首席数据分析师 (自动识别+人工反馈版)")

# --- 2. AI 客户端配置 ---
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    st.error("❌ 未检测到 API Key。请在侧边栏或 Secrets 中配置 DEEPSEEK_API_KEY")
    st.stop()

client = OpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com"
)

# --- 3. 侧边栏：业务记忆管理 ---
if "business_memory" not in st.session_state:
    st.session_state.business_memory = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("🧠 业务知识库")
    context_input = st.text_area(
        "输入业务背景与字段定义：",
        value=st.session_state.business_memory,
        placeholder="例如：filesize是字节，usetime是毫秒...",
        height=150
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

# --- 4. 数据处理与动态 Prompt 生成 ---
if uploaded_file:
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)
    
    df = load_data(uploaded_file)
    con = duckdb.connect(database=':memory:')
    con.register('data_table', df)

    # ✨ 核心改进：动态获取表结构和样例数据
    sample_df = df.head(3)
    dynamic_column_info = ""
    for col in df.columns:
        dtype = str(df[col].dtype)
        # 提取两个非空样例
        samples = ", ".join(map(str, sample_df[col].dropna().unique()[:2]))
        dynamic_column_info += f"- {col} ({dtype})。样例: [{samples}]\n"

    # 构建发送给 AI 的系统提示词
    sys_msg = f"""
    你是一个资深的 DuckDB SQL 专家。表名为 data_table。
    【实时表结构】：
    {dynamic_column_info}
    【业务背景】：
    {st.session_state.business_memory}
    【任务要求】：
    1. 只能返回 SQL 代码块，格式为 ```sql (代码) ```。
    2. 使用 DuckDB 语法，字符串匹配用 LIKE。
    3. 速度公式通常：filesize * 1000.0 / NULLIF(usetime, 0)。
    """

    # AI 业务背景同步提示
    if st.session_state.get("needs_feedback", False) and st.session_state.business_memory:
        with st.status("AI 正在同步业务背景...", expanded=False):
            check_res = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": "简短总结你对背景的理解。"}, {"role": "user", "content": st.session_state.business_memory}]
            )
            st.info(f"✅ **AI 记忆同步成功：** {check_res.choices[0].message.content}")
            st.session_state.needs_feedback = False

    # 渲染历史
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- 5. 核心对话与反馈纠错逻辑 ---
    if prompt := st.chat_input("基于业务背景提问..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        if "temp_sql" in st.session_state:
            del st.session_state.temp_sql
        st.rerun()

    if st.session_state.chat_history:
        last_msg = st.session_state.chat_history[-1]
        
        if last_msg["role"] == "user":
            with st.chat_message("assistant"):
                # A. 初始 SQL 生成
                if "temp_sql" not in st.session_state:
                    with st.status("🧠 AI 正在编写 SQL 指令...", expanded=True):
                        res = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[
                                {"role": "system", "content": sys_msg},
                                {"role": "user", "content": last_msg["content"]}
                            ]
                        )
                        raw_content = res.choices[0].message.content
                        if "```sql" in raw_content:
                            st.session_state.temp_sql = raw_content.split("```sql")[1].split("```")[0].strip()
                        else:
                            st.session_state.temp_sql = raw_content.strip()

                # B. SQL 展示与【反馈重写】区
                st.markdown("### 🛠️ SQL 指令微调")
                edited_sql = st.text_area("检查或手动修改 SQL：", value=st.session_state.temp_sql, height=120)
                st.session_state.temp_sql = edited_sql

                # ✨ 核心改进：人工反馈输入框
                user_feedback = st.text_input("💡 觉得 AI 理解不对？在此输入改进建议（如：'换个单位'、'过滤异常值'）：")
                
                c1, c2 = st.columns([1, 4])
                with c1:
                    execute_btn = st.button("🚀 执行指令")
                with c2:
                    if st.button("🔄 根据建议重新生成") and user_feedback:
                        with st.spinner("AI 正在根据建议修正 SQL..."):
                            res = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[
                                    {"role": "system", "content": sys_msg},
                                    {"role": "user", "content": last_msg["content"]},
                                    {"role": "assistant", "content": st.session_state.temp_sql},
                                    {"role": "user", "content": f"请根据此反馈修改 SQL：{user_feedback}"}
                                ]
                            )
                            new_raw = res.choices[0].message.content
                            st.session_state.temp_sql = new_raw.split("```sql")[1].split("```")[0].strip() if "```sql" in new_raw else new_raw
                            st.rerun()

                # C. 执行结果展示
                if execute_btn:
                    with st.spinner("⏳ 正在计算并解读..."):
                        try:
                            query_result = con.execute(st.session_state.temp_sql).df()
                            
                            if not query_result.empty:
                                st.markdown("### 📝 分析结果")
                                st.dataframe(query_result, use_container_width=True)
                                
                                # 图表展示
                                if len(query_result.columns) >= 2:
                                    fig = px.bar(query_result, x=query_result.columns[0], y=query_result.columns[1], title="数据分布图")
                                    st.plotly_chart(fig, use_container_width=True)

                                # AI 解读结论
                                interpret_res = client.chat.completions.create(
                                    model="deepseek-chat",
                                    messages=[
                                        {"role": "system", "content": f"基于背景分析数据：{st.session_state.business_memory}"},
                                        {"role": "user", "content": f"原始问题：{last_msg['content']}\n查询结果：\n{query_result.head(10).to_string()}"}
                                    ]
                                )
                                final_ans = interpret_res.choices[0].message.content
                                st.success(final_ans)
                                st.session_state.chat_history.append({"role": "assistant", "content": final_ans})
                            else:
                                st.warning("查询结果为空，请调整查询条件。")
                        except Exception as e:
                            st.error(f"❌ SQL 执行出错：{e}")
                            st.info("提示：您可以直接在上面的反馈框描述这个错误，让 AI 帮您修复。")

else:
    st.info("👋 欢迎！请在侧边栏上传 CSV 日志文件开始分析。")