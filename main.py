import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
from googletrans import Translator

load_dotenv()
# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="FAQ Chatbot Prototype", layout="wide")

# Load API key (set in environment variable before running)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

# Translator init
translator = Translator()

# Utility functions
# -----------------------------
def detect_language(text):
    try:
        lang = translator.detect(text).lang
        return lang
    except:
        return "en"  # fallback

def translate_text(text, target_lang="en"):
    try:
        return translator.translate(text, dest=target_lang).text
    except:
        return text


# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ratings" not in st.session_state:
    st.session_state.ratings = {}  # store {index: rating}

if "token_usage" not in st.session_state:
    st.session_state.token_usage = 0

if "show_faq" not in st.session_state:
    st.session_state.show_faq = False

# -----------------------------
# Sidebar Controls
# -----------------------------
st.header("📘 FAQ Chatbot Prototype")
st.sidebar.header("⚙️ Controls")
faq_context = ""

with st.sidebar:
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="""
You are Priya, a helpful AI assistant working at Narayan Seva Sansthan (NSS). You answer questions based on provided FAQ context, uploaded documents, and chat history. 
Rules:
- Introduce yourself as Priya from NSS when greeting new users
- FAQ data is in English
- Always process queries in English (translate if needed)
- Respond back in the same language as the user query
- Provide accurate answers based on the given context
- Reference uploaded documents when relevant
- Maintain conversation continuity
- Be concise but comprehensive
- Maintain continuity, use a friendly and professional tone""",
        height=200
    )

# Token Usage Meter
st.sidebar.subheader("📊 Token Usage Meter")
st.sidebar.write(f"**Total Tokens Used:** {st.session_state.token_usage}")

# Reset session
if st.sidebar.button("🔄 Reset Session"):
    st.session_state.chat_history = []
    st.session_state.token_usage = 0
    st.session_state.ratings = []
    st.success("Session reset!")


# File Upload
# -----------------------------

faq_file = st.file_uploader("Upload FAQ Excel", type=["xlsx"])


if faq_file:
    df = pd.read_excel(faq_file)

    # Show/Hide FAQ Data Toggle
    if st.button("👀 View FAQ Data"):
        st.session_state.show_faq = not st.session_state.show_faq

    if st.session_state.show_faq:
        st.write("### FAQ Data Loaded:")
        st.dataframe(df)

    # Prepare FAQ context string
    faq_context = "\n".join([
        f"[{row['SL']}] {row['Area']} - Q: {row['Question']} | A: {row['Answer']}"
        for _, row in df.iterrows()
    ])
else:
    st.warning("Please upload an FAQ Excel file to start.")
    st.stop()

# Chat Interface
# -----------------------------
st.subheader("💬 Chat")

# 👋 Show greeting only before first user query (not in chat history)
if len(st.session_state.chat_history) == 0:
    st.info("Hello! 👋 I’m **Priya**, your helpful AI assistant from Narayan Seva Sansthan (NSS). Ask me anything about NSS from the FAQ!")


# Show chat history ABOVE the input box (latest near input)
for i, chat in enumerate(st.session_state.chat_history):
    with st.container():
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['assistant']}")
        st.caption(f"Tokens used: {chat['tokens']}")

    # Feedback only if not already given
    if i not in st.session_state.ratings:
        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            if st.button("👍", key=f"up_{i}"):
                st.session_state.ratings[i] = "👍"
                st.success("Feedback recorded: 👍")
        with col2:
            if st.button("👎", key=f"down_{i}"):
                st.session_state.ratings[i] = "👎"
                st.error("Feedback recorded: 👎")
    else:
        st.caption(f"✅ Feedback: {st.session_state.ratings[i]}")

st.markdown("---")

# Query input with SEND button on left side
col_input = st.columns([0.8, 0.2])
with col_input[0]:
    default_value = st.session_state.pop("user_query", "")
    user_query = st.text_input("Ask something:", value=default_value, key="user_query")
with col_input[1]:
    send_btn = st.button("Send")

# Query Handling with Translation
# -----------------------------
if send_btn and user_query.strip():
    # Detect query language
    user_lang = detect_language(user_query)
    query_for_faq = user_query

    # If not English, translate to English
    if user_lang != "en":
        query_for_faq = translate_text(user_query, "en")

    # Build full context in English
    full_context = f"""
System Prompt: {system_prompt}

Conversation History:
{st.session_state.chat_history[-5:]}

FAQ Data (English only):
{faq_context}

User Query (translated to English if needed): {query_for_faq}
    """

    try:
        response = model.generate_content(full_context)
        bot_answer_en = response.text if response.text else "Sorry, I couldn't find an answer."

        # Translate back to user language if needed
        bot_answer = bot_answer_en
        if user_lang != "en":
            bot_answer = translate_text(bot_answer_en, user_lang)

        # Tokens
        tokens_used = len(user_query.split()) + len(bot_answer_en.split())
        if hasattr(response, "usage_metadata"):
            tokens_used = response.usage_metadata.total_token_count
        st.session_state.token_usage += tokens_used

        # Save in history
        st.session_state.chat_history.append(
            {"user": user_query, "assistant": bot_answer, "tokens": tokens_used}
        )

        st.session_state.pop("user_query", None)
        st.rerun()

    except Exception as e:
        st.error(f"Error: {str(e)}")