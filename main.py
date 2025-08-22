import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="FAQ Chatbot Prototype", layout="wide")

# Load API key (set in environment variable before running)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")
# -----------------------------
# Session State Initialization
# -----------------------------
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
You should:
- Introduce yourself as Priya from NSS when greeting new users
- Provide accurate answers based on the given context
- Reference uploaded documents when relevant
- Maintain conversation continuity
- Be concise but comprehensive
- Use a friendly and professional tone""",
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

# Query Handling
# ------------------------
if send_btn and user_query.strip() != "":
    # Prepare prompt
    context = f"System Prompt: {system_prompt}\n\n"
    context += f"Here is the FAQ data:\n{faq_context}\n\n"
    context += f"User Query: {user_query}\nAnswer only from FAQ."

    try:
        response = model.generate_content(context)
        bot_answer = response.text if response.text else "Sorry, I couldn't find an answer."

        # # Capture tokens
        tokens_used = len(user_query.split()) + len(bot_answer.split())
        if hasattr(response, "usage_metadata"):
            tokens_used = response.usage_metadata.total_token_count
            st.session_state.token_usage += tokens_used

        # Save in history
        st.session_state.chat_history.append(
            {"user": user_query, "assistant": bot_answer, "tokens": tokens_used}
        )

        
    except Exception as e:
        st.error(f"Error: {str(e)}")

if send_btn and user_query:
    # Conversation history (last 5 turns)
    # Prepare conversation history (last few turns)
    history_context = [
        {"role": "user", "parts": h['user']} for h in st.session_state.chat_history[-5:]
    ] + [
        {"role": "model", "parts": h['assistant']} for h in st.session_state.chat_history[-5:]
    ]

    # Build messages for Gemini
    messages = [
    {"role": "user", "parts": f"""
    You are Priya, an AI assistant for hotel-related queries.
    Always be helpful and concise.

    Conversation History:
    {history_context}

    FAQ Data:
    {faq_context}

    User Query: {user_query}
    """}
    ]

    # Generate response
    response = model.generate_content(messages)
    answer = response.text


    # Final input to model
    full_context = f"""
System Prompt: {system_prompt}

Conversation History:
{history_context}

FAQ Data:
{faq_context}

User Query: {user_query}
    """
    st.session_state.pop("user_query", None)
    st.rerun()


    # Call Gemini Flash
    response = model.generate_content(full_context)
    answer = response.text

    # Update history
    st.session_state.chat_history.append({"user": user_query, "assistant": answer})

    # Update token usage (approximate)
    st.session_state.token_usage += len(user_query.split()) + len(answer.split())

