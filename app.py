# import os
# import json
# import logging
# import numpy as np
# from flask import Flask, request, jsonify
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from groq import Groq
# from dotenv import load_dotenv


# load_dotenv()
# GROQ_KEY=os.getenv('GROQ_API_KEY')
# if not GROQ_KEY:
#     raise ValueError("GROQ_API_KEY is not found in env variables")
# groq_client=Groq(api_key=GROQ_KEY)

# SIMILARITY_THRESHOLD = 0.3
# MAX_SNIPPET_CHARS = 2000

# app = Flask(__name__)

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Load FAQ data
# faq_data = []
# with open("faq_clean.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         faq_data.append(json.loads(line))

# faq_questions = [q["question"] for q in faq_data]

# # Build TF-IDF retriever
# vectorizer = TfidfVectorizer()
# faq_matrix = vectorizer.fit_transform(faq_questions)

# def retrieve_context(user_query, threshold=SIMILARITY_THRESHOLD):
#     query_vec = vectorizer.transform([user_query])
#     similarities = cosine_similarity(query_vec, faq_matrix).flatten()

#     # Get all indices where similarity >= threshold
#     valid_indices = np.where(similarities >= threshold)[0]

#     # Sort by similarity (descending)
#     valid_indices = sorted(valid_indices, key=lambda i: similarities[i], reverse=True)

#     if not valid_indices:
#         return None  # No relevant FAQs found

#     snippets = []
#     for idx in valid_indices:
#         ans = faq_data[idx]["answer"]
#         if len(ans) > MAX_SNIPPET_CHARS:
#             ans = ans[:MAX_SNIPPET_CHARS] + "..."
#         snippets.append(
#             f"(score={similarities[idx]:.2f})\nQ: {faq_data[idx]['question']}\nA: {ans}"
#         )

#     return "\n\n".join(snippets)

# def ask_llm(user_query):
#     context = retrieve_context(user_query)

#     if not context:
#         return "I couldn’t find any relevant information in the FAQ."

#     prompt = f"""
# You are a helpful assistant answering user questions based only on the provided FAQ context.

# FAQ Context:
# {context}

# User Question:
# {user_query}

# Answer clearly and concisely, using only the FAQ context.
# If no relevant answer exists, say "I don’t know based on the available FAQ."
# """
#     response = groq_client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2,
#     )
#     return response.choices[0].message["content"]


# @app.route("/ask", methods=["POST"])
# def ask():
#     user_query = request.json.get("query", "")
#     if not user_query.strip():
#         return jsonify({"error": "Query cannot be empty"}), 400

#     logging.info(f"User query: {user_query}")
#     answer = ask_llm(user_query)
#     return jsonify({"answer": answer})

# if __name__ == "__main__":
#     app.run(port=8000, debug=True)