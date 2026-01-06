import json
import os
import requests
import streamlit as st

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen15b-cloudqa")

st.set_page_config(page_title=f"Ollama Chat - {MODEL}")
st.title(f"Chat: {MODEL}")
st.caption(f"Ollama: {OLLAMA_HOST}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a senior platform engineer. Answer..."}
    ]

for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Ask something about AWS/EKS/Kubernetes/GCP/Java...")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        buf = ""

        # Ollama chat API (streaming)
        url = f"{OLLAMA_HOST}/api/chat"
        payload = {
            "model": MODEL,
            "messages": st.session_state.messages,
            "stream": True,
        }

        try:
            with requests.post(url, json=payload, stream=True, timeout=600) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = (evt.get("message") or {}).get("content")
                    if delta:
                        buf += delta
                        placeholder.markdown(buf)
        except Exception as e:
            placeholder.markdown(f"**Error talking to Ollama:** {e}")

    st.session_state.messages.append({"role": "assistant", "content": buf})
