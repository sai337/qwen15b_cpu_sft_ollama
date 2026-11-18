import json
import os

import requests
import streamlit as st

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen15b-cloudqa")


@st.cache_data(ttl=15)
def list_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return []


st.set_page_config(page_title="Fine-tuned Qwen chat")

with st.sidebar:
    models = list_models()
    if not models:
        st.error(f"No models found at {OLLAMA_HOST}. Is Ollama running and the model created?")
        models = [DEFAULT_MODEL]
    default_idx = next((i for i, m in enumerate(models) if m.split(":")[0] == DEFAULT_MODEL), 0)
    model = st.selectbox("Model", models, index=default_idx, help="Switch to the base model to compare answers.")
    system = st.text_area("System prompt override", "", help="Leave empty to use the SYSTEM from the Modelfile (what the model was trained with).")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
    if st.button("Clear chat"):
        st.session_state.messages = []

st.title(f"Chat: {model}")
st.caption(f"Ollama: {OLLAMA_HOST}")

st.session_state.setdefault("messages", [])
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Ask something about AWS/EKS/Kubernetes/GCP...")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    msgs = ([{"role": "system", "content": system}] if system.strip() else []) + st.session_state.messages
    with st.chat_message("assistant"):
        placeholder = st.empty()
        buf = ""
        try:
            with requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": model, "messages": msgs, "stream": True, "options": {"temperature": temperature}},
                stream=True,
                timeout=600,
            ) as r:
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
    if buf:
        st.session_state.messages.append({"role": "assistant", "content": buf})
