import streamlit as st

import rag

st.set_page_config(page_title="Docs Assistant", page_icon="📚")
st.title("📚 Docs Assistant")
st.caption(f"Answers grounded in your documents · {rag.LLM_MODEL} + {rag.EMBED_MODEL} via Ollama · pgvector")


@st.cache_resource(show_spinner="First start: downloading models and indexing documents...")
def setup():
    if rag.is_indexed():
        rag.ensure_models()
    else:
        rag.ingest()


setup()

with st.sidebar:
    st.write("Drop `.md`, `.txt` or `.pdf` files in the `docs/` folder, then re-index.")
    if st.button("Re-index documents"):
        with st.spinner("Indexing..."):
            st.success(f"{rag.ingest()} chunks indexed")

messages = st.session_state.setdefault("messages", [])
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Each question is answered independently (no chat-history rewriting of the query).
if question := st.chat_input("Ask a question about the documents"):
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    hits = rag.retrieve(question)
    with st.chat_message("assistant"):
        text = st.write_stream(rag.answer_stream(question, hits))
        with st.expander("Sources"):
            for i, h in enumerate(hits, 1):
                st.markdown(f"**[{i}] {h['source']}** · similarity {h['score']:.2f}\n\n{h['content']}")
    messages.append({"role": "assistant", "content": text})
