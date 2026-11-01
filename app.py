"""Streamlit UI for agentic_chatbot_hitl_backend.py (threaded RAG + HITL)."""
import os
import tempfile
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from backend import chatbot, get_all_threads, ingest_rag_document

st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖")
st.title("Agentic Chatbot with LangGraph")


def config_for(thread_id):
    return {"configurable": {"thread_id": str(thread_id)}}


def new_thread():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_threads.append(st.session_state.thread_id)


def snapshot_for(thread_id):
    return chatbot.get_state(config_for(thread_id))


def pending_interrupt(thread_id):
    """Read pending HITL from the persisted checkpoint, not session state."""
    snapshot = snapshot_for(thread_id)
    direct = getattr(snapshot, "interrupts", ()) or ()
    if direct:
        return direct[0]
    for task in getattr(snapshot, "tasks", ()) or ():
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            return interrupts[0]
    return None


def text_of(content):
    """Convert string or LangChain content blocks to displayable text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
        )
    return ""


def history_for(thread_id):
    """Reconstruct UI history from LangGraph's saved messages."""
    messages = snapshot_for(thread_id).values.get("messages", [])
    result = []
    for message in messages:
        if isinstance(message, HumanMessage):
            result.append(("user", text_of(message.content)))
        elif isinstance(message, AIMessage):
            # Tool-call messages aren't user-facing answers.
            if getattr(message, "tool_calls", None):
                continue
            content = text_of(message.content)
            if content.strip():
                result.append(("assistant", content))
    return result


def interrupt_description(interruption):
    value = interruption.value
    if isinstance(value, dict):
        return value.get("question") or str(value)
    return str(value)


def run_graph(graph_input, thread_id):
    """Stream text while allowing tool execution and interrupt persistence."""
    status = None
    try:
        with st.chat_message("assistant"):
            def assistant_tokens():
                nonlocal status
                for chunk, metadata in chatbot.stream(
                    graph_input,
                    config=config_for(thread_id),
                    stream_mode="messages",
                ):
                    if isinstance(chunk, ToolMessage):
                        name = getattr(chunk, "name", None) or "tool"
                        if status is None:
                            status = st.status(f"🔧 Using {name}...", expanded=False)
                        else:
                            status.update(label=f"🔧 Using {name}...", state="running")
                    elif isinstance(chunk, AIMessage):
                        # Exclude tool calls, including their streamed argument fragments.
                        if getattr(chunk, "tool_calls", None) or getattr(chunk, "tool_call_chunks", None):
                            continue
                        # Skip non-chat nodes if their metadata is present.
                        node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
                        if node and node != "chat_node":
                            continue
                        content = text_of(chunk.content)
                        if content:
                            yield content

            st.write_stream(assistant_tokens())
            interruption = pending_interrupt(thread_id)
            if interruption:
                st.warning("⏸️ Human approval required. Use the buttons below.")
            if status is not None:
                status.update(
                    label="⏸️ Waiting for approval" if interruption else "✅ Tools finished",
                    state="complete", expanded=False,
                )
        st.rerun()
    except Exception as exc:
        st.error(f"Chat execution failed: {exc}")


if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = list(dict.fromkeys(get_all_threads()))
if "thread_id" not in st.session_state:
    new_thread()
if st.session_state.thread_id not in st.session_state.chat_threads:
    st.session_state.chat_threads.append(st.session_state.thread_id)

with st.sidebar:
    st.title("My Conversations")
    if st.button("➕ New Chat", use_container_width=True):
        new_thread()
        st.rerun()
    for tid in reversed(st.session_state.chat_threads):
        if st.button(tid, key=f"thread_{tid}", use_container_width=True):
            st.session_state.thread_id = tid
            st.rerun()

thread_id = st.session_state.thread_id

# Render persisted history on each run. No duplicate Streamlit-only history.
for role, content in history_for(thread_id):
    with st.chat_message(role):
        st.markdown(content)

interruption = pending_interrupt(thread_id)
if interruption:
    st.warning("🧑 Human approval required: " + interrupt_description(interruption))
    yes_col, no_col = st.columns(2)
    with yes_col:
        approve = st.button("✅ Approve Purchase", type="primary", use_container_width=True)
    with no_col:
        reject = st.button("❌ Reject Purchase", use_container_width=True)
    if approve or reject:
        # Same thread_id as the original interrupted invocation is essential.
        run_graph(Command(resume="yes" if approve else "no"), thread_id)

submission = st.chat_input(
    "Ask a question or attach PDF files",
    accept_file="multiple",
    file_type=["pdf"],
    disabled=interruption is not None,
)

if submission:
    user_text = (submission.text or "").strip()
    uploaded_files = submission.files or []
    indexed = []

    for uploaded_pdf in uploaded_files:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                temp.write(uploaded_pdf.getvalue())
                temp_path = temp.name
            with st.spinner(f"Indexing {uploaded_pdf.name}..."):
                result = ingest_rag_document(
                    file_path=temp_path,
                    thread_id=thread_id,
                    filename=uploaded_pdf.name,
                )
            indexed.append(uploaded_pdf.name)
            st.toast(result, icon="✅")
        except Exception as exc:
            st.error(f"Could not index {uploaded_pdf.name}: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    if not user_text and indexed:
        user_text = "Summarize the uploaded PDF document(s): " + ", ".join(indexed)

    if not user_text and uploaded_files and not indexed:
        st.warning("No PDFs were indexed. Please retry with a text-based PDF.")

    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
        run_graph({"messages": [HumanMessage(content=user_text)]}, thread_id)