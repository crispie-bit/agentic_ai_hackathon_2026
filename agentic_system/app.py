from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agentic_system.services.answering import answer_question
from agentic_system.services.azure_setup import azure_status
from agentic_system.services.aws_setup import aws_status
from agentic_system.services.preparation import WorkspacePreparation
from agentic_system.services.speech_service import speech_ready, speak, transcribe_audio
from agentic_system.services.workspace_store import WorkspaceStore
from agentic_system.services.workday_planner import rank_tasks
from agentic_system.tools.ntulearn_tool import login_to_ntulearn, save_ntulearn_session

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
st.set_page_config(page_title="Agentic Workday OS", page_icon="A", layout="wide")

if "ntulearn_browser" not in st.session_state:
    st.session_state.ntulearn_browser = None
if "ntulearn_ready" not in st.session_state:
    st.session_state.ntulearn_ready = Path(__file__).with_name("ntulearn_session.json").exists()
if "outlook_ready" not in st.session_state:
    st.session_state.outlook_ready = False
if "prepared" not in st.session_state:
    st.session_state.prepared = False
if "demo_ready" not in st.session_state:
    st.session_state.demo_ready = False

store = WorkspaceStore()
preparation = WorkspacePreparation(store)
azure = azure_status()
aws = aws_status()

with st.sidebar:
    st.title("Workday OS")
    st.caption("Personal course and inbox assistant")
    st.divider()
    st.subheader("Connection status")
    st.write(f"{'OK' if azure['ready'] else '!!'} Microsoft Graph")
    st.write(f"{'OK' if aws['ready'] else '!!'} AWS credentials")
    st.write(f"{'OK' if st.session_state.ntulearn_ready else '!!'} NTULearn session")
    st.write(f"{'OK' if st.session_state.outlook_ready else '!!'} Outlook session")
    st.write(f"{'OK' if st.session_state.demo_ready else '--'} Demo data")
    counts = store.counts()
    st.caption(f"Indexed: {counts.get('ntulearn', 0)} course items, {counts.get('outlook', 0)} emails")

st.title("Agentic Workday OS")
st.caption("Your private workspace, prepared from course and calendar data.")

if not st.session_state.prepared:
    st.info("Use fictional demo data now, or connect NTULearn and Outlook when those integrations are available.")
    if st.button("Use safe demo data", type="primary"):
        added = preparation.load_demo_data()
        st.session_state.demo_ready = True
        st.session_state.prepared = True
        st.success(f"Loaded {added} fictional records. No account data was accessed.")
        st.rerun()
    st.divider()
    st.caption("Optional live integrations")
    st.subheader("1. NTULearn")
    st.write("Open the visible browser, complete NTU single sign-on, then save the session.")
    first, second = st.columns(2)
    with first:
        if st.button("Open NTULearn login", type="primary", disabled=st.session_state.ntulearn_browser is not None):
            try:
                st.session_state.ntulearn_browser = login_to_ntulearn(headless=False)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not open NTULearn: {exc}")
    with second:
        if st.button("Save NTULearn login", disabled=st.session_state.ntulearn_browser is None):
            try:
                save_ntulearn_session(st.session_state.ntulearn_browser)
                st.session_state.ntulearn_ready = True
                st.session_state.ntulearn_browser.close()
                st.session_state.ntulearn_browser = None
                st.success("NTULearn session saved.")
            except Exception as exc:
                st.error(f"Could not save NTULearn session: {exc}")

    st.subheader("2. Outlook")
    if not azure["ready"]:
        st.warning("Add MICROSOFT_TENANT_ID, MICROSOFT_CLIENT_ID, and MICROSOFT_REDIRECT_URI to .env before Outlook login.")
    if st.button("Sign in and sync Outlook", disabled=not bool(azure["ready"])):
        try:
            with st.spinner("Opening Microsoft sign-in and reading your mailbox..."):
                added = preparation.run_outlook_sync()
            st.session_state.outlook_ready = True
            st.success(f"Outlook connected. Indexed {added} messages.")
        except Exception as exc:
            st.error(f"Outlook sync failed: {exc}")

    st.subheader("3. Prepare workspace")
    if st.session_state.ntulearn_ready and st.button("Sync NTULearn course content"):
        try:
            with st.spinner("Reading authenticated course pages and indexing materials..."):
                count = preparation.run_ntulearn_sync()
            st.success(f"NTULearn sync complete. Indexed {count} course items.")
        except Exception as exc:
            st.error(f"NTULearn sync failed: {exc}")

    can_prepare = st.session_state.ntulearn_ready
    if st.session_state.ntulearn_ready and not st.session_state.outlook_ready:
        st.caption("NTULearn-only mode is available. Outlook is optional for this demo.")
    if st.button("Prepare with NTULearn only", type="primary", disabled=not can_prepare):
        st.session_state.prepared = True
        st.rerun()
    if not can_prepare:
        st.caption("Complete NTULearn login and save the session first.")
else:
    st.success("Workspace prepared. The assistant is ready.")
    st.subheader("What matters next")
    tasks = rank_tasks(preparation.get_tasks())
    if tasks:
        for task in tasks:
            st.markdown(f"**{task.title}** · due {task.due_label} · {task.estimated_minutes} min")
            st.caption(f"{task.source} · {task.action}")
    else:
        st.info("No structured tasks found yet. Sync course content or load demo data first.")

    question = st.chat_input("Ask about an exam, deadline, email, or document")
    audio = st.audio_input("Or ask by voice")
    if audio is not None:
        if not speech_ready():
            st.warning("Voice dependencies are not installed. Run setup_prereqs.py, then restart the app.")
        else:
            try:
                question = transcribe_audio(audio.getvalue(), ".wav")
                st.info(f"Heard: {question}")
                st.caption("Voice requests are submitted after the recording is stopped. Follow-up prompts appear below.")
            except Exception as exc:
                st.error(f"Voice transcription failed: {exc}")
    if question:
        response = answer_question(question, preparation)
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.markdown(response)
            if speech_ready() and st.button("Speak response", key=f"speak-{question}"):
                try:
                    st.audio(speak(response), format="audio/wav", autoplay=True)
                except Exception as exc:
                    st.error(f"Speech output failed: {exc}")

            st.caption("Useful next questions")
            st.write(" · ".join(preparation.follow_up_prompts()))

    st.subheader("Workspace")
    current_counts = store.counts()
    st.metric("Indexed items", sum(current_counts.values()))
    st.caption("Answers are grounded in locally indexed source content.")
