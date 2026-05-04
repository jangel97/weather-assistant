"""Streamlit chat UI for the weather assistant."""

import httpx
import streamlit as st

AGENT_URL = "http://localhost:8002"
CHAT_ENDPOINT = f"{AGENT_URL}/weather/chat"

st.set_page_config(page_title="Weather Assistant", page_icon=":sun_with_face:")
st.title(":sun_with_face: Weather Assistant")
st.caption("Powered by small LLMs (8B-20B) via the agent framework")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            with st.expander("Tool calls"):
                for tc in msg["tool_calls"]:
                    st.code(f'{tc["tool"]}({tc.get("arguments", "")})', language="text")

if prompt := st.chat_input("Ask about the weather..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = httpx.post(
                    CHAT_ENDPOINT,
                    json={"message": prompt, "history": history or None},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("message", "")
                tool_calls = data.get("tool_calls") or []
                trace = data.get("trace", {})

                st.markdown(answer)

                if tool_calls:
                    with st.expander("Tool calls"):
                        for tc in tool_calls:
                            st.code(
                                f'{tc["tool"]}({tc.get("arguments", "")})',
                                language="text",
                            )

                if trace:
                    rounds = trace.get("rounds", [])
                    total_ms = trace.get("total_ms", 0)
                    rewritten = trace.get("rewritten_question")
                    with st.expander("Trace"):
                        if rewritten:
                            st.write(f"**Rewritten question:** {rewritten}")
                        for r in rounds:
                            st.write(
                                f"Round {r.get('round')}: "
                                f"action={r.get('action')} "
                                f"tool={r.get('tool', '-')} "
                                f"({r.get('classifier_ms', 0)}+"
                                f"{r.get('selector_ms', 0)}+"
                                f"{r.get('argument_ms', 0)}ms)"
                            )
                        st.write(f"**Total:** {total_ms}ms")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "tool_calls": tool_calls,
                })

            except httpx.ConnectError:
                st.error(
                    "Cannot reach the agent at "
                    f"{AGENT_URL}. Start it with: "
                    "`python -m weather_assistant.main`"
                )
            except Exception as e:
                st.error(f"Error: {e}")
