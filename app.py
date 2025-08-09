# app.py
import streamlit as st
from main import SalesSystem
import json
import re
import pandas as pd
# from utils.helpers import generate_markdown_output

# ---------- Helper Function ----------
def prettify_response(raw_text):
    """
    Format assistant response as either:
    - A pandas DataFrame (if JSON list of dicts)
    - A pretty JSON string (if single JSON object)
    - Markdown-enhanced string (fallback)
    """
    try:
        # Clean out triple backticks or language tags like ```json
        cleaned_text = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE)

        # Try parsing as list of dictionaries
        parsed = json.loads(cleaned_text)
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            return pd.DataFrame(parsed)
    except Exception:
        pass

    # Try parsing as single JSON object
    try:
        parsed = json.loads(cleaned_text)
        return "```json\n" + json.dumps(parsed, indent=2) + "\n```"
    except Exception:
        pass

    # Markdown enhancements for plain text
    raw_text = re.sub(r'(?m)^([A-Za-z _\-]+):\s*(.+)$', r'**\1**: \2', raw_text)
    raw_text = re.sub(r'(?m)^- ', '• ', raw_text)
    raw_text = re.sub(r'(?m)^#{1,6} (.+)', r'**\1**', raw_text)

    return raw_text

# ---------- Page Config ----------
st.set_page_config(
    page_title="AI Sales System",
    page_icon="",
    layout="centered"
)

# ---------- Session State ----------
if "sales_system" not in st.session_state:
    st.session_state.sales_system = None
    st.session_state.system_ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Header ----------
st.title("AI Sales System")
st.markdown("---")

# ---------- System Initialization ----------
if not st.session_state.system_ready:
    with st.spinner("Initializing Sales System..."):
        try:
            st.session_state.sales_system = SalesSystem()
            status = st.session_state.sales_system.get_system_status()

            if all(s == "Ready" for s in status.values()):
                st.session_state.system_ready = True
                st.success("System initialized successfully!")
            else:
                st.error("System initialization failed")
                st.write("**Status:**")
                for component, status_msg in status.items():
                    st.write(f"- {component}: {status_msg}")
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")

# ---------- System Status ----------
if st.session_state.system_ready:
    with st.expander("System Status"):
        status = st.session_state.sales_system.get_system_status()
        for component, status_msg in status.items():
            st.write(f"**{component}**: {status_msg}")

# ---------- Help Section ----------
with st.expander(" Example Queries"):
    st.markdown("""
    ** Prospecting:**
    - Find computer contractors in Texas
    - Show me businesses with low local presence
    - Find IT companies missing Google Places listings

    ** Analysis:**
    - Get details for [Business Name]
    - Analyze [Business Name]

    ** Communication:**
    - Draft an email for [Business Name]
    - Write a sales message for [Business Name]
    """)

# ---------- Chat Interface ----------
st.markdown("###  Chat")

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# ---------- Chat Input ----------
if prompt := st.chat_input("Enter your query..."):
    if not st.session_state.system_ready:
        st.error(" System not ready. Please wait for initialization.")
    else:
        # Add user input to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    result = st.session_state.sales_system.run_query(prompt)
                    # res_formatted = generate_markdown_output(result)      
                    if result and 'agent_out' in result:
                        response = result['agent_out']['messages'][0].content
                        pretty_response = prettify_response(response)

                        # Show as table if DataFrame
                        if isinstance(pretty_response, pd.DataFrame):
                            st.dataframe(pretty_response, use_container_width=True)
                            # Save as markdown in history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": pretty_response.to_markdown(index=False)
                            })
                        else:
                            st.markdown(pretty_response, unsafe_allow_html=True)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": pretty_response
                            })
                    else:
                        error_msg = " No result returned. Please try a different query."
                        st.markdown(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

                except Exception as e:
                    error_msg = f" Error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# ---------- Clear Chat Button ----------
if st.session_state.messages:
    if st.button(" Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
