import streamlit as st
import random
import uuid
from langchain_core.messages import HumanMessage

from io import BytesIO
from fpdf import FPDF
from groq import Groq
from streamlit_chat import message
from agent import Agent
from main import get_response

def initialize_agent():
    if 'agent' not in st.session_state:
        st.session_state.agent = Agent()

def process_input(user_input):
    initialize_agent()
    thread_id = str(uuid.uuid4())
    st.session_state.thread_id = thread_id
    messages = [HumanMessage(content=user_input)]
    config = {'configurable': {'thread_id': thread_id}}
    result = st.session_state.agent.graph.invoke({'messages': messages}, config=config)
    response = result['messages'][-1].content
    return response

# Initialize the Groq client
client = Groq()

# Function to connect to ChatGroq and get a summarized response
def get_groq_response(messages):
    completion = client.chat.completions.create(
        model="llama3-groq-70b-8192-tool-use-preview",
        messages=messages,
        temperature=0.5,
        top_p=0.65,
        stream=True,
        stop=None,
    )

    # Collect response chunks
    response_text = ""
    for chunk in completion:
        response_text += chunk.choices[0].delta.content or ""
    return response_text

# Function to summarize chat history
def summarize_chat_history(chat_history):
    full_conversation = "\n".join(
        [f"User: {msg}\nAssistant: {response}" for msg, response in zip(chat_history["user_inputs"], chat_history["assistant_responses"])]
    )
    # Prepare prompt to summarize conversation
    messages = [
        {"role": "user", "content": f"Summarize the following conversation, use appropiate headings, bullet ponts and tabels if required:\n{full_conversation}"}
    ]
    summary = get_groq_response(messages)
    return summary

def generate_pdf_report(summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Chat Summary Report", 0, 1, "C")
    
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, summary_text)
    
    # Save PDF in a BytesIO object
    pdf_buffer = BytesIO()
    pdf_buffer.write(pdf.output(dest='S').encode('latin1'))
    pdf_buffer.seek(0)
    return pdf_buffer



st.header("HR Chatbot")
st.markdown("Ask your HR-related questions here.")

if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "input_message_key" not in st.session_state:
    st.session_state["input_message_key"] = str(random.random())

chat_container = st.container()

user_input = st.text_input("Type your message and click send.", key=st.session_state["input_message_key"])

if st.button("Send"):
    response = process_input(user_input)

    st.session_state["past"].append(user_input)
    st.session_state["generated"].append(response)

    st.session_state["input_message_key"] = str(random.random())

    st.experimental_rerun()

if st.session_state["generated"]:
    with chat_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))

    if st.button("Generate Report"):
        # Combine both user and assistant messages for summarization
        chat_history = {
            "user_inputs": st.session_state["past"],
            "assistant_responses": st.session_state["generated"]
        }

        # Get summary of the conversation
        summary_text = summarize_chat_history(chat_history)

        # Generate PDF from the summary
        pdf_buffer = generate_pdf_report(summary_text)

        # Provide download link for the summarized PDF
        st.download_button(
            label="Download Summarized Report as PDF",
            data=pdf_buffer,
            file_name="summarized_chat_report.pdf",
            mime="application/pdf"
        )

