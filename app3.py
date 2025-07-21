import os
import tempfile
import streamlit as st
from decouple import config
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
persist_directory = 'db'

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    os.remove(temp_file_path)
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    chunks = text_spliter.split_documents(documents=docs)
    return chunks

def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
        return vector_store
    return None

def add_to_vector_store(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
    return vector_store

def ask_question(model, query, vector_store):
    llm = ChatOpenAI(model=model)
    retriever = vector_store.as_retriever()
    system_prompt = '''
    Use the context to answer the questions.
    If you cannot find an answer in the context,
    explain that there is no information available.
    Respond in markdown format and with elaborate and interactive visualizations.
    Context: {context}
    '''
    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', '{input}'))
    prompt = ChatPromptTemplate.from_messages(messages)
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    response = chain.invoke({'input': query})
    return response.get('answer')

def get_chat_history_txt():
    lines = []
    for m in st.session_state.messages:
        user = "User" if m["role"] == "user" else "AI"
        lines.append(f"{user}: {m['content']}")
    return "\n".join(lines)

def get_chat_history_md():
    lines = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            lines.append(f"**User:** {m['content']}\n")
        else:
            lines.append(f"**AI:** {m['content']}\n")
    return "\n".join(lines)

try:
    import fpdf
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

def get_chat_history_pdf():
    if not PDF_AVAILABLE:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for m in st.session_state.messages:
        user = "User" if m["role"] == "user" else "AI"
        txt = f"{user}: {m['content']}\n"
        pdf.multi_cell(0, 10, txt)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        tmpfile.seek(0)
        data = tmpfile.read()
    os.unlink(tmpfile.name)
    return data

vector_store = load_existing_vector_store()

st.set_page_config(
    page_title='Chat RAG',
    page_icon='📄',
)

st.header('🤖 Chat with your documents (RAG)')

with st.sidebar:
    st.header('Upload files 📄')
    uploaded_files = st.file_uploader(
        label='Upload PDF files',
        type=['pdf'],
        accept_multiple_files=True,
    )
    if uploaded_files:
        with st.spinner('Processing documents...'):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(file=uploaded_file)
                all_chunks.extend(chunks)
            vector_store = add_to_vector_store(
                chunks=all_chunks,
                vector_store=vector_store,
            )

    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    ]
    selected_model = st.sidebar.selectbox(
        label='Select the LLM model',
        options=model_options,
    )

    st.markdown("---")
    st.subheader("Chat History Options")

    # Botões lado a lado
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset Chat History"):
            st.session_state['messages'] = []
            st.success("Chat history reset!")
    with col2:
        download_clicked = st.button("Download Chat History")

    # Seleção de formato aparece somente quando clicar em download
    if download_clicked:
        if 'messages' in st.session_state and st.session_state['messages']:
            format_option = st.selectbox(
                "Select file format", 
                options=["txt", "md"] + (["pdf"] if PDF_AVAILABLE else []), 
                key="download_format"
            )
            if format_option == "txt":
                st.download_button(
                    label="Download .txt",
                    data=get_chat_history_txt(),
                    file_name="chat_history.txt",
                    mime="text/plain"
                )
            elif format_option == "md":
                st.download_button(
                    label="Download .md",
                    data=get_chat_history_md(),
                    file_name="chat_history.md",
                    mime="text/markdown"
                )
            elif format_option == "pdf" and PDF_AVAILABLE:
                pdf_data = get_chat_history_pdf()
                if pdf_data:
                    st.download_button(
                        label="Download .pdf",
                        data=pdf_data,
                        file_name="chat_history.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.warning("PDF export unavailable. Install 'fpdf'.")
        else:
            st.info("No chat history to download.")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

question = st.chat_input('How can I help you?')

if vector_store and question:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))
    st.chat_message('user').write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})

    with st.spinner('Searching for an answer...'):
        response = ask_question(
            model=selected_model,
            query=question,
            vector_store=vector_store,
        )
        st.chat_message('ai').write(response)
        st.session_state.messages.append({'role': 'ai', 'content': response})

