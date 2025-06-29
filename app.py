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

#Diretório de RAG que criamos para armazenar os vetores
persist_directory = 'db'


# Função para processar o arquivo PDF e dividir em chunks
# Utiliza o PyPDFLoader para carregar o PDF e o RecursiveCharacterTextSplitter para dividir
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    os.remove(temp_file_path)

    # Dividir os documentos em chunks
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    chunks = text_spliter.split_documents(documents=docs)
    return chunks

# Função para carregar o vetor store existente, se existir
def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
        return vector_store
    return None

# Função para adicionar os chunks ao vetor store
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

# Função para fazer a pergunta ao modelo e obter a resposta
# Utiliza o modelo LLM selecionado e o vetor store para buscar a resposta
def ask_question(model, query, vector_store):
    llm = ChatOpenAI(model=model)
    retriever = vector_store.as_retriever()

    system_prompt = '''
    Use o contexto para responder as perguntas.
    Se não encontrar uma resposta no contexto,
    explique que não há informações disponíveis.
    Responda em formato de markdown e com visualizações
    elaboradas e interativas.
    Contexto: {context}
    '''
    # O prompt do sistema é uma mensagem que define o comportamento da IA
    # O prompt é a primeira mensagem a ser passada para a IA
    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    response = chain.invoke({'input': query})
    return response.get('answer')

#Essa função é inicizalizada junto com a aplicação, para carregar o vetor store existente
vector_store = load_existing_vector_store()

st.set_page_config(
    page_title='Chat PyGPT',
    page_icon='📄',
)

# Título do aplicativo
st.header('🤖 Chat com seus documentos (RAG)')

# Barra lateral para upload de arquivos PDF
with st.sidebar:
    st.header('Upload de arquivos 📄')
    uploaded_files = st.file_uploader(
        label='Faça o upload de arquivos PDF',
        type=['pdf'],
        accept_multiple_files=True,
    )

    #Função para receber os arquivos e quebrar em chunks e adicionar ao vetor store
    if uploaded_files:
        with st.spinner('Processando documentos...'):
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
        label='Selecione o modelo LLM',
        options=model_options,
    )

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Barra de mensagem para enviar perguntas
question = st.chat_input('Como posso ajudar?')


# O session state armazena o histórico de conversa
if vector_store and question:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))


    # Exibir a pergunta do usuário
    # O que o usuário digitou é armazenado em `question` com um ícone
    # Caso tenha sido o user, exibe um ícone de humano. Caso tenha sido o AI, exibe um ícone de robô.
    st.chat_message('user').write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})

# Quando o usuário envia uma pergunta, chama a função ask_question
    with st.spinner('Buscando resposta...'):
        response = ask_question(
            model=selected_model,
            query=question,
            vector_store=vector_store,
        )

        st.chat_message('ai').write(response)
        st.session_state.messages.append({'role': 'ai', 'content': response})
