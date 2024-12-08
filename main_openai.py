import logging
from dotenv import load_dotenv
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import re
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage
import time
import tiktoken


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
        logging.debug(f"Initialized chat history for session: {session_id}")
    return st.session_state.store[session_id]

def chat_interface():

    openai_api_key = os.getenv("OPENAI_API_KEY")
    st.subheader("Chat with uploaded documents...")
    
    # Define the system message template
    system_message = "You are a helpful assistant knowledgeable about the uploaded PDF documents."

    # Initialize vectorstore and LLM
    persist_directory = 'chroma_db'
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
      
    llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o-mini")
    logging.info("OpenAI initialized successfully.")

    # Initialize or retrieve the message history
    if 'message_history' not in st.session_state:
        st.session_state.message_history = []
    
    # Display existing messages from the history
    for message in st.session_state.message_history:
        with st.chat_message(message['role']):
            st.write(message['content'])

    # Display chat container
    with st.container():
        user_input = st.chat_input("Ask a question...")  # Chat input field
        
        if user_input:
            # Append user message to the history
            st.session_state.message_history.append({"role": "user", "content": user_input})
            with st.spinner():
    
                # Get response from LLM    # Retrieve similar documents from the vectorstore
                retriever = vectorstore.as_retriever(search_type="similarity")
                logging.info("Retriever initialized from vectorstore.")

                relevant_docs = retriever.invoke(user_input)
                logging.info(f"Relevent documents retreieved from vector db {relevant_docs}")
                

                # Display relevant documents in a proper format
                st.write("### Retrieved Relevant Documents from VectorDB:")
                for idx, doc in enumerate(relevant_docs):
                    st.write(f"**Document {idx + 1}:**")
                    # st.write(doc.page_content.replace("$", "\$"))
                    st.write(' '.join(doc.page_content.replace("$", "\$").split()[:30]) + " ...")
                    st.write(f"- **File Name:** {doc.metadata.get('source', 'Unknown')} || **Page Number:** {doc.metadata.get('page_number', 'Unknown')}")
                    # st.write(f"- **Page Number:** {doc.metadata.get('page_number', 'Unknown')}")
                    # st.write(f"")
                    st.write("---")
                
                # Combine relevant documents into context
                context = "\n".join(doc.page_content for doc in relevant_docs)

                # Calculate input tokens using tiktoken
                encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
                input_tokens = len(encoding.encode(system_message + user_input + context))
                logging.info(f"input token count*****************: {input_tokens}")

                # Create messages
                system_message_obj = SystemMessage(content=system_message)
                user_message_obj = HumanMessage(content=f"{user_input}\n\nContext:\n{context}")

                # Generate response
                response = llm([system_message_obj, user_message_obj]).content

                # Calculate output tokens using tiktoken
                output_tokens = len(encoding.encode(response))
                logging.info(f"output token count*****************: {output_tokens}")
                total_tokens = input_tokens + output_tokens
                # Append assistant response to the history
                st.session_state.message_history.append({"role": "assistant", "content": response})
                logging.info(f"**********response: {response}")
                
                # Extract token usage and time from response metadata
                # Note: The ChatOpenAI class does not have 'get_response_metadata' method
                response_metadata = {}
                usage_metadata = {}
                # input_tokens is calculated above
                # output_tokens is calculated above
                # total_tokens is calculated above
                total_time = 'N/A'
                      
                # Debug: Print the entire response object and additional_kwargs
                logging.debug(f"Full response object: {response}")
                logging.debug(f"response metadata : {response_metadata}")

                # Display user input and assistant response in chat
                with st.chat_message("user"):
                    st.write(user_input)

                with st.chat_message("assistant"):
                    st.write(response.replace("$", "\$"))
                    st.write(f"Input Tokens: {input_tokens} | Output Tokens: {output_tokens} | Total Tokens: {input_tokens +output_tokens }" )

   

# Preprocess the text to perform cleanup
def clean_text(text):
    """Clean text while retaining structure."""

    # Remove unwanted patterns like headers/footers
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'(cid:.*?)', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Define patterns for disclaimers (adjust based on your data)
    disclaimer_patterns = [
        r"(?i)Disclaimer[:\s].*?(?=\n|\Z)",  # Matches lines starting with 'Disclaimer:'
        r"(?i)This document is confidential.*",  # Example pattern for a common disclaimer
        r"(?i)The information contained herein.*"  # Another common pattern
    ]

    # Remove each disclaimer pattern from the text
    for pattern in disclaimer_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    return text

# Function to process the uploaded files
def process_files(uploaded_files):
    
    documents = []
    documents_cleaned = []
    
    #Initialize OpenAI Embeddings
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model = "text-embedding-ada-002" )
    logging.info("Embeddings initialized successfully.")

    persist_directory = 'chroma_db'

    with st.spinner('Processing...'):
        message_placeholder = st.empty()
        for uploaded_file in uploaded_files:
            message_placeholder.write(f"Processing file: {uploaded_file.name}")
            temppdf = uploaded_file.name
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                message_placeholder.write(f"Saved uploaded PDF file: {temppdf} to disk")
            
            message_placeholder.write(f"Text processing started for: {temppdf}")
            # process pdf files
            loader = PyMuPDFLoader(temppdf)
            docs = loader.load()
            logging.info(f"Loaded {len(docs)} documents from PDF: {uploaded_file.name}")
            message_placeholder.write(f"Loaded {len(docs)} documents from PDF: {uploaded_file.name}")
            documents.extend(docs)
            for doc in docs:
                original_text = doc.page_content
                logging.info(f"Original text: {original_text}")
                cleaned_text = clean_text(original_text)
                logging.info(f"Cleaned text: {cleaned_text}")
                doc.page_content = cleaned_text
                doc.metadata["source"] = temppdf
                doc.metadata["page_number"] = doc.metadata.get("page", "Unknown")
                logging.info(f"\n Source: {doc.metadata['source'] } Page Number: {doc.metadata['page_number']}")
                documents_cleaned.append(doc)
            message_placeholder.write(f"Text processing completed for: {temppdf}")

        try:
            # Split and create embeddings for the documents
            message_placeholder.write(f"Processing recursive split and generating chunks...")
            #logging.info(documents_cleaned)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents_cleaned)
            logging.info(f"Split documents into {len(splits)} chunks.")
        except Exception as e:
            logging.error(f"Error splitting documents into chunks:\n{e}")
            message_placeholder.write(f"{e}")
        else:
            message_placeholder.write(f"{len(splits)} Chunks generated successfully...")

        # Embed documents and create a vector store
        message_placeholder.write(f"Enbedding chunks has started")            
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logging.info("Chroma vectorstore created successfully.")
        message_placeholder.write(f"Enbeddings are generated and stored in vectordb {vectorstore}")            
            
   
            # Add further file processing logic here

def main():
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # ## Langsmith Tracking
    # os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
    # os.environ["LANGCHAIN_TRACING_V2"]="true"
    # os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
    load_dotenv()
    ## set up Streamlit 
    st.title("GGU-Upgrad - Gen AI Assignment01")
    st.write("NVIDIA Annual Reports Analysis for the years 2022 and 2023")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize session state for uploaded files
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    uploaded_files = st.file_uploader("Drag and drop files here or click to upload", type="pdf", accept_multiple_files=True)
    # message_placeholder = st.empty()
    # Check if new files are uploaded
    if uploaded_files and uploaded_files != st.session_state.uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"{len(uploaded_files)} new file(s) uploaded!")
        process_files(uploaded_files)
    elif not uploaded_files:
        st.info("No files selected. Please upload a file to proceed.")
    
    session_id = st.sidebar.text_input("Session ID", value="Session1")
    logging.debug(f"Session ID set to: {session_id}")
    ## statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}
        logging.debug("Initialized session state store.")

    # Initialize vectorstore and LLM
    persist_directory = 'chroma_db'
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # user_input = st.chat_input("Ask a question...")  # Chat input field
    if st.session_state.uploaded_files or vectorstore:
        # Display the chat interface only when there are uploaded files
        chat_interface()

if __name__ == "__main__":
    main()
