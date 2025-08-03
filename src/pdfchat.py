import streamlit as st
import os
import tempfile
import pathlib
import traceback
import time
import json
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables from the neurofetch-ui directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Import all agents
from agents.structured_data_agent import StructuredDataExtractionAgent
from agents.query_reformulation_agent import QueryReformulationAgent
from agents.retrieval_agent import AdaptiveRetrievalAgent

# Initialize all agents
structured_agent = StructuredDataExtractionAgent()
query_reformulation_agent = QueryReformulationAgent()
retrieval_agent = AdaptiveRetrievalAgent()  # Will be updated with vectorstore later

def get_agent_display_name(agent_id):
    agent_names = {
        "structured_data_extraction": "📊 Structured Data Agent",
        "query_reformulation": "🔄 Query Reformulation Agent",
        "adaptive_retrieval": "🔍 Adaptive Retrieval Agent",
        "rag_system": "🤖 RAG System"
    }
    return agent_names.get(agent_id, f"Agent: {agent_id}")

def get_user_info():
    try:
        token = st.query_params.get("token")
        if isinstance(token, list):
            token = token[0]
        if not token:
            return None
        response = requests.get(
            "http://localhost:3000/api/auth/verify",
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.ok:
            user_data = response.json()
            return user_data.get("user", {})
        else:
            return None
    except Exception as e:
        st.error(f"Error getting user info: {e}")
        return None

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.markdown(
        """
        <meta http-equiv="refresh" content="0; url='http://localhost:5173'" />
        <script>
            window.location.replace('http://localhost:5173');
        </script>
        """,
        unsafe_allow_html=True
    )

def get_document_text_and_images(uploaded_files):
    all_text = ""
    temp_dir = "temp_uploaded_files"
    os.makedirs(temp_dir, exist_ok=True)
    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            st.error(f"Error saving temporary file {uploaded_file.name}: {e}")
            continue
        loader = None
        try:
            if file_extension == ".pdf":
                pdf_reader = PdfReader(temp_file_path)
                pdf_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                all_text += pdf_text + "\n"
            elif file_extension == ".csv":
                from langchain_community.document_loaders import CSVLoader
                loader = CSVLoader(file_path=temp_file_path, encoding="utf-8")
            elif file_extension == ".txt" or file_extension == ".md":
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path=temp_file_path, encoding="utf-8")
            else:
                st.warning(f"Unsupported file type: {file_extension}. Skipping {uploaded_file.name}.")
                continue
            if loader:
                documents = loader.load()
                loader_text = "\n".join(doc.page_content for doc in documents)
                all_text += loader_text + "\n"
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    try:
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except OSError:
        pass
    return all_text, []

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    try:
        embeddings = OpenAIEmbeddings()
        if not text_chunks:
            st.warning("No text chunks found to create vector store.")
            return None
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_conversation_chain(vectorstore):
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        if vectorstore:
            return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory
            )
        return None
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def handle_user_input(user_question):
    if st.session_state.conversation is None and not st.session_state.get('current_pdf_filename'):
        st.error("Please process your documents first using the sidebar before asking questions.")
        return
    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({'role': 'user', 'content': user_question})
    with st.spinner("🔄 Reformulating query..."):
        reformulated_query = query_reformulation_agent.process({
            "query": user_question,
            "context": "document_qa"
        })
        if reformulated_query["success"]:
            final_query = reformulated_query["data"]["primary_query"]
            st.session_state.chat_history.append({
                'role': 'system', 
                'content': f"Query reformulated by {get_agent_display_name('query_reformulation')}"
            })
        else:
            final_query = user_question
    data_type = structured_agent.detect_data_type(final_query)
    pdf_filename = st.session_state.get('current_pdf_filename')
    pdf_path = os.path.join("temp_uploaded_files", pdf_filename) if pdf_filename else None
    if data_type in ["table", "chat"] and pdf_path and os.path.exists(pdf_path):
        with st.spinner(f"📊 Extracting {data_type}s using {get_agent_display_name('structured_data_extraction')}..."):
            result = structured_agent.process({
                "pdf_path": pdf_path,
                "data_type": data_type,
                "pages": "all"
            })
            if result["success"]:
                agent_used = get_agent_display_name(result["agent_id"])
                if data_type == "table" and result["data"].get("tables"):
                    for i, table in enumerate(result["data"]["tables"]):
                        df = pd.DataFrame(table["data"]) if isinstance(table, dict) and "data" in table else pd.DataFrame(table)
                        st.session_state.chat_history.append({
                            'role': 'bot',
                            'content': f'Table {i+1}:<br>{df.to_html(index=False, classes="table-auto w-full text-xs")}',
                        })
                elif data_type == "chat" and result["data"].get("chat_segments"):
                    for segment in result["data"]["chat_segments"]:
                        st.session_state.chat_history.append({
                            'role': 'bot',
                            'content': f'<pre>{segment}</pre>',
                        })
                else:
                    st.session_state.chat_history.append({
                        'role': 'bot',
                        'content': f'No {data_type}s could be extracted from the PDF. (Agent: {agent_used})'
                    })
            else:
                st.session_state.chat_history.append({
                    'role': 'bot',
                    'content': f'Failed to extract {data_type}s.'
                })
        display_chat_history()
        return
    if st.session_state.conversation:
        with st.spinner(f"🔍 Searching documents using {get_agent_display_name('adaptive_retrieval')}..."):
            retrieval_result = retrieval_agent.process({
                "queries": [final_query],
                "original_query": final_query
            })
            if retrieval_result["success"]:
                enhanced_query = final_query
                st.session_state.chat_history.append({
                    'role': 'system', 
                    'content': f"Query processed by {get_agent_display_name('adaptive_retrieval')}"
                })
            else:
                enhanced_query = final_query
        try:
            with st.spinner("🤖 Generating response..."):
                start_time = time.time()
                response = st.session_state.conversation({'question': enhanced_query})
                end_time = time.time()
                fetch_time = end_time - start_time
                response_content = response['answer']
                agent_used = get_agent_display_name('rag_system')
                response_with_agent = f"{response_content}\n\n---\n*Response generated by {agent_used} in {fetch_time:.2f} seconds*"
                st.session_state.chat_history.append({
                    'role': 'bot', 
                    'content': response_with_agent
                })
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
            import sys
            print("--- Detailed Error Traceback ---")
            traceback.print_exc(file=sys.stdout)
            print("------------------------------")
            if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'user':
                st.session_state.chat_history.pop()
    else:
        st.error("No conversation chain available. Please process documents first.")
    display_chat_history()

def display_chat_history():
    # Inject Tailwind and custom styles (from ai.html)
    st.markdown('''
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
        .dark-mode { background-color: #1e293b; color: #e2e8f0; }
        .dark-mode .bg-white { background-color: #334155; }
        .dark-mode .text-gray-800 { color: #e2e8f0; }
        .dark-mode .text-gray-600 { color: #94a3b8; }
        .dark-mode .text-gray-500 { color: #cbd5e1; }
        .dark-mode .text-gray-400 { color: #94a3b8; }
        .dark-mode .hover\:bg-gray-100:hover { background-color: #475569; }
        .dark-mode .border-gray-200 { border-color: #475569; }
        .dark-mode .hover\:text-gray-900:hover { color: #f1f5f9; }
        .dark-mode .ring-gray-300 { ring-color: #475569; }
        .dark-mode .focus\:ring-indigo-500:focus { ring-color: #818cf8; }
        .dark-mode .focus\:border-indigo-500:focus { border-color: #818cf8; }
        .dark-mode .upgrade-button { background: linear-gradient(180deg, #4F46E5 0%, #C026D3 100%); }
        .upgrade-button { writing-mode: vertical-rl; text-orientation: mixed; }
        .rounded-xl { border-radius: 0.75rem; }
        .rounded-lg { border-radius: 0.5rem; }
        .rounded-md { border-radius: 0.375rem; }
        .rounded-full { border-radius: 9999px; }
    </style>
    ''', unsafe_allow_html=True)
    # Main chat area
    st.markdown('<div class="flex h-screen dark-mode">', unsafe_allow_html=True)
    # Sidebar (static for now)
    st.markdown('''
    <div class="w-80 bg-white p-6 flex flex-col justify-between rounded-r-xl shadow-lg">
      <div>
        <div class="flex items-center justify-between mb-8">
          <h1 class="text-2xl font-bold text-gray-800">CHAT <span class="text-indigo-500">A.I+</span></h1>
        </div>
        <button class="w-full bg-indigo-500 text-white py-3 px-4 rounded-lg flex items-center justify-center text-sm font-semibold hover:bg-indigo-600 transition-colors mb-6">
          <span class="material-icons mr-2">add</span>New chat
        </button>
        <div class="flex justify-between items-center mb-3">
          <h2 class="text-xs text-gray-500 font-semibold uppercase">Your conversations</h2>
          <button class="text-xs text-indigo-500 hover:text-indigo-600 font-semibold">Clear All</button>
        </div>
        <nav class="space-y-2">
          <a class="flex items-center px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-md" href="#">Example conversation 1</a>
          <a class="flex items-center px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-md" href="#">Example conversation 2</a>
        </nav>
      </div>
      <div class="border-t border-gray-200 pt-6">
        <a class="flex items-center px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-md" href="#"><span class="material-icons mr-3">settings</span>Settings</a>
      </div>
    </div>
    ''', unsafe_allow_html=True)
    # Chat area
    st.markdown('<div class="flex-1 p-8 overflow-y-auto"><div class="max-w-3xl mx-auto">', unsafe_allow_html=True)
    # Render chat history
    for message in st.session_state.get('chat_history', []):
        if message['role'] == 'user':
            st.markdown(f'''
            <div class="flex items-start space-x-4 mb-8">
              <img alt="User avatar" class="w-8 h-8 rounded-full" src="https://lh3.googleusercontent.com/aida-public/AB6AXuCTKiKgYUhUJLTATbsBocmlwAhJ6KXz5gEWfWRhjlZGFdzj1rNYaNe97XyAXYqwhMeXAXjGqYUJtckcex3AiBHPeKbBJr3h1ZhLKp_IoX-WMgqIsfxP_ybp0cj3fZsTns7wG5H-ahlADU69aAhkFLOucZYzM4XQ_txofc3u92jEcbRAORl7-P1_ylWbMCWGCymlGIgN7YoRr1oraSvTW8e-YoI77ldyJKcScYH80ffZ-9XA1Q3LsxC4jhlSOmmgYxw4jqOhlZoqdmZW"/>
              <div>
                <p class="font-semibold text-gray-800">You</p>
                <div class="bg-white p-4 rounded-lg mt-1 shadow">
                  <p class="text-sm text-gray-800">{message['content']}</p>
                </div>
              </div>
            </div>
            ''', unsafe_allow_html=True)
        elif message['role'] == 'bot':
            st.markdown(f'''
            <div class="flex items-start space-x-4 mb-8">
              <img alt="AI avatar" class="w-8 h-8 rounded-full bg-indigo-500 p-1" src="https://lh3.googleusercontent.com/aida-public/AB6AXuACOkTUYP_YJHCYbH9gYY362HMhRHDJxlH3GhLfd5oJNhyQAnG_laWj_p7N4C2irUjcAYc6B1ugNvBoZuUFsmLn5YjEXQdy31Gq_UDFXbU7BOTitTrpa5ZaIDzq8YTzAPNSj0N_irs8fDnXeNI-4YpcFTjDxU_tKpNKtk48-k3dfvwE3L9vvWqHlKn2qwpi45eCxbifP6e3FfO82eRsmU-_j-HWkTm_QquWrIkHAwo6m9-kt9EYxiGaNDDbzDzgSP6Dwg-MK9VSaVzV"/>
              <div>
                <p class="font-semibold text-gray-800">CHAT <span class="text-indigo-500">A.I+</span></p>
                <div class="bg-white p-4 rounded-lg mt-1 shadow">
                  <p class="text-sm text-gray-800">{message['content']}</p>
                </div>
              </div>
            </div>
            ''', unsafe_allow_html=True)
        elif message['role'] == 'system':
            st.info(f"⚙️ {message['content']}")
    # Sticky input at bottom
    st.markdown('''
      <div class="sticky bottom-0 pb-8 bg-opacity-80 backdrop-filter backdrop-blur-md dark-mode:bg-opacity-80 dark-mode:backdrop-filter dark-mode:backdrop-blur-md dark-mode:bg-slate-800">
        <form action="#" method="post">
        <div class="relative">
          <input name="user_input" id="user_input" class="w-full py-4 pl-6 pr-16 text-sm text-gray-800 bg-white border border-gray-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 shadow-sm dark-mode:placeholder-gray-400" placeholder="What's in your mind..." type="text" autocomplete="off"/>
          <button type="submit" class="absolute right-3 top-1/2 transform -translate-y-1/2 bg-indigo-500 text-white p-2.5 rounded-lg hover:bg-indigo-600 transition-colors">
            <span class="material-icons">send</span>
          </button>
        </div>
        </form>
      </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div></div></div>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat A.I+ PDF", page_icon="🤖", layout="wide")
    user_info = get_user_info()
    if user_info:
        name = user_info.get("name") or "User"
        st.markdown(f"<span style='font-size:1.1rem;'>👤 <b>{name}</b></span>", unsafe_allow_html=True)
        if st.button("Logout", key="logout_btn", help="Logout"):
            logout()
    else:
        st.error("⚠️ Authentication required. Please login through the main application.")
        st.stop()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "images" not in st.session_state:
        st.session_state.images = []
    display_chat_history()
    # Handle user input from the custom HTML form
    user_input = st.experimental_get_query_params().get("user_input", [""])[0]
    if user_input:
        handle_user_input(user_input)
    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "Upload (PDF, CSV, TXT, MD) and click 'Process'",
            accept_multiple_files=True,
            type=["pdf", "csv", "txt", "md"]
        )
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    raw_text, images = get_document_text_and_images(uploaded_files)
                    if not raw_text.strip():
                        st.error("No text could be extracted. Please check your documents.")
                        st.session_state.conversation = None
                        st.session_state.images = []
                        st.session_state.chat_history = []
                        return
                    vectorstore = None
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                    if vectorstore:
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.session_state.images = []
                        retrieval_agent.update_vectorstore(vectorstore)
                        if st.session_state.conversation:
                            st.success("Documents processed successfully! You can now ask questions.")
                            st.session_state.chat_history = []
                        else:
                            st.error("Failed to create conversation chain. Check Ollama setup and ensure 'llama3' model is pulled.")
                            st.session_state.conversation = None
                            st.session_state.images = []
                            st.session_state.chat_history = []
                    else:
                        st.error("Failed to process documents. No text found for processing or vector store creation failed.")
                        st.session_state.conversation = None
                        st.session_state.images = []
                        st.session_state.chat_history = []
                for uploaded_file in uploaded_files:
                    if uploaded_file.name.lower().endswith('.pdf'):
                        st.session_state['current_pdf_filename'] = uploaded_file.name
                        break
            else:
                st.warning("Please upload at least one document file.")
                st.session_state.conversation = None
                st.session_state.images = []
                st.session_state.chat_history = []

if __name__ == '__main__':
    main() 