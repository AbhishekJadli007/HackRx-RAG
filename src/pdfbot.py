import streamlit as st
import os
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
        params = st.query_params
        token = params.get("token")
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
    st.experimental_rerun()

def get_document_text_and_images(uploaded_files):
    all_text = ""
    temp_dir = pathlib.Path("./temp_uploaded_files")
    temp_dir.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files:
        temp_file_path = temp_dir / uploaded_file.name
        file_extension = temp_file_path.suffix.lower()
        st.write(f"[DEBUG] Processing file: {uploaded_file.name}, type: {file_extension}")
        print(f"[DEBUG] Processing file: {uploaded_file.name}, type: {file_extension}")
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            st.error(f"Error saving temporary file {uploaded_file.name}: {e}")
            print(f"[ERROR] Error saving temporary file {uploaded_file.name}: {e}")
            continue
        loader = None
        try:
            if file_extension == ".pdf":
                pdf_reader = PdfReader(temp_file_path)
                pdf_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                st.write(f"[DEBUG] Extracted text length from PDF: {len(pdf_text)}")
                print(f"[DEBUG] Extracted text length from PDF: {len(pdf_text)}")
                all_text += pdf_text + "\n"
            elif file_extension == ".csv":
                from langchain_community.document_loaders import CSVLoader
                loader = CSVLoader(file_path=str(temp_file_path), encoding="utf-8")
            elif file_extension == ".txt" or file_extension == ".md":
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path=str(temp_file_path), encoding="utf-8")
            else:
                st.warning(f"Unsupported file type: {file_extension}. Skipping {uploaded_file.name}.")
                print(f"[WARNING] Unsupported file type: {file_extension}. Skipping {uploaded_file.name}.")
                continue
            if loader:
                documents = loader.load()
                loader_text = "\n".join(doc.page_content for doc in documents)
                st.write(f"[DEBUG] Extracted text length from loader: {len(loader_text)}")
                print(f"[DEBUG] Extracted text length from loader: {len(loader_text)}")
                all_text += loader_text + "\n"
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
            print(f"[ERROR] Error processing file {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    try:
        if temp_dir.exists() and not list(temp_dir.iterdir()):
            temp_dir.rmdir()
    except OSError:
        pass
    st.write(f"[DEBUG] Total extracted text length: {len(all_text)}")
    print(f"[DEBUG] Total extracted text length: {len(all_text)}")
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
        if vectorstore:
            retriever = vectorstore.as_retriever()
            st.session_state.retriever = retriever  # Store retriever for later use
            def get_message_history(session_id: str):
                return InMemoryChatMessageHistory()
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
            )
            return RunnableWithMessageHistory(
                chain,
                get_message_history,
                input_messages_key="question",
                history_messages_key="chat_history"
            )
        return None
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def build_eligibility_prompt(policy_text, user_question):
    return f"""
You are an insurance policy expert. ONLY use the provided policy text to answer the claim scenario below. If the answer is not explicitly stated in the policy, reply: 'Not found in policy.'

Policy Excerpt:
{policy_text}

Claim Scenario:
{user_question}

Instructions:
- Answer 'Yes' or 'No' ONLY if the policy text clearly supports it.
- Quote or reference the exact policy clause/section.
- If the answer is not found, say: 'Not found in policy.'

Answer in the following format:
Eligibility: [Yes/No/Not found in policy]
Justification: [Short explanation, quote or reference the policy clause/section]
"""

# Add a simple keyword re-ranker for policy chunks
KEYWORDS = ["claim", "exclusion", "waiting period", "surgery", "operation", "hospitalization", "pre-existing", "accident", "repudiation", "eligibility", "cover", "covered", "not covered", "benefit", "treatment", "procedure"]
def rerank_policy_chunks(chunks):
    def score(chunk):
        text = chunk.page_content.lower()
        return sum(1 for kw in KEYWORDS if kw in text)
    # Sort by keyword match count, descending
    return sorted(chunks, key=score, reverse=True)

def handle_user_input(user_question):
    if st.session_state.conversation is None and not st.session_state.get('current_pdf_filename'):
        st.error("Please process your documents first using the sidebar before asking questions.")
        st.session_state.chat_history.append({'role': 'bot', 'content': '❗ No document processed. Please upload and process a PDF first.'})
        display_chat_history()
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
            agent_used = get_agent_display_name(result.get("agent_id", "structured_data_extraction"))
            if result["success"]:
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
                        'content': f'❗ No {data_type}s could be extracted from the PDF. (Agent: {agent_used})'
                    })
            else:
                error_msg = result.get('error', f'Failed to extract {data_type}s.')
                st.session_state.chat_history.append({
                    'role': 'bot',
                    'content': f'❗ Extraction error: {error_msg}'
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
            with st.spinner("🤖 Generating eligibility answer..."):
                start_time = time.time()
                # Retrieve the most relevant policy text (top 7 chunks)
                retrieved_docs = st.session_state.retriever.get_relevant_documents(enhanced_query)
                reranked_docs = rerank_policy_chunks(retrieved_docs)
                selected_docs = reranked_docs[:7]
                policy_text = "\n\n".join([doc.page_content for doc in selected_docs])
                # Build the eligibility prompt
                eligibility_prompt = build_eligibility_prompt(policy_text, enhanced_query)
                # Call the LLM directly (bypassing the chain for this special prompt)
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
                eligibility_response = llm(eligibility_prompt)
                end_time = time.time()
                fetch_time = end_time - start_time
                agent_used = get_agent_display_name('rag_system')
                response_with_agent = f"{eligibility_response}\n\n---\n*Response generated by {agent_used} in {fetch_time:.2f} seconds*"
                st.session_state.chat_history.append({
                    'role': 'bot', 
                    'content': response_with_agent
                })
                # Show sources (retrieved policy text)
                st.markdown("<b>Sources (Policy Excerpt):</b>", unsafe_allow_html=True)
                for i, doc in enumerate(selected_docs):
                    st.markdown(f"<div style='background:#f3f4f6; border-radius:0.5em; padding:0.5em; margin-bottom:0.5em;'><b>Chunk {i+1}:</b><br><pre style='white-space:pre-wrap'>{doc.page_content}</pre></div>", unsafe_allow_html=True)
        except Exception as e:
            st.session_state.chat_history.append({
                'role': 'bot',
                'content': f'❗ Error processing your question: {str(e)}'
            })
            st.error(f"Error processing your question: {str(e)}")
            import sys
            print("--- Detailed Error Traceback ---")
            traceback.print_exc(file=sys.stdout)
            print("------------------------------")
            if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'user':
                st.session_state.chat_history.pop()
    else:
        st.session_state.chat_history.append({'role': 'bot', 'content': '❗ No conversation chain available. Please process documents first.'})
        st.error("No conversation chain available. Please process documents first.")
    display_chat_history()

def display_chat_history():
    chat_messages_html = ""
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                chat_messages_html += f'''
                <div class="flex items-start space-x-4 mb-8">
                    <img alt="User avatar" class="w-8 h-8 rounded-full" src="https://i.ibb.co/C8y3Gz2/user-avatar.png"/>
                    <div>
                        <p class="font-semibold text-gray-800 dark:text-gray-200">You</p>
                        <div class="bg-white p-4 rounded-lg mt-1 shadow dark:bg-gray-700">
                            <p class="text-sm text-gray-800 dark:text-gray-200">{message['content']}</p>
                        </div>
                    </div>
                </div>
                '''
            elif message['role'] == 'system':
                chat_messages_html += f'''
                <div class="flex items-start space-x-4 mb-4">
                    <img alt="System avatar" class="w-8 h-8 rounded-full bg-blue-500 p-1" src="https://i.ibb.co/hK8b7XW/system-avatar.png"/>
                    <div>
                        <p class="font-semibold text-blue-500">System</p>
                        <div class="bg-blue-100 p-2 rounded-lg mt-1 shadow dark:bg-blue-900/50">
                            <p class="text-xs text-blue-800 dark:text-blue-200">{message['content']}</p>
                        </div>
                    </div>
                </div>
                '''
            else:  # role == 'bot'
                message_key = f"bot_message_{len(st.session_state.chat_history)}"
                chat_messages_html += f'''
                <div class="flex items-start space-x-4 mb-8">
                    <img alt="AI avatar" class="w-8 h-8 rounded-full bg-indigo-500 p-1" src="https://i.ibb.co/kH65yqF/ai-avatar.png"/>
                    <div>
                        <p class="font-semibold text-gray-800 dark:text-gray-200">CHAT <span class="text-indigo-500">A.I+</span></p>
                        <div class="bg-white p-4 rounded-lg mt-1 shadow dark:bg-gray-700">
                            <div id="{message_key}"></div> </div>
                        <div class="flex items-center space-x-4 mt-3">
                            <button class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                                <span class="material-icons">thumb_up</span>
                            </button>
                            <button class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                                <span class="material-icons">thumb_down</span>
                            </button>
                            <button class="flex items-center text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                                <span class="material-icons mr-1 text-base">refresh</span>
                                Regenerate
                            </button>
                        </div>
                    </div>
                </div>
                '''
    st.session_state.chat_display_html = chat_messages_html

def main():
    st.set_page_config(page_title="NeuroFetch PDFBot", page_icon="🤖", layout="wide")
    # Inject custom CSS
    with open(os.path.join(os.path.dirname(__file__), "streamlit_custom.css")) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "images" not in st.session_state:
        st.session_state.images = []
    if "current_pdf_filename" not in st.session_state:
        st.session_state.current_pdf_filename = None
    if "user_input_value" not in st.session_state:
        st.session_state.user_input_value = ""
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False

    st.title("NeuroFetch PDFBot")
    st.write("Upload your documents and chat with them using AI-powered retrieval and extraction.")

    uploaded_files = st.file_uploader(
        "Upload (PDF, CSV, TXT, MD) and click 'Process'",
        accept_multiple_files=True,
        type=["pdf", "csv", "txt", "md"]
    )
    # Reset processed flag if new files are uploaded
    if uploaded_files:
        st.session_state.documents_processed = False
    # Show success message and green 'Processed' button if processed
    if st.session_state.get("documents_processed"):
        st.success("✅ Documents processed successfully! You can now ask questions.")
        st.markdown('<button style="background-color: #22c55e; color: white; border: none; padding: 0.5em 1.5em; border-radius: 0.5em; font-size: 1.1em; font-weight: bold; cursor: not-allowed;">Processed</button>', unsafe_allow_html=True)
    else:
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Extracting text and building vectorstore (this may take a moment)..."):
                    raw_text, images = get_document_text_and_images(uploaded_files)
                if not raw_text.strip():
                    st.error("No text could be extracted. Please check your documents.")
                    st.session_state.conversation = None
                    st.session_state.images = []
                    st.session_state.chat_history = []
                    st.session_state.documents_processed = False
                    st.write("[DEBUG] No text extracted from documents.")
                    print("[DEBUG] No text extracted from documents.")
                else:
                    with st.spinner("Splitting text into chunks and embedding (please wait)..."):
                        text_chunks = get_text_chunks(raw_text)
                        st.write(f"[DEBUG] Number of text chunks: {len(text_chunks)}")
                        print(f"[DEBUG] Number of text chunks: {len(text_chunks)}")
                        vectorstore = get_vectorstore(text_chunks)
                        st.write(f"[DEBUG] Vectorstore created: {vectorstore is not None}")
                        print(f"[DEBUG] Vectorstore created: {vectorstore is not None}")
                    if vectorstore:
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.session_state.images = []
                        retrieval_agent.update_vectorstore(vectorstore)
                        st.session_state.chat_history = []
                        st.session_state.documents_processed = True
                        for uploaded_file in uploaded_files:
                            if uploaded_file.name.lower().endswith('.pdf'):
                                st.session_state['current_pdf_filename'] = uploaded_file.name
                                break
                    else:
                        st.error("Failed to create conversation chain. Check Ollama setup and ensure 'llama3' model is pulled.")
                        st.session_state.conversation = None
                        st.session_state.images = []
                        st.session_state.chat_history = []
                        st.session_state.documents_processed = False
                        st.write("[DEBUG] Vectorstore creation failed.")
                        print("[DEBUG] Vectorstore creation failed.")
            else:
                st.warning("Please upload at least one document file.")
                st.session_state.conversation = None
                st.session_state.images = []
                st.session_state.chat_history = []
                st.session_state.documents_processed = False

    # Chat area
    st.subheader("Chat")
    user_input = st.text_input("Type your question and press Enter", value=st.session_state.user_input_value, key="user_input")
    if st.button("Send"):
        if user_input.strip():
            handle_user_input(user_input)
            st.session_state.user_input_value = ""

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'bot':
            st.markdown(
                f'<div class="chat-message-container">'
                f'<div class="chat-avatar" style="font-size:2rem;">🤖</div>'
                f'<div class="chat-message-bot">{message["content"]}</div></div>',
                unsafe_allow_html=True
            )
        elif message['role'] == 'system':
            st.markdown(
                f'<div class="chat-message-container">'
                f'<div class="chat-avatar" style="font-size:2rem;">🤖</div>'
                f'<div class="chat-message-bot">{message["content"]}</div></div>',
                unsafe_allow_html=True
            )
        elif message['role'] == 'user':
            st.markdown(
                f'<div class="chat-message-container user">'
                f'<div class="chat-message-user">{message["content"]}</div>'
                f'<div class="chat-avatar" style="font-size:2rem;">🧑</div></div>',
                unsafe_allow_html=True
            )

if __name__ == '__main__':
    main()