# NeuroFetch

NeuroFetch is a modern, full-stack AI-powered document understanding and retrieval system. It allows users to upload documents (PDF, CSV, TXT, MD), extract structured data (tables, chat, etc.), and interact with the content using a conversational UI.

---

## Features

- **Modern React + Vite Frontend**: Beautiful, responsive UI with chat and document upload.
- **Node.js/Express Backend**: Handles authentication, user management, and API endpoints.
- **Python Flask API**: Orchestrates document processing and LLM-powered chat.
- **MCP Server (Python)**: Handles advanced document parsing, table extraction, and agent logic.
- **MongoDB Integration**: For user authentication and data storage.
- **Table Extraction**: Extracts tables from PDFs and displays them in structured, readable format.
- **LLM Integration**: Uses LLMs for chat, query reformulation, and document Q&A.

---

## Tech Stack

- **Frontend**: React, Vite, Tailwind CSS, react-markdown
- **Backend**: Node.js, Express, Mongoose, MongoDB
- **Python**: Flask, LangChain, PyPDF2, pandas, camelot, pdfplumber, and more
- **Other**: concurrently, cross-env

---

## Requirements

- Node.js (v18+ recommended)
- npm
- Python 3.10+
- MongoDB (local or cloud)
- (Optional) Ollama or other LLM backend for advanced features

---

## Installation

### 1. Clone the repository

```sh
git clone https://github.com/yourusername/neurofetch.git
cd neurofetch/neurofetch-ui
```

### 2. Install Node.js dependencies

```sh
npm install
```

### 3. Install Python dependencies

```sh
pip install -r requirements.txt
```

### 4. Set up MongoDB

- Make sure MongoDB is running locally or update your `MONGODB_URI` in `server/server.js` to point to your MongoDB instance.

---

## Running the Project

### **All-in-one (recommended):**

This will start the React frontend, Node.js backend, Flask API, and MCP server all together:

```sh
npm run dev:all
```

- Frontend: [http://localhost:5173](http://localhost:5173)
- Node.js API: [http://localhost:3000](http://localhost:3000)
- Flask API: [http://localhost:5000](http://localhost:5000)
- MCP server: (Python, port as configured)

---

### **Run Each Service Separately**

**Frontend (React):**
```sh
npm run dev
```

**Node.js Backend:**
```sh
npm run server
```

**Flask API:**
```sh
npm run flask
```

**MCP Server:**
```sh
npm run mcp
```

---

## Usage

1. Open the frontend in your browser.
2. Sign up or log in.
3. Upload your documents (PDF, CSV, TXT, MD).
4. Click "Process Documents".
5. Chat with NeuroFetch to extract tables, ask questions, or retrieve structured data.

---

## Python Requirements

See `requirements.txt` for all Python dependencies.

---

## Troubleshooting

- If you get a port error, make sure no other process is using the port (3000, 5000, 5173).
- If you see `ModuleNotFoundError: No module named 'agents'`, make sure to run the MCP server from the `src` directory or use the provided npm script.
- For MongoDB errors, ensure your database is running and the URI is correct.

---

## License

MIT

---

## Acknowledgements

- Inspired by modern UI/UX best practices and open-source LLM tools.

---

## Multi-Agent MCP Server

NeuroFetch uses a Python-based Multi-Agent Control and Processing (MCP) server to orchestrate advanced document understanding and retrieval. The MCP server manages a set of specialized agents, each responsible for a distinct aspect of document analysis or query handling. This architecture enables modular, extensible, and robust document intelligence.

### Agents Overview

| Agent Name                    | Class Name                        | Purpose                                                                                  |
|-------------------------------|-----------------------------------|------------------------------------------------------------------------------------------|
| Query Reformulation Agent     | `QueryReformulationAgent`         | Reformulates and expands user queries for better retrieval and LLM understanding.         |
| Adaptive Retrieval Agent      | `AdaptiveRetrievalAgent`          | Performs intelligent document retrieval, chunking, embedding, and re-ranking of results.  |
| Structured Data Extraction Agent | `StructuredDataExtractionAgent` | Extracts structured data (tables, chat logs) from PDFs using multiple extraction methods. |

#### Agent Details

- **QueryReformulationAgent**
  - *Role*: Reformulates user queries to improve search and LLM results. Expands queries with synonyms, intent detection, and keyword extraction.
  - *Example*: Turns "What is the capital of France?" into ["What is the capital of France?", "define capital of France", "main city of France", ...].

- **AdaptiveRetrievalAgent**
  - *Role*: Chunks and embeds documents, then retrieves and re-ranks relevant content using vector similarity and additional heuristics.
  - *Example*: Given a set of reformulated queries, returns the most relevant document passages, removing duplicates and ranking by quality.

- **StructuredDataExtractionAgent**
  - *Role*: Extracts tables and chat/conversation data from PDFs using tools like Camelot and pdfplumber. Handles both tabular and conversational data.
  - *Example*: Extracts all tables from a financial report PDF, or chat logs from a support transcript.

### How the MCP Server Works

- The MCP server manages agent registration, health, and communication.
- It exposes a Flask API with endpoints for:
  - `/route_query`: Main orchestration endpoint. Routes user queries to the LLM and/or the appropriate agent(s) based on intent.
  - `/agents`: Lists all available agents and their health status.
  - `/health/<agent_name>`: Health check for each agent.

#### Query Routing Logic

1. **LLM First:** The server first tries to answer using the LLM.
2. **Agent Fallback:** If the LLM is not confident, the server detects the query intent and routes to the appropriate agent:
   - Table/chat → StructuredDataExtractionAgent
   - Retrieval/other → AdaptiveRetrievalAgent
3. **LLM with Tools:** If the agent cannot answer, the LLM is prompted again with the agent's output for a final answer.

#### Example API Usage

- **Check agent health:**
  ```
  GET http://localhost:8000/agents
  ```

- **Route a query:**
  ```
  POST http://localhost:8000/route_query
  Content-Type: application/json

  {
    "query": "Extract all tables from the attached PDF.",
    "context": {
      "pdf_path": "/path/to/file.pdf",
      "data_type": "table"
    }
  }
  ```

- **Sample response:**
  ```json
  {
    "agent": "structured_data_extraction",
    "response": {
      "success": true,
      "data": {
        "tables": [ ... ],
        "extraction_method": "camelot",
        "total_tables": 3
      }
    },
    "trace": ["llm", "structured_data_extraction"],
    "elapsed": 1.23
  }
  ```

---

## PDFBot: Streamlit Document Chat

NeuroFetch also includes a standalone Streamlit app (`pdfbot.py`) for interactive document chat and extraction. PDFBot allows you to:

- Upload PDF, CSV, TXT, or MD files
- Process and chunk documents for retrieval
- Chat with your documents using LLM-powered Q&A
- Extract tables and chat logs from PDFs
- Enjoy a beautiful interface styled with `streamlit_custom.css` (matching the main app theme)

### How to Use PDFBot

1. **Install requirements** (from the project root):
   ```sh
   pip install -r requirements.txt
   ```
2. **Run the Streamlit app** (from the `neurofetch-ui` directory):
   ```sh
   streamlit run src/pdfbot.py
   ```
3. **Open your browser** to [http://localhost:8501](http://localhost:8501)
4. **Upload your documents** and start chatting!

### Features
- No authentication required—just upload and chat.
- Uses the same agent logic as the main system (retrieval, query reformulation, structured extraction).
- All chat bubbles are styled with custom CSS and use emojis for avatars.
- Supports table and chat extraction from PDFs, with results shown in the chat.

---
