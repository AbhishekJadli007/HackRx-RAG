import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { heroBackground, forLight } from "../assets";
import curve from "../assets/hero/curve.png";
import chatgptLogo from "../assets/hero/ChatGPT Image Jun 30, 2025, 12_02_36 AM.png";
import ReactMarkdown from 'react-markdown';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isConversationReady, setIsConversationReady] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isThinking, setIsThinking] = useState(false);
  const [currentAgent, setCurrentAgent] = useState(null);
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();
  const [agents, setAgents] = useState({});

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Check authentication on component mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem("token");
      if (!token) {
        navigate('/');
        return;
      }

      try {
        const response = await fetch(`${import.meta.env.VITE_AUTH_API_URL || "http://localhost:3000"}/api/auth/verify`, {
          headers: {
            "Authorization": `Bearer ${token}`
          }
        });

        if (response.ok) {
          setIsAuthenticated(true);
        } else {
          localStorage.removeItem("token");
          navigate('/');
        }
      } catch (error) {
        console.error("Auth check failed:", error);
        localStorage.removeItem("token");
        navigate('/');
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [navigate]);

  useEffect(() => {
    // Fetch available agents for sidebar
            fetch(`${import.meta.env.VITE_CHAT_API_URL || "http://localhost:5000"}/agents`)
      .then(res => res.json())
      .then(setAgents)
      .catch(() => setAgents({}));
  }, [navigate]);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files);
    // Prevent duplicates by name and size
    const newFiles = files.filter(
      file => !uploadedFiles.some(f => f.name === file.name && f.size === file.size)
    );
    setUploadedFiles(prev => [...prev, ...newFiles]);
  };

  const removeFile = (index) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const processDocuments = async () => {
    if (uploadedFiles.length === 0) {
      alert("Please upload at least one document file.");
      return;
    }

    setIsProcessing(true);
    const formData = new FormData();
    uploadedFiles.forEach(file => {
      formData.append('files', file);
    });

    try {
              const response = await fetch(`${import.meta.env.VITE_CHAT_API_URL || "http://localhost:5000"}/api/upload`, {
        method: "POST",
        body: formData,
      });
      
      const data = await response.json();
      if (data.success) {
        setIsConversationReady(true);
        setMessages([{ 
          role: "system", 
          content: "Documents processed successfully! You can now ask questions.",
          agentsStatus: true
        }]);
      } else {
        setMessages([{ 
          role: "system", 
          content: "Failed to process documents. Please try again." 
        }]);
      }
    } catch (error) {
      setMessages([{ 
        role: "system", 
        content: "Error processing documents. Please check your connection." 
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userMessage = { role: "user", content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");

    if (!isConversationReady) {
      setMessages(prev => [...prev, {
        role: "bot",
        content: "Please process your documents first using the sidebar before asking questions.",
        agent_name: "System"
      }]);
      return;
    }

    setCurrentAgent(null);
    setIsThinking(true);

    try {
              const response = await fetch(`${import.meta.env.VITE_CHAT_API_URL || "http://localhost:5000"}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });
      
      const data = await response.json();
      setMessages(prev => [...prev, {
        role: "bot",
        content: typeof data.response === 'string' ? data.response : JSON.stringify(data.response),
        agent_name: data.agent || "Unknown",
        trace: data.trace || []
      }]);
      setCurrentAgent(data.agent || null);
    } catch (error) {
      setMessages(prev => [...prev, { role: "bot", content: "Error: Could not reach backend.", agent_name: "System" }]);
      setCurrentAgent(null);
    } finally {
      setIsThinking(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setIsConversationReady(false);
    setUploadedFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const logout = () => {
    localStorage.removeItem("token");
    navigate('/');
  };

  // Show loading screen while checking authentication
  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-100 dark:bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  // Show authentication error if not authenticated
  if (!isAuthenticated) {
    return null; // Will redirect to login
  }

  const renderMessage = (message, index) => {
    const isUser = message.role === "user";
    const isBot = message.role === "bot";
    const isSystem = message.role === "system";
    const agentName = message.agent_name || "NeuroFetch";
    const textColor = isDarkMode ? '#FFFFFF' : '#232946';
    const bubbleStyle = {
      background: 'transparent',
      color: textColor,
      boxShadow: 'none',
      border: 'none',
      padding: '1.25rem 1.5rem',
      marginBottom: '1.5rem',
      fontSize: '1.08rem',
      fontWeight: 600,
      wordBreak: 'break-word',
    };
    if (isUser) {
      return (
        <div key={index} className="flex items-start space-x-4 mb-8">
          <span style={{fontSize: '2rem'}}>🧑</span>
          <div>
            <p className="font-semibold" style={{color: textColor}}>You</p>
            <div style={bubbleStyle}>
              <p className="text-sm" style={{color: textColor}}>{message.content}</p>
            </div>
          </div>
        </div>
      );
    }
    if (isBot || isSystem) {
      // Extract response time from message.content if present
      let content = message.content;
      let responseTime = null;
      // Try to extract response time (e.g., 'in 44.98 seconds') from the end of the message
      const timeMatch = content.match(/in ([\d.]+) seconds\*?$/);
      if (timeMatch) {
        responseTime = timeMatch[1];
        // Remove the agent/response time line from the content
        content = content.replace(/---.*?\*Response generated by.*?in [\d.]+ seconds\*?/, '').trim();
        content = content.replace(/\*?Response generated by.*?in [\d.]+ seconds\*?/, '').trim();
      }
      // Remove trailing '---' and any whitespace before/after
      content = content.replace(/---+$/, '').trim();
      return (
        <div key={index} className="mb-8">
          <div className="font-semibold text-base text-indigo-300 mb-1 flex items-center">
            <span className="mr-2" style={{fontSize: '1.5rem'}}>🤖</span> NeuroFetch
          </div>
          <div style={bubbleStyle}>
            {/* Render markdown tables if present */}
            <ReactMarkdown>{content}</ReactMarkdown>
            {/* Render raw table data if present */}
            {message.tables && Array.isArray(message.tables) && message.tables.map((table, i) => (
              <Table key={i} data={table.data} />
            ))}
            {responseTime && (
              <div className="text-xs mt-2 text-indigo-400">Answered in {responseTime} seconds</div>
            )}
            {message.agentsStatus && (
              <div className="text-green-600 font-semibold mt-2">All agents are active and running...</div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  // Table component for rendering raw table data
  function Table({ data }) {
    if (!data || data.length === 0) return null;
    const columns = Object.keys(data[0]);
    return (
      <div className="overflow-x-auto my-4">
        <table className="min-w-full border border-gray-300">
          <thead>
            <tr>
              {columns.map(col => <th key={col} className="px-3 py-2 border-b bg-gray-100 text-xs font-bold text-gray-700">{col}</th>)}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i}>
                {columns.map(col => <td key={col} className="px-3 py-2 border-b text-sm">{row[col]}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <div
      className={`flex h-screen relative ${isDarkMode ? 'dark-mode' : ''}`}
      style={
        isDarkMode
          ? {
              backgroundImage: `url(${heroBackground}), linear-gradient(120deg, #181c2a 0%, #232946 100%)`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
              backgroundRepeat: 'no-repeat',
              minHeight: '100vh',
              width: '100vw',
            }
          : {
              background: 'linear-gradient(120deg, #F8F8F8 0%, #F5CFFF 60%, #E8D8F1 100%)',
              minHeight: '100vh',
              width: '100vw',
            }
      }
    >
      {/* Sidebar */}
      <div className="min-w-[16rem] max-w-[20rem] w-full md:w-80 h-full p-6 flex flex-col justify-between rounded-r-xl shadow-lg z-10">
        <div>
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-2xl font-bold flex flex-col items-start">
              {isDarkMode ? (
                <>
                  <span
                    style={{
                      color: '#F5F5F5',
                      fontWeight: 900,
                      letterSpacing: "0.01em"
                    }}
                  >
                    NeuroFetch
                  </span>
                  <img
                    src={curve}
                    alt="curve underline"
                    style={{ width: "110px", marginTop: "-8px", marginLeft: "2px", filter: "drop-shadow(0 2px 4px rgba(35,41,70,0.10))" }}
                  />
                </>
              ) : (
                <span
                  style={{
                    background: "linear-gradient(90deg, #232946 0%, #3A2D6C 50%, #6A5ACD 100%)",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                    backgroundClip: "text",
                    textFillColor: "transparent",
                    textShadow: "0 2px 8px rgba(35,41,70,0.10)",
                    fontWeight: 900,
                    letterSpacing: "0.01em"
                  }}
                >
                  NeuroFetch
                </span>
              )}
            </h1>
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600"
            >
              <span className="material-icons text-gray-600 dark:text-gray-300">
                {isDarkMode ? 'light_mode' : 'dark_mode'}
              </span>
            </button>
          </div>
          
          <button 
            onClick={clearChat}
            className="w-full bg-indigo-500 text-white py-3 px-4 rounded-lg flex items-center justify-center text-sm font-semibold hover:bg-indigo-600 transition-colors mb-6"
          >
            <span className="material-icons mr-2">add</span>
            New chat
          </button>
          
          <div className="flex justify-between items-center mb-3">
            <h2 className="text-xs text-gray-500 font-semibold uppercase">Your documents</h2>
          </div>
          
          <div className="mb-6">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.csv,.txt,.md"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="w-full border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-4 text-center text-gray-500 dark:text-gray-400 hover:border-indigo-500 dark:hover:border-indigo-400 transition-colors"
            >
              <span className="material-icons mb-2">upload_file</span>
              <div>Upload files (PDF, CSV, TXT, MD)</div>
            </button>
            
            {uploadedFiles.length > 0 && (
              <div className="mt-3">
                <div className="text-xs text-gray-500 mb-2">Uploaded files:</div>
                {uploadedFiles.map((file, index) => (
                  <div key={index} className="flex items-center text-xs text-gray-600 dark:text-gray-400 mb-1">
                    <span className="flex-1 truncate">{file.name}</span>
                    <button
                      onClick={() => removeFile(index)}
                      className="flex items-center justify-center ml-2 px-2 py-1 bg-indigo-100 text-indigo-700 font-semibold rounded-md dark:bg-indigo-500/20 dark:text-indigo-400 hover:bg-red-100 dark:hover:bg-red-400 transition-colors"
                      title="Remove file"
                    >
                      <span className="material-icons text-xs">delete</span>
                    </button>
                  </div>
                ))}
                <button
                  onClick={processDocuments}
                  disabled={isProcessing || isConversationReady}
                  className={`w-full mt-3 ${isConversationReady ? 'bg-green-400 cursor-not-allowed' : 'bg-green-500 hover:bg-green-600'} text-white py-2 px-4 rounded-lg text-sm font-semibold transition-colors disabled:opacity-50`}
                >
                  {isProcessing ? "Processing..." : isConversationReady ? "Processed" : "Process Documents"}
                </button>
                {isConversationReady && (
                  <button
                    onClick={() => {
                      setIsConversationReady(false);
                      setTimeout(() => fileInputRef.current?.click(), 0);
                    }}
                    className="w-full mt-2 bg-indigo-500 hover:bg-indigo-600 text-white py-2 px-4 rounded-lg text-sm font-semibold transition-colors"
                  >
                    Add More Documents
                  </button>
                )}
              </div>
            )}
          </div>
          
          <h2 className="text-xs text-gray-500 font-semibold uppercase mt-8 mb-3">Your conversations</h2>
          <nav className="space-y-2">
            <a className="flex items-center px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-md dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-gray-200" href="#">
              Create Html Game Environment..
            </a>
            <a className="flex items-center px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-md dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-gray-200" href="#">
              Apply To Leave For Emergency
            </a>
            <a className="flex items-center px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-md dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-gray-200" href="#">
              What is UI UX Design?
            </a>
            <a className="flex items-center justify-between px-3 py-2 text-sm bg-indigo-100 text-indigo-700 font-semibold rounded-md dark:bg-indigo-500/20 dark:text-indigo-400" href="#">
              <span>Current Chatbot GPT...</span>
              <div className="flex space-x-1">
                <span className="material-icons text-xs">edit</span>
                <span className="material-icons text-xs">delete</span>
              </div>
            </a>
            <a className="flex items-center px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-md dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-gray-200" href="#">
              How Chat GPT Work?
            </a>
          </nav>
          {/* AGENT LIST */}
          {/*
          <div className="mt-8">
            <h2 className="text-xs text-gray-500 font-semibold uppercase mb-2">Available Agents</h2>
            <ul className="text-xs">
              {Object.entries(agents).map(([name, status]) => (
                <li key={name} className="mb-1 flex items-center">
                  <span className="font-bold mr-2">{name}</span>
                  <span className={status.status === 'ok' ? 'text-green-500' : 'text-red-500'}>
                    {status.status}
                  </span>
                </li>
              ))}
            </ul>
          </div>
          */}
        </div>
        
        <div className="border-t border-gray-200 pt-6 dark:border-gray-700">
          <a className="flex items-center px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-md dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-gray-200" href="#">
            <span className="material-icons mr-3">settings</span>Settings
          </a>
          <button
            onClick={logout}
            className="flex items-center px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-md mt-2 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-gray-200 w-full"
          >
            <span style={{fontSize: '1.5rem'}}>👤</span>
            Logout
          </button>
        </div>
      </div>
      
      {/* Chat area */}
      <div className="flex-1 min-w-0 h-full p-4 md:p-8 overflow-y-auto">
        <div className="max-w-3xl mx-auto">
          <div className="mb-8">
            {messages.map((message, index) => (
              <div key={index} className={
                message.role === 'user' || message.role === 'system'
                  ? (isDarkMode ? 'dark-bubble' : 'copilot-bubble')
                  : (isDarkMode ? 'dark-bubble' : 'copilot-bubble')
              }>
                {renderMessage(message, index)}
              </div>
            ))}
          </div>
          <div ref={chatEndRef} />
        </div>
        
        <div className="sticky bottom-0 pb-8 bg-opacity-80 backdrop-filter backdrop-blur-md dark-mode:bg-slate-800 dark:bg-opacity-80 dark:backdrop-filter dark:backdrop-blur-md z-10">
          <div className="max-w-3xl mx-auto">
            <div className="relative flex items-center">
              <input
                className={`w-full py-4 pl-6 pr-16 text-sm border focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 shadow-sm ${isDarkMode ? 'dark-input' : 'copilot-input'}`}
                placeholder="What's in your mind..."
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              />
              <button
                onClick={sendMessage}
                className={`absolute right-2 top-1/2 transform -translate-y-1/2 ${isDarkMode ? 'dark-btn' : 'copilot-btn'} p-2`}
              >
                <span className="material-icons">send</span>
              </button>
            </div>
            {isThinking && (
              <div className="w-full flex flex-col items-center justify-center mt-4">
                <div className="flex items-center space-x-2">
                  <span style={{fontSize: '2rem'}}>🔄</span>
                  <span className="animate-spin inline-block h-6 w-6 border-4 border-indigo-300 border-t-transparent rounded-full"></span>
                  <span style={{color: isDarkMode ? '#FFFFFF' : '#232946', fontWeight: 600, fontSize: '1.08rem'}}>
                    Generating response...{currentAgent ? ` ${currentAgent} Processing` : ''}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Upgrade button */}
      <div className="fixed right-0 top-1/2 transform -translate-y-1/2 z-50">
        <button
          className="upgrade-button text-white font-semibold py-4 px-2 rounded-l-lg shadow-md hover:opacity-90 transition-opacity flex items-center space-x-2"
          onClick={() => window.location.href = 'http://localhost:5173/#pricing'}
        >
          <span className="material-icons transform rotate-90 text-lg">arrow_back_ios</span>
          <span className="text-sm">Upgrade To Pro</span>
        </button>
      </div>
    </div>
  );
};

export default Chatbot; 