# 🤖 CLAUDE.md - Projet 1: RAG Application (Chatbot Spécialisé)

## 📌 Objectif du Projet
Créer un chatbot alimenté par IA qui répond à des questions basées sur un corpus de documents personnalisé. Cette application démontre la maîtrise des embeddings vectoriels, de la recherche sémantique, et de l'intégration d'APIs LLM.

---

## 🛠️ Stack Technologique Complète

### Backend
- **Framework :** FastAPI (Python 3.11+)
- **LLM :** Claude API (Anthropic SDK) OU OpenAI API
- **Orchestration RAG :** LangChain v0.1.x
- **Vector Store :** PostgreSQL + pgvector extension (self-hosted)
- **Database :** PostgreSQL 15+
- **Async :** Asyncio, aiohttp
- **Environment :** python-dotenv

### Frontend
- **Framework :** React 18 + Vite
- **Styling :** Tailwind CSS 3.3+
- **HTTP Client :** Axios
- **State Management :** React Query v4
- **UI Components :** shadcn/ui (optionnel)

### Infrastructure
- **Backend Hosting :** Railway.app (gratuit)
- **Frontend Hosting :** Vercel (gratuit)
- **Database Hosting :** Railway.app (PostgreSQL managed)

---

## 📐 Architecture Système

```
┌──────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Chat Interface | Document Panel | Settings      │   │
│  └─────────────────────┬──────────────────────────┘   │
│                        │                               │
│                   (HTTPS REST)                         │
│                        │                               │
│  ┌─────────────────────▼──────────────────────────┐   │
│  │        Backend API (FastAPI)                   │   │
│  ├──────────────┬──────────────┬──────────────┐   │   │
│  │ Documents    │ Chat Logic   │ RAG Pipeline │   │   │
│  │ Endpoints    │ Orchestr.    │ Management   │   │   │
│  └──────────────┼──────────────┼──────────────┘   │   │
│                 │              │                  │   │
│  ┌──────────────▼──┐  ┌────────▼──────────────┐   │   │
│  │ LLM Service     │  │ Vector Database      │   │   │
│  │ (Claude/GPT API)│  │ (PostgreSQL pgvector)│   │   │
│  └─────────────────┘  └─────────────────────┘   │   │
│                                                  │   │
└──────────────────────────────────────────────────────┘
```

---

## 📋 Phase 1: Setup Initial & Infrastructure

### 1.1 Repository Setup
```bash
mkdir rag-application
cd rag-application

# Structure du projet
mkdir -p backend frontend
mkdir -p backend/app/{api,core,models,schemas,utils}
mkdir -p frontend/src/{components,pages,services,styles}

# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install fastapi uvicorn python-dotenv anthropic langchain psycopg2-binary python-multipart aiohttp

# Frontend
cd ../frontend
npm create vite@latest . -- --template react
npm install tailwindcss axios react-query

# Root .gitignore
echo "
*.env
.env.local
venv/
node_modules/
.DS_Store
__pycache__/
*.db
.vscode
" > ../.gitignore
```

### 1.2 Environment Variables

**Backend (.env)**
```
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/rag_db
DB_POOL_SIZE=20

# LLM API
ANTHROPIC_API_KEY=sk-ant-...
# OU pour OpenAI:
OPENAI_API_KEY=sk-...

# App Config
API_PORT=8000
API_HOST=0.0.0.0
DEBUG=True
CORS_ORIGINS=["http://localhost:5173"]

# RAG Config
CHUNK_SIZE=1024
CHUNK_OVERLAP=204
MIN_RELEVANCE_SCORE=0.7
TOP_K_RETRIEVAL=5

# Vector DB
PGVECTOR_DIMENSION=1536
```

**Frontend (.env.local)**
```
VITE_API_URL=http://localhost:8000
VITE_API_TIMEOUT=30000
```

### 1.3 Database Setup

**PostgreSQL Setup (Local ou Railway)**

```sql
-- Connect to PostgreSQL
psql -U postgres -h localhost

-- Create database
CREATE DATABASE rag_db;
\c rag_db

-- Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  filename VARCHAR(255) NOT NULL,
  content_type VARCHAR(50),
  file_size_bytes BIGINT,
  uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  total_chunks INT,
  status VARCHAR(20) DEFAULT 'processing'
);

CREATE TABLE chunks (
  id SERIAL PRIMARY KEY,
  document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,
  text TEXT NOT NULL,
  embedding vector(1536),
  metadata JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT unique_chunk UNIQUE(document_id, chunk_index)
);

CREATE TABLE conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_message_at TIMESTAMP,
  total_messages INT DEFAULT 0,
  title VARCHAR(255)
);

CREATE TABLE messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role VARCHAR(10) NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  retrieved_chunk_ids INT[] DEFAULT ARRAY[]::INT[],
  tokens_used INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indices
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
CREATE INDEX idx_conversations_created_at ON conversations(created_at);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE rag_db TO postgres;
```

---

## 📋 Phase 2: Backend Development

### 2.1 Core Models & Schemas

**File: `backend/app/models.py`**
```python
from sqlalchemy import Column, Integer, String, DateTime, JSONB, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    content_type = Column(String(50))
    file_size_bytes = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    total_chunks = Column(Integer)
    status = Column(String(20), default="processing")

class Chunk(Base):
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(ARRAY(Float))  # vector(1536)
    metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_message_at = Column(DateTime)
    total_messages = Column(Integer, default=0)
    title = Column(String(255))

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(10), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    retrieved_chunk_ids = Column(ARRAY(Integer))
    tokens_used = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
```

**File: `backend/app/schemas.py`**
```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from uuid import UUID

class DocumentUploadRequest(BaseModel):
    filename: str
    content_type: str

class DocumentResponse(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    total_chunks: int
    status: str
    
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[UUID] = None
    model: str = "claude-3-5-sonnet"
    temperature: float = 0.7
    k: int = 5

class RetrievedChunk(BaseModel):
    id: int
    text: str
    document_id: int
    similarity_score: float

class ChatResponse(BaseModel):
    response: str
    conversation_id: UUID
    message_id: UUID
    retrieved_chunks: List[RetrievedChunk]
    tokens_used: int
    processing_time_ms: float

class ConversationResponse(BaseModel):
    id: UUID
    created_at: datetime
    title: Optional[str]
    messages: List['MessageResponse']

class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: datetime
    
    class Config:
        from_attributes = True
```

### 2.2 LangChain RAG Pipeline

**File: `backend/app/rag_pipeline.py`**
```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self):
        # Initialize embeddings
        if os.getenv("ANTHROPIC_API_KEY"):
            # Use Claude for chat
            self.llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=float(os.getenv("TEMPERATURE", 0.7))
            )
            # Use OpenAI embeddings (cheaper)
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            # Fallback to OpenAI
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1024)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 204)),
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        return self.text_splitter.split_text(text)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        return self.embeddings.embed_documents(texts)
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        return self.embeddings.embed_query(query)
    
    async def generate_response(self, query: str, context_chunks: List[str]) -> str:
        """Generate LLM response using retrieved context"""
        context_text = "\n\n".join(context_chunks)
        
        system_prompt = SystemMessage(content="""You are a helpful AI assistant.
Answer questions based on the provided context.
If the context doesn't contain relevant information, say so clearly.
Provide accurate, concise, and helpful responses.
Cite the source when referencing information from the context.""")
        
        user_prompt = HumanMessage(content=f"""Context:
{context_text}

Question: {query}

Provide a comprehensive answer based on the context above.""")
        
        response = await self.llm.agenerate(
            messages=[[system_prompt, user_prompt]]
        )
        
        return response.generations[0][0].text
```

### 2.3 Vector Database Service

**File: `backend/app/vector_db.py`**
```python
import asyncpg
from typing import List, Tuple
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

class VectorDB:
    def __init__(self):
        self.connection_string = os.getenv("DATABASE_URL")
        self.pool = None
    
    async def connect(self):
        """Create connection pool"""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=5,
            max_size=int(os.getenv("DB_POOL_SIZE", 20))
        )
    
    async def disconnect(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
    
    async def store_chunks(self, document_id: int, chunks: List[Tuple[str, List[float], dict]]) -> int:
        """
        Store chunks with embeddings
        chunks: [(text, embedding, metadata), ...]
        """
        async with self.pool.acquire() as conn:
            count = 0
            for idx, (text, embedding, metadata) in enumerate(chunks):
                await conn.execute(
                    """INSERT INTO chunks 
                    (document_id, chunk_index, text, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5)""",
                    document_id, idx, text, embedding, metadata
                )
                count += 1
            return count
    
    async def search(self, query_embedding: List[float], top_k: int = 5, min_score: float = 0.7) -> List[dict]:
        """Search for similar chunks using cosine similarity"""
        async with self.pool.acquire() as conn:
            results = await conn.fetch(
                """SELECT id, document_id, text, 1 - (embedding <=> $1::vector) as similarity
                FROM chunks
                WHERE 1 - (embedding <=> $1::vector) > $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3""",
                query_embedding, min_score, top_k
            )
            
            return [dict(r) for r in results]
    
    async def get_document_chunks(self, document_id: int) -> List[dict]:
        """Retrieve all chunks for a document"""
        async with self.pool.acquire() as conn:
            results = await conn.fetch(
                "SELECT id, chunk_index, text FROM chunks WHERE document_id = $1 ORDER BY chunk_index",
                document_id
            )
            return [dict(r) for r in results]
    
    async def delete_document(self, document_id: int) -> int:
        """Delete document and all its chunks"""
        async with self.pool.acquire() as conn:
            # Get count of chunks before deletion
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE document_id = $1",
                document_id
            )
            
            # Delete chunks
            await conn.execute("DELETE FROM chunks WHERE document_id = $1", document_id)
            
            # Delete document
            await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
            
            return count
```

### 2.4 FastAPI Routes

**File: `backend/app/main.py`**
```python
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime
import json
import time
import uuid

from app.schemas import (
    DocumentResponse, ChatRequest, ChatResponse, 
    ConversationResponse, RetrievedChunk
)
from app.models import Document, Message, Conversation
from app.rag_pipeline import RAGPipeline
from app.vector_db import VectorDB

load_dotenv()

app = FastAPI(title="RAG Application API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_pipeline = None
vector_db = None

@app.on_event("startup")
async def startup():
    global rag_pipeline, vector_db
    rag_pipeline = RAGPipeline()
    vector_db = VectorDB()
    await vector_db.connect()

@app.on_event("shutdown")
async def shutdown():
    if vector_db:
        await vector_db.disconnect()

# ============ Document Endpoints ============

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Read file content
        content = await file.read()
        text = content.decode("utf-8")
        
        # Split into chunks
        chunks = rag_pipeline.split_text(text)
        
        # Generate embeddings
        embeddings = rag_pipeline.get_embeddings(chunks)
        
        # Prepare chunk data
        chunk_data = [
            (chunk, embedding, {"filename": file.filename})
            for chunk, embedding in zip(chunks, embeddings)
        ]
        
        # Store in database
        # TODO: Use SQLAlchemy session to create Document record
        document_id = 1  # Placeholder
        stored_count = await vector_db.store_chunks(document_id, chunk_data)
        
        return DocumentResponse(
            id=document_id,
            filename=file.filename,
            uploaded_at=datetime.utcnow(),
            total_chunks=stored_count,
            status="completed"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents"""
    # TODO: Query database for documents
    return []

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int):
    """Delete a document"""
    deleted_chunks = await vector_db.delete_document(doc_id)
    return {"status": "deleted", "chunks_deleted": deleted_chunks}

# ============ Chat Endpoints ============

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Send a question and get an answer"""
    try:
        start_time = time.time()
        
        # Get query embedding
        query_embedding = rag_pipeline.get_query_embedding(request.question)
        
        # Retrieve relevant chunks
        chunks_data = await vector_db.search(
            query_embedding,
            top_k=request.k,
            min_score=float(os.getenv("MIN_RELEVANCE_SCORE", 0.7))
        )
        
        # Extract chunk texts
        chunk_texts = [chunk["text"] for chunk in chunks_data]
        
        # Generate response
        response_text = await rag_pipeline.generate_response(
            request.question,
            chunk_texts
        )
        
        # Create conversation if not exists
        if not request.conversation_id:
            request.conversation_id = uuid.uuid4()
        
        # Save to database
        # TODO: Use SQLAlchemy to save messages
        
        processing_time = (time.time() - start_time) * 1000
        
        return ChatResponse(
            response=response_text,
            conversation_id=request.conversation_id,
            message_id=uuid.uuid4(),
            retrieved_chunks=[
                RetrievedChunk(
                    id=chunk["id"],
                    text=chunk["text"],
                    document_id=chunk["document_id"],
                    similarity_score=chunk["similarity"]
                )
                for chunk in chunks_data
            ],
            tokens_used=0,  # TODO: Calculate actual tokens
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    # TODO: Query messages for conversation
    return {"messages": []}

@app.get("/api/stats")
async def get_stats():
    """Get application statistics"""
    return {
        "total_documents": 0,
        "total_chunks": 0,
        "total_queries": 0,
        "tokens_used": 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 📋 Phase 3: Frontend Development

### 3.1 Main App Component

**File: `frontend/src/App.jsx`**
```jsx
import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import ChatInterface from './components/ChatInterface';
import DocumentPanel from './components/DocumentPanel';
import './App.css';

function App() {
  const [documents, setDocuments] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/documents`);
      const data = await response.json();
      setDocuments(data);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const handleDocumentUpload = async (file) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/api/documents/upload`,
        {
          method: 'POST',
          body: formData
        }
      );
      
      if (response.ok) {
        fetchDocuments();
      }
    } catch (error) {
      console.error('Error uploading document:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <Navbar />
      <div className="main-container">
        <DocumentPanel 
          documents={documents}
          onUpload={handleDocumentUpload}
          loading={loading}
        />
        <ChatInterface 
          conversationId={conversationId}
          onConversationCreate={setConversationId}
        />
      </div>
    </div>
  );
}

export default App;
```

### 3.2 Chat Component

**File: `frontend/src/components/ChatInterface.jsx`**
```jsx
import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

export default function ChatInterface({ conversationId, onConversationCreate }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message to UI
    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `${import.meta.env.VITE_API_URL}/api/chat`,
        {
          question: input,
          conversation_id: conversationId,
          k: 5,
          temperature: 0.7
        },
        { timeout: 30000 }
      );

      // Create new conversation if needed
      if (!conversationId && response.data.conversation_id) {
        onConversationCreate(response.data.conversation_id);
      }

      // Add assistant message
      const assistantMessage = {
        role: 'assistant',
        content: response.data.response,
        retrievedChunks: response.data.retrieved_chunks
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (err) {
      setError(err.message);
      console.error('Error sending message:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-interface">
      <div className="messages-container">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-content">
              {msg.content}
            </div>
            {msg.retrievedChunks && (
              <div className="sources">
                <small>Sources: {msg.retrievedChunks.length} chunks</small>
              </div>
            )}
          </div>
        ))}
        {loading && <div className="message assistant"><span className="typing">...</span></div>}
        {error && <div className="message error">{error}</div>}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSendMessage} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={loading}
          className="input-field"
        />
        <button type="submit" disabled={loading} className="send-button">
          Send
        </button>
      </form>
    </div>
  );
}
```

---

## 📋 Phase 4: Déploiement

### 4.1 Backend Deployment (Railway)

```bash
# 1. Create Railway project and link
railway login
railway init

# 2. Add PostgreSQL plugin in Railway dashboard
# 3. Set environment variables in Railway dashboard

# 4. Deploy
git push

# 5. Get public URL
railway open
```

### 4.2 Frontend Deployment (Vercel)

```bash
# 1. Build
npm run build

# 2. Deploy
vercel deploy --prod

# 3. Add environment variables in Vercel dashboard
VITE_API_URL=https://your-railway-backend.up.railway.app
```

---

## ✅ Checklist de Développement

- [ ] Database setup PostgreSQL + pgvector
- [ ] Backend: Models & Schemas
- [ ] Backend: RAG Pipeline (LangChain)
- [ ] Backend: Vector DB service
- [ ] Backend: FastAPI routes (upload, chat)
- [ ] Backend: Authentication (optional)
- [ ] Backend: Error handling & logging
- [ ] Backend: Unit tests
- [ ] Frontend: Chat interface component
- [ ] Frontend: Document upload
- [ ] Frontend: Real-time messaging
- [ ] Frontend: Responsive design
- [ ] Frontend: Error handling
- [ ] Integration: Backend + Frontend
- [ ] Testing: E2E flows
- [ ] Deployment: Railway (Backend)
- [ ] Deployment: Vercel (Frontend)
- [ ] Documentation: README + API docs
- [ ] Blog post: Launch announcement

---

## 🎯 Critères de Succès

✅ **Fonctionnalité:**
- Upload de 5+ documents fonctionnel
- Retrieval avec score > 0.7
- Réponses en < 5 secondes
- Historique persistant

✅ **Performance:**
- Backend Lighthouse > 90
- Frontend Lighthouse > 85
- First Contentful Paint < 1.5s

✅ **Documentation:**
- README complet avec setup instructions
- API documentation (OpenAPI/Swagger)
- Screenshots/GIFs du fonctionnement

✅ **Code Quality:**
- TypeScript everywhere
- Linting avec Eslint + Prettier
- Type safety
- Error handling complet

✅ **Déploiement:**
- Live sur domaine public
- HTTPS activé
- Environment variables sécurisées
- CI/CD avec GitHub Actions

---

## 📞 Support & Debugging

### Erreurs Courantes

**PG Vector not installed:**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**CORS issues:**
Vérifier `CORS_ORIGINS` dans `.env`

**Embeddings timeout:**
Augmenter timeout dans axios: `timeout: 60000`

**Out of memory:**
Réduire `CHUNK_SIZE` ou `TOP_K_RETRIEVAL`

---

## 🚀 Prochaines Étapes (Post-MVP)

- [ ] Authentification utilisateur
- [ ] Batch document processing
- [ ] Reranking avec cross-encoders
- [ ] Web crawling + auto-indexing
- [ ] Advanced filtering
- [ ] Analytics dashboard
- [ ] Cost tracking
- [ ] Custom model fine-tuning

---

**📅 Timeline estimée: 2-3 semaines**

Bonne chance! 🎯
