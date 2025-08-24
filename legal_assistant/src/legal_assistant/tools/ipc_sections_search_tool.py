import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from crewai.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import json


@tool("IPC Sections Search Tool")
def search_ipc_sections(query: str, top_k: int = 3) -> str:
    """
    Search IPC vector database for sections relevant to the input query.
    
    Args:
        query (str): User query in natural language describing the legal issue
        top_k (int): Number of top results to return (default: 3)
    
    Returns:
        str: Formatted string with matching IPC sections including section numbers,
             titles, descriptions, and relevant details
    
    Example:
        search_ipc_sections("theft of property from a house")
    """
    # Load environment variables
    load_dotenv()
    
    # Resolve vector DB path
    persist_dir = os.getenv("PERSIST_DIRECTORY_PATH")
    if not persist_dir:
        return "❌ Error: 'PERSIST_DIRECTORY_PATH' is not set in .env file"
    
    # Check if database exists
    if not Path(persist_dir).exists():
        return "❌ Error: Vector database not found. Please run setup first."
    
    collection_name = os.getenv("IPC_COLLECTION_NAME", "ipc_collection")
    
    try:
        # Initialize embedding function
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load vectorstore
        vector_db = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embedding_function
        )
        
        # Perform similarity search
        docs = vector_db.similarity_search(query, k=top_k)
        
        if not docs:
            return "No relevant IPC sections found for the given query."
        
        # Format results for better readability
        results = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            section_info = f"""
**Result {i}:**
📗 **Section {metadata.get('section', 'N/A')}**: {metadata.get('section_title', 'N/A')}

**Chapter**: {metadata.get('chapter', 'N/A')} - {metadata.get('chapter_title', 'N/A')}

**Description**: {metadata.get('description', doc.page_content[:200])}

**Punishment**: {metadata.get('punishment', 'Not specified')}

**Legal Details**:
- Bailable: {metadata.get('is_bailable', 'Not specified')}
- Cognizable: {metadata.get('is_cognizable', 'Not specified')}
- Triable By: {metadata.get('triable_by', 'Not specified')}
---"""
            results.append(section_info)
        
        return "\n".join(results)
        
    except Exception as e:
        return f"❌ Error searching IPC sections: {str(e)}"


# Alternative implementation with caching for better performance
class IPCSectionSearcher:
    """
    Enhanced IPC searcher with caching and batch processing capabilities.
    """
    
    def __init__(self):
        load_dotenv()
        self.vector_db = None
        self.embedding_function = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the vector database connection."""
        persist_dir = os.getenv("PERSIST_DIRECTORY_PATH")
        if not persist_dir or not Path(persist_dir).exists():
            raise ValueError("Vector database not found. Please run setup first.")
        
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_db = Chroma(
            collection_name=os.getenv("IPC_COLLECTION_NAME", "ipc_collection"),
            persist_directory=persist_dir,
            embedding_function=self.embedding_function
        )
    
    def search(self, query: str, top_k: int = 3, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search with optional filters.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional filters like {'chapter': 'XVI'}
        """
        where_clause = filters if filters else None
        docs = self.vector_db.similarity_search(
            query, 
            k=top_k,
            filter=where_clause
        )
        
        return [
            {
                "section": doc.metadata.get("section"),
                "section_title": doc.metadata.get("section_title"),
                "chapter": doc.metadata.get("chapter"),
                "chapter_title": doc.metadata.get("chapter_title"),
                "description": doc.metadata.get("description"),
                "punishment": doc.metadata.get("punishment"),
                "is_bailable": doc.metadata.get("is_bailable"),
                "is_cognizable": doc.metadata.get("is_cognizable"),
                "content": doc.page_content
            }
            for doc in docs
        ]
        

