import os 
import json
from typing import List, Dict, Any
from dotenv import load_dotenv 
from langchain_chroma import Chroma, ChromaDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from tqdm import tqdm
import shutil


class IPCVectorDBSetup:
    """ Sets up Chroma Vector database with IPC sections using HuggingFace embeddinngs"""
    
    def __init__(self):
        load_dotenv()
        # Get paths from environment
        self.ipc_json_path = os.environ.get('IPC_JSON_PATH')
        self.persist_directory = os.environ.get('PERSIST_DIRECTORY')
        self.collection_name = os.environ.get('COLLECTION_NAME')
        
        # Validate paths
        if not self.ipc_json_path:
            raise ValueError("IPC_JSON_PATH not set in .env file.")
        
        if not self.persist_directory:
            raise ValueError("PERSIST_DIRECTORY not set in .env file.")
        
        #Initialize embeddings
        print("Initializing HuggingFace Embeddings...")
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        
    def load_ipc_data(self) -> List[Dict[str, Any]]:
        """Load IPC data from JSON file"""
        print(f"Loading IPC data from {self.ipc_json_path}...")
        
        if not os.path.exists(self.ipc_json_path):
            raise FileNotFoundError(f"IPC JSON file not found at {self.ipc_json_path}")
        
        with open(self.ipc_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(ipc_data)} IPC sections.")
        return data
    
    def prepare_documents(self, ipc_data: List[Dict[str, Any]]) -> List[Document]:
        """Convert IPC data to Langchain Document format."""
        documents = []
        
        for section in tqdm(ipc_data, desc="Preparing documents"):
            # Create searchable content
            content = f"""
            Section {section.get('Section', '')}: {section.get('Section_Title', '')}
            
            Description: {section.get('Description', '')}
            
            Chapter: {section.get('Chapter', '')} - {section.get('Chapter_Title', '')}
            
            Offense Type: {section.get('Offense_Type', '')}
            Punishment: {section.get('Punishment', '')}
            Bailable: {section.get('Is_Bailable', 'Not specified')}
            Cognizable: {section.get('Is_Cognizable', 'Not specified')}
            Triable By: {section.get('Triable_By', 'Not specified')}
            """
            
            # Prepare metadata
            metadata = {
                "section": section.get('Section', ''),
                "section_title": section.get('Section_Title', ''),
                "chapter": section.get('Chapter', ''),
                "chapter_title": section.get('Chapter_Title', ''),
                "description": section.get('Description', ''),
                "offense_type": section.get('Offense_Type', ''),
                "punishment": section.get('Punishment', ''),
                "is_bailable": section.get('Is_Bailable', ''),
                "is_cognizable": section.get('Is_Cognizable', ''),
                "triable_by": section.get('Triable_By', '')
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def create_vector_db(self, documents: List[Document], reset: bool = False):
        """Create or update Chroma vector database."""
        
        # Reset database if requested
        if reset and os.path.exists(self.persist_directory):
            print(f"Removing existing database at {self.persist_directory}...")
            shutil.rmtree(self.persist_directory)
        
        print(f"Creating Chroma vector database at {self.persist_directory}...")
        
        # Create vector store
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        print(f"Vector database created with {len(documents)} documents")
        return vector_db
    
    def test_search(self, vector_db: Chroma):
        """Test the vector database with sample queries."""
        test_queries = [
            "murder and homicide",
            "theft of property",
            "assault and hurt",
            "cheating and fraud",
            "kidnapping"
        ]
        
        print("\n" + "="*60)
        print("Testing Vector Database Search")
        print("="*60)
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = vector_db.similarity_search(query, k=3)
            
            for i, doc in enumerate(results, 1):
                print(f"  {i}. Section {doc.metadata['section']}: {doc.metadata['section_title']}")
    
    def setup(self, reset: bool = False):
        """Main setup method."""
        try:
            # Load IPC data
            ipc_data = self.load_ipc_data()
            
            # Prepare documents
            documents = self.prepare_documents(ipc_data)
            
            # Create vector database
            vector_db = self.create_vector_db(documents, reset=reset)
            
            # Test search functionality
            self.test_search(vector_db)
            
            print("\n✅ Vector database setup complete!")
            print(f"Database location: {self.persist_directory}")
            print(f"Collection name: {self.collection_name}")
            
        except Exception as e:
            print(f"❌ Error during setup: {str(e)}")
            raise

if __name__ == "__main__":
    setup = IPCVectorDBSetup()
    setup.setup(reset=True)  # Set reset=False to append to existing DB
    
