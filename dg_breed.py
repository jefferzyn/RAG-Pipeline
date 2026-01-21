
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json

# Haystack imports
from haystack import Document, Pipeline
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever


class DogBreedScraper:
    """Scrapes dog breed information from alldogbreeds.net"""
    
    def __init__(self, base_url: str = "https://alldogbreeds.net"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_breed_list(self) -> List[str]:
        """
        Scrape the list of all dog breeds
        Note: This site loads content dynamically, so we'll need to handle that
        """
        # This is a placeholder - you may need to use Selenium or 
        # find the API endpoint the site uses
        url = f"{self.base_url}/all-dog-breeds-complete-list/"
        response = self.session.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract breed links - adjust selectors based on actual HTML structure
        breed_links = []
        # This selector is an example - you'll need to inspect the actual page
        for link in soup.find_all('a', href=True):
            if '/dog-breeds/' in link['href']:
                breed_links.append(link['href'])
        
        return breed_links
    
    def scrape_breed_details(self, breed_url: str) -> Dict:
        """Scrape details for a specific breed"""
        response = self.session.get(breed_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract breed information
        # Adjust these selectors based on the actual page structure
        breed_data = {
            'url': breed_url,
            'name': '',
            'origin': '',
            'size': '',
            'temperament': '',
            'description': '',
            'characteristics': {}
        }
        
        # Example extraction (adjust based on actual HTML)
        title = soup.find('h1')
        if title:
            breed_data['name'] = title.text.strip()
        
        # Extract all text content for now
        breed_data['description'] = soup.get_text(separator=' ', strip=True)
        
        return breed_data


class DogBreedRAG:
    """RAG Pipeline for Dog Breed Information"""
    
    def __init__(self, use_embeddings: bool = True):
        """
        Initialize the RAG pipeline
        
        Args:
            use_embeddings: If True, use semantic search; if False, use BM25
        """
        self.use_embeddings = use_embeddings
        self.document_store = InMemoryDocumentStore()
        self.pipeline = None
        
    def load_documents_from_json(self, json_file: str):
        """Load dog breed documents from JSON file"""
        with open(json_file, 'r') as f:
            breeds_data = json.load(f)
        
        documents = []
        for breed in breeds_data:
            # Create a rich text representation of the breed
            content = f"""
            Breed Name: {breed.get('name', 'Unknown')}
            Origin: {breed.get('origin', 'Unknown')}
            Size: {breed.get('size', 'Unknown')}
            Temperament: {breed.get('temperament', 'Unknown')}
            Description: {breed.get('description', '')}
            """
            
            doc = Document(
                content=content.strip(),
                meta={
                    'name': breed.get('name'),
                    'origin': breed.get('origin'),
                    'size': breed.get('size'),
                    'url': breed.get('url')
                }
            )
            documents.append(doc)
        
        return documents
    
    def load_documents_from_scraper(self, scraper: DogBreedScraper, breed_urls: List[str]):
        """Load documents by scraping breed pages"""
        documents = []
        
        for url in breed_urls[:10]:  # Limit for testing
            try:
                breed_data = scraper.scrape_breed_details(url)
                content = f"""
                Breed Name: {breed_data.get('name', 'Unknown')}
                Origin: {breed_data.get('origin', 'Unknown')}
                Size: {breed_data.get('size', 'Unknown')}
                Temperament: {breed_data.get('temperament', 'Unknown')}
                Description: {breed_data.get('description', '')}
                """
                
                doc = Document(
                    content=content.strip(),
                    meta=breed_data
                )
                documents.append(doc)
                print(f"Scraped: {breed_data.get('name', url)}")
            except Exception as e:
                print(f"Error scraping {url}: {e}")
        
        return documents
    
    def index_documents(self, documents: List[Document]):
        """Index documents in the document store"""
        if self.use_embeddings:
            # Use embeddings for semantic search
            embedder = SentenceTransformersDocumentEmbedder(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            embedder.warm_up()
            
            # Embed documents
            docs_with_embeddings = embedder.run(documents)
            self.document_store.write_documents(docs_with_embeddings["documents"])
        else:
            # Just write documents for BM25 search
            self.document_store.write_documents(documents)
        
        print(f"Indexed {len(documents)} documents")
    
    def build_rag_pipeline(self, api_key: str | None = None):
        """
        Build the RAG pipeline with retriever and generator
        
        Args:
            api_key: OpenAI API key (if using OpenAI generator)
        """
        template = """
        You are a helpful dog breed expert assistant. Use the following information to answer the question.
        If you don't know the answer, just say so - don't make up information.
        
        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}
        
        Question: {{ question }}
        
        Answer:
        """
        
        prompt_builder = PromptBuilder(template=template)
        
        if self.use_embeddings:
            # Semantic search with embeddings
            text_embedder = SentenceTransformersTextEmbedder(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            retriever = InMemoryEmbeddingRetriever(
                document_store=self.document_store,
                top_k=3
            )
            
            self.pipeline = Pipeline()
            self.pipeline.add_component("text_embedder", text_embedder)
            self.pipeline.add_component("retriever", retriever)
            self.pipeline.add_component("prompt_builder", prompt_builder)
            
            if api_key:
                generator = OpenAIGenerator(api_key=Secret.from_token(api_key), model="gpt-3.5-turbo")
                self.pipeline.add_component("llm", generator)
                
                self.pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
                self.pipeline.connect("retriever", "prompt_builder.documents")
                self.pipeline.connect("prompt_builder", "llm")
            else:
                self.pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
                self.pipeline.connect("retriever", "prompt_builder.documents")
        else:
            # BM25 keyword search
            retriever = InMemoryBM25Retriever(
                document_store=self.document_store,
                top_k=3
            )
            
            self.pipeline = Pipeline()
            self.pipeline.add_component("retriever", retriever)
            self.pipeline.add_component("prompt_builder", prompt_builder)
            
            if api_key:
                generator = OpenAIGenerator(api_key=Secret.from_token(api_key), model="gpt-3.5-turbo")
                self.pipeline.add_component("llm", generator)
                
                self.pipeline.connect("retriever", "prompt_builder.documents")
                self.pipeline.connect("prompt_builder", "llm")
            else:
                self.pipeline.connect("retriever", "prompt_builder.documents")
    
    def query(self, question: str) -> Dict:
        """Query the RAG pipeline"""
        if not self.pipeline:
            raise ValueError("Pipeline not built. Call build_rag_pipeline() first.")
        
        if self.use_embeddings:
            result = self.pipeline.run({
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question}
            })
        else:
            result = self.pipeline.run({
                "retriever": {"query": question},
                "prompt_builder": {"question": question}
            })
        
        return result


# Example usage
def main():
    """Example of how to use the Dog Breed RAG system"""
    
    # Option 1: Create sample data for testing
    sample_breeds = [
        {
            'name': 'Labrador Retriever',
            'origin': 'Canada',
            'size': 'Large',
            'temperament': 'Friendly, Active, Outgoing',
            'description': 'Labrador Retrievers are friendly, outgoing, and active dogs. They are one of the most popular dog breeds.',
            'url': 'https://example.com/labrador'
        },
        {
            'name': 'German Shepherd',
            'origin': 'Germany',
            'size': 'Large',
            'temperament': 'Confident, Courageous, Smart',
            'description': 'German Shepherds are confident, courageous, and intelligent. They are excellent working dogs.',
            'url': 'https://example.com/german-shepherd'
        }
    ]
    
    # Save sample data
    with open('sample_breeds.json', 'w') as f:
        json.dump(sample_breeds, f, indent=2)
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = DogBreedRAG(use_embeddings=True)
    
    # Load and index documents
    print("Loading documents...")
    documents = rag.load_documents_from_json('sample_breeds.json')
    
    print("Indexing documents...")
    rag.index_documents(documents)
    
    # Build pipeline (without OpenAI for now)
    print("Building pipeline...")
    rag.build_rag_pipeline(api_key=None)
    
    # Query the system
    print("\nQuerying the system...")
    question = "What can you tell me about Labrador Retrievers?"
    result = rag.query(question)
    
    print(f"\nQuestion: {question}")
    print("\nRetrieved Documents:")
    for doc in result['prompt_builder']['prompt'].split('Context:')[1].split('Question:')[0].strip().split('\n\n'):
        if doc.strip():
            print(f"- {doc.strip()[:200]}...")
    
    # Option 2: Scrape actual data (commented out for safety)
    """
    print("\\nScraping dog breed data...")
    scraper = DogBreedScraper()
    breed_urls = scraper.scrape_breed_list()
    documents = rag.load_documents_from_scraper(scraper, breed_urls)
    rag.index_documents(documents)
    """


if __name__ == "__main__":
    main()