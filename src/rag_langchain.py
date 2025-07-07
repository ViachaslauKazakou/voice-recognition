from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)
# Add this block to enable console logging
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] - %(levelname)s [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class AdvancedRAG:
    def __init__(self, documents_path: str = "knowledge_base"):
        self.documents_path = documents_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.retriever = None
        self.setup_rag()
    
    def setup_rag(self):
        """Настройка RAG с LangChain"""
        # Загружаем документы
        logger.info(f"Загрузка документов из {self.documents_path}")
        loader = DirectoryLoader(
            self.documents_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        
        try:
            documents = loader.load()
            
            # Разбиваем на чанки
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            
            texts = text_splitter.split_documents(documents)
            
            # Создаем векторное хранилище
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            
            # Создаем retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
        except Exception as e:
            print(f"Ошибка при настройке RAG: {e}")
            self.vectorstore = None
            self.retriever = None
    
    def get_relevant_docs(self, query: str) -> List[str]:
        """Получает релевантные документы"""
        logger
        if not self.retriever:
            logger.warning("Retriever не инициализирован. Проверьте настройку RAG.")
            return []
            
        try:
            docs = self.retriever.get_relevant_documents(query)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger
            return []