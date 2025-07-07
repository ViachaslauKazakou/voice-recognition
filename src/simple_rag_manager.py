import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger("SimpleRAG")
# Add this block to enable console logging
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] - %(levelname)s [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Простая реализация RAG (Retrieval-Augmented Generation)

class SimpleRAG:
    def __init__(self, documents_path: str = "knowledge_base"):
        self.documents_path = documents_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_documents()
        
    def load_documents(self):
        """Загружает документы из папки и создает индекс"""
        documents = []
        
        # Загружаем документы из файлов
        if os.path.exists(self.documents_path):
            for filename in os.listdir(self.documents_path):
                logger.info(f"Обработка файла: {filename}")
                if filename.endswith('.txt'):
                    with open(os.path.join(self.documents_path, filename), 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Разбиваем на чанки
                        chunks = self.chunk_text(content)
                        documents.extend(chunks)
        
        self.documents = documents

        logger.info(f"Загружено {len(documents)} документов из {self.documents_path}")
        
        if documents:
            # Создаем эмбеддинги
            logger
            self.embeddings = self.embedding_model.encode(documents)
            
            # Создаем FAISS индекс
            logger
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings.astype('float32'))
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Разбивает текст на чанки"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        logger.info(f"Разбито на {len(chunks)} чанков")
            
        return chunks
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Ищет релевантные документы"""
        if not self.documents or self.index is None:
            return []
        logger.info(f"Поиск по запросу: {query}")
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        relevant_docs = []
        for idx in indices[0]:
            if idx < len(self.documents):
                relevant_docs.append(self.documents[idx])
                
        return relevant_docs