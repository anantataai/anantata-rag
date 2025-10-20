"""
Модуль для завантаження розмов ChatGPT в Qdrant
"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ChatGPTToQdrant:
    """Клас для роботи з RAG базою на Qdrant"""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        """Ініціалізація"""
        self.client = QdrantClient(url=qdrant_url)
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.vector_size = 384
        
        logger.info(f"Підключено до Qdrant: {qdrant_url}")
    
    def ensure_collection(self, collection_name: str):
        """Створює колекцію якщо не існує"""
        try:
            self.client.get_collection(collection_name)
            logger.info(f"Колекція '{collection_name}' вже існує")
        except:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✅ Створено колекцію '{collection_name}'")
    
    def parse_chatgpt_export(self, filepath: str) -> List[Dict]:
        """Парсинг експорту ChatGPT"""
        logger.info(f"Парсинг файлу: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = []
        
        for conv in data:
            conv_id = conv.get('id', 'unknown')
            title = conv.get('title', 'Без назви')
            create_time = conv.get('create_time', 0)
            
            # Збираємо всі повідомлення
            messages = []
            mapping = conv.get('mapping', {})
            
            for node_id, node_data in mapping.items():
                message_data = node_data.get('message')
                if not message_data:
                    continue
                
                author = message_data.get('author', {}).get('role', 'unknown')
                content = message_data.get('content', {})
                
                if isinstance(content, dict):
                    parts = content.get('parts', [])
                    if parts and parts[0]:
                        text = parts[0] if isinstance(parts[0], str) else str(parts[0])
                        messages.append({
                            'role': author,
                            'content': text
                        })
            
            if messages:
                conversations.append({
                    'id': conv_id,
                    'title': title,
                    'create_time': create_time,
                    'messages': messages
                })
        
        logger.info(f"✅ Знайдено {len(conversations)} розмов")
        return conversations
    
    def create_chunks(self, conversations: List[Dict], chunk_size: int = 3) -> List[Dict]:
        """Розбиття розмов на чанки"""
        chunks = []
        
        for conv in conversations:
            messages = conv['messages']
            
            # Розбиваємо по chunk_size повідомлень
            for i in range(0, len(messages), chunk_size):
                chunk_messages = messages[i:i + chunk_size]
                
                # Створюємо текст чанку
                text_parts = []
                for msg in chunk_messages:
                    role = msg['role'].upper()
                    content = msg['content']
                    text_parts.append(f"{role}: {content}")
                
                chunk_text = "\n\n".join(text_parts)
                
                chunks.append({
                    'conversation_id': conv['id'],
                    'conversation_title': conv['title'],
                    'text': chunk_text,
                    'timestamp': datetime.fromtimestamp(conv['create_time']).isoformat(),
                    'source': 'ChatGPT'
                })
        
        logger.info(f"✅ Створено {len(chunks)} чанків")
        return chunks
    
    def upload_to_qdrant(self, chunks: List[Dict], collection_name: str = "chatgpt_conversations"):
        """Завантаження в Qdrant"""
        self.ensure_collection(collection_name)
        
        logger.info(f"Завантаження {len(chunks)} чанків в '{collection_name}'...")
        
        # Генеруємо embeddings
        texts = [chunk['text'] for chunk in chunks]
        vectors = self.model.encode(texts, show_progress_bar=True)
        
        # Створюємо points
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(
                PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload=chunk
                )
            )
        
        # Завантажуємо в Qdrant
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"✅ Завантажено {len(points)} points в Qdrant!")
    
    def load_from_file(self, filepath: str, collection_name: str = "chatgpt_conversations"):
        """Повний цикл завантаження з файлу"""
        logger.info("=" * 50)
        logger.info("Початок завантаження в RAG")
        
        # Парсинг
        conversations = self.parse_chatgpt_export(filepath)
        
        # Створення чанків
        chunks = self.create_chunks(conversations)
        
        # Завантаження
        self.upload_to_qdrant(chunks, collection_name)
        
        logger.info("=" * 50)
    
    def search(self, query: str, collection_name: str = "chatgpt_conversations", top_k: int = 5):
        """Пошук в RAG базі"""
        # Генеруємо вектор запиту
        query_vector = self.model.encode([query])[0]
        
        # Шукаємо
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        
        return results
