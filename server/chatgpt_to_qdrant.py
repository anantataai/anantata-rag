
import os
import json
from datetime import datetime
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


class ChatGPTToQdrant:
    def __init__(self, qdrant_url: str = None):
        """
        Ініціалізація клієнта Qdrant та моделі ембедингів.
        """
        load_dotenv()

        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = os.getenv("QDRANT_COLLECTION", "chatgpt_conversations")
        self.api_key = os.getenv("QDRANT_API_KEY", None)

        self.client = QdrantClient(url=self.qdrant_url, api_key=self.api_key)
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.vector_size = 384

        print(f"��� Підключено до Qdrant: {self.qdrant_url}")
        print(f"��� Колекція за замовчуванням: {self.collection_name}")

    def parse_conversation(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = []
        mapping = conversation.get("mapping", {})
        title = conversation.get("title", "Без назви")

        for node_id, node_data in mapping.items():
            message = node_data.get("message")
            if not message:
                continue

            author = message.get("author", {})
            role = author.get("role")
            if role == "system":
                continue

            content = message.get("content", {})
            parts = content.get("parts", [])
            text = " ".join([str(p) for p in parts if p])

            if not text.strip():
                continue

            create_time = message.get("create_time")
            timestamp = None
            if create_time:
                try:
                    if 0 < create_time < 2147483647:
                        timestamp = datetime.fromtimestamp(create_time).isoformat()
                except Exception:
                    pass

            messages.append({
                "id": message.get("id"),
                "role": role,
                "text": text.strip(),
                "conversation_title": title,
                "timestamp": timestamp
            })
        return messages

    def create_chunks(self, messages: List[Dict[str, Any]], chunk_size: int = 2) -> List[Dict[str, Any]]:
        chunks = []
        for i in range(0, len(messages), chunk_size):
            chunk_messages = messages[i:i + chunk_size]
            combined_text = "\n\n".join([f"{m['role'].upper()}: {m['text']}" for m in chunk_messages])
            chunks.append({
                "text": combined_text,
                "conversation_title": chunk_messages[0]["conversation_title"],
                "roles": [m["role"] for m in chunk_messages],
                "timestamp": chunk_messages[0]["timestamp"],
                "message_ids": [m["id"] for m in chunk_messages]
            })
        return chunks

    def create_collection(self, collection_name: str = None):
        collection_name = collection_name or self.collection_name
        try:
            self.client.delete_collection(collection_name)
            print(f"���️ Існуючу колекцію '{collection_name}' видалено.")
        except Exception:
            pass

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )
        print(f"✅ Створено колекцію '{collection_name}'")

    def load_to_qdrant(self, json_file_path: str, collection_name: str = None):
        collection_name = collection_name or self.collection_name
        print(f"��� Завантаження '{json_file_path}'...")

        with open(json_file_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)

        print(f"��� Знайдено {len(conversations)} розмов.")
        self.create_collection(collection_name)

        all_chunks = []
        for idx, conv in enumerate(conversations):
            msgs = self.parse_conversation(conv)
            if msgs:
                all_chunks.extend(self.create_chunks(msgs))
            if (idx + 1) % 10 == 0:
                print(f"��� Оброблено {idx + 1}/{len(conversations)}")

        print(f"Всіх чанків: {len(all_chunks)}")

        batch_size = 32
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [ch["text"] for ch in batch]
            embeddings = self.model.encode(texts, show_progress_bar=False)
            points = [
                PointStruct(
                    id=i + j,
                    vector=emb.tolist(),
                    payload={
                        "text": ch["text"],
                        "conversation_title": ch["conversation_title"],
                        "roles": ch["roles"],
                        "timestamp": ch["timestamp"],
                        "message_ids": ch["message_ids"]
                    }
                )
                for j, (ch, emb) in enumerate(zip(batch, embeddings))
            ]
            self.client.upsert(collection_name=collection_name, points=points)
        print(f"✅ {len(all_chunks)} чанків завантажено до Qdrant!")

    def search(self, query: str, collection_name: str = None, top_k: int = 5):
        collection_name = collection_name or self.collection_name
        query_vector = self.model.encode(query).tolist()
        results = self.client.search(collection_name=collection_name, query_vector=query_vector, limit=top_k)
        return results


if __name__ == "__main__":
    loader = ChatGPTToQdrant()
    loader.load_to_qdrant("conversations.json")
    results = loader.search("різниця між RAG серверами", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. ({r.score:.4f}) {r.payload['conversation_title']}")
        print(r.payload["text"][:200])

