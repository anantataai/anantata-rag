"""
Автоматичне оновлення RAG пам'яті з ChatGPT та Claude
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from chatgpt_to_qdrant import ChatGPTToQdrant

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/rag_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ініціалізація RAG
QDRANT_URL = "http://localhost:6333"
rag = ChatGPTToQdrant(qdrant_url=QDRANT_URL)

# Шляхи до файлів (через latest)
DATA_DIR = Path("/root/rag-memory/data")
LATEST_DIR = DATA_DIR / "latest"
CHATGPT_FILE = LATEST_DIR / "chatgpt_conversations.json"
CLAUDE_FILE = LATEST_DIR / "claude_conversations.json"


def update_from_chatgpt():
    """Оновлення з ChatGPT"""
    logger.info("=" * 50)
    logger.info("📱 ChatGPT: Початок оновлення")
    
    if not CHATGPT_FILE.exists():
        logger.warning(f"❌ Файл не знайдено: {CHATGPT_FILE}")
        return False
    
    try:
        logger.info(f"📂 Читаю: {CHATGPT_FILE}")
        
        # Використовуємо існуючий метод
        rag.load_from_file(str(CHATGPT_FILE))
        
        logger.info("✅ ChatGPT успішно оновлено!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Помилка ChatGPT: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def parse_claude_conversations(data):
    """Парсинг формату Claude"""
    conversations = []
    
    if isinstance(data, list):
        for conv in data:
            conv_id = conv.get('uuid', 'unknown')
            title = conv.get('name', 'Без назви')
            created_at = conv.get('created_at', '')
            
            messages = []
            chat_messages = conv.get('chat_messages', [])
            
            for msg in chat_messages:
                sender = msg.get('sender', 'unknown')
                text = msg.get('text', '')
                
                if text:
                    messages.append({
                        'role': 'user' if sender == 'human' else 'assistant',
                        'content': text
                    })
            
            if messages:
                conversations.append({
                    'id': conv_id,
                    'title': title,
                    'create_time': created_at,
                    'messages': messages
                })
    
    return conversations


def update_from_claude():
    """Оновлення з Claude"""
    logger.info("=" * 50)
    logger.info("🤖 Claude: Початок оновлення")
    
    if not CLAUDE_FILE.exists():
        logger.warning(f"❌ Файл не знайдено: {CLAUDE_FILE}")
        return False
    
    try:
        logger.info(f"📂 Читаю: {CLAUDE_FILE}")
        
        with open(CLAUDE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = parse_claude_conversations(data)
        
        logger.info(f"📊 Знайдено {len(conversations)} розмов Claude")
        
        if not conversations:
            logger.warning("⚠️ Розмов не знайдено")
            return False
        
        # Конвертуємо в простий формат для завантаження
        # Поки що просто логуємо інформацію
        for conv in conversations[:5]:  # Перші 5 для прикладу
            logger.info(f"  📝 {conv['title']}: {len(conv['messages'])} повідомлень")
        
        logger.info("✅ Claude прочитано!")
        logger.info("⚠️ Повне завантаження Claude в RAG - в розробці")
        return True
        
    except Exception as e:
        logger.error(f"❌ Помилка Claude: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def get_rag_stats():
    """Статистика RAG"""
    try:
        collections_info = rag.client.get_collections()
        
        logger.info("=" * 50)
        logger.info("📊 Статистика RAG бази:")
        
        for collection in collections_info.collections:
            points = collection.points_count if hasattr(collection, 'points_count') else 'N/A'
            logger.info(f"  📦 {collection.name}: {points} записів")
            
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Помилка статистики: {e}")


def main():
    """Головна функція"""
    logger.info("\n" + "=" * 50)
    logger.info(f"🚀 Початок оновлення: {datetime.now()}")
    logger.info("=" * 50)
    
    results = {
        'chatgpt': update_from_chatgpt(),
        'claude': update_from_claude()
    }
    
    # Статистика
    get_rag_stats()
    
    # Підсумок
    logger.info("=" * 50)
    logger.info("📋 Підсумок:")
    for source, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"  {status} {source.upper()}")
    
    logger.info(f"🏁 Завершено: {datetime.now()}")
    logger.info("=" * 50 + "\n")


if __name__ == "__main__":
    main()
