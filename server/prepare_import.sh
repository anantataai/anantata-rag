#!/bin/bash

# Отримуємо поточну дату
DATE=$(date +%Y-%m-%d)
IMPORT_DIR="/root/rag-memory/data/$DATE"
LATEST_LINK="/root/rag-memory/data/latest"

# Створюємо папку з датою
mkdir -p "$IMPORT_DIR"

echo "✅ Створено папку: $IMPORT_DIR"

# Видаляємо старе символічне посилання
rm -f "$LATEST_LINK"

# Створюємо нове посилання на останню папку
ln -s "$IMPORT_DIR" "$LATEST_LINK"

echo "✅ Посилання 'latest' оновлено → $DATE"
echo ""
echo "📂 Тепер копіюй файли в:"
echo "   $IMPORT_DIR/"
echo ""
echo "============================================"
echo "Команди для Windows PowerShell:"
echo "============================================"
echo ""
echo "# ChatGPT:"
echo "scp \"C:\\Users\\Admin\\OneDrive\\Desktop\\Dani LLM\\ChatGPT\\conversations\" root@46.62.204.28:$IMPORT_DIR/chatgpt_conversations.json"
echo ""
echo "# Claude:"
echo "scp \"C:\\Users\\Admin\\OneDrive\\Desktop\\Dani LLM\\Cloud\\data-2025-10-16-20-59-24-batch-0000\\conversations\" root@46.62.204.28:$IMPORT_DIR/claude_conversations.json"
echo ""
echo "============================================"

# Залишаємо тільки останні 5 імпортів
echo ""
echo "🧹 Очищення старих імпортів..."
cd /root/rag-memory/data
ls -t | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' | tail -n +6 | xargs -I {} rm -rf {}
echo "✅ Залишено останні 5 імпортів"
