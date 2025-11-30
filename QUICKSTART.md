# Быстрый старт LLM Tags

## Установка и настройка (5 минут)

### 1. Установите зависимости

```bash
cd /home/alex/tradeML/llm_tags
pip install -r requirements.txt
```

### 2. Запустите Ollama

```bash
# В отдельном терминале
ollama serve

# Скачайте модель (если еще не установлена)
ollama pull qwen2.5:32b
```

## Использование

### Минимальный пример

```python
import pandas as pd
from llm_tags import TaggingPipeline

# Загрузка данных
df = pd.read_parquet("your_data.parquet")
# или
df = pd.read_csv("your_data.csv.gz", compression="gzip")

# Создание pipeline
pipeline = TaggingPipeline()

# Промпт
prompt = """
Проанализируй обращения клиентов и определи теги для каждого.
Теги должны отражать основную тему обращения.
"""

# Тегирование
result_df, tags_dict = pipeline.tag(
    df,
    text_column="text",  # Название колонки с текстом
    tag_prompt=prompt
)

# Сохранение
result_df.to_csv("tagged_data.csv", index=False)

# Вывод словаря тегов
print("Найденные теги:")
for tag, desc in tags_dict.items():
    print(f"  {tag}: {desc}")
```

### Повторный прогон (только новые обращения)

```python
# Первый прогон
result_df1, tags_dict1 = pipeline.tag(df1, text_column="text", tag_prompt=prompt)

# Второй прогон - используем теги из первого
result_df2, tags_dict2 = pipeline.tag(
    df2,
    text_column="text",
    tag_prompt=prompt,
    existing_tags=tags_dict1,  # Передаем существующие теги
    skip_if_tags_count=0        # Обработать только строки без тегов
)
```

## Параметры

### Основные параметры `tag()`

- `text_column` - название колонки с текстом (по умолчанию "text")
- `tag_prompt` - промпт с инструкциями по тегированию
- `existing_tags` - словарь существующих тегов (опционально)
- `skip_if_tags_count` - пропустить строки с >= N тегов (по умолчанию 10)
- `max_tags` - максимум тегов на обращение (по умолчанию 5)

### Настройка pipeline

```python
pipeline = TaggingPipeline(
    batch_size=50,      # Размер батча (30-100)
    num_workers=5,      # Параллельных потоков (3-10)
    cache_dir="./cache" # Директория для кеша
)
```

## Примеры

Запустите файл с примерами:

```bash
python example.py
```

## Troubleshooting

### Ollama не отвечает

```bash
# Проверьте что Ollama запущена
ollama serve

# Проверьте список моделей
ollama list

# Установите модель если нужно
ollama pull qwen2.5:32b
```

### Медленная обработка

Уменьшите параметры:

```python
pipeline = TaggingPipeline(
    batch_size=30,   # Меньше батч
    num_workers=3    # Меньше потоков
)
```

## Полная документация

См. [README.md](README.md) для подробной документации.

