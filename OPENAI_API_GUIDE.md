# Руководство по использованию OpenAI-совместимых API

## Обзор

Класс `OpenAICompatibleLLM` позволяет использовать любые OpenAI-совместимые API для тегирования обращений:
- **OpenAI API** (GPT-4, GPT-3.5-turbo)
- **vLLM** (локальный сервер для запуска open-source моделей)
- **LM Studio** (GUI приложение для локальных моделей)
- **Любые другие** OpenAI-совместимые сервисы

## Быстрый старт

### 1. OpenAI API

```python
from llm_tags import TaggingPipeline, OpenAICompatibleLLM
import pandas as pd

# Инициализация
llm = OpenAICompatibleLLM(
    api_url="https://api.openai.com/v1",
    model="gpt-4",
    api_key="your-api-key-here"
)

# Создание pipeline
pipeline = TaggingPipeline(llm=llm)

# Тегирование
df = pd.read_csv("data.csv")
result_df, tags_dict = pipeline.tag(
    df,
    text_column="text",
    tag_prompt="Определи теги для обращений клиентов"
)
```

### 2. vLLM (локальный сервер)

**Установка vLLM:**
```bash
pip install vllm
```

**Запуск сервера:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000
```

**Использование:**
```python
from llm_tags import TaggingPipeline, OpenAICompatibleLLM

llm = OpenAICompatibleLLM(
    api_url="http://localhost:8000/v1",
    model="meta-llama/Llama-2-7b-chat-hf"
)

pipeline = TaggingPipeline(llm=llm)
```

### 3. LM Studio

**Настройка:**
1. Скачайте и установите [LM Studio](https://lmstudio.ai/)
2. Загрузите модель через интерфейс
3. Включите "Local Server" (по умолчанию порт 1234)

**Использование:**
```python
from llm_tags import TaggingPipeline, OpenAICompatibleLLM

llm = OpenAICompatibleLLM(
    api_url="http://localhost:1234/v1",
    model="local-model"  # LM Studio использует "local-model"
)

pipeline = TaggingPipeline(llm=llm)
```

## Параметры класса OpenAICompatibleLLM

### Обязательные параметры

- **`api_url`** (str): URL API
  - OpenAI: `"https://api.openai.com/v1"`
  - vLLM: `"http://localhost:8000/v1"`
  - LM Studio: `"http://localhost:1234/v1"`

- **`model`** (str): Название модели
  - OpenAI: `"gpt-4"`, `"gpt-3.5-turbo"`
  - vLLM: полное название модели, например `"meta-llama/Llama-2-7b-chat-hf"`
  - LM Studio: `"local-model"`

### Опциональные параметры

- **`api_key`** (str, optional): API ключ
  - Требуется для OpenAI
  - Не требуется для локальных серверов (vLLM, LM Studio)

- **`temperature`** (float, default=0.7): Температура генерации
  - Диапазон: 0.0 - 2.0
  - 0.0 - детерминированные результаты
  - 0.7 - баланс
  - 1.0+ - более креативные результаты

- **`max_tokens`** (int, default=4096): Максимум токенов в ответе
  - Рекомендуется: 2048-4096 для тегирования

- **`top_p`** (float, default=1.0): Nucleus sampling
  - Диапазон: 0.0 - 1.0
  - 1.0 - без ограничений
  - 0.9 - более сфокусированные результаты

- **`frequency_penalty`** (float, default=0.0): Штраф за повторения
  - Диапазон: 0.0 - 2.0
  - 0.0 - без штрафа
  - 0.5-1.0 - уменьшает повторения

- **`presence_penalty`** (float, default=0.0): Штраф за присутствие токенов
  - Диапазон: 0.0 - 2.0
  - 0.0 - без штрафа
  - 0.5-1.0 - стимулирует новые темы

## Примеры использования

### Пример 1: Базовое тегирование с OpenAI

```python
from llm_tags import TaggingPipeline, OpenAICompatibleLLM
import pandas as pd

# Данные
df = pd.DataFrame({
    "text": [
        "Не могу войти в личный кабинет",
        "Забыл пароль от аккаунта",
        "Товар не пришел уже неделю"
    ]
})

# LLM
llm = OpenAICompatibleLLM(
    api_url="https://api.openai.com/v1",
    model="gpt-4",
    api_key="sk-..."
)

# Pipeline
pipeline = TaggingPipeline(llm=llm)

# Промпт
prompt = """
Определи теги для обращений клиентов.
Категории: авторизация, пароль, доставка, тарифы.
"""

# Тегирование
result_df, tags_dict = pipeline.tag(
    df,
    text_column="text",
    tag_prompt=prompt
)

print(result_df[["text", "tags"]])
```

### Пример 2: Настройка параметров генерации

```python
llm = OpenAICompatibleLLM(
    api_url="https://api.openai.com/v1",
    model="gpt-4",
    api_key="sk-...",
    temperature=0.5,           # Меньше креативности
    max_tokens=2048,           # Меньше токенов
    top_p=0.9,                 # Более сфокусированные результаты
    frequency_penalty=0.5,     # Уменьшаем повторения
    presence_penalty=0.3       # Стимулируем разнообразие
)
```

### Пример 3: Батчинг и многопоточность

```python
from llm_tags import TaggingPipeline, OpenAICompatibleLLM

llm = OpenAICompatibleLLM(
    api_url="https://api.openai.com/v1",
    model="gpt-3.5-turbo",  # Быстрая модель
    api_key="sk-..."
)

# Pipeline с настройками производительности
pipeline = TaggingPipeline(
    llm=llm,
    batch_size=100,    # Большой батч для GPT-3.5
    num_workers=10     # Много параллельных запросов
)

# Обработка большого датасета
large_df = pd.read_csv("large_data.csv")
result_df, tags_dict = pipeline.tag(
    large_df,
    text_column="text",
    tag_prompt=prompt
)
```

### Пример 4: Использование с vLLM

```python
from llm_tags import TaggingPipeline, OpenAICompatibleLLM

# Запустите vLLM сервер в отдельном терминале:
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3-8B-Instruct \
#     --port 8000

llm = OpenAICompatibleLLM(
    api_url="http://localhost:8000/v1",
    model="meta-llama/Llama-3-8B-Instruct",
    temperature=0.7
)

pipeline = TaggingPipeline(
    llm=llm,
    batch_size=50,
    num_workers=5
)

result_df, tags_dict = pipeline.tag(
    df,
    text_column="text",
    tag_prompt=prompt
)
```

### Пример 5: Повторный прогон с существующими тегами

```python
# Первый прогон
result_df1, tags_dict1 = pipeline.tag(
    df1,
    text_column="text",
    tag_prompt=prompt
)

# Сохраняем словарь тегов
import json
with open("tags_dict.json", "w") as f:
    json.dump(tags_dict1, f, ensure_ascii=False, indent=2)

# Второй прогон с использованием существующих тегов
result_df2, tags_dict2 = pipeline.tag(
    df2,
    text_column="text",
    tag_prompt=prompt,
    existing_tags=tags_dict1  # Передаем теги из первого прогона
)
```

## Сравнение с Ollama

| Параметр | Ollama | OpenAI API | vLLM | LM Studio |
|----------|--------|------------|------|-----------|
| **Стоимость** | Бесплатно | Платно | Бесплатно | Бесплатно |
| **Скорость** | Средняя | Быстрая | Быстрая | Средняя |
| **Качество** | Зависит от модели | Высокое | Зависит от модели | Зависит от модели |
| **Настройка** | Простая | Простая | Средняя | Простая |
| **Требования** | GPU 8GB+ | Интернет | GPU 16GB+ | GPU 8GB+ |
| **Модели** | Любые Ollama | GPT-3.5, GPT-4 | Любые HuggingFace | Любые GGUF |

## Рекомендации

### Когда использовать OpenAI API:
- Нужно высокое качество результатов
- Нет мощного GPU
- Нужна быстрая обработка
- Бюджет позволяет

### Когда использовать vLLM:
- Есть мощный GPU (16GB+ VRAM)
- Нужна высокая скорость обработки
- Большие объемы данных
- Нужен полный контроль над моделью

### Когда использовать LM Studio:
- Нужен простой GUI
- Экспериментирование с разными моделями
- Небольшие объемы данных
- Есть GPU (8GB+ VRAM)

### Когда использовать Ollama:
- Простота настройки
- Средние объемы данных
- Есть GPU (8GB+ VRAM)
- Нужна стабильность

## Troubleshooting

### Ошибка подключения к API

```
ConnectionError: Не удалось подключиться к API
```

**Решение:**
1. Проверьте URL API
2. Проверьте API ключ (для OpenAI)
3. Убедитесь что сервер запущен (для vLLM/LM Studio)

### Ошибка 401 Unauthorized

```
API error: 401
```

**Решение:**
1. Проверьте API ключ
2. Убедитесь что ключ активен
3. Проверьте баланс аккаунта (для OpenAI)

### Ошибка 429 Rate Limit

```
API error: 429
```

**Решение:**
1. Уменьшите `num_workers` (например, до 3)
2. Добавьте задержки между запросами
3. Увеличьте лимиты в настройках аккаунта

### Медленная обработка

**Решение:**
1. Используйте более быструю модель (GPT-3.5 вместо GPT-4)
2. Увеличьте `batch_size` (до 100)
3. Увеличьте `num_workers` (до 10)
4. Используйте локальный vLLM сервер

## Дополнительные ресурсы

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [vLLM Documentation](https://docs.vllm.ai/)
- [LM Studio](https://lmstudio.ai/)
- [Полные примеры](openai_example.py)

