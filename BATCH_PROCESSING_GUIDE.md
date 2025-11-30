# Руководство по обработке батчей

## Обзор

`TaggingPipeline` поддерживает два режима обработки батчей:

1. **Последовательная обработка** (`num_workers=1`)
2. **Параллельная обработка** (`num_workers>1`)

## Последовательная обработка (num_workers=1)

### Как работает:
- Батчи обрабатываются один за другим
- Каждый следующий батч видит теги, найденные во всех предыдущих батчах
- Теги автоматически накапливаются и переиспользуются

### Преимущества:
✅ Максимальная консистентность тегов  
✅ Избегание дублирования похожих тегов  
✅ Автоматическое накопление словаря тегов  

### Недостатки:
❌ Медленнее, чем параллельная обработка  
❌ Не использует многопоточность  

### Когда использовать:
- **Первая итерация** обработки данных
- Создание начального словаря тегов
- Когда важна консистентность тегов

### Пример:

```python
from llm_tags import TaggingPipeline, OllamaLLM

llm = OllamaLLM()
pipeline = TaggingPipeline(llm=llm, batch_size=50, num_workers=1)

# Обрабатываем данные последовательно
result_df, tags_dict = pipeline.tag(
    df,
    text_column="text",
    tag_prompt="Проанализируй обращения и определи теги.",
    max_tags=3
)

print(f"Найдено тегов: {len(tags_dict)}")
```

### Как это работает внутри:

```
Батч 1: [] → обрабатывается → находит теги [A, B]
Батч 2: [A, B] → обрабатывается → находит теги [A, C] (переиспользует A)
Батч 3: [A, B, C] → обрабатывается → находит теги [B, D] (переиспользует B)
Результат: [A, B, C, D]
```

## Параллельная обработка (num_workers>1)

### Как работает:
- Несколько батчей обрабатываются одновременно
- Все батчи получают одинаковый начальный словарь тегов
- Теги объединяются после завершения всех батчей

### Преимущества:
✅ Значительно быстрее  
✅ Эффективное использование ресурсов  
✅ Подходит для больших датасетов  

### Недостатки:
❌ Возможны похожие теги из разных батчей  
❌ Меньшая консистентность  

### Когда использовать:
- **Последующие итерации** обработки данных
- Когда словарь тегов уже накоплен
- Когда скорость важнее консистентности
- При передаче `existing_tags` из предыдущих итераций

### Пример:

```python
from llm_tags import TaggingPipeline, OllamaLLM

llm = OllamaLLM()
pipeline = TaggingPipeline(llm=llm, batch_size=50, num_workers=5)

# Обрабатываем данные параллельно
result_df, tags_dict = pipeline.tag(
    df,
    text_column="text",
    tag_prompt="Проанализируй обращения и определи теги.",
    existing_tags=previous_tags,  # Передаем накопленные теги
    max_tags=3
)

print(f"Найдено тегов: {len(tags_dict)}")
```

### Как это работает внутри:

```
Батч 1: [] → обрабатывается параллельно → находит [A, B]
Батч 2: [] → обрабатывается параллельно → находит [A', C] (создает похожий A')
Батч 3: [] → обрабатывается параллельно → находит [B', D] (создает похожий B')
Результат: [A, B, A', C, B', D] (возможны дубликаты)
```

## Рекомендуемая стратегия

### Итерация 1: Создание словаря тегов

```python
# Используем последовательную обработку для консистентности
pipeline = TaggingPipeline(llm=llm, batch_size=50, num_workers=1)

result1, tags1 = pipeline.tag(
    df1,
    text_column="text",
    tag_prompt=prompt,
    max_tags=3
)

# Сохраняем словарь тегов
import json
with open("tags_dict.json", "w", encoding="utf-8") as f:
    json.dump(tags1, f, ensure_ascii=False, indent=2)
```

### Итерация 2+: Использование накопленных тегов

```python
# Загружаем словарь тегов
import json
with open("tags_dict.json", "r", encoding="utf-8") as f:
    existing_tags = json.load(f)

# Используем параллельную обработку для скорости
pipeline = TaggingPipeline(llm=llm, batch_size=50, num_workers=5)

result2, tags2 = pipeline.tag(
    df2,
    text_column="text",
    tag_prompt=prompt,
    existing_tags=existing_tags,  # Передаем накопленные теги
    max_tags=3
)

# Обновляем словарь тегов
with open("tags_dict.json", "w", encoding="utf-8") as f:
    json.dump(tags2, f, ensure_ascii=False, indent=2)
```

## Сравнение производительности

| Параметр | num_workers=1 | num_workers=5 |
|----------|---------------|---------------|
| Скорость | 1x (базовая) | ~4-5x быстрее |
| Консистентность | Высокая | Средняя |
| Использование CPU | Низкое | Высокое |
| Накопление тегов | Автоматическое | Требует existing_tags |
| Рекомендуется | Первая итерация | Последующие итерации |

## Примеры использования

### Пример 1: Обработка большого датасета (первый раз)

```python
# Разбиваем на части
df_part1 = df.iloc[:10000]
df_part2 = df.iloc[10000:20000]
df_part3 = df.iloc[20000:]

# Обрабатываем первую часть последовательно
pipeline = TaggingPipeline(llm=llm, num_workers=1)
result1, tags1 = pipeline.tag(df_part1, text_column="text", tag_prompt=prompt)

# Остальные части обрабатываем параллельно с накопленными тегами
pipeline = TaggingPipeline(llm=llm, num_workers=5)
result2, tags2 = pipeline.tag(df_part2, text_column="text", tag_prompt=prompt, existing_tags=tags1)
result3, tags3 = pipeline.tag(df_part3, text_column="text", tag_prompt=prompt, existing_tags=tags2)

# Объединяем результаты
result_df = pd.concat([result1, result2, result3])
```

### Пример 2: Ежедневная обработка новых данных

```python
# День 1: создаем словарь тегов
pipeline = TaggingPipeline(llm=llm, num_workers=1)
result_day1, tags_day1 = pipeline.tag(df_day1, text_column="text", tag_prompt=prompt)

# День 2: используем накопленные теги
pipeline = TaggingPipeline(llm=llm, num_workers=5)
result_day2, tags_day2 = pipeline.tag(
    df_day2, 
    text_column="text", 
    tag_prompt=prompt,
    existing_tags=tags_day1
)

# День 3: продолжаем накапливать
result_day3, tags_day3 = pipeline.tag(
    df_day3, 
    text_column="text", 
    tag_prompt=prompt,
    existing_tags=tags_day2
)
```

## Демонстрация

Запустите демонстрационный скрипт для сравнения режимов:

```bash
python demo_sequential_vs_parallel.py
```

Этот скрипт покажет разницу между последовательной и параллельной обработкой на тестовых данных.

---

**Рекомендация:** Начинайте с `num_workers=1` для создания качественного словаря тегов, затем переходите на `num_workers=5` для ускорения обработки больших объемов данных.

