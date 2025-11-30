#!/usr/bin/env python3
"""
Скрипт для диагностики проблем с тегированием.
Тестирует на небольшом батче и показывает детальную информацию.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, '/home/alex/tradeML/llm_tags')

from llm_tags import TaggingPipeline, OllamaLLM

# Загрузка данных
data_path = "/home/alex/tradeML/llm_tags/demo_sample.csv.gz"
df = pd.read_csv(data_path, compression='gzip')

print(f"Загружено {len(df)} обращений")
print(f"\nПервые 3 обращения:")
for i, row in df.head(3).iterrows():
    print(f"\n{i+1}. ID: {row['request_id']}")
    print(f"   Speaker: {row['speaker']}")
    print(f"   Text: {row['text'][:150]}...")

# Инициализация LLM
llm = OllamaLLM(
    api_url="http://localhost:11434/api",
    model="qwen3:32b",
    temperature=0.7
)

# Тестовый промпт
tag_prompt = """
Проанализируй обращения клиентов телеком-оператора Union Mobile и определи теги для каждого.
Теги должны отражать основную тему и проблему обращения.

Возможные категории тегов:
- авторизация: проблемы со входом в систему, личный кабинет
- пароль: восстановление или смена пароля
- тарифы: вопросы по тарифным планам, изменение тарифа
- биллинг: вопросы по оплате, списаниям, счетам
- технические_проблемы: неполадки с услугами (интернет, звонки, SMS)
- роуминг: вопросы по роумингу
- настройки: настройка телефона, функций, приложений
- устройство: проблемы с устройством, покупка нового телефона
- sim_карта: вопросы по SIM-карте, замена, активация
- доставка: вопросы по доставке товаров/сим-карт
- жалоба: жалобы на обслуживание, качество связи
- консультация: общие вопросы, информация об услугах
- обслуживание: работа агента, приветствия, прощания
- диалог: короткие ответы в диалоге

Можешь создавать новые теги если существующие не подходят.
Используй от 1 до 3 тегов на обращение.
"""

print("\n" + "="*80)
print("ТЕСТИРОВАНИЕ НА МАЛЕНЬКОМ БАТЧЕ (5 обращений)")
print("="*80)

# Берем только первые 5 обращений
test_df = df.head(5).copy()

# Тестируем напрямую метод tag_batch
print("\n1. Прямой вызов llm.tag_batch():")
print("-" * 80)

requests = test_df['text'].tolist()
print(f"Количество обращений: {len(requests)}")

try:
    results = llm.tag_batch(
        requests=requests,
        prompt=tag_prompt,
        existing_tags=None,
        max_tags=3
    )
    
    print(f"\nПолучено результатов: {len(results)}")
    print(f"Тип результатов: {type(results)}")
    
    for i, result in enumerate(results):
        print(f"\nРезультат {i+1}:")
        print(f"  Тип: {type(result)}")
        print(f"  Содержимое: {result}")
        
        if isinstance(result, dict):
            tags = result.get("tags", [])
            descriptions = result.get("descriptions", {})
            print(f"  Теги: {tags}")
            print(f"  Описания: {descriptions}")
        else:
            print(f"  ОШИБКА: Результат не является словарем!")

except Exception as e:
    print(f"\nОШИБКА при вызове tag_batch: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("2. Тестирование через Pipeline:")
print("-" * 80)

pipeline = TaggingPipeline(
    llm=llm,
    batch_size=5,
    num_workers=1  # Используем 1 поток для упрощения отладки
)

try:
    result_df, tags_dict = pipeline.tag(
        tags=test_df,
        text_column="text",
        tag_prompt=tag_prompt,
        id_column="request_id",
        skip_if_tags_count=999,
        max_tags=3
    )
    
    print(f"\nРезультаты:")
    print(f"  Всего обращений: {len(result_df)}")
    print(f"  Всего тегов в словаре: {len(tags_dict)}")
    
    # Проверяем сколько обращений получили теги
    has_tags = result_df['tags'].apply(lambda x: bool(x and str(x).strip()))
    print(f"  Обращений с тегами: {has_tags.sum()}")
    print(f"  Обращений без тегов: {(~has_tags).sum()}")
    
    print("\nДетали по каждому обращению:")
    for i, row in result_df.iterrows():
        print(f"\n{i+1}. ID: {row['request_id']}")
        print(f"   Text: {row['text'][:100]}...")
        print(f"   Tags: '{row['tags']}'")
        print(f"   Has tags: {bool(row['tags'] and str(row['tags']).strip())}")
    
    if tags_dict:
        print(f"\nСловарь тегов:")
        for tag, desc in tags_dict.items():
            print(f"  - {tag}: {desc}")
    else:
        print("\nСловарь тегов ПУСТ!")

except Exception as e:
    print(f"\nОШИБКА при вызове pipeline.tag: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("ДИАГНОСТИКА ЗАВЕРШЕНА")
print("="*80)

