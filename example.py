#!/usr/bin/env python3
"""
Пример использования llm_tags для тегирования обращений клиентов.

Этот скрипт демонстрирует:
1. Базовое использование TaggingPipeline
2. Работу с существующими тегами
3. Фильтрацию по количеству тегов
4. Загрузку данных из различных форматов
"""

import pandas as pd
from llm_tags import TaggingPipeline
from lmstudio_llm import LMStudioLLM


# Глобальная LLM модель для всех примеров (LM Studio)
LLM = LMStudioLLM(
    api_url="http://192.168.1.26:1234/v1",
    model="qwen3-30b-a3b-instruct-2507",
    temperature=0.7,
    max_tokens=8192*2,
    timeout=600,
    disable_thinking=True   
)



def example_load_from_file():
    """Пример загрузки данных из файла."""
    print("\n" + "=" * 80)
    print("ПРИМЕР 5: Загрузка данных из файла")
    print("=" * 80)
    
    data_path = "/home/alex/tradeML/llm_tags/demo_sample.csv.gz"


    df = pd.read_csv(data_path, compression='gzip')

    print(f"Загружено {len(df)} обращений")
    print(f"Колонки: {list(df.columns)}")
    print(f"\nПервые 5 строк:")
    print(df.head())

    
    # Создаем pipeline с передачей глобальной LLM модели
    pipeline = TaggingPipeline(llm=LLM, batch_size=20)
    
    prompt = """
    Определи теги для обращений клиентов телеком-оператора.
    Теги должны отражать основную проблему обращения.
    Ты должна создавать новые теги, если существующие не подходят.
    Формируей теги на русском языке.
    """
    
    # Тегирование холодный старт - последовательная обработка
    result_df, tags_dict = pipeline.tag(
        df.sample(100),
        text_column="text",
        tag_prompt=prompt,
        max_tags=5,
        num_workers=1,
        batch_size=20
    )

    # Тегирование горячий старт - параллельная обработка с накопленными тегами
    result_df, tags_dict = pipeline.tag(
        df,
        existing_tags=tags_dict,
        text_column="text",
        tag_prompt=prompt,
        max_tags=5,
        num_workers=10,
        batch_size=20
    )

    
    # Сохраняем результаты

    print("\nПервые 5 результатов:")
    print(result_df[["request_id", "text", "tags"]].head())
    
    # 1. Раскидываем теги по столбцам
    print("\n" + "=" * 80)
    print("РАСКИДЫВАНИЕ ТЕГОВ ПО СТОЛБЦАМ")
    print("=" * 80)
    expanded_df = pipeline.expand_tags_to_columns(result_df, tags_column="tags")
    tag_columns = [col for col in expanded_df.columns if col.startswith("tag_")]
    print(f"\nСоздано {len(tag_columns)} столбцов с тегами")
    print(f"Примеры столбцов: {tag_columns[:10]}")
    print(f"\nПервые 5 строк с тегами:")
    display_columns = ["request_id", "text"] + tag_columns[:5]
    print(expanded_df[display_columns].head())
    
    # 2. Анализ тегов: статистика и корреляции
    print("\n" + "=" * 80)
    print("АНАЛИЗ ТЕГОВ: СТАТИСТИКА И КОРРЕЛЯЦИИ")
    print("=" * 80)
    analysis = pipeline.analyze_tags(result_df, tags_column="tags")
    
    print("\n--- Статистика по тегам ---")
    print(analysis['statistics'].head(20))
    
    print("\n--- Топ-10 пар тегов, которые чаще всего встречаются вместе ---")
    print(analysis['top_pairs'])
    
    print("\n--- Корреляционная матрица (первые 10x10) ---")
    corr_matrix = analysis['correlation']
    print(corr_matrix.iloc[:10, :10])
    
    # 3. Объединение редких и похожих тегов
    print("\n" + "=" * 80)
    print("ОБЪЕДИНЕНИЕ РЕДКИХ И ПОХОЖИХ ТЕГОВ")
    print("=" * 80)
    
    print(f"\nДо объединения: {len(tags_dict)} тегов")
    print(f"Примеры редких тегов (частота < 5):")
    rare_tags_stats = analysis['statistics'][analysis['statistics']['count'] < 5]
    if len(rare_tags_stats) > 0:
        print(rare_tags_stats.head(10))
    
    # Объединяем теги
    merged_df, merged_tags_dict, tag_mapping = pipeline.merge_similar_tags(
        result_df,
        tags_dict,
        tags_column="tags",
        min_frequency=5,  # Теги с частотой < 5 считаются редкими
        similarity_threshold=0.7,  # Порог схожести для объединения
        use_llm_for_similarity=True,  # Использовать LLM для семантического сравнения
        merge_rare_to_common=True  # Объединять редкие с частыми
    )
    
    print(f"\nПосле объединения: {len(merged_tags_dict)} тегов")
    print(f"\nПримеры объединений (первые 10):")
    for i, (old_tag, new_tag) in enumerate(list(tag_mapping.items())[:10]):
        print(f"  {old_tag} -> {new_tag}")
    
    # Показываем статистику после объединения
    merged_analysis = pipeline.analyze_tags(merged_df, tags_column="tags")
    print(f"\n--- Статистика по тегам после объединения (топ-20) ---")
    print(merged_analysis['statistics'].head(20))
    
    return result_df, tags_dict, expanded_df, analysis, merged_df, merged_tags_dict, tag_mapping



if __name__ == "__main__":
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ LLM_TAGS")
    print("=" * 80)
    print()
    print("Убедитесь что LM Studio запущена и сервер активен")
    print("(API доступен на http://192.168.1.26:1234/v1)")
    print("Модель: qwen3-32b-128k")
    print()
    
    try:
        # Запускаем примеры
        result5, tags5, expanded_df, analysis, merged_df, merged_tags_dict, tag_mapping = example_load_from_file()

        #result5, tags5 = example_load_from_file()
        # result6, tags6 = example_custom_ollama()  # Раскомментируйте если нужно
        
        print("\n" + "=" * 80)
        print("ВСЕ ПРИМЕРЫ УСПЕШНО ВЫПОЛНЕНЫ!")
        print("=" * 80)
        print(f"\nИтоговая статистика:")
        print(f"  Исходных тегов: {len(tags5)}")
        print(f"  После объединения: {len(merged_tags_dict)}")
        print(f"  Объединено тегов: {len(tag_mapping)}")
        
    except Exception as e:
        print(f"\nОшибка: {e}")
        print("\nУбедитесь что:")
        print("  1. LM Studio запущена и сервер активен")
        print("  2. Модель qwen3-32b-128k загружена в LM Studio")
        print("  3. API сервер доступен на http://192.168.1.26:1234/v1")

