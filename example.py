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
from llm_tags import TaggingPipeline, OllamaLLM


# Глобальная LLM модель для всех примеров
LLM = OllamaLLM(
    api_url="http://localhost:11434/api",
    model="qwen2.5:32b",
    temperature=0.7
)


def example_basic():
    """Базовый пример тегирования."""
    print("=" * 80)
    print("ПРИМЕР 1: Базовое тегирование")
    print("=" * 80)
    
    # Создаем тестовые данные
    df = pd.DataFrame({
        "text": [
            "Не могу войти в личный кабинет",
            "Забыл пароль от аккаунта",
            "Товар не пришел уже неделю",
            "Как изменить тариф?",
            "Проблемы с интернетом",
            "Хочу подключить роуминг",
            "Не работает мобильное приложение",
            "Списали деньги дважды",
        ]
    })
    
    # Создаем pipeline с передачей глобальной LLM модели
    pipeline = TaggingPipeline(llm=LLM, batch_size=4, num_workers=2)
    
    # Промпт для тегирования
    prompt = """
    Проанализируй обращения клиентов телеком-оператора и определи теги для каждого.
    Теги должны отражать основную тему обращения.
    
    Возможные категории тегов:
    - авторизация: проблемы со входом в систему
    - пароль: восстановление или смена пароля
    - доставка: вопросы по доставке товаров/сим-карт
    - тарифы: вопросы по тарифным планам
    - технические_проблемы: неполадки с услугами
    - роуминг: вопросы по роумингу
    - приложение: проблемы с мобильным приложением
    - биллинг: вопросы по оплате и списаниям
    
    Можешь создавать новые теги если нужно.
    """
    
    # Тегирование
    result_df, tags_dict = pipeline.tag(
        df,
        text_column="text",
        tag_prompt=prompt,
        max_tags=3
    )
    
    print("\nРезультаты тегирования:")
    print(result_df[["text", "tags"]])
    
    print("\nСловарь тегов:")
    for tag, desc in tags_dict.items():
        print(f"  • {tag}: {desc}")
    
    return result_df, tags_dict


def example_with_existing_tags(previous_tags_dict):
    """Пример с использованием существующих тегов."""
    print("\n" + "=" * 80)
    print("ПРИМЕР 2: Тегирование с использованием существующих тегов")
    print("=" * 80)
    
    # Новые данные
    df = pd.DataFrame({
        "text": [
            "Не могу зайти в аккаунт",
            "Потерял пароль",
            "Медленный интернет",
            "Хочу сменить тариф на более дешевый",
        ]
    })
    
    # Создаем pipeline с передачей глобальной LLM модели
    pipeline = TaggingPipeline(llm=LLM)
    
    prompt = """
    Проанализируй обращения клиентов телеком-оператора и определи теги для каждого.
    Используй существующие теги если они подходят, или создай новые.
    """
    
    # Тегирование с существующими тегами
    result_df, tags_dict = pipeline.tag(
        df,
        text_column="text",
        tag_prompt=prompt,
        existing_tags=previous_tags_dict,  # Передаем ранее созданные теги
        max_tags=2
    )
    
    print("\nРезультаты тегирования:")
    print(result_df[["text", "tags"]])
    
    print(f"\nВсего тегов в словаре: {len(tags_dict)}")
    
    return result_df, tags_dict


def example_skip_tagged():
    """Пример с пропуском уже размеченных обращений."""
    print("\n" + "=" * 80)
    print("ПРИМЕР 3: Пропуск обращений с существующими тегами")
    print("=" * 80)
    
    # Данные с уже проставленными тегами
    df = pd.DataFrame({
        "text": [
            "Не могу войти",
            "Забыл пароль",
            "Проблемы с интернетом",
            "Новое обращение без тегов",
            "Еще одно новое обращение",
        ],
        "tags": [
            "авторизация",
            "пароль, авторизация",
            "технические_проблемы, интернет",
            "",  # Нет тегов
            "",  # Нет тегов
        ]
    })
    
    # Создаем pipeline с передачей глобальной LLM модели
    pipeline = TaggingPipeline(llm=LLM)
    
    prompt = """
    Проанализируй обращения клиентов и определи теги.
    """
    
    # Обработать только строки без тегов (skip_if_tags_count=0)
    print("\nОбработка только строк без тегов (skip_if_tags_count=0):")
    result_df, tags_dict = pipeline.tag(
        df,
        text_column="text",
        tag_prompt=prompt,
        skip_if_tags_count=0,  # Пропустить строки с тегами
        max_tags=2
    )
    
    print("\nРезультаты:")
    print(result_df[["text", "tags"]])
    
    return result_df, tags_dict


def example_retag_with_limit():
    """Пример переразметки с ограничением."""
    print("\n" + "=" * 80)
    print("ПРИМЕР 4: Переразметка строк с малым количеством тегов")
    print("=" * 80)
    
    # Данные с разным количеством тегов
    df = pd.DataFrame({
        "text": [
            "Обращение 1",
            "Обращение 2",
            "Обращение 3",
            "Обращение 4",
        ],
        "tags": [
            "",  # 0 тегов
            "тег1",  # 1 тег
            "тег1, тег2",  # 2 тега
            "тег1, тег2, тег3",  # 3 тега
        ]
    })
    
    # Создаем pipeline с передачей глобальной LLM модели
    pipeline = TaggingPipeline(llm=LLM)
    
    prompt = "Определи теги для обращений."
    
    # Обработать строки с менее чем 2 тегами
    print("\nОбработка строк с < 2 тегами (skip_if_tags_count=2):")
    result_df, tags_dict = pipeline.tag(
        df,
        text_column="text",
        tag_prompt=prompt,
        skip_if_tags_count=2,  # Обработать строки с 0 или 1 тегом
        max_tags=3
    )
    
    print("\nРезультаты:")
    print(result_df[["text", "tags"]])
    
    return result_df, tags_dict


def example_load_from_file():
    """Пример загрузки данных из файла."""
    print("\n" + "=" * 80)
    print("ПРИМЕР 5: Загрузка данных из файла")
    print("=" * 80)
    
    # Создаем тестовый файл
    test_data = pd.DataFrame({
        "request_id": [f"req-{i:04d}" for i in range(10)],
        "text": [
            "Вопрос по тарифу",
            "Проблема с интернетом",
            "Не работает SMS",
            "Хочу подключить услугу",
            "Списали деньги",
            "Как отключить подписку",
            "Плохая связь",
            "Не приходят уведомления",
            "Хочу вернуть деньги",
            "Проблема с роумингом",
        ]
    })
    
    # Сохраняем в файл
    test_data.to_csv("test_data.csv", index=False)
    print("Создан тестовый файл: test_data.csv")
    
    # Загружаем из файла
    df = pd.read_csv("test_data.csv")
    print(f"Загружено {len(df)} обращений")
    
    # Создаем pipeline с передачей глобальной LLM модели
    pipeline = TaggingPipeline(llm=LLM, batch_size=5)
    
    prompt = """
    Определи теги для обращений клиентов телеком-оператора.
    Используй категории: тарифы, технические_проблемы, услуги, биллинг, роуминг.
    """
    
    # Тегирование
    result_df, tags_dict = pipeline.tag(
        df,
        text_column="text",
        tag_prompt=prompt,
        max_tags=2
    )
    
    # Сохраняем результаты
    result_df.to_csv("test_data_tagged.csv", index=False)
    print("\nРезультаты сохранены в: test_data_tagged.csv")
    
    print("\nПервые 5 результатов:")
    print(result_df[["request_id", "text", "tags"]].head())
    
    return result_df, tags_dict


def example_custom_ollama():
    """Пример с кастомными настройками Ollama."""
    print("\n" + "=" * 80)
    print("ПРИМЕР 6: Кастомные настройки Ollama")
    print("=" * 80)
    

    
    # Создаем pipeline с кастомным LLM
    pipeline = TaggingPipeline(
        llm=LLM,
        batch_size=3,
        num_workers=2
    )
    
    df = pd.DataFrame({
        "text": [
            "Тестовое обращение 1",
            "Тестовое обращение 2",
            "Тестовое обращение 3",
        ]
    })
    
    prompt = "Определи теги для обращений."
    
    result_df, tags_dict = pipeline.tag(
        df,
        text_column="text",
        tag_prompt=prompt,
        max_tags=2
    )
    
    print("\nРезультаты:")
    print(result_df[["text", "tags"]])
    
    return result_df, tags_dict


if __name__ == "__main__":
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ LLM_TAGS")
    print("=" * 80)
    print()
    print("Убедитесь что Ollama запущена: ollama serve")
    print("И модель установлена: ollama pull qwen2.5:32b")
    print()
    
    try:
        # Запускаем примеры
        result5, tags5 = example_load_from_file()
        result1, tags1 = example_basic()
        result2, tags2 = example_with_existing_tags(tags1)
        result3, tags3 = example_skip_tagged()
        result4, tags4 = example_retag_with_limit()
        #result5, tags5 = example_load_from_file()
        # result6, tags6 = example_custom_ollama()  # Раскомментируйте если нужно
        
        print("\n" + "=" * 80)
        print("ВСЕ ПРИМЕРЫ УСПЕШНО ВЫПОЛНЕНЫ!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nОшибка: {e}")
        print("\nУбедитесь что:")
        print("  1. Ollama запущена: ollama serve")
        print("  2. Модель установлена: ollama pull qwen2.5:32b")

