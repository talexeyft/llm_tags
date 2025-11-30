#!/usr/bin/env python3
"""
Пример использования OpenAI-совместимого API для тегирования обращений.

Поддерживаемые сервисы:
- OpenAI API (GPT-4, GPT-3.5-turbo)
- vLLM (локальный сервер)
- LM Studio (локальный сервер)
- Любые другие OpenAI-совместимые API
"""

import pandas as pd
from llm_tags import OpenAICompatibleLLM, TaggingPipeline


def example_openai():
    """Пример с OpenAI API"""
    print("=" * 60)
    print("Пример с OpenAI API")
    print("=" * 60)
    
    # Создаем тестовый DataFrame
    test_df = pd.DataFrame({
        "text": [
            "Не могу войти в личный кабинет",
            "Забыл пароль от аккаунта",
            "Товар не пришел уже неделю",
            "Как изменить тариф?",
            "Проблемы с интернетом"
        ]
    })
    
    # Инициализируем OpenAI LLM
    llm = OpenAICompatibleLLM(
        api_url="https://api.openai.com/v1",
        model="gpt-4",
        api_key="your-api-key-here",  # Замените на ваш API ключ
        temperature=0.7,
        max_tokens=4096
    )
    
    # Создаем pipeline
    pipeline = TaggingPipeline(llm=llm, batch_size=50, num_workers=5)
    
    # Промпт для тегирования
    prompt = """
    Проанализируй обращения клиентов телеком-оператора и определи теги для каждого.
    Теги должны отражать основную тему обращения.
    Возможные категории: авторизация, пароль, доставка, тарифы, технические_проблемы.
    """
    
    # Тегирование
    result_df, tags_dict = pipeline.tag(
        tags=test_df,
        text_column="text",
        tag_prompt=prompt,
        max_tags=3
    )
    
    print("\nРезультаты:")
    print(result_df[["text", "tags"]])
    print("\nСловарь тегов:")
    for tag, desc in tags_dict.items():
        print(f"  {tag}: {desc}")


def example_vllm():
    """Пример с локальным vLLM сервером"""
    print("\n" + "=" * 60)
    print("Пример с vLLM (локальный сервер)")
    print("=" * 60)
    print("\nЗапустите vLLM сервер:")
    print("python -m vllm.entrypoints.openai.api_server \\")
    print("    --model meta-llama/Llama-2-7b-chat-hf \\")
    print("    --port 8000")
    print()
    
    # Создаем тестовый DataFrame
    test_df = pd.DataFrame({
        "text": [
            "Не могу войти в личный кабинет",
            "Забыл пароль от аккаунта",
        ]
    })
    
    # Инициализируем vLLM
    llm = OpenAICompatibleLLM(
        api_url="http://localhost:8000/v1",
        model="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.7,
        max_tokens=4096
    )
    
    # Создаем pipeline
    pipeline = TaggingPipeline(llm=llm)
    
    # Промпт для тегирования
    prompt = """
    Проанализируй обращения клиентов и определи теги для каждого.
    """
    
    # Тегирование
    result_df, tags_dict = pipeline.tag(
        tags=test_df,
        text_column="text",
        tag_prompt=prompt,
        max_tags=3
    )
    
    print("\nРезультаты:")
    print(result_df[["text", "tags"]])


def example_lm_studio():
    """Пример с LM Studio"""
    print("\n" + "=" * 60)
    print("Пример с LM Studio")
    print("=" * 60)
    print("\n1. Запустите LM Studio")
    print("2. Загрузите модель")
    print("3. Включите локальный сервер (по умолчанию порт 1234)")
    print()
    
    # Создаем тестовый DataFrame
    test_df = pd.DataFrame({
        "text": [
            "Не могу войти в личный кабинет",
            "Забыл пароль от аккаунта",
        ]
    })
    
    # Инициализируем LM Studio
    llm = OpenAICompatibleLLM(
        api_url="http://localhost:1234/v1",
        model="local-model",  # LM Studio использует "local-model" как название
        temperature=0.7,
        max_tokens=4096
    )
    
    # Создаем pipeline
    pipeline = TaggingPipeline(llm=llm)
    
    # Промпт для тегирования
    prompt = """
    Проанализируй обращения клиентов и определи теги для каждого.
    """
    
    # Тегирование
    result_df, tags_dict = pipeline.tag(
        tags=test_df,
        text_column="text",
        tag_prompt=prompt,
        max_tags=3
    )
    
    print("\nРезультаты:")
    print(result_df[["text", "tags"]])


def example_custom_parameters():
    """Пример с настройкой всех параметров"""
    print("\n" + "=" * 60)
    print("Пример с настройкой всех параметров OpenAI API")
    print("=" * 60)
    
    # Инициализируем с полным набором параметров
    llm = OpenAICompatibleLLM(
        api_url="https://api.openai.com/v1",
        model="gpt-3.5-turbo",
        api_key="your-api-key-here",
        temperature=0.7,           # Креативность (0.0-2.0)
        max_tokens=4096,           # Максимум токенов в ответе
        top_p=0.9,                 # Nucleus sampling (0.0-1.0)
        frequency_penalty=0.5,     # Штраф за повторения (0.0-2.0)
        presence_penalty=0.5       # Штраф за присутствие токенов (0.0-2.0)
    )
    
    print("\nПараметры:")
    print(f"  API URL: {llm.api_url}")
    print(f"  Model: {llm.model}")
    print(f"  Temperature: {llm.temperature}")
    print(f"  Max tokens: {llm.max_tokens}")
    print(f"  Top P: {llm.top_p}")
    print(f"  Frequency penalty: {llm.frequency_penalty}")
    print(f"  Presence penalty: {llm.presence_penalty}")


if __name__ == "__main__":
    print("Примеры использования OpenAI-совместимого API\n")
    
    # Раскомментируйте нужный пример:
    
    # example_openai()
    # example_vllm()
    # example_lm_studio()
    example_custom_parameters()
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)

