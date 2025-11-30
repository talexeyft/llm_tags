#!/usr/bin/env python3
"""
Модуль для обработки батчей обращений.

Содержит логику:
- Нормализации формата запросов
- Парсинга ответов от LLM
- Обработки результатов тегирования
"""

import json
import re
from typing import Optional


def normalize_requests(
    requests: list[tuple[str, str]] | list[dict]
) -> list[tuple[str, str]]:
    """
    Нормализовать формат запросов к единому виду [(id, text), ...].
    
    Args:
        requests: Список обращений в различных форматах:
            - [(id, text), ...]
            - [{"id": id, "text": text}, ...]
            - [str, ...] (обратная совместимость)
            
    Returns:
        Нормализованный список кортежей [(id, text), ...]
    """
    normalized_requests = []
    for req in requests:
        if isinstance(req, tuple):
            req_id, req_text = req
        elif isinstance(req, dict):
            req_id = req.get("id", str(hash(str(req))))
            req_text = req.get("text", "")
        else:
            # Обратная совместимость: если передан список строк
            req_id = str(hash(str(req)))
            req_text = str(req)
        normalized_requests.append((req_id, req_text))
    
    return normalized_requests


def clean_markdown_code_blocks(content: str) -> str:
    """
    Удалить markdown блоки кода из контента.
    
    Args:
        content: Текст с возможными markdown блоками кода
        
    Returns:
        Очищенный текст
    """
    if content.startswith("```"):
        # Удаляем markdown блоки кода
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]  # Удаляем первую строку с ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # Удаляем последнюю строку с ```
        content = "\n".join(lines).strip()
    
    return content


def parse_llm_response(
    content: str,
    normalized_requests: list[tuple[str, str]]
) -> tuple[list[dict], dict[str, str]]:
    """
    Распарсить ответ от LLM и извлечь результаты тегирования.
    
    Args:
        content: Сырой ответ от LLM
        normalized_requests: Список нормализованных запросов [(id, text), ...]
        
    Returns:
        Кортеж (список результатов, словарь новых тегов):
        - results: [{"id": "id1", "tags": ["тег1", "тег2"]}, ...]
        - new_tags: {"новый_тег1": "описание1", ...}
    """
    # Очистка контента от markdown блоков если есть
    content = clean_markdown_code_blocks(content)
    
    try:
        result = json.loads(content)
        
        # Ожидаем формат: {"results": [...], "new_tags": {...}}
        if isinstance(result, dict):
            results_list = result.get("results", [])
            new_tags = result.get("new_tags", {})
        else:
            # Обратная совместимость: если вернули массив
            results_list = result if isinstance(result, list) else [result]
            new_tags = {}
        
        # Проверяем что results_list - список
        if not isinstance(results_list, list):
            results_list = [results_list]
        
        # Создаем словарь для быстрого поиска по ID
        results_dict = {}
        for item in results_list:
            if isinstance(item, dict):
                req_id = item.get("id")
                if req_id:
                    results_dict[req_id] = item
        
        # Формируем результат в правильном порядке с дополнением недостающих
        final_results = []
        for req_id, req_text in normalized_requests:
            if req_id in results_dict:
                result_item = results_dict[req_id]
                # Убеждаемся что есть поле tags
                if "tags" not in result_item:
                    result_item["tags"] = []
                final_results.append({
                    "id": req_id,
                    "tags": result_item["tags"]
                })
            else:
                # Если ID не найден, создаем пустой результат
                final_results.append({
                    "id": req_id,
                    "tags": []
                })
        
        # Убеждаемся что new_tags - словарь
        if not isinstance(new_tags, dict):
            new_tags = {}
        
        return final_results, new_tags
        
    except json.JSONDecodeError as e:
        # Если не удалось распарсить JSON, пробуем альтернативные методы
        print(f"[ERROR] Failed to parse JSON: {e}")
        # Показываем больше контента для диагностики
        if len(content) <= 500:
            print(f"[ERROR] Full content: {content}")
        else:
            print(f"[ERROR] Content (first 300 chars): {content[:300]}")
            print(f"[ERROR] Content (last 300 chars): {content[-300:]}")
        
        # Попытка 1: Попробовать найти объект с results и new_tags
        try:
            # Ищем JSON объект с фигурными скобками
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                if isinstance(result, dict):
                    results_list = result.get("results", [])
                    new_tags = result.get("new_tags", {})
                    
                    results_dict = {}
                    for item in results_list:
                        if isinstance(item, dict) and "id" in item:
                            results_dict[item["id"]] = item
                    
                    final_results = []
                    for req_id, req_text in normalized_requests:
                        if req_id in results_dict:
                            final_results.append({
                                "id": req_id,
                                "tags": results_dict[req_id].get("tags", [])
                            })
                        else:
                            final_results.append({"id": req_id, "tags": []})
                    
                    print(f"[SUCCESS] Attempt 1 succeeded! Parsed {len(final_results)} results")
                    return final_results, new_tags if isinstance(new_tags, dict) else {}
        except Exception as e1:
            print(f"[ERROR] Attempt 1 failed: {e1}")
        
        # Попытка 2: Если не нашли структуру, возвращаем пустые результаты
        print(f"[ERROR] All parsing attempts failed, returning empty results")
        return (
            [{"id": req_id, "tags": []} for req_id, _ in normalized_requests],
            {}
        )

