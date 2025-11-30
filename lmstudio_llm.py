#!/usr/bin/env python3
"""
Модуль для работы с LM Studio через OpenAI-совместимый API.

Специализированный класс для работы с LM Studio, оптимизированный
для работы с локальными моделями через LM Studio сервер.
"""

import json
import re
from typing import Optional
import urllib3

from prompt_builder import build_tagging_prompt
from batch_processor import normalize_requests, parse_llm_response

# Отключаем предупреждения urllib3
urllib3.disable_warnings()


class LMStudioLLM:
    """
    Класс для работы с LM Studio через OpenAI-совместимый API.
    
    Оптимизирован для работы с локальными моделями через LM Studio сервер.
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:1234/v1",
        model: str = "local-model",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        timeout: int = 600,
        disable_thinking: bool = False
    ):
        """
        Инициализация LM Studio LLM.
        
        Args:
            api_url: URL API LM Studio (по умолчанию http://localhost:1234/v1)
            model: Название модели в LM Studio (по умолчанию "local-model")
            temperature: Температура генерации (по умолчанию 0.7)
            max_tokens: Максимальное количество токенов в ответе (по умолчанию 8192)
            timeout: Таймаут запроса в секундах (по умолчанию 600)
            disable_thinking: Если True, добавляет "/nothink" в системный промпт для отключения размышлений
        """
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.disable_thinking = disable_thinking
        
    def _make_request(self, endpoint: str, payload: dict) -> dict:
        """
        Выполнить HTTP запрос к LM Studio API.
        
        Args:
            endpoint: Конечная точка API (например, "chat/completions")
            payload: Данные запроса
            
        Returns:
            Ответ от API в виде словаря
        """
        url = f"{self.api_url}/{endpoint}"
        
        # Парсим URL
        if url.startswith("http://"):
            protocol = "http"
            url = url[7:]
        elif url.startswith("https://"):
            protocol = "https"
            url = url[8:]
        else:
            protocol = "http"
            # Если нет протокола, добавляем http://
            if not url.startswith("http"):
                url = f"http://{url}"
                protocol = "http"
                url = url[7:]
        
        host_port = url.split("/", 1)
        host = host_port[0]
        path = "/" + host_port[1] if len(host_port) > 1 else "/"
        
        if ":" in host:
            host, port = host.split(":")
            port = int(port)
        else:
            port = 1234 if protocol == "http" else 443
        
        try:
            # Используем urllib3 для HTTP запросов
            if protocol == "https":
                http = urllib3.PoolManager(
                    num_pools=1,
                    maxsize=1,
                    timeout=urllib3.Timeout(connect=10, read=self.timeout),
                    cert_reqs='CERT_NONE'
                )
            else:
                http = urllib3.PoolManager(
                    num_pools=1,
                    maxsize=1,
                    timeout=urllib3.Timeout(connect=10, read=self.timeout)
                )
            
            headers = {
                "Content-Type": "application/json",
                "Connection": "close"
            }
            
            # LM Studio обычно не требует API ключ, но если нужен - можно добавить
            # if self.api_key:
            #     headers["Authorization"] = f"Bearer {self.api_key}"
            
            full_url = f"{protocol}://{host}:{port}{path}"
            
            response = http.request(
                "POST",
                full_url,
                body=json.dumps(payload).encode("utf-8"),
                headers=headers
            )
            
            if response.status != 200:
                error_text = response.data.decode("utf-8", errors='ignore')
                print(f"[LMStudio] Ошибка API: статус {response.status}")
                print(f"[LMStudio] Ответ: {error_text[:500]}")
                raise ConnectionError(
                    f"LM Studio API error: {response.status}. Response: {error_text[:500]}"
                )
            
            response_data = json.loads(response.data.decode("utf-8"))
            return response_data
            
        except json.JSONDecodeError as e:
            error_text = response.data.decode("utf-8", errors='ignore') if 'response' in locals() else "No response"
            print(f"[LMStudio] Ошибка парсинга JSON: {e}")
            print(f"[LMStudio] Ответ: {error_text[:500]}")
            raise ConnectionError(
                f"Не удалось распарсить ответ от LM Studio. Ошибка: {e}. Ответ: {error_text[:500]}"
            )
        except Exception as e:
            print(f"[LMStudio] Общая ошибка: {e}")
            raise ConnectionError(
                f"Не удалось подключиться к LM Studio на {self.api_url}. Ошибка: {e}"
            )
    
    def _extract_content(self, response: dict) -> str:
        """
        Извлечь контент из ответа LM Studio API.
        
        Args:
            response: Ответ от API LM Studio
            
        Returns:
            Текст контента
            
        Raises:
            ValueError: Если формат ответа неожиданный
        """
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0].get("message", {}).get("content", "").strip()
            return content
        else:
            print(f"[LMStudio] Неожиданный формат ответа: {list(response.keys())}")
            print(f"[LMStudio] Полный ответ: {json.dumps(response, indent=2, ensure_ascii=False)[:1000]}")
            raise ValueError(f"Неожиданный формат ответа от LM Studio: {response}")
    
    def tag_batch(
        self,
        requests: list[tuple[str, str]] | list[dict],
        prompt: str,
        existing_tags: Optional[dict[str, str]] = None,
        max_tags: int = 5,
        allow_new_tags: bool = True,
        disable_thinking: Optional[bool] = None
    ) -> tuple[list[dict], dict[str, str]]:
        """
        Тегировать батч обращений.
        
        Args:
            requests: Список обращений в формате [(id, text), ...] или [{"id": id, "text": text}, ...]
            prompt: Промпт с инструкциями по тегированию
            existing_tags: Существующие теги {tag_name: description}
            max_tags: Максимальное количество тегов на обращение
            allow_new_tags: Разрешить создание новых тегов (по умолчанию True)
            disable_thinking: Если True, добавляет "/nothink" в системный промпт для отключения размышлений.
                             Если None, используется значение из конструктора.
            
        Returns:
            Кортеж (список результатов с id и тегами, словарь новых тегов)
            Формат: (
                [{"id": "id1", "tags": ["тег1", "тег2"]}, ...],
                {"новый_тег1": "описание1", "новый_тег2": "описание2", ...}
            )
        """
        # Нормализуем формат запросов
        normalized_requests = normalize_requests(requests)
        
        # Формируем промпт
        full_prompt = build_tagging_prompt(
            base_prompt=prompt,
            requests=normalized_requests,
            existing_tags=existing_tags,
            max_tags=max_tags,
            allow_new_tags=allow_new_tags
        )
        
        # Используем значение из параметра или из конструктора
        use_disable_thinking = disable_thinking if disable_thinking is not None else self.disable_thinking
        
        # Запрос к LM Studio (OpenAI API формат)
        # Некоторые модели в LM Studio могут не поддерживать response_format, поэтому пробуем без него сначала
        messages = []
        
        # Добавляем системное сообщение для отключения размышлений, если нужно
        if use_disable_thinking:
            messages.append({"role": "system", "content": "/nothink"})
        
        # Добавляем пользовательское сообщение с промптом
        messages.append({"role": "user", "content": full_prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # LM Studio API поддерживает только "json_schema" или "text" для response_format.type
        # Не добавляем response_format, так как модель должна возвращать JSON согласно промпту
        # Это избавляет от лишних ошибок и повторных запросов
        
        response = self._make_request("chat/completions", payload)
        
        # Извлекаем контент из ответа
        content = self._extract_content(response)
        
        # Парсим ответ
        final_results, new_tags = parse_llm_response(content, normalized_requests)
        
        return final_results, new_tags

