#!/usr/bin/env python3
"""
LLM Tags - Упрощенная система тегирования обращений клиентов с использованием LLM.

Основные возможности:
- Тегирование обращений через Ollama (qwen32b по умолчанию)
- Батчинг (30-100 обращений)
- Многопоточность (5 параллельных запросов)
- Фильтрация по количеству существующих тегов
"""

import json
import urllib3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from difflib import SequenceMatcher

from prompt_builder import build_tagging_prompt
from batch_processor import normalize_requests, parse_llm_response

# Отключаем предупреждения urllib3
urllib3.disable_warnings()


class OllamaLLM:
    """
    Класс для работы с LLM через Ollama.
    
    По умолчанию использует qwen32b модель для тегирования обращений.
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:11434/api",
        model: str = "qwen2.5:32b",
        temperature: float = 0.7
    ):
        """
        Инициализация Ollama LLM.
        
        Args:
            api_url: URL API Ollama (по умолчанию localhost:11434)
            model: Название модели (по умолчанию qwen2.5:32b)
            temperature: Температура генерации (по умолчанию 0.7)
        """
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        
    def _make_request(self, endpoint: str, payload: dict) -> dict:
        """
        Выполнить HTTP запрос к Ollama API.
        
        Args:
            endpoint: Конечная точка API (например, "chat")
            payload: Данные запроса
            
        Returns:
            Ответ от API в виде словаря
        """
        url = f"{self.api_url}/{endpoint}"
        
        # Парсим URL для urllib3
        if url.startswith("http://"):
            url = url[7:]
        elif url.startswith("https://"):
            url = url[8:]
        
        host_port = url.split("/", 1)
        host = host_port[0]
        path = "/" + host_port[1] if len(host_port) > 1 else "/"
        
        if ":" in host:
            host, port = host.split(":")
            port = int(port)
        else:
            port = 11434
        
        try:
            http = urllib3.PoolManager(
                num_pools=1,
                maxsize=1,
                timeout=urllib3.Timeout(connect=10, read=300)
            )
            
            response = http.request(
                "POST",
                f"http://{host}:{port}{path}",
                body=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Connection": "close"
                }
            )
            
            if response.status != 200:
                raise ConnectionError(
                    f"Ollama API error: {response.status}. "
                    f"Убедитесь что Ollama запущена: 'ollama serve'"
                )
            
            return json.loads(response.data.decode("utf-8"))
            
        except Exception as e:
            raise ConnectionError(
                f"Не удалось подключиться к Ollama на {self.api_url}. "
                f"Убедитесь что Ollama запущена: 'ollama serve'. Ошибка: {e}"
            )
    
    def _extract_content(self, response: dict) -> str:
        """
        Извлечь контент из ответа Ollama API.
        
        Args:
            response: Ответ от API Ollama
            
        Returns:
            Текст контента
        """
        return response.get("message", {}).get("content", "").strip()
    
    def tag_batch(
        self,
        requests: list[tuple[str, str]] | list[dict],
        prompt: str,
        existing_tags: Optional[dict[str, str]] = None,
        max_tags: int = 5,
        allow_new_tags: bool = True
    ) -> tuple[list[dict], dict[str, str]]:
        """
        Тегировать батч обращений.
        
        Args:
            requests: Список обращений в формате [(id, text), ...] или [{"id": id, "text": text}, ...]
            prompt: Промпт с инструкциями по тегированию
            existing_tags: Существующие теги {tag_name: description}
            max_tags: Максимальное количество тегов на обращение
            allow_new_tags: Разрешить создание новых тегов (по умолчанию True)
            
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
        
        # Запрос к LLM (специфичный для Ollama)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": 8192,
            },
        }
        
        response = self._make_request("chat", payload)
        content = self._extract_content(response)
        
        # Парсим ответ
        return parse_llm_response(content, normalized_requests)


class OpenAICompatibleLLM:
    """
    Класс для работы с OpenAI-совместимыми API (OpenAI, vLLM, LM Studio, etc).
    
    Поддерживает стандартный OpenAI API формат запросов.
    """
    
    def __init__(
        self,
        api_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ):
        """
        Инициализация OpenAI-совместимого LLM.
        
        Args:
            api_url: URL API (например, "https://api.openai.com/v1" или "http://localhost:8000/v1")
            model: Название модели (например, "gpt-4", "gpt-3.5-turbo")
            api_key: API ключ (если требуется, по умолчанию None)
            temperature: Температура генерации (по умолчанию 0.7)
            max_tokens: Максимальное количество токенов в ответе (по умолчанию 4096)
            top_p: Nucleus sampling параметр (по умолчанию 1.0)
            frequency_penalty: Штраф за частоту повторений (по умолчанию 0.0)
            presence_penalty: Штраф за присутствие токенов (по умолчанию 0.0)
        """
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        
    def _make_request(self, endpoint: str, payload: dict) -> dict:
        """
        Выполнить HTTP запрос к OpenAI-совместимому API.
        
        Args:
            endpoint: Конечная точка API (например, "chat/completions")
            payload: Данные запроса
            
        Returns:
            Ответ от API в виде словаря
        """
        url = f"{self.api_url}/{endpoint}"
        
        # Парсим URL для urllib3
        if url.startswith("https://"):
            protocol = "https"
            url = url[8:]
        elif url.startswith("http://"):
            protocol = "http"
            url = url[7:]
        else:
            protocol = "https"
        
        host_port = url.split("/", 1)
        host = host_port[0]
        path = "/" + host_port[1] if len(host_port) > 1 else "/"
        
        if ":" in host:
            host, port = host.split(":")
            port = int(port)
        else:
            port = 443 if protocol == "https" else 80
        
        try:
            if protocol == "https":
                http = urllib3.PoolManager(
                    num_pools=1,
                    maxsize=1,
                    timeout=urllib3.Timeout(connect=10, read=300),
                    cert_reqs='CERT_NONE'
                )
            else:
                http = urllib3.PoolManager(
                    num_pools=1,
                    maxsize=1,
                    timeout=urllib3.Timeout(connect=10, read=300)
                )
            
            headers = {
                "Content-Type": "application/json",
                "Connection": "close"
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = http.request(
                "POST",
                f"{protocol}://{host}:{port}{path}",
                body=json.dumps(payload).encode("utf-8"),
                headers=headers
            )
            
            if response.status != 200:
                error_text = response.data.decode("utf-8")
                raise ConnectionError(
                    f"API error: {response.status}. Response: {error_text}"
                )
            
            return json.loads(response.data.decode("utf-8"))
            
        except Exception as e:
            raise ConnectionError(
                f"Не удалось подключиться к API на {self.api_url}. Ошибка: {e}"
            )
    
    def _extract_content(self, response: dict) -> str:
        """
        Извлечь контент из ответа OpenAI-совместимого API.
        
        Args:
            response: Ответ от API
            
        Returns:
            Текст контента
        """
        return response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    
    def tag_batch(
        self,
        requests: list[tuple[str, str]] | list[dict],
        prompt: str,
        existing_tags: Optional[dict[str, str]] = None,
        max_tags: int = 5,
        allow_new_tags: bool = True
    ) -> tuple[list[dict], dict[str, str]]:
        """
        Тегировать батч обращений.
        
        Args:
            requests: Список обращений в формате [(id, text), ...] или [{"id": id, "text": text}, ...]
            prompt: Промпт с инструкциями по тегированию
            existing_tags: Существующие теги {tag_name: description}
            max_tags: Максимальное количество тегов на обращение
            allow_new_tags: Разрешить создание новых тегов (по умолчанию True)
            
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
        
        # Запрос к LLM (специфичный для OpenAI API)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "response_format": {"type": "json_object"}
        }
        
        response = self._make_request("chat/completions", payload)
        content = self._extract_content(response)
        
        # Парсим ответ
        return parse_llm_response(content, normalized_requests)


class TaggingPipeline:
    """
    Pipeline для тегирования обращений клиентов.
    
    Поддерживает:
    - Батчинг (30-100 обращений)
    - Многопоточность (5 параллельных запросов)
    - Фильтрацию по количеству существующих тегов
    - Автоматическое накопление тегов между батчами
    """
    
    def __init__(
        self,
        llm = None,
        batch_size: int = 50,
    ):
        """
        Инициализация pipeline.
        
        Args:
            llm: Экземпляр OllamaLLM или OpenAICompatibleLLM (если None, создается OllamaLLM по умолчанию)
            batch_size: Размер батча обращений (по умолчанию 50)
            num_workers: Количество параллельных потоков (по умолчанию 5)
                - num_workers=1: последовательная обработка, каждый батч видит теги предыдущих (медленнее, но консистентнее)
                - num_workers>1: параллельная обработка (быстрее, но возможны похожие теги из разных батчей)
        """
        self.llm = llm or OllamaLLM()
        self.batch_size = batch_size
        
    def _count_tags(self, tags_str: str) -> int:
        """
        Подсчитать количество тегов в строке.
        
        Args:
            tags_str: Строка с тегами через запятую
            
        Returns:
            Количество тегов
        """
        if pd.isna(tags_str) or not tags_str or tags_str.strip() == "":
            return 0
        return len([t.strip() for t in str(tags_str).split(",") if t.strip()])
    
    def _process_batch(
        self,
        batch_df: pd.DataFrame,
        text_column: str,
        id_column: Optional[str],
        tag_prompt: str,
        existing_tags: Optional[dict[str, str]],
        max_tags: int,
        allow_new_tags: bool = True
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Обработать один батч обращений.
        
        Args:
            batch_df: DataFrame с батчем обращений
            text_column: Название колонки с текстом
            id_column: Название колонки с ID (если None, используется индекс)
            tag_prompt: Промпт для тегирования
            existing_tags: Существующие теги
            max_tags: Максимальное количество тегов
            allow_new_tags: Разрешить добавление новых тегов (по умолчанию True)
            
        Returns:
            Кортеж (словарь {id: tags_string}, обновленный словарь тегов с новыми тегами)
        """
        # Формируем список кортежей (id, text) для передачи в tag_batch
        requests_with_ids = []
        id_to_idx = {}  # Маппинг id -> индекс в batch_df
        
        for idx, row in batch_df.iterrows():
            if id_column and id_column in batch_df.columns:
                req_id = str(row[id_column])
            else:
                req_id = str(idx)
            
            req_text = str(row[text_column])
            requests_with_ids.append((req_id, req_text))
            id_to_idx[req_id] = idx
        
        # Получаем теги от LLM (новый формат: кортеж из results и new_tags)
        results, new_tags = self.llm.tag_batch(
            requests=requests_with_ids,
            prompt=tag_prompt,
            existing_tags=existing_tags,
            max_tags=max_tags,
            allow_new_tags=allow_new_tags
        )
        
        # Формируем результаты: словарь {id: tags_string}
        tags_dict = {}
        updated_tags_dict = existing_tags.copy() if existing_tags else {}
        
        # Добавляем новые теги в словарь только если разрешено
        if allow_new_tags:
            updated_tags_dict.update(new_tags)
        
        # Обрабатываем результаты от LLM
        for result in results:
            if not isinstance(result, dict):
                continue
            
            req_id = result.get("id")
            tags = result.get("tags", [])
            
            if req_id:
                # Формируем строку с тегами
                tags_str = ", ".join(tags) if tags else ""
                tags_dict[req_id] = tags_str
        
        return tags_dict, updated_tags_dict
    
    def tag(
        self,
        tags: pd.DataFrame,
        text_column: str,
        tag_prompt: str,
        id_column: Optional[str] = None,
        existing_tags: Optional[dict[str, str]] = None,
        skip_if_tags_count: int = 10,
        max_tags: int = 5,
        num_workers: int = 1,
        batch_size: int = 50,
        allow_new_tags: bool = True
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        Тегировать обращения в DataFrame.
        
        Args:
            tags: DataFrame с обращениями и существующими тегами
            text_column: Название колонки с текстом обращений
            tag_prompt: Промпт с инструкциями по тегированию
            id_column: Название колонки с ID (если None, используется индекс DataFrame)
            existing_tags: Существующие теги {tag_name: description}
            skip_if_tags_count: Пропустить строки с количеством тегов >= этого значения
            max_tags: Максимальное количество тегов на обращение
            num_workers: Количество параллельных потоков (по умолчанию None, используется self.num_workers)
            batch_size: Размер батча для обработки (по умолчанию 50)
            allow_new_tags: Разрешить добавление новых тегов в словарь (по умолчанию True)
            
        Returns:
            Кортеж (DataFrame с колонкой tags, словарь тегов {tag_name: description})
        """
        # Копируем DataFrame
        result_df = tags.copy()
        
        # Инициализируем колонку tags если нет
        if "tags" not in result_df.columns:
            result_df["tags"] = ""
        
        # Фильтруем строки для обработки
        mask = result_df.apply(
            lambda row: self._count_tags(row["tags"]) < skip_if_tags_count,
            axis=1
        )
        to_process_df = result_df[mask].copy()
        
        print(f"Всего обращений: {len(result_df)}")
        print(f"Для обработки: {len(to_process_df)}")
        print(f"Пропущено (уже есть {skip_if_tags_count}+ тегов): {len(result_df) - len(to_process_df)}")
        
        if len(to_process_df) == 0:
            print("Нет обращений для обработки")
            return result_df, existing_tags or {}
        
        # Инициализируем словарь тегов
        tags_dict = existing_tags.copy() if existing_tags else {}
        
        # Разбиваем на батчи
        batches = []
        for i in range(0, len(to_process_df), batch_size):
            batch = to_process_df.iloc[i:i+batch_size]
            batches.append(batch)
        
        print(f"Батчей для обработки: {len(batches)}")
        
        # Обрабатываем батчи с учетом num_workers
        # num_workers=1: последовательная обработка, каждый батч видит теги предыдущих
        # num_workers>1: параллельная обработка (до num_workers одновременно), каждый новый батч видит обновленные теги от завершенных батчей
        batch_results = {}


        if num_workers == 1:
            # Последовательная обработка - каждый батч видит накопленные теги
            print(f"[LOG] Последовательная обработка: {len(batches)} батчей, начальное количество тегов: {len(tags_dict)}")
            with tqdm(total=len(batches), desc="Обработка батчей (последовательно)", unit="batch") as pbar:
                for batch_idx, batch in enumerate(batches):
                    try:
                        tags_before = len(tags_dict)
                        tags_before_keys = set(tags_dict.keys())
                        print(f"[LOG] Запуск батча {batch_idx + 1}/{len(batches)} с {tags_before} тегами в словаре")
                        
                        # Каждый батч получает обновленный словарь тегов
                        tags_dict_batch, updated_tags_dict = self._process_batch(
                            batch,
                            text_column,
                            id_column,
                            tag_prompt,
                            tags_dict,  # Передаем текущий словарь тегов
                            max_tags,
                            allow_new_tags
                        )
                        
                        # Определяем новые теги до обновления tags_dict
                        new_tags_keys = set(updated_tags_dict.keys()) - tags_before_keys
                        new_tags_count = len(new_tags_keys)
                        
                        # Обновляем словарь тегов для следующего батча
                        tags_dict = updated_tags_dict
                        tags_after = len(tags_dict)
                        
                        if new_tags_count > 0:
                            new_tags_list = list(new_tags_keys)[:5]
                            new_tags_str = ", ".join(new_tags_list)
                            if new_tags_count > 5:
                                new_tags_str += f" ... (+{new_tags_count - 5} еще)"
                            print(f"[LOG] Батч {batch_idx + 1} завершен: добавлено {new_tags_count} новых тегов ({tags_before} -> {tags_after}): {new_tags_str}")
                        else:
                            print(f"[LOG] Батч {batch_idx + 1} завершен: новых тегов нет ({tags_before} -> {tags_after})")
                        
                        batch_results[batch_idx] = (tags_dict_batch, updated_tags_dict)
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            "батч": batch_idx + 1,
                            "тегов": len(tags_dict)
                        })
                    except Exception as e:
                        print(f"\n[ERROR] Ошибка в батче {batch_idx + 1}: {e}")
                        batch_results[batch_idx] = ({}, tags_dict.copy())
                        pbar.update(1)
            print(f"[LOG] Последовательная обработка завершена: итоговое количество тегов: {len(tags_dict)}")
        else:
            # Параллельная обработка с последовательным обновлением tags_dict
            # Запускаем батчи последовательно, но одновременно может работать до num_workers батчей
            # Каждый новый батч видит обновленный tags_dict от завершенных батчей
            print(f"[LOG] Параллельная обработка: {len(batches)} батчей, max_workers={num_workers}, начальное количество тегов: {len(tags_dict)}")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                completed_count = 0
                
                def submit_next_batch(batch_idx):
                    """Запустить следующий батч с текущим tags_dict"""
                    # Создаем копию текущего словаря тегов для этого батча
                    current_tags = tags_dict.copy()
                    tags_count = len(current_tags)
                    print(f"[LOG] Запуск батча {batch_idx + 1}/{len(batches)} с {tags_count} тегами в словаре")
                    return executor.submit(
                        self._process_batch,
                        batches[batch_idx],
                        text_column,
                        id_column,
                        tag_prompt,
                        current_tags,  # Каждый батч получает актуальный словарь тегов
                        max_tags,
                        allow_new_tags
                    )
                
                # Запускаем первые num_workers батчей
                print(f"[LOG] Параллельная обработка: запускаем первые {min(num_workers, len(batches))} батчей")
                for i in range(min(num_workers, len(batches))):
                    future = submit_next_batch(i)
                    futures[future] = i
                
                next_batch_idx = num_workers
                
                with tqdm(total=len(batches), desc="Обработка батчей (параллельно с обновлением)", unit="batch") as pbar:
                    while futures:
                        # Ждем завершения любого батча
                        done, not_done = [], []
                        for future in futures:
                            if future.done():
                                done.append(future)
                            else:
                                not_done.append(future)
                        
                        # Обрабатываем завершенные батчи
                        for future in done:
                            batch_idx = futures[future]
                            try:
                                tags_dict_batch, updated_tags_dict = future.result()
                                
                                # Логируем до обновления
                                tags_before = len(tags_dict)
                                new_tags_count = len(updated_tags_dict)
                                
                                # Обновляем общий словарь тегов
                                tags_dict.update(updated_tags_dict)
                                
                                # Логируем после обновления
                                tags_after = len(tags_dict)
                                if new_tags_count > 0:
                                    new_tags_list = list(updated_tags_dict.keys())[:5]  # Показываем первые 5
                                    new_tags_str = ", ".join(new_tags_list)
                                    if new_tags_count > 5:
                                        new_tags_str += f" ... (+{new_tags_count - 5} еще)"
                                    print(f"[LOG] Батч {batch_idx + 1} завершен: добавлено {new_tags_count} новых тегов ({tags_before} -> {tags_after}): {new_tags_str}")
                                else:
                                    print(f"[LOG] Батч {batch_idx + 1} завершен: новых тегов нет ({tags_before} -> {tags_after})")
                                
                                batch_results[batch_idx] = (tags_dict_batch, updated_tags_dict)
                                
                                completed_count += 1
                                pbar.update(1)
                                pbar.set_postfix({
                                    "батч": batch_idx + 1,
                                    "тегов": len(tags_dict),
                                    "активных": len(not_done)
                                })
                            except Exception as e:
                                print(f"\n[ERROR] Ошибка в батче {batch_idx + 1}: {e}")
                                batch_results[batch_idx] = ({}, tags_dict.copy())
                                completed_count += 1
                                pbar.update(1)
                            
                            # Удаляем завершенный батч из словаря
                            del futures[future]
                            
                            # Запускаем следующий батч, если есть
                            if next_batch_idx < len(batches):
                                new_future = submit_next_batch(next_batch_idx)
                                futures[new_future] = next_batch_idx
                                next_batch_idx += 1
                        
                        # Если нет завершенных батчей, небольшая задержка
                        if not done:
                            import time
                            time.sleep(0.1)
            
            print(f"[LOG] Параллельная обработка завершена: итоговое количество тегов: {len(tags_dict)}")
        
        # Применяем результаты к DataFrame
        for batch_idx in sorted(batch_results.keys()):
            tags_dict_batch, _ = batch_results[batch_idx]
            batch = batches[batch_idx]
            
            for idx, row in batch.iterrows():
                # Определяем ID для этой строки
                if id_column and id_column in batch.columns:
                    req_id = str(row[id_column])
                else:
                    req_id = str(idx)
                
                # Получаем теги из словаря по ID
                tags_str = tags_dict_batch.get(req_id, "")
                result_df.at[idx, "tags"] = tags_str
        
        print(f"Обработка завершена. Всего тегов в словаре: {len(tags_dict)}")
        
        return result_df, tags_dict
    
    def expand_tags_to_columns(
        self,
        df: pd.DataFrame,
        tags_column: str = "tags"
    ) -> pd.DataFrame:
        """
        Раскидывает теги по отдельным столбцам.
        
        Для каждого уникального тега создается столбец tag_[name_of_tag] с булевыми значениями.
        
        Args:
            df: DataFrame с колонкой tags
            tags_column: Название колонки с тегами (по умолчанию "tags")
            
        Returns:
            DataFrame с дополнительными столбцами tag_[name_of_tag]
        """
        result_df = df.copy()
        
        if tags_column not in result_df.columns:
            raise ValueError(f"Колонка '{tags_column}' не найдена в DataFrame")
        
        # Парсим теги один раз для всех строк и собираем уникальные теги
        def parse_tags(tags_str):
            """Парсит строку тегов и возвращает множество тегов"""
            if pd.notna(tags_str) and tags_str and str(tags_str).strip():
                return set(t.strip() for t in str(tags_str).split(",") if t.strip())
            return set()
        
        # Парсим теги один раз для всех строк
        tags_sets = result_df[tags_column].apply(parse_tags)
        
        # Собираем все уникальные теги
        all_tags = set()
        for tag_set in tags_sets:
            all_tags.update(tag_set)
        
        # Нормализуем имена тегов для столбцов заранее
        tag_to_column = {}
        for tag in all_tags:
            tag_normalized = tag.replace(" ", "_").replace("/", "_").replace("\\", "_")
            tag_normalized = "".join(c if c.isalnum() or c == "_" else "_" for c in tag_normalized)
            tag_to_column[tag] = f"tag_{tag_normalized}"
        
        # Создаем все столбцы сразу, используя векторизованные операции
        new_columns = {}
        for tag, column_name in tag_to_column.items():
            new_columns[column_name] = tags_sets.apply(lambda tag_set: tag in tag_set)
        
        # Добавляем все столбцы одним присваиванием
        result_df = result_df.assign(**new_columns)
            
        
        print(f"Создано {len(all_tags)} столбцов для тегов")
        
        return result_df
    
    def analyze_tags(
        self,
        df: pd.DataFrame,
        tags_column: str = "tags"
    ) -> dict:
        """
        Собирает статистику по тегам и выполняет корреляционный анализ.
        
        Args:
            df: DataFrame с колонкой tags
            tags_column: Название колонки с тегами (по умолчанию "tags")
            
        Returns:
            Словарь с ключами:
            - 'statistics': DataFrame со статистикой по каждому тегу (количество, процент)
            - 'correlation': DataFrame с корреляционной матрицей тегов
            - 'top_pairs': DataFrame с топ-10 парами тегов, которые чаще всего встречаются вместе
        """
        if tags_column not in df.columns:
            raise ValueError(f"Колонка '{tags_column}' не найдена в DataFrame")
        
        # Собираем все уникальные теги и создаем маппинг нормализованное имя -> оригинальный тег
        all_tags = set()
        tag_to_normalized = {}
        normalized_to_tag = {}
        
        for tags_str in df[tags_column]:
            if pd.notna(tags_str) and tags_str and str(tags_str).strip():
                tags_list = [t.strip() for t in str(tags_str).split(",") if t.strip()]
                all_tags.update(tags_list)
        
        # Создаем маппинг
        for tag in all_tags:
            tag_normalized = tag.replace(" ", "_").replace("/", "_").replace("\\", "_")
            tag_normalized = "".join(c if c.isalnum() or c == "_" else "_" for c in tag_normalized)
            tag_to_normalized[tag] = tag_normalized
            normalized_to_tag[tag_normalized] = tag
        
        if not all_tags:
            return {
                'statistics': pd.DataFrame(),
                'correlation': pd.DataFrame(),
                'top_pairs': pd.DataFrame()
            }
        
        # Создаем бинарную матрицу тегов
        tag_matrix = {}
        for tag in all_tags:
            tag_normalized = tag_to_normalized[tag]
            tag_matrix[tag_normalized] = df[tags_column].apply(
                lambda x: tag in [t.strip() for t in str(x).split(",") if t.strip()] 
                if pd.notna(x) and x and str(x).strip() else False
            ).astype(int)
        
        tag_data_df = pd.DataFrame(tag_matrix)
        
        # 1. Статистика по тегам
        tag_stats = []
        total_rows = len(df)
        
        for tag_normalized, tag_original in normalized_to_tag.items():
            count = tag_data_df[tag_normalized].sum()
            percentage = (count / total_rows * 100) if total_rows > 0 else 0
            
            tag_stats.append({
                'tag': tag_original,
                'count': count,
                'percentage': round(percentage, 2)
            })
        
        statistics_df = pd.DataFrame(tag_stats).sort_values('count', ascending=False)
        
        # 2. Корреляционный анализ
        correlation_matrix = tag_data_df.corr()
        # Переименовываем индексы и столбцы на оригинальные имена тегов
        correlation_matrix.index = [normalized_to_tag.get(idx, idx) for idx in correlation_matrix.index]
        correlation_matrix.columns = [normalized_to_tag.get(col, col) for col in correlation_matrix.columns]
        
        # 3. Топ пар тегов, которые встречаются вместе
        pairs_data = []
        tag_columns = list(tag_data_df.columns)
        
        for i, col1 in enumerate(tag_columns):
            for col2 in tag_columns[i+1:]:
                tag1 = normalized_to_tag.get(col1, col1)
                tag2 = normalized_to_tag.get(col2, col2)
                
                # Количество строк, где оба тега присутствуют
                both_count = ((tag_data_df[col1] == 1) & (tag_data_df[col2] == 1)).sum()
                
                # Количество строк с каждым тегом отдельно
                tag1_count = tag_data_df[col1].sum()
                tag2_count = tag_data_df[col2].sum()
                
                # Процент совместного появления от общего количества строк
                both_percentage = (both_count / total_rows * 100) if total_rows > 0 else 0
                
                # Процент совместного появления от количества тега1
                tag1_joint_percentage = (both_count / tag1_count * 100) if tag1_count > 0 else 0
                
                # Процент совместного появления от количества тега2
                tag2_joint_percentage = (both_count / tag2_count * 100) if tag2_count > 0 else 0
                
                if both_count > 0:
                    pairs_data.append({
                        'tag1': tag1,
                        'tag2': tag2,
                        'both_count': both_count,
                        'both_percentage': round(both_percentage, 2),
                        'tag1_count': tag1_count,
                        'tag2_count': tag2_count,
                        'tag1_joint_percentage': round(tag1_joint_percentage, 2),
                        'tag2_joint_percentage': round(tag2_joint_percentage, 2),
                        'correlation': correlation_matrix.loc[tag1, tag2]
                    })
        
        top_pairs_df = pd.DataFrame(pairs_data).sort_values('both_count', ascending=False).head(10)
        
        return {
            'statistics': statistics_df,
            'correlation': correlation_matrix,
            'top_pairs': top_pairs_df
        }
    
    def merge_similar_tags(
        self,
        df: pd.DataFrame,
        tags_dict: dict[str, str],
        tags_column: str = "tags",
        min_frequency: int = 5,
        similarity_threshold: float = 0.7,
        use_llm_for_similarity: bool = True,
        merge_rare_to_common: bool = True
    ) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
        """
        Схлопывает редкие и похожие теги в единые, расширяя описание и имя тега.
        
        Логика работы:
        1. Находит редкие теги (частота < min_frequency)
        2. Находит похожие теги (семантически или по названию)
        3. Объединяет их в единые теги с расширенным описанием
        4. Обновляет DataFrame и словарь тегов
        
        Args:
            df: DataFrame с колонкой tags
            tags_dict: Словарь тегов {tag_name: description}
            tags_column: Название колонки с тегами (по умолчанию "tags")
            min_frequency: Минимальная частота тега, ниже которой он считается редким
            similarity_threshold: Порог схожести для объединения тегов (0.0-1.0)
            use_llm_for_similarity: Использовать LLM для определения похожих тегов (True)
                                    или только строковое сравнение (False)
            merge_rare_to_common: Объединять редкие теги с похожими частыми (True)
                                  или только объединять похожие теги независимо от частоты (False)
        
        Returns:
            Кортеж (обновленный DataFrame, обновленный словарь тегов, маппинг старых->новых тегов)
            Формат маппинга: {old_tag: new_tag}
        """
        result_df = df.copy()
        
        # 1. Получаем статистику по тегам
        analysis = self.analyze_tags(result_df, tags_column=tags_column)
        tag_stats = analysis['statistics']
        
        if len(tag_stats) == 0:
            return result_df, tags_dict, {}
        
        # 2. Определяем редкие теги
        rare_tags = set(tag_stats[tag_stats['count'] < min_frequency]['tag'].tolist())
        common_tags = set(tag_stats[tag_stats['count'] >= min_frequency]['tag'].tolist())
        
        print(f"Редких тегов (< {min_frequency}): {len(rare_tags)}")
        print(f"Частых тегов (>= {min_frequency}): {len(common_tags)}")
        
        # 3. Создаем маппинг для объединения тегов
        tag_mapping = {}  # {old_tag: new_tag}
        merged_tags_info = {}  # {new_tag: {'description': ..., 'merged_from': [tags]}}
        
        # 4. Функция для вычисления схожести названий
        def name_similarity(tag1: str, tag2: str) -> float:
            """Вычисляет схожесть названий тегов"""
            # Нормализуем названия (убираем пробелы, нижний регистр)
            norm1 = re.sub(r'[_\s]+', '_', tag1.lower().strip())
            norm2 = re.sub(r'[_\s]+', '_', tag2.lower().strip())
            
            # Проверяем точное совпадение после нормализации
            if norm1 == norm2:
                return 1.0
            
            # Проверяем вхождение одного в другое
            if norm1 in norm2 or norm2 in norm1:
                return 0.9
            
            # Используем SequenceMatcher для схожести
            return SequenceMatcher(None, norm1, norm2).ratio()
        
        # 5. Функция для определения похожих тегов через LLM
        def find_similar_tags_llm(tag: str, tag_description: str, candidate_tags: list[str]) -> Optional[str]:
            """Использует LLM для поиска похожего тега среди кандидатов"""
            if not candidate_tags:
                return None
            
            # Формируем промпт для LLM
            candidates_text = "\n".join([
                f"- {cand}: {tags_dict.get(cand, 'нет описания')}"
                for cand in candidate_tags
            ])
            
            prompt = f"""Проанализируй тег и найди наиболее похожий тег из списка кандидатов.

Текущий тег:
Название: {tag}
Описание: {tag_description}

Кандидаты для сравнения:
{candidates_text}

Верни ТОЛЬКО название наиболее похожего тега из списка кандидатов, или "НЕТ" если похожего нет.
Если теги семантически похожи (описывают одну и ту же проблему/категорию), верни название кандидата.
Если теги разные, верни "НЕТ".

Ответ (только название тега или "НЕТ"):"""
            
            try:
                # Определяем формат запроса в зависимости от типа LLM
                if hasattr(self.llm, 'api_key') or 'openai' in str(type(self.llm)).lower() or 'lmstudio' in str(type(self.llm)).lower():
                    # OpenAI-совместимый API
                    payload = {
                        "model": self.llm.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 50
                    }
                    response = self.llm._make_request("chat/completions", payload)
                else:
                    # Ollama API
                    payload = {
                        "model": self.llm.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 50,
                        },
                    }
                    response = self.llm._make_request("chat", payload)
                
                content = self.llm._extract_content(response).strip()
                
                # Парсим ответ
                content = content.strip().strip('"').strip("'")
                if content.upper() == "НЕТ" or content.upper() == "NO":
                    return None
                
                # Проверяем, что ответ - это один из кандидатов
                for cand in candidate_tags:
                    if cand.lower() in content.lower() or content.lower() in cand.lower():
                        return cand
                
                return None
            except Exception as e:
                print(f"Ошибка при LLM сравнении для тега {tag}: {e}")
                return None
        
        # 6. Обрабатываем редкие теги (объединяем с частыми похожими)
        if merge_rare_to_common and rare_tags:
            print(f"\nОбработка {len(rare_tags)} редких тегов...")
            
            for rare_tag in tqdm(rare_tags, desc="Объединение редких тегов"):
                if rare_tag in tag_mapping:
                    continue  # Уже обработан
                
                rare_desc = tags_dict.get(rare_tag, "")
                
                # Ищем похожий частый тег
                best_match = None
                best_similarity = 0.0
                
                # Сначала проверяем строковую схожесть
                for common_tag in common_tags:
                    if common_tag in tag_mapping:
                        continue  # Уже объединен
                    
                    similarity = name_similarity(rare_tag, common_tag)
                    if similarity > best_similarity and similarity >= similarity_threshold:
                        best_similarity = similarity
                        best_match = common_tag
                
                # Если не нашли по строке и используем LLM
                if not best_match and use_llm_for_similarity:
                    candidate_common_tags = [t for t in common_tags if t not in tag_mapping]
                    if candidate_common_tags:
                        best_match = find_similar_tags_llm(
                            rare_tag, 
                            rare_desc, 
                            candidate_common_tags[:20]  # Ограничиваем для производительности
                        )
                        if best_match:
                            best_similarity = 0.8  # Доверяем LLM
                
                if best_match:
                    # Объединяем редкий тег с частым
                    tag_mapping[rare_tag] = best_match
                    
                    # Расширяем описание частого тега
                    if best_match not in merged_tags_info:
                        merged_tags_info[best_match] = {
                            'description': tags_dict.get(best_match, ""),
                            'merged_from': [best_match]
                        }
                    
                    merged_tags_info[best_match]['merged_from'].append(rare_tag)
                    # Добавляем информацию из редкого тега в описание
                    if rare_desc:
                        if merged_tags_info[best_match]['description']:
                            merged_tags_info[best_match]['description'] += f"; также включает: {rare_desc}"
                        else:
                            merged_tags_info[best_match]['description'] = rare_desc
        
        # 7. Обрабатываем похожие теги среди всех (независимо от частоты)
        print(f"\nПоиск похожих тегов среди всех...")
        all_tags = set(tag_stats['tag'].tolist())
        processed_tags = set()
        
        for tag1 in tqdm(all_tags, desc="Объединение похожих тегов"):
            if tag1 in tag_mapping or tag1 in processed_tags:
                continue
            
            tag1_desc = tags_dict.get(tag1, "")
            similar_tags = []
            
            # Ищем похожие теги
            for tag2 in all_tags:
                if tag2 == tag1 or tag2 in tag_mapping or tag2 in processed_tags:
                    continue
                
                similarity = name_similarity(tag1, tag2)
                if similarity >= similarity_threshold:
                    similar_tags.append((tag2, similarity))
            
            # Если используем LLM, проверяем семантическую схожесть
            if use_llm_for_similarity and not similar_tags:
                candidates = [t for t in all_tags if t != tag1 and t not in tag_mapping and t not in processed_tags]
                if candidates:
                    similar_tag = find_similar_tags_llm(tag1, tag1_desc, candidates[:10])  # Ограничиваем кандидатов
                    if similar_tag:
                        similar_tags.append((similar_tag, 0.8))
            
            if similar_tags:
                # Выбираем тег с наибольшей частотой как основной
                similar_tags_with_freq = []
                for tag, sim in similar_tags:
                    tag_row = tag_stats[tag_stats['tag'] == tag]
                    freq = tag_row['count'].iloc[0] if len(tag_row) > 0 else 0
                    similar_tags_with_freq.append((tag, sim, freq))
                
                tag1_row = tag_stats[tag_stats['tag'] == tag1]
                tag1_freq = tag1_row['count'].iloc[0] if len(tag1_row) > 0 else 0
                similar_tags_with_freq.append((tag1, 1.0, tag1_freq))
                
                # Сортируем по частоте (убывание), затем по схожести
                similar_tags_with_freq.sort(key=lambda x: (x[2], x[1]), reverse=True)
                main_tag = similar_tags_with_freq[0][0]
                
                # Объединяем все похожие теги в основной
                if main_tag not in merged_tags_info:
                    merged_tags_info[main_tag] = {
                        'description': tags_dict.get(main_tag, ""),
                        'merged_from': [main_tag]
                    }
                
                for other_tag, _, _ in similar_tags_with_freq[1:]:
                    tag_mapping[other_tag] = main_tag
                    merged_tags_info[main_tag]['merged_from'].append(other_tag)
                    
                    other_desc = tags_dict.get(other_tag, "")
                    if other_desc:
                        if merged_tags_info[main_tag]['description']:
                            merged_tags_info[main_tag]['description'] += f"; также включает: {other_desc}"
                        else:
                            merged_tags_info[main_tag]['description'] = other_desc
                
                processed_tags.add(main_tag)
                processed_tags.update([tag for tag, _, _ in similar_tags_with_freq[1:]])
        
        # 8. Обновляем словарь тегов
        updated_tags_dict = {}
        for tag, desc in tags_dict.items():
            if tag not in tag_mapping:
                # Тег не объединен, оставляем как есть или обновляем если был расширен
                if tag in merged_tags_info:
                    updated_tags_dict[tag] = merged_tags_info[tag]['description']
                else:
                    updated_tags_dict[tag] = desc
        
        # Добавляем только основные теги из объединенных
        for main_tag, info in merged_tags_info.items():
            if main_tag not in tag_mapping:  # Это основной тег, не объединенный в другой
                updated_tags_dict[main_tag] = info['description']
        
        # 9. Обновляем DataFrame - заменяем старые теги на новые
        def replace_tags(tags_str):
            """Заменяет теги в строке согласно маппингу"""
            if pd.isna(tags_str) or not tags_str or str(tags_str).strip() == "":
                return ""
            
            tags_list = [t.strip() for t in str(tags_str).split(",") if t.strip()]
            updated_tags = []
            for tag in tags_list:
                new_tag = tag_mapping.get(tag, tag)
                if new_tag not in updated_tags:  # Убираем дубликаты
                    updated_tags.append(new_tag)
            
            return ", ".join(updated_tags) if updated_tags else ""
        
        result_df[tags_column] = result_df[tags_column].apply(replace_tags)
        
        print(f"\n✓ Объединение завершено:")
        print(f"  Объединено тегов: {len(tag_mapping)}")
        print(f"  Итоговое количество тегов: {len(updated_tags_dict)} (было {len(tags_dict)})")
        
        return result_df, updated_tags_dict, tag_mapping


# Пример использования (для тестирования)
if __name__ == "__main__":
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
    
    # Инициализируем LLM модель
    llm = OllamaLLM(
        api_url="http://localhost:11434/api",
        model="qwen2.5:32b",
        temperature=0.7
    )
    
    # Создаем pipeline с передачей LLM модели
    pipeline = TaggingPipeline(llm=llm)
    
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
    
    # Пример с OpenAI-совместимым API
    print("\n" + "="*50)
    print("Пример с OpenAI-совместимым API:")
    print("="*50)
    
    # Инициализируем OpenAI-совместимую LLM модель
    # Например, для OpenAI:
    # openai_llm = OpenAICompatibleLLM(
    #     api_url="https://api.openai.com/v1",
    #     model="gpt-4",
    #     api_key="your-api-key-here",
    #     temperature=0.7
    # )
    
    # Или для локального vLLM сервера:
    # vllm_llm = OpenAICompatibleLLM(
    #     api_url="http://localhost:8000/v1",
    #     model="meta-llama/Llama-2-7b-chat-hf",
    #     temperature=0.7
    # )
    
    # Или для LM Studio:
    # lmstudio_llm = OpenAICompatibleLLM(
    #     api_url="http://localhost:1234/v1",
    #     model="local-model",
    #     temperature=0.7
    # )
    
    # Создаем pipeline с OpenAI-совместимой моделью
    # openai_pipeline = TaggingPipeline(llm=openai_llm)
    
    # Тегирование
    # result_df, tags_dict = openai_pipeline.tag(
    #     tags=test_df,
    #     text_column="text",
    #     tag_prompt=prompt,
    #     max_tags=3
    # )
    
    print("Примеры закомментированы. Раскомментируйте нужный вариант.")

