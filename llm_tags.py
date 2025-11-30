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
from tqdm import tqdm

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
    
    def tag_batch(
        self,
        requests: list[str],
        prompt: str,
        existing_tags: Optional[dict[str, str]] = None,
        max_tags: int = 5
    ) -> list[dict]:
        """
        Тегировать батч обращений.
        
        Args:
            requests: Список текстов обращений
            prompt: Промпт с инструкциями по тегированию
            existing_tags: Существующие теги {tag_name: description}
            max_tags: Максимальное количество тегов на обращение
            
        Returns:
            Список словарей с тегами для каждого обращения
            Формат: [{"tags": ["тег1", "тег2"], "descriptions": {"тег1": "описание1", ...}}, ...]
        """
        # Формируем промпт с существующими тегами
        existing_tags_text = ""
        if existing_tags:
            existing_tags_text = "\n\nСуществующие теги (используй их если подходят):\n"
            for tag_name, description in existing_tags.items():
                existing_tags_text += f"- {tag_name}: {description}\n"
        
        # Формируем список обращений
        requests_text = "\n".join([f"{i+1}. {req}" for i, req in enumerate(requests)])
        
        # Полный промпт
        full_prompt = f"""{prompt}

{existing_tags_text}

Обращения для тегирования:
{requests_text}

Верни JSON массив, где для каждого обращения указаны теги и их описания.
Формат: [{{"tags": ["тег1", "тег2"], "descriptions": {{"тег1": "описание1", "тег2": "описание2"}}}}, ...]
Максимум {max_tags} тегов на обращение.
Если тег определить невозможно, верни {{"tags": [], "descriptions": {{}}}}.
"""
        
        # Запрос к LLM
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": 4096,
            },
        }
        
        response = self._make_request("chat", payload)
        content = response.get("message", {}).get("content", "").strip()
        
        try:
            result = json.loads(content)
            
            # Если результат - словарь, пытаемся извлечь массив из него
            if isinstance(result, dict):
                # Ищем массив в различных возможных ключах
                for key in ["issues", "results", "items", "data", "tags", "responses", "requests"]:
                    if key in result and isinstance(result[key], list):
                        result = result[key]
                        break
                else:
                    # Если не нашли массив, оборачиваем словарь в список
                    result = [result]
            
            # Проверяем что результат - список
            if not isinstance(result, list):
                result = [result]
            
            # Дополняем до нужного количества если LLM вернул меньше
            while len(result) < len(requests):
                result.append({"tags": [], "descriptions": {}})
            
            return result[:len(requests)]
            
        except json.JSONDecodeError as e:
            # Если не удалось распарсить JSON, возвращаем пустые теги
            print(f"[ERROR] Failed to parse JSON: {e}")
            print(f"[ERROR] Content was: {content[:200]}...")
            return [{"tags": [], "descriptions": {}} for _ in requests]


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
    
    def tag_batch(
        self,
        requests: list[str],
        prompt: str,
        existing_tags: Optional[dict[str, str]] = None,
        max_tags: int = 5
    ) -> list[dict]:
        """
        Тегировать батч обращений.
        
        Args:
            requests: Список текстов обращений
            prompt: Промпт с инструкциями по тегированию
            existing_tags: Существующие теги {tag_name: description}
            max_tags: Максимальное количество тегов на обращение
            
        Returns:
            Список словарей с тегами для каждого обращения
            Формат: [{"tags": ["тег1", "тег2"], "descriptions": {"тег1": "описание1", ...}}, ...]
        """
        # Формируем промпт с существующими тегами
        existing_tags_text = ""
        if existing_tags:
            existing_tags_text = "\n\nСуществующие теги (используй их если подходят):\n"
            for tag_name, description in existing_tags.items():
                existing_tags_text += f"- {tag_name}: {description}\n"
        
        # Формируем список обращений
        requests_text = "\n".join([f"{i+1}. {req}" for i, req in enumerate(requests)])
        
        # Полный промпт
        full_prompt = f"""{prompt}

{existing_tags_text}

Обращения для тегирования:
{requests_text}

Верни JSON массив, где для каждого обращения указаны теги и их описания.
Формат: [{{"tags": ["тег1", "тег2"], "descriptions": {{"тег1": "описание1", "тег2": "описание2"}}}}, ...]
Максимум {max_tags} тегов на обращение.
Если тег определить невозможно, верни {{"tags": [], "descriptions": {{}}}}.
"""
        
        # Запрос к LLM (OpenAI API формат)
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
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        try:
            result = json.loads(content)
            
            # Если результат - словарь, пытаемся извлечь массив из него
            if isinstance(result, dict):
                # Ищем массив в различных возможных ключах
                for key in ["issues", "results", "items", "data", "tags", "responses", "requests"]:
                    if key in result and isinstance(result[key], list):
                        result = result[key]
                        break
                else:
                    # Если не нашли массив, оборачиваем словарь в список
                    result = [result]
            
            # Проверяем что результат - список
            if not isinstance(result, list):
                result = [result]
            
            # Дополняем до нужного количества если LLM вернул меньше
            while len(result) < len(requests):
                result.append({"tags": [], "descriptions": {}})
            
            return result[:len(requests)]
            
        except json.JSONDecodeError as e:
            # Если не удалось распарсить JSON, возвращаем пустые теги
            print(f"[ERROR] Failed to parse JSON: {e}")
            print(f"[ERROR] Content was: {content[:200]}...")
            return [{"tags": [], "descriptions": {}} for _ in requests]


class TaggingPipeline:
    """
    Pipeline для тегирования обращений клиентов.
    
    Поддерживает:
    - Батчинг (30-100 обращений)
    - Многопоточность (5 параллельных запросов)
    - Фильтрацию по количеству существующих тегов
    """
    
    def __init__(
        self,
        llm = None,
        batch_size: int = 50,
        num_workers: int = 5
    ):
        """
        Инициализация pipeline.
        
        Args:
            llm: Экземпляр OllamaLLM или OpenAICompatibleLLM (если None, создается OllamaLLM по умолчанию)
            batch_size: Размер батча обращений (по умолчанию 50)
            num_workers: Количество параллельных потоков (по умолчанию 5)
        """
        self.llm = llm or OllamaLLM()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
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
        tag_prompt: str,
        existing_tags: Optional[dict[str, str]],
        max_tags: int
    ) -> tuple[list[str], dict[str, str]]:
        """
        Обработать один батч обращений.
        
        Args:
            batch_df: DataFrame с батчем обращений
            text_column: Название колонки с текстом
            tag_prompt: Промпт для тегирования
            existing_tags: Существующие теги
            max_tags: Максимальное количество тегов
            
        Returns:
            Кортеж (список тегов для каждой строки, обновленный словарь тегов)
        """
        requests = batch_df[text_column].tolist()
        
        # Получаем теги от LLM
        results = self.llm.tag_batch(
            requests=requests,
            prompt=tag_prompt,
            existing_tags=existing_tags,
            max_tags=max_tags
        )
        
        # Формируем результаты
        tags_list = []
        new_tags_dict = existing_tags.copy() if existing_tags else {}
        
        for result in results:
            tags = result.get("tags", [])
            descriptions = result.get("descriptions", {})
            
            # Обновляем словарь тегов
            for tag in tags:
                if tag and tag not in new_tags_dict:
                    new_tags_dict[tag] = descriptions.get(tag, "")
            
            # Формируем строку с тегами
            tags_str = ", ".join(tags) if tags else ""
            tags_list.append(tags_str)
        
        return tags_list, new_tags_dict
    
    def tag(
        self,
        tags: pd.DataFrame,
        text_column: str,
        tag_prompt: str,
        id_column: Optional[str] = None,
        existing_tags: Optional[dict[str, str]] = None,
        skip_if_tags_count: int = 10,
        max_tags: int = 5
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
        for i in range(0, len(to_process_df), self.batch_size):
            batch = to_process_df.iloc[i:i+self.batch_size]
            batches.append(batch)
        
        print(f"Батчей для обработки: {len(batches)}")
        
        # Обрабатываем батчи параллельно
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    self._process_batch,
                    batch,
                    text_column,
                    tag_prompt,
                    tags_dict,
                    max_tags
                ): i
                for i, batch in enumerate(batches)
            }
            
            batch_results = {}
            with tqdm(total=len(batches), desc="Обработка батчей", unit="batch") as pbar:
                for future in as_completed(futures):
                    batch_idx = futures[future]
                    try:
                        tags_list, new_tags_dict = future.result()
                        batch_results[batch_idx] = (tags_list, new_tags_dict)
                        tags_dict.update(new_tags_dict)
                        pbar.update(1)
                        pbar.set_postfix({"последний батч": batch_idx + 1})
                    except Exception as e:
                        print(f"\nОшибка в батче {batch_idx + 1}: {e}")
                        batch_results[batch_idx] = ([""] * len(batches[batch_idx]), {})
                        pbar.update(1)
        
        # Применяем результаты к DataFrame
        for batch_idx in sorted(batch_results.keys()):
            tags_list, _ = batch_results[batch_idx]
            batch = batches[batch_idx]
            
            for (idx, row), tags_str in zip(batch.iterrows(), tags_list):
                result_df.at[idx, "tags"] = tags_str
        
        print(f"Обработка завершена. Всего тегов в словаре: {len(tags_dict)}")
        
        return result_df, tags_dict


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

