import re
import json
import requests
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sklearn_normalize
from sentence_transformers import SentenceTransformer
import warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings('ignore')


_RE_HTML = re.compile(r'<[^>]+>')
_RE_SPECIAL = re.compile(r'[^a-zA-Zа-яА-ЯёЁ0-9\s.,!?;:\-\(\)\[\]{}\'\"]')
_RE_SINGLE = re.compile(r'\b[a-zA-Zа-яА-ЯёЁ]\b')
_RE_MULTI_SPACE = re.compile(r'\s+')


# ==================== КЛАСС ДЛЯ ОЧИСТКИ И НОРМАЛИЗАЦИИ ТЕКСТА ====================

class TextCleaner:
    """Очистка и нормализация текста для векторизации"""

    STOP_WORDS = {
        "привет", "здравствуйте", "хай", "хелло", "здарова",
        "подскажите", "пожалуйста", "пожалуста", "благодарю", "спасибо",
        "помогите", "напишите", "расскажите", "ответьте", "скажите",
        "добрый", "день", "утро", "вечер", "ночь",
        "можно", "нужно", "хочу", "поиск", "найти", "информацию"
    }

    # Компилируем паттерн стоп-слов один раз
    _STOP_PATTERN = re.compile(
        r'\b(?:' + '|'.join(map(re.escape, sorted(STOP_WORDS, key=len, reverse=True))) + r')\b',
        re.IGNORECASE
    )

    @staticmethod
    def clean_text(text: str, for_query: bool = True) -> str:
        """
        Очищает текст от шума.
        for_query: если True — удаляем стоп-слова (для запросов), если False — оставляем (для базы знаний)
        """
        if not text:
            return ""

        # 1. Удаляем HTML-теги
        text = _RE_HTML.sub('', text)

        # 2. Удаляем спецсимволы (оставляем буквы, цифры, базовую пунктуацию)
        text = _RE_SPECIAL.sub('', text)

        # 3. Удаляем одиночные буквы (шум)
        text = _RE_SINGLE.sub('', text)

        # 4. Для запросов — удаляем стоп-слова (для базы знаний оставляем контекст)
        if for_query:
            text = TextCleaner._STOP_PATTERN.sub('', text)

        # 5. Приводим множественные пробелы к одному, убираем края
        text = _RE_MULTI_SPACE.sub(' ', text).strip()

        # 6. Приводим к нижнему регистру для консистентности векторизации
        return text.lower()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
        """
        Разбивает текст на чанки с помощью LangChain RecursiveCharacterTextSplitter.
        chunk_size и chunk_overlap — в символах.
        """
        cleaned = TextCleaner.clean_text(text, for_query=False)  # для БЗ не удаляем стоп-слова
        if not cleaned:
            return []

        # RecursiveCharacterTextSplitter умнее: сначала пробует разбить по абзацам, потом по предложениям
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", '##', "\n", ". ", "! ", "? ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

        return splitter.split_text(cleaned)


# ==================== ВЕКТОРИЗАТОР С НОРМАЛИЗАЦИЕЙ ====================

class EmbeddingVectorizer:
    """Векторизатор с поддержкой мультиязычных моделей и L2-нормализацией"""

    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: List[str], apply_normalization: bool = True) -> np.ndarray:
        """
        Векторизует список текстов.
        apply_normalization: если True — применяет L2-нормализацию
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Исправлено: используем алиас sklearn_normalize и переименованный параметр
        if apply_normalization and embeddings.ndim == 2:
            embeddings = sklearn_normalize(embeddings, norm='l2', axis=1)

        return embeddings

    def encode_single(self, text: str, apply_normalization: bool = True) -> np.ndarray:
        """Векторизует один текст, возвращает вектор формы (1, dim)"""
        return self.encode([text], apply_normalization=apply_normalization)

# ==================== БАЗА ДАННЫХ С КОРРЕКТНОЙ ВЕКТОРИЗАЦИЕЙ ====================

class RAGDatabase:
    """База знаний с предобработкой и векторизацией"""

    def __init__(self, name: str, embedding_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.name = name
        self.vectorizer = EmbeddingVectorizer(model_name=embedding_model_name)
        self.documents: List[str] = []
        self.chunks: List[str] = []
        self.chunk_embeddings: np.ndarray = None

    def add_documents(self, documents: List[str], chunk_size: int = 500, chunk_overlap: int = 100):
        """Добавляет документы, разбивает на чанки и векторизует"""
        self.documents.extend(documents)
        self._process_documents(chunk_size, chunk_overlap)

    def _process_documents(self, chunk_size: int, chunk_overlap: int):
        """Очистка → чанкинг → векторизация"""
        self.chunks = []

        for doc in self.documents:
            # Разбиваем на чанки с перекрытием для сохранения контекста
            chunks = TextCleaner.chunk_text(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.chunks.extend(chunks)

        if self.chunks:
            # Векторизуем ВСЕ чанки сразу с нормализацией
            self.chunk_embeddings = self.vectorizer.encode(self.chunks, apply_normalization=True)
        else:
            self.chunk_embeddings = None

    def search(self, query_embedding: np.ndarray, top_k: int = 3, min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """
        Ищет наиболее похожие чанки по косинусному сходству.
        min_similarity: порог — результаты с меньшим сходством отбрасываются
        """
        if self.chunk_embeddings is None or self.chunk_embeddings.shape[0] == 0:
            return []

        # Косинусное сходство (благодаря L2-нормализации это просто dot product)
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]

        # Фильтруем по порогу
        valid_mask = similarities >= min_similarity
        if not np.any(valid_mask):
            return []

        # Берём top_k среди валидных
        valid_indices = np.where(valid_mask)[0]
        valid_sims = similarities[valid_mask]
        top_local_indices = np.argsort(valid_sims)[::-1][:top_k]
        top_indices = valid_indices[top_local_indices]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))

        return results

    def get_chunk_count(self) -> int:
        """Возвращает количество чанков в базе"""
        return len(self.chunks)


# ==================== УЛУЧШЕННЫЙ ПРОМПТ ДЛЯ LLAMA 3.2 ====================

def create_rag_prompt(context: str, query: str, db_name: str) -> str:
    """
    Создаёт структурированный промпт для Llama 3.2 в формате инструкций.
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Ты — интеллектуальный ассистент для ответов на вопросы по базе знаний.
Твоя задача: дать точный, краткий и полезный ответ на вопрос пользователя, используя ТОЛЬКО предоставленный контекст.

ПРАВИЛА:
1. Ответь на русском языке.
2. Опирайся на информацию из раздела "Контекст".
3. Если в контексте нет информации для ответа, напиши: "В предоставленных материалах нет информации для ответа на этот вопрос."
4. Будь конкретным: избегай общих фраз вроде "это зависит от обстоятельств", если контекст даёт чёткий ответ.
5. Если вопрос требует перечисления — используй маркированный список.
6. Не повторяй контекст дословно — перефразируй и синтезируй ответ.
7. Не упоминай, что ты "ИИ" или "модель" — просто отвечай как эксперт.

<|eot_id|><|start_header_id|>user<|end_header_id|>

База знаний: {db_name}

Контекст:
{context}

Вопрос: {query}

Ответ:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


# ==================== ОСНОВНОЙ КЛАСС RAG-СИСТЕМЫ ====================

class OllamaRAG:
    """RAG-система с двумя базами данных и корректной векторизацией"""

    def __init__(
            self,
            model: str = "llama3.2",
            host: str = "http://localhost:11434",
            threshold: float = 0.35,
            embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    ):
        self.model_name = model
        self.host = host
        self.api_url = f"{host}/api/generate"
        self.threshold = threshold

        # Векторизатор для запросов (ТА ЖЕ модель, что и для БД!)
        self.vectorizer = EmbeddingVectorizer(model_name=embedding_model)

        # Базы данных с той же моделью векторизации
        self.db1 = RAGDatabase("Математика_узкая", embedding_model_name=embedding_model)
        self.db2 = RAGDatabase("Математика_общая", embedding_model_name=embedding_model)

        self._initialize_databases()

    def _initialize_databases(self):
        """Инициализация баз данных с корректным чанкингом"""

        with open('L:\PythonProject2\RAG\узкая база.md', 'r', encoding='utf-8') as file:
            content = file.read()
            math_thin = [content]

        with open('L:\PythonProject2\RAG\широкая база.md', 'r', encoding='utf-8') as file:
            content = file.read()
            math_all = [content]

        # Добавляем документы с оптимальными параметрами чанкинга
        self.db1.add_documents(math_thin, chunk_size=500, chunk_overlap=100)
        self.db2.add_documents(math_all, chunk_size=500, chunk_overlap=100)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Векторизация запроса с очисткой и нормализацией"""
        cleaned_query = TextCleaner.clean_text(query, for_query=True)
        return self.vectorizer.encode_single(cleaned_query, apply_normalization=True)

    def _query_ollama(self, prompt: str) -> str:
        """Отправка запроса к Ollama API с улучшенной обработкой ошибок"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Низкая температура для фактологических ответов
                "top_p": 0.9,
                "num_predict": 500  # Ограничиваем длину ответа
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
            return answer if answer else "Ошибка: модель вернула пустой ответ"
        except requests.exceptions.ConnectionError:
            return "Ошибка подключения: не удалось соединиться с Ollama. Убедитесь, что служба запущена командой 'ollama serve'."
        except requests.exceptions.Timeout:
            return "Ошибка: превышено время ожидания ответа от модели. Попробуйте упростить запрос."
        except requests.exceptions.HTTPError as e:
            return f"HTTP ошибка от Ollama: {e}"
        except json.JSONDecodeError:
            return "Ошибка: не удалось распарсить ответ от Ollama"
        except Exception as e:
            return f"Непредвиденная ошибка: {type(e).__name__}: {str(e)}"

    def query(self, user_prompt: str) -> Dict:
        """
        Основной метод RAG-системы:
        1. Векторизует запрос (с очисткой и нормализацией)
        2. Ищет в БД по косинусному сходству (с порогом)
        3. Формирует структурированный промпт для Llama
        4. Возвращает ответ + метаданные
        """
        print(f"\n{'=' * 70}")
        print(f"📝 ЗАПРОС: '{user_prompt}'")
        print(f"{'=' * 70}")

        # 1. Векторизуем запрос (очистка + нормализация)
        query_embedding = self._get_query_embedding(user_prompt)

        # 2. Ищем в первой базе
        db1_results = self.db1.search(query_embedding, top_k=2, min_similarity=0.0)
        best_score_db1 = db1_results[0][1] if db1_results else 0.0

        print(f"\n🔍 Поиск в Базе 1 ({self.db1.name}):")
        print(f"   Найдено чанков: {len(db1_results)}, лучшее сходство: {best_score_db1:.4f}")

        # 3. Выбираем базу по порогу
        if best_score_db1 >= self.threshold:
            print(f"   ✓ Сходство >= {self.threshold} → Используем базу: {self.db1.name}")
            selected_db = self.db1
            best_context = db1_results[0][0]
            similarity_score = best_score_db1
            all_contexts = [r[0] for r in db1_results]
        else:
            print(f"   ✗ Сходство < {self.threshold} → Переключаемся на Базу 2")

            db2_results = self.db2.search(query_embedding, top_k=2, min_similarity=0.0)
            best_score_db2 = db2_results[0][1] if db2_results else 0.0

            print(f"\n🔍 Поиск в Базе 2 ({self.db2.name}):")
            print(f"   Найдено чанков: {len(db2_results)}, лучшее сходство: {best_score_db2:.4f}")

            selected_db = self.db2
            best_context = db2_results[0][0] if db2_results else "Контекст не найден"
            similarity_score = best_score_db2
            all_contexts = [r[0] for r in db2_results] if db2_results else []

        # 4. Формируем контекст (объединяем несколько чанков если нужно)
        combined_context = "\n\n".join(all_contexts) if all_contexts else best_context

        # 5. Создаём промпт для Llama 3.2
        rag_prompt = create_rag_prompt(
            context=combined_context,
            query=user_prompt,
            db_name=selected_db.name
        )

        # 6. Отправляем в Ollama
        print(f"\n🤖 Генерация ответа через {self.model_name}...")
        print(f"   Контекст: {combined_context[:150]}...")

        llm_response = self._query_ollama(rag_prompt)

        print(f"   ✓ Ответ получен")

        return {
            "query": user_prompt,
            "selected_database": selected_db.name,
            "similarity_score": similarity_score,
            "context_used": combined_context,
            "all_chunks_found": all_contexts,
            "answer": llm_response,
            "model_used": self.model_name
        }


# ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================

if __name__ == "__main__":
    print("=" * 70)
    print("RAG-СИСТЕМА: корректная векторизация + улучшенный промптинг")
    print("=" * 70)

    # Создаём систему с мультиязычной моделью эмбеддингов
    rag = OllamaRAG(
        model="llama3.2",
        threshold=0.35,
        embedding_model='paraphrase-multilingual-MiniLM-L12-v2'  # Важно для русского!
    )

    test_queries = [
        "Что такое точка?",
        "Что такое сверху?",
        "что такое луч?",
        "дай задач на сложение чисел"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"ВОПРОС #{i} из {len(test_queries)}")
        print(f"{'=' * 70}")

        result = rag.query(query)

        print(f"\n{'=' * 70}")
        print(f"📊 РЕЗУЛЬТАТ:")
        print(f"{'=' * 70}")
        print(f"📚 База данных: {result['selected_database']}")
        print(f"📐 Косинусное сходство: {result['similarity_score']:.4f}")
        print(f"🔢 Чанков в ответе: {len(result['all_chunks_found'])}")
        print(f"\n📄 Контекст:")
        for j, chunk in enumerate(result['all_chunks_found'], 1):
            print(f"   [{j}] {chunk[:200]}...")
        print(f"\n💬 Ответ от {result['model_used']}:")
        print(f"   {result['answer']}")
        print(f"{'=' * 70}")

        if i < len(test_queries):
            input("\nНажмите Enter для следующего вопроса...")
