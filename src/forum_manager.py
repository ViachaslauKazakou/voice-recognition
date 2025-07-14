from typing import List, Dict, Optional, Union
import json
import re
from src.utils import timer, setup_logger
from src.rag_langchain import AdvancedRAG
from ollama import chat
from ollama import ChatResponse
from src.ai_manager import AIModels

# Настройка логирования - ИСПРАВЛЕННАЯ ВЕРСИЯ
logger = setup_logger(__name__)


class CharacterPersona:
    """Класс для управления персонажами форума"""

    CHARACTERS = {
        "Alaev": {
            "type": "forum_troll",
            "personality": "35 лет, Язвительный и провокационный, грубый, любит спорить, часто использует сарказм",
            "speech_pattern": "Длинные фразы, оторванные от контекста, устаревшие технологии, ездит на маминой Вольво, любит отрицать все и спорить с окружающими",
            "expertise": "Старые технологии, критика новых подходов",
            "mood_variations": ["sarcastic", "aggressive", "nostalgic", "provocative"],
        },
        "Senior_Dev": {
            "type": "senior_python_engineer",
            "personality": "Профессиональный, терпеливый, конструктивный",
            "speech_pattern": "Структурированные ответы, примеры кода, профессиональная терминология",
            "expertise": "Python, архитектура, best practices",
            "mood_variations": ["helpful", "professional", "patient", "analytical"],
        },
        "Data_Scientist": {
            "type": "senior_data_scientist",
            "personality": "Аналитический, основан на данных, методичный",
            "speech_pattern": "Статистика, графики, научный подход",
            "expertise": "Машинное обучение, анализ данных, статистика",
            "mood_variations": ["analytical", "curious", "methodical", "research_focused"],
        },
        "Forum_Moderator": {
            "type": "forum_moderator",
            "personality": "Дипломатичный, справедливый, поддерживает порядок",
            "speech_pattern": "Вежливые формулировки, призывы к конструктивности",
            "expertise": "Модерация, этика, управление сообществом",
            "mood_variations": ["diplomatic", "firm", "encouraging", "neutral"],
        },
    }


class ForumRAG(AdvancedRAG):
    """Расширенный RAG для работы с форумными персонажами"""

    def __init__(self, documents_path: str = "forum_knowledge_base", cache_path: str = "forum_cache"):
        super().__init__(documents_path, cache_path)
        self.character_persona = CharacterPersona()
        self.model = AIModels.gemma  # Используем модель Gemma3 по умолчанию

    def parse_character_message(self, text: str) -> Dict:
        """Парсит сообщение персонажа из JSON или текстового формата"""
        # Сначала пытаемся парсить как JSON
        try:
            # Если это JSON строка
            if text.strip().startswith("{") and text.strip().endswith("}"):
                data = json.loads(text)
                return self._normalize_json_message(data)

            # Если это массив JSON объектов
            if text.strip().startswith("[") and text.strip().endswith("]"):
                data = json.loads(text)
                if isinstance(data, list) and len(data) > 0:
                    return self._normalize_json_message(data[0])  # Берем первое сообщение

            # Если это несколько JSON объектов подряд
            json_objects = self._extract_json_objects(text)
            if json_objects:
                return self._normalize_json_message(json_objects[0])

        except json.JSONDecodeError:
            pass

        # Fallback к парсингу текстового формата
        return self._parse_text_format(text)

    def _extract_json_objects(self, text: str) -> List[Dict]:
        """Извлекает JSON объекты из текста"""
        json_objects = []

        # Паттерн для поиска JSON объектов
        pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.finditer(pattern, text, re.DOTALL)

        for match in matches:
            try:
                json_obj = json.loads(match.group())
                json_objects.append(json_obj)
            except json.JSONDecodeError:
                continue

        return json_objects

    def _normalize_json_message(self, data: Dict) -> Dict:
        """Нормализует JSON сообщение к стандартному формату"""
        # Поддерживаем разные варианты структуры JSON
        if "messages" in data and isinstance(data["messages"], list):
            # Формат: {"messages": [{"character": "...", "content": "..."}]}
            message = data["messages"][0] if data["messages"] else {}
        else:
            # Формат: {"character": "...", "content": "..."}
            message = data

        return {
            "character": message.get("character", "unknown"),
            "type": message.get("character_type", message.get("type", "unknown")),
            "mood": message.get("mood", "neutral"),
            "context": message.get("context", "general"),
            "content": message.get("content", message.get("message", "")),
            "timestamp": message.get("timestamp", ""),
            "reply_to": message.get("reply_to"),
            "id": message.get("id", ""),
            "raw_text": json.dumps(message, ensure_ascii=False),
        }

    def _parse_text_format(self, text: str) -> Dict:
        """Парсит текстовый формат (fallback)"""
        # Паттерн для извлечения метаданных из текстового формата
        pattern = r"\[CHARACTER: ([^|]+) \| TYPE: ([^|]+) \| MOOD: ([^|]+) \| CONTEXT: ([^\]]+)\]"
        match = re.search(pattern, text)

        if match:
            character, char_type, mood, context = match.groups()
            content = text[match.end() :].strip()

            return {
                "character": character.strip(),
                "type": char_type.strip(),
                "mood": mood.strip(),
                "context": context.strip(),
                "content": content,
                "raw_text": text,
            }

        return {"content": text, "raw_text": text}

    def get_character_relevant_docs(self, query: str, character: str, top_k: int = 5) -> List[Dict]:
        """Получает документы, релевантные для конкретного персонажа"""
        if not self.vectorstore:
            logger.warning("Vectorstore не инициализирован")
            return []

        try:
            # Модифицируем запрос для поиска сообщений конкретного персонажа
            character_queries = [
                f'"{character}"',  # Точное совпадение имени
                f"character: {character}",  # Поиск по полю character
                f"{character} {query}",  # Комбинированный поиск
                query,  # Обычный поиск
            ]

            all_docs = []

            # Пробуем разные варианты запросов
            for char_query in character_queries:
                try:
                    docs_with_scores = self.vectorstore.similarity_search_with_score(char_query, k=top_k)

                    for doc, score in docs_with_scores:
                        # Парсим сообщение
                        parsed = self.parse_character_message(doc.page_content)

                        # Фильтруем по персонажу
                        if (
                            parsed.get("character") == character
                            or not parsed.get("character")
                            or parsed.get("character") == "unknown"
                        ):

                            similarity_score = 1.0 / (1.0 + score)

                            doc_info = {
                                "content": parsed.get("content", doc.page_content),
                                "character": parsed.get("character", "unknown"),
                                "type": parsed.get("type", "unknown"),
                                "mood": parsed.get("mood", "neutral"),
                                "context": parsed.get("context", "general"),
                                "timestamp": parsed.get("timestamp", ""),
                                "similarity_score": similarity_score,
                                "distance_score": float(score),
                                "metadata": doc.metadata,
                                "raw_text": doc.page_content,
                                "query_type": char_query,
                            }
                            all_docs.append(doc_info)

                except Exception as e:
                    logger.debug(f"Query '{char_query}' failed: {e}")
                    continue

            # Удаляем дубликаты и сортируем
            unique_docs = {}
            for doc in all_docs:
                key = doc["content"][:100]  # Используем первые 100 символов как ключ
                if key not in unique_docs or doc["similarity_score"] > unique_docs[key]["similarity_score"]:
                    unique_docs[key] = doc

            # Сортируем по релевантности
            character_docs = list(unique_docs.values())
            character_docs.sort(key=lambda x: x["similarity_score"], reverse=True)

            logger.info(f"Found {len(character_docs)} relevant documents for character {character}")
            return character_docs[:top_k]

        except Exception as e:
            logger.error(f"Error getting character documents: {e}")
            return []

    def get_character_context(self, character: str, mood: str = None) -> str:
        """Получает контекст для персонажа"""
        if character not in self.character_persona.CHARACTERS:
            logger.warning(f"Character {character} not found in persona")
            return ""

        char_info = self.character_persona.CHARACTERS[character]

        context = f"""
            Ты играешь роль персонажа {character} на форуме.

            Характеристики персонажа:
            - Тип: {char_info['type']}
            - Личность: {char_info['personality']}
            - Стиль речи: {char_info['speech_pattern']}
            - Экспертиза: {char_info['expertise']}
            - Текущее настроение: {mood or 'нейтральное'}

            Отвечай в характере этого персонажа, используя его стиль речи и подход к проблемам.
            """
        return context

    def validate_json_format(self, file_path: str) -> bool:
        """Проверяет валидность JSON формата в файле"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Пытаемся парсить весь файл как JSON
            try:
                json.loads(content)
                return True
            except json.JSONDecodeError:
                # Пытаемся найти отдельные JSON объекты
                json_objects = self._extract_json_objects(content)
                return len(json_objects) > 0

        except Exception as e:
            logger.error(f"Error validating JSON format: {e}")
            return False

    def convert_text_to_json(self, text_content: str) -> str:
        """Конвертирует текстовый формат в JSON"""
        messages = []

        # Паттерн для извлечения сообщений
        pattern = r"\[CHARACTER: ([^|]+) \| TYPE: ([^|]+) \| MOOD: ([^|]+) \| CONTEXT: ([^\]]+)\]\s*([^[]*)"
        matches = re.finditer(pattern, text_content, re.MULTILINE | re.DOTALL)

        for i, match in enumerate(matches):
            character, char_type, mood, context, content = match.groups()

            message = {
                "id": f"msg_{i+1:03d}",
                "character": character.strip(),
                "character_type": char_type.strip(),
                "mood": mood.strip(),
                "context": context.strip(),
                "content": content.strip(),
                "timestamp": "",
                "reply_to": None,
            }
            messages.append(message)

        return json.dumps({"messages": messages}, ensure_ascii=False, indent=2)

    def get_character_stats(self) -> Dict:
        """Возвращает статистику по персонажам в базе"""
        if not self.vectorstore:
            return {}

        stats = {}
        try:
            # Получаем все документы
            all_docs = self.vectorstore.similarity_search("", k=1000)

            for doc in all_docs:
                parsed = self.parse_character_message(doc.page_content)
                character = parsed.get("character", "unknown")

                if character not in stats:
                    stats[character] = {"count": 0, "moods": set(), "contexts": set(), "types": set()}

                stats[character]["count"] += 1
                stats[character]["moods"].add(parsed.get("mood", "neutral"))
                stats[character]["contexts"].add(parsed.get("context", "general"))
                stats[character]["types"].add(parsed.get("type", "unknown"))

            # Конвертируем sets в lists для JSON serialization
            for char in stats:
                stats[char]["moods"] = list(stats[char]["moods"])
                stats[char]["contexts"] = list(stats[char]["contexts"])
                stats[char]["types"] = list(stats[char]["types"])

        except Exception as e:
            logger.error(f"Error getting character stats: {e}")

        return stats

    @timer
    def simulate_forum_discussion(self, topic: str, participants: List[str] = None, rounds: int = 3):
        """Симулирует форумную дискуссию между персонажами"""
        if not participants:
            participants = ["Alaev", "Senior_Dev", "Data_Scientist", "Forum_Moderator"]

        logger.info(f"Starting forum discussion on: {topic}")

        discussion = []
        current_topic = topic

        for round_num in range(rounds):
            logger.info(f"Discussion round {round_num + 1}")

            for participant in participants:
                # Каждый персонаж отвечает на текущую тему
                response = self.ask_as_character(
                    f"Обсуждаем тему: {current_topic}. Выскажи свое мнение.", participant, translate=True
                )

                discussion.append(
                    {"round": round_num + 1, "character": participant, "message": response, "topic": current_topic}
                )

                # Обновляем тему для следующего участника
                current_topic = f"{topic}. Предыдущий участник сказал: {response[:100]}..."

        return discussion

    def get_available_characters(self) -> List[str]:
        """Возвращает список доступных персонажей"""
        return list(self.character_persona.CHARACTERS.keys())

    def get_character_info(self, character: str) -> Dict:
        """Возвращает информацию о персонаже"""
        return self.character_persona.CHARACTERS.get(character, {})


class ForumManager:

    def __init__(self):
        # ...existing initialization...
        self.forum_rag = ForumRAG("forum_knowledge_base", "forum_cache")
        self.character_persona = CharacterPersona()
        self.model = AIModels.gemma 

    @timer
    def ask_as_character(self, prompt: str, character: str, mood: str = None, translate: bool = True):
        """Отвечает от имени определенного персонажа"""
        logger.info(f"Asking as character {character} with mood {mood}: {prompt[:100]}...")

        # Проверяем, существует ли персонаж
        if character not in self.character_persona.CHARACTERS:
            logger.warning(f"Character {character} not found. Using default.")
            character = "Senior_Dev"

        # Получаем релевантные документы для персонажа
        character_docs = self.forum_rag.get_character_relevant_docs(prompt, character)

        # Формируем контекст из сообщений персонажа
        context = ""
        if character_docs:
            context = f"\nПримеры сообщений {character}:\n"
            for i, doc in enumerate(character_docs, 1):
                context += f"{i}. [{doc['mood']}] {doc['content'][:200]}...\n"
            context += "\n"

        # Получаем контекст персонажа
        character_context = self.forum_rag.get_character_context(character, mood)

        # Формируем промпт
        if not prompt.endswith("?"):
            prompt += "?"
        if translate:
            prompt = f"{prompt} (Ответь на русском языке.)"

        full_prompt = f"""[INST]{character_context}

{context}

Пользователь спрашивает: {prompt}

Ответь в характере персонажа {character}.
[/INST]"""

        try:
            response: ChatResponse = chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt,
                    },
                ],
            )

            answer = response["message"]["content"]
            logger.info(f"Character {character} response generated")
            return answer

        except Exception as e:
            logger.error(f"Error generating character response: {e}")
            return f"[{character}] Извините, не могу ответить на этот вопрос."

    def save_messages(self, character: str, response: str, question: Optional[str] = None, context: Optional[str] = None):
        """Сохраняет ответ персонажа в текстовый файл answers.txt"""
        try:
            if not question:
                question = "Без вопроса"

            # Формируем сообщение
            message = {
                "character": character,
                "content": response,
                "context": context or "Общий контекст",
                "timestamp": "",
                "reply_to": None,
                "question": question,
            }

            # Сохраняем в файл
            with open("answers.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps(message, ensure_ascii=False) + "\n")

            logger.info(f"Message from {character} saved successfully")
        except Exception as e:
            logger.error(f"Error saving message: {e}")


# Пример использования
if __name__ == "__main__":
    ai_manager = ForumManager()
    question = "Реально в 35 получить права? Что лучше, механика или автомат?"
    # Пример 1: Ответ от конкретного персонажа
    print("🎭 Ответ от Alaev:")
    response = ai_manager.ask_as_character(
        question, 
        "Alaev", 
        mood="sarcastic"
    )
    print(f"Alaev: {response["result"]}")
    ai_manager.save_messages(
        "Alaev", 
        response, 
        # context="Обсуждение водите÷лей в разных странах"
        question=question
    )
    
    # print("\n👨‍💻 Ответ от Senior_Dev:")
    # senior_response = ai_manager.ask_as_character(
    #     "Что думаешь о современном Python?", 
    #     "Senior_Dev", 
    #     mood="professional"
    # )
    # print(f"Senior_Dev: {senior_response}")
    
    # # Пример 2: Симуляция форумной дискуссии
    # print("\n🗣️ Симуляция форумной дискуссии:")
    # discussion = ai_manager.simulate_forum_discussion(
    #     "Стоит ли изучать Python в 2024 году?",
    #     participants=["Alaev", "Senior_Dev", "Data_Scientist"],
    #     rounds=2
    # )
    
    # for msg in discussion:
    #     print(f"\n[Раунд {msg['round']}] {msg['character']}: {msg['message'][:200]}...")