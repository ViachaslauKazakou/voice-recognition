import requests
from ollama import chat
from ollama import ChatResponse
from langchain.chains import LLMChain, ConversationChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from src.utils import timer, setup_logger
from src.simple_rag_manager import SimpleRAG
from src.rag_langchain import AdvancedRAG
from langchain.chains import RetrievalQA

# Настройка логирования - ИСПРАВЛЕННАЯ ВЕРСИЯ
logger = setup_logger(__name__)


class Role:
    """
    Class to manage roles for AI interactions.
    """

    senior_python_engineer: str = "You are a Senior software engineer with main skill Python, answer the next question:"
    senior_frontend_engineer: str = "You are a Senior frontend engineer with main skill JavaScript, answer the next question:"
    senior_backend_engineer: str = "You are a Senior backend engineer with main skill Python, answer the next question:"
    senior_data_scientist: str = "You are a Senior data scientist with main skill Python, answer the next question:"
    forum_moderator: str = "You are a forum moderator, answer the next question:"
    forum_troll: str = "You are a forum talkative troll names Alaev, answer the next question:"


class AIModels:
    """
    Class to manage AI models and their identifiers.
    """

    light: str = "google/flan-t5-base"
    medium: str = "google/flan-t5-large"
    heavy: str = "google/flan-t5-xl"
    light_1: str = "mistralai/Mistral-7B-Instruct-v0.2"
    distilgpt2: str = "distilgpt2"
    llama_mistral: str = "mistral:latest"
    deepseek: str = "deepseek-r1"
    gemma: str = "gemma3:latest"
    # heavy: str = "mistralai/Mistral-7B-Instruct-v0.2"
    # light: str = "meta-llama/Llama-2-7b-chat-hf"
    # medium: str = "meta-llama/Llama-2-7b-chat-hf"


class AiManager:
    """
    Class to manage AI models and their interactions.
    """

    def __init__(self):
        # Initialize Ollama with LangChain
        self.model = AIModels.gemma
        self.llm = Ollama(model=self.model)
        self.memory = ConversationBufferMemory()
        # Create a conversation chain
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=True)

        # Настройка логгера для этого класса
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

        self.logger.info("AiManager initialized successfully")

    def ask_ollama_api(self, prompt):
        """
        Ask a question to the Ollama API and return the response. Using internal api for request"""
        modified_prompt = (
            f"You are a Senior software engineer with main skill Python, answer the next question: {prompt}"
        )
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": modified_prompt, "stream": False},  # или llama2, gemma и т.п.
        )
        print(response.json()["response"])
        return response.json()["response"]

    @timer
    def ask_ollama(self, prompt, translate=True):
        self.logger.info(f"Simple Asking Ollama: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        role = "You are a Senior software engineer with main skill Python, answer the next question:"
        if not prompt.endswith("?"):
            prompt += "?"
        if translate:
            prompt = f"{prompt} (Ответ должен быть на русском языке.)"

        modified_prompt = f"[INST]{role} {prompt}[/INST]"

        try:
            response: ChatResponse = chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": modified_prompt,
                    },
                ],
            )

            result = response["message"]["content"]
            self.logger.info(f"Response received: {len(result)} characters")
            return result

        except Exception as e:
            self.logger.error(f"Error in ask_ollama: {str(e)}")
            raise

    @timer
    def ask_ollama_memory(self, prompt, translate=True) -> str:
        if not prompt.endswith("?"):
            prompt += "?"
        if translate:
            prompt += " (Ответ должен быть на русском языке.)"

        # Use the conversation chain which maintains history
        response = self.conversation.predict(input=prompt)
        print(response)
        return response

    def question_chain(self, initial_prompt, follow_ups=2, translate=True):
        """
        Create a chain of questions where each follow-up is based on previous answers.

        Args:
            initial_prompt: The starting question
            follow_ups: Number of follow-up questions to ask
            translate: Whether to translate responses to Russian

        Returns:
            List of question-answer pairs
        """
        qa_chain = []

        # Ask initial question
        current_prompt = initial_prompt
        response = self.ask_ollama_memory(current_prompt, translate)
        qa_chain.append({"question": current_prompt, "answer": response})

        # Generate follow-up questions
        for i in range(follow_ups):
            # Create a follow-up question based on previous answer
            response_preview = response['result'][:100] if len(response) > 100 else response
            follow_up_prompt = (
                f"Based on your previous answer that '{response_preview}...', can "
                "you elaborate further on the most important aspect mentioned?"
            )

            # Get response to follow-up
            response = self.ask_ollama_memory(follow_up_prompt, translate)
            qa_chain.append({"question": follow_up_prompt, "answer": response})

        return qa_chain

    def guided_learning_chain(self, topic, depth=3, translate=True):
        """
        Create a sequential learning experience that builds understanding step by step.

        Args:
            topic: The topic to learn about
            depth: How many levels to explore
            translate: Whether to translate responses to Russian

        Returns:
            Dictionary with the learning progression
        """
        # Initial broad question about the topic
        initial_q = f"What is {topic} and why is it important?"
        intro = self.ask_ollama_memory(initial_q, translate)

        # Ask for core concepts
        concepts_q = f"What are the 3 most important concepts in {topic} that I should understand first?"
        concepts = self.ask_ollama_memory(concepts_q, translate)

        # Extract concepts and explore each one
        detailed_explanations = []

        # For each level of depth, ask increasingly specific questions
        for level in range(depth):
            depth_q = (
                f"For someone at a {level+1}/5 understanding level of {topic}, what should they focus on learning next?"
            )
            explanation = self.ask_ollama_memory(depth_q, translate)
            detailed_explanations.append({"level": level + 1, "focus": explanation})

        # Final practical application
        apply_q = f"Give me a practical example or exercise to apply what I've learned about {topic}"
        application = self.ask_ollama_memory(apply_q, translate)

        return {
            "topic": topic,
            "introduction": intro,
            "core_concepts": concepts,
            "learning_path": detailed_explanations,
            "practical_application": application,
        }

    def decision_tree_chain(self, problem_statement, max_depth=3, translate=True):
        """
        Create a decision tree by asking follow-up questions based on previous answers.

        Args:
            problem_statement: The initial problem to solve
            max_depth: Maximum depth of the decision tree
            translate: Whether to translate responses to Russian

        Returns:
            A dictionary representing the decision tree
        """

        def explore_branch(question, current_depth=0):
            if current_depth >= max_depth:
                return {"question": question, "answer": self.ask_ollama_memory(question, translate)}

            response = self.ask_ollama_memory(question, translate)

            # Generate two alternative paths to explore
            options_q = (
                f"Based on the answer '{response['result'][:100]}...', what are two different "
                "approaches or perspectives we could explore next?"
            )
            options = self.ask_ollama_memory(options_q, translate)

            # Generate specific questions for each path
            path_q = "Based on these options, formulate two specific questions that would explore each path."
            path_questions_text = self.ask_ollama_memory(path_q, translate)

            # For simplicity, we'll assume the model returns reasonably formatted questions
            # In production, you might want to use more structured prompting
            path1_q = f"Let's explore the first approach: {
                path_questions_text.split('1.')[1].split('2.')[0] if '1.' in path_questions_text else 'Tell me more about the first approach'}"
            path2_q = f"Let's explore the second approach: {
                path_questions_text.split('2.')[1] if '2.' in path_questions_text else 'Tell me more about the second approach'}"
            # Recursively explore both paths

            return {
                "question": question,
                "answer": response,
                "paths": [explore_branch(path1_q, current_depth + 1), explore_branch(path2_q, current_depth + 1)],
            }

        # Start with the problem statement
        return explore_branch(problem_statement)

    @timer
    def ask_ollama_with_rag(self, prompt, translate=True, use_rag=True):
        """
        Ask Ollama with RAG support
        """
        logger.info(f"==== Processing prompt: {prompt} with simple RAG: {use_rag}")
        self.rag_simple = SimpleRAG("knowledge_base")  # Ensure RAG is initialized
        context = ""

        if use_rag:
            # Retrieve relevant documents
            relevant_docs = self.rag_simple.search(prompt, top_k=3)

            if relevant_docs:
                logger.info(f"Found {len(relevant_docs)} relevant documents for RAG.")
                context = "\n\nКонтекст из базы знаний:\n"
                for i, doc in enumerate(relevant_docs, 1):
                    context += f"{i}. {doc}\n"
                context += "\n"

        # Prepare the prompt
        role = "You are a Senior software engineer with main skill Python. Use the provided context to answer the question accurately."

        if not prompt.endswith("?"):
            prompt += "?"
        if translate:
            prompt = f"{prompt} (Ответ должен быть на русском языке.)"

        # Combine context and prompt
        full_prompt = f"[INST]{role}\n{context}Вопрос: {prompt}[/INST]"

        response: ChatResponse = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                },
            ],
        )

        print(response["message"]["content"])
        return response.message.content

    @timer
    def ask_ollama_memory_with_rag(self, prompt, translate=True, use_rag=True):
        """
        Ask Ollama with both memory and RAG
        """
        context = ""

        if use_rag:
            relevant_docs = self.rag_simple.search(prompt, top_k=3)

            if relevant_docs:
                context = "Контекст из базы знаний:\n"
                for i, doc in enumerate(relevant_docs, 1):
                    context += f"{i}. {doc}\n"
                context += "\n"

        if not prompt.endswith("?"):
            prompt += "?"
        if translate:
            prompt += " (Ответ должен быть на русском языке.)"

        # Add context to the prompt
        full_prompt = f"{context}Вопрос: {prompt}"

        # Use conversation chain with enhanced prompt
        response = self.conversation.predict(input=full_prompt)
        print(response)
        return response

    @timer
    def ask_ollama_langchain_rag(self, prompt, translate=True):
        """
        Ask Ollama using LangChain RAG
        """
        logger.info(f" ===== Processing prompt: {prompt} with LangChain RAG")
        self.rag = AdvancedRAG("knowledge_base")
        # Create RAG chain with LangChain
        if self.rag.retriever:
            logger.info("Creating RAG chain with LangChain")
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm, 
                chain_type="stuff", 
                retriever=self.rag.retriever, 
                return_source_documents=True
            )
        else:
            logger.warning("RAG retriever is not initialized. Using fallback method.")
            self.rag_chain = None
        if not hasattr(self, "rag_chain") or not self.rag_chain:
            # Fallback to regular ask_ollama
            return self.ask_ollama(prompt, translate)

        if not prompt.endswith("?"):
            prompt += "?"
        if translate:
            prompt += " (Ответ должен быть на русском языке.)"

        try:
            # Use RAG chain
            response = self.rag_chain.invoke({"query": prompt})

            answer = response["result"]
            sources = response.get("source_documents", [])

            print(f"Answer: {answer}")
            if sources:
                print(f"Sources: {len(sources)} documents used")

            return answer

        except Exception as e:
            print(f"Ошибка RAG: {e}")
            # Fallback to regular method
            return self.ask_ollama(prompt, translate)

    @timer
    def ask_ollama_langchain_rag_enhanced(self, prompt, translate=True, rag_method="adaptive", role=None):
        """
        Ask Ollama using enhanced LangChain RAG with improved ranking
        """
        logger.info(f"==== Processing prompt with enhanced RAG using method: {rag_method}")
        logger.info(f"Processing prompt with enhanced RAG: {prompt[:100]}...")
        
        self.rag = AdvancedRAG("knowledge_base")
        
        if not self.rag.retriever:
            logger.warning("RAG retriever is not initialized. Using fallback method.")
            return self.ask_ollama(prompt, translate)
        
        # Получаем релевантные документы с улучшенным ранжированием
        relevant_docs = self.rag.get_relevant_docs(prompt, method=rag_method)
        
        if not relevant_docs:
            logger.warning("No relevant documents found. Using fallback method.")
            return self.ask_ollama(prompt, translate)
        logger.info(f"Found {len(relevant_docs)} relevant documents using {rag_method} method")

        # Создаем контекст с информацией о качестве
        logger.info("Preparing context for response")
        context = "Контекст из базы знаний (ранжирован по релевантности):\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"{i}. {doc}\n"
        context += "\n"
        
        # Подготавливаем промпт
        logger.info("Preparing full prompt for Ollama")
        role = role or "You are a Senior software engineer with main skill Python. Use the provided context to answer the question accurately."
        # role = "You are a Senior software engineer. Use the provided context (ranked by relevance) to answer accurately."
        
        if not prompt.endswith("?"):
            prompt += "?"
        if translate:
            prompt = f"{prompt} (Ответ должен быть на русском языке.)"
        
        full_prompt = f"[INST]{role}\n{context}Вопрос: {prompt}[/INST]"
        
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
            logger.info(f"Enhanced RAG response generated using {len(relevant_docs)} documents")
            return answer
            
        except Exception as e:
            logger.error(f"Error in enhanced RAG: {e}")
            return self.ask_ollama(prompt, translate)
        
    def enhanced_adaptig_rag(self, prompt=None, translate=True, compare_methods=None, analyze_quality=False):
        """
        Ask Ollama using an enhanced RAG chain with improved ranking.
        """
        # Тестирование разных методов RAG
        test_prompt = prompt if prompt else input("Введите тестовый запрос для RAG: ")

        print("\n🔍 Сравнение методов RAG:")
        print("-" * 50)
        if not compare_methods:
            methods = ["adaptive", "filtered", "ranked", "contextual"]
            for method in methods:
                print(f"\n📊 Метод: {method}")
                try:
                    response = ai_manager.ask_ollama_langchain_rag_enhanced(
                        test_prompt, 
                        translate=translate, 
                        rag_method=method
                    )
                    print(f"Ответ: {response['result'][:500]}...")
                except Exception as e:
                    print(f"Ошибка: {e}")
        else:
            if compare_methods not in ["adaptive", "filtered", "ranked", "contextual"]:
                print(f"❌ Неверный метод: {compare_methods}. Доступные методы: adaptive, filtered, ranked, contextual")
                return
            # Сравнение с указанным методом
            print(f"\n📊 Метод: {compare_methods}")
            try:
                response = self.ask_ollama_langchain_rag_enhanced(
                    test_prompt, 
                    translate=translate, 
                    rag_method=compare_methods
                )
                print(f"Ответ: {response['result'][:500]}...")
            except Exception as e:
                print(f"Ошибка: {e}")
        if analyze_quality:
            self.analyze_docs_quality(test_prompt)

    def analyze_docs_quality(self, prompt):
        # Анализ качества документов
        print("\n📈 Анализ качества найденных документов:")
        rag = ai_manager.rag
        docs_info = rag.get_ranked_relevant_docs(prompt)
        
        for i, doc in enumerate(docs_info):
            print(f"{i+1}. Score: {doc['similarity_score']:.4f}, Source: {doc['source']}")
            print(f"   Preview: {doc['content'][:100]}...")
        


# Создайте папку knowledge_base и добавьте туда .txt файлы
# Например: knowledge_base/python_basics.txt, knowledge_base/algorithms.txt

if __name__ == "__main__":
    print("=" * 100)
    print("🚀 Инициализация AI Manager...")
    ai_manager = AiManager()

    print("\n" + "=" * 100)
    print("🤖 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ AI MANAGER".center(100))
    print("=" * 100)

    # 1. Обычный вопрос без RAG
    print("=" * 100)
    print("\n📝 1. ОБЫЧНЫЙ ВОПРОС (без RAG)")
    
    # response1 = ai_manager.ask_ollama("Что такое декоратор в Python?")
    # print(f"Ответ: {response1['result'][:200]}...")
    print("-" * 120)
    # 2. Вопрос с Simple RAG
    print("=" * 100)
    print("\n🔍 2. ВОПРОС С SIMPLE RAG")
   
    # response2 = ai_manager.ask_ollama_with_rag(
    #     "Что такое декоратор в Python? Приведи практический пример.", use_rag=True
    # )
    # print(f"Ответ с RAG: {response2}")
    print("-" * 120)

    # 3. Вопрос с LangChain RAG
    print("=" * 100)
    print("🧠 3. ВОПРОС С LANGCHAIN RAG".center(120))
    # response3 = ai_manager.ask_ollama_langchain_rag(
    #     "Что такое декоратор в Python? Приведи пример использования.", translate=True
    # )
    # print(f"Ответ с LangChain RAG: {response3['result'][:100]}...")
    print("-" * 100)

    # 4. Цепочка вопросов (Question Chain)
    print("\n" + "=" * 100)
    print("\n🔗 4. ЦЕПОЧКА ВОПРОСОВ")
    # qa_chain = ai_manager.question_chain("Объясни принципы ООП в Python", follow_ups=2, translate=True)
    # print(f"Создано {len(qa_chain)} вопросов-ответов в цепочке:")
    # for i, qa in enumerate(qa_chain, 1):
    #     print(f"  {i}. Вопрос: {qa['question']}...")
    #     print(f"     Ответ: {qa['answer']}...")
    print("-" * 100)

    # 5. Управляемое обучение (Guided Learning)
    print("=" * 100)
    print("\n🎓 5. УПРАВЛЯЕМОЕ ОБУЧЕНИЕ")
    print("=" * 100)  
    # learning_path = ai_manager.guided_learning_chain("асинхронное программирование в Python", depth=2, translate=True)
    # print(f"Тема обучения: {learning_path['topic']}")
    # print(f"Введение: {learning_path['introduction'][:100]}...")
    # print(f"Ключевые концепции: {learning_path['core_concepts']}...")
    # print(f"Шагов обучения: {len(learning_path['learning_path'])}")
    print("-" * 100)
 
    # 6. Дерево решений (Decision Tree)
    print("  ")
    print("=" * 100)
    print("🌳 6. ДЕРЕВО РЕШЕНИЙ".center(100))
    print("=" * 100)
    # decision_tree = ai_manager.decision_tree_chain(
    #     "Как выбрать подходящую структуру данных в Python?", max_depth=2, translate=True
    # )
    # print(f"Проблема: {decision_tree['question'][:60]}...")
    # print(f"Основной ответ: {decision_tree['answer'][:100]}...")
    # if "paths" in decision_tree:
    #     print(f"Альтернативных путей: {len(decision_tree['paths'])}")

    # 7. Сравнение методов с памятью
    print("  ")
    print("=" * 100)
    print("🧠 7. СРАВНЕНИЕ: ОБЫЧНЫЙ vs С ПАМЯТЬЮ".center(100))
    print("=" * 100)

    # Обычный метод
    # regular_response = ai_manager.ask_ollama("Какие есть типы данных в Python?", translate=True)
    # print(f"Обычный ответ: {regular_response['message'][:100]}...")

    # Метод с памятью
    # memory_response = ai_manager.ask_ollama_memory(
    #     "А какой из этих типов лучше всего подходит для хранения уникальных значений?", translate=True
    # )
    # print(f"Ответ с памятью: {memory_response['message'][:100]}...")

    # 8. Работа с RAG и память одновременно
    print("  ")
    print("=" * 100)
    print("\n🔄 8. RAG + ПАМЯТЬ")

    # rag_memory_response = ai_manager.ask_ollama_memory_with_rag(
    #     "Покажи пример использования set для удаления дубликатов", translate=True, use_rag=True
    # )
    # print(f"RAG + Память: {rag_memory_response[:100]}...")
    print("-" * 100)

    # 9. Демонстрация кеширования (если доступно)
    print("  ")
    print("=" * 100)
    print("\n💾 9. ИНФОРМАЦИЯ О КЕШИРОВАНИИ")
    print("=" * 100)
    try:
        # Попытаемся получить информацию о кеше RAG
        if hasattr(ai_manager, "rag_simple") and ai_manager.rag_simple:
            cache_info = ai_manager.rag_simple.get_cache_info()
            print(f"Simple RAG кеш: {cache_info}")

        if hasattr(ai_manager, "rag") and ai_manager.rag:
            advanced_cache_info = ai_manager.rag.get_cache_info()
            print(f"Advanced RAG кеш: {advanced_cache_info}")
        else:
            print("Advanced RAG кеш не инициализирован.")

    except Exception as e:
        print(f"Информация о кеше недоступна: {e}")

    # 10. Практический пример: Помощник по коду
    print("  ")
    print("=" * 100)
    print("\n👨‍💻 10. ПРАКТИЧЕСКИЙ ПРИМЕР: ПОМОЩНИК ПО КОДУ")
    print("=" * 100)
    # code_question = """
    # У меня есть список чисел [1, 2, 2, 3, 3, 3, 4]. 
    # Как найти число, которое встречается чаще всего? 
    # Покажи несколько способов решения.
    # """

    # code_response = ai_manager.ask_ollama_with_rag(code_question, translate=True, use_rag=True)
    # print(f"Решение задачи: {code_response['result'][:200]}...")
    print("-" * 100)

    # 10.1 Практический пример: Помощник по коду
    print("  ")
    print("=" * 100)
    print("\n👨‍💻 10.1 РАСШИРЕННЫЙ RAG: КЭШ И РЕРАНКИНГ")
    print("=" * 100)
    prompt = "Кто такой Владимир Ленин? Расскажи о его вкладе в историю России и мира. Приведи ключевые даты и события."
    print(f"Тестовый запрос: {prompt}")
    ai_manager.enhanced_adaptig_rag(prompt, translate=True, compare_methods=None, analyze_quality=True)
    print("-" * 100)


    # 11. Интерактивный режим (опционально)
    print("  ")
    print("=" * 100)
    print("\n🎮 11. ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("-" * 50)
    print("Для запуска интерактивного режима раскомментируйте следующий блок:")

    """
    # Интерактивный режим - раскомментируйте для использования
    while True:
        user_input = input("\n❓ Введите ваш вопрос (или 'quit' для выхода): ")
        if user_input.lower() in ['quit', 'exit', 'выход']:
            print("👋 До свидания!")
            break
        
        print("\nВыберите метод:")
        print("1 - Обычный ответ")
        print("2 - С Simple RAG")
        print("3 - С LangChain RAG")
        print("4 - С памятью")
        
        choice = input("Ваш выбор (1-4): ")
        
        try:
            if choice == "1":
                response = ai_manager.ask_ollama(user_input, translate=True)
            elif choice == "2":
                response = ai_manager.ask_ollama_with_rag(user_input, translate=True, use_rag=True)
            elif choice == "3":
                response = ai_manager.ask_ollama_langchain_rag(user_input, translate=True)
            elif choice == "4":
                response = ai_manager.ask_ollama_memory(user_input, translate=True)
            else:
                print("❌ Неверный выбор")
                continue
                
            print(f"\n🤖 Ответ: {response}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    """

    print("\n✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 80)

    # 12. Сводная статистика
    print("\n📊 СВОДНАЯ ИНФОРМАЦИЯ")
    print("-" * 50)
    print("📚 Доступные методы:")
    print("  • ask_ollama() - базовые вопросы")
    print("  • ask_ollama_with_rag() - с Simple RAG")
    print("  • ask_ollama_langchain_rag() - с LangChain RAG")
    print("  • ask_ollama_memory() - с контекстной памятью")
    print("  • question_chain() - цепочки вопросов")
    print("  • guided_learning_chain() - управляемое обучение")
    print("  • decision_tree_chain() - дерево решений")

    print("\n🔧 Полезные команды:")
    print("  • ai_manager.rag_simple.clear_cache() - очистить Simple RAG кеш")
    print("  • ai_manager.rag.rebuild_index() - пересоздать LangChain индекс")
    print("  • ai_manager.rag.get_cache_info() - информация о кеше")

    print("\n💡 Советы по использованию:")
    print("  • Создайте папку 'knowledge_base' с .txt файлами для RAG")
    print("  • Используйте RAG для вопросов, требующих контекста из документов")
    print("  • Используйте память для диалогов и уточняющих вопросов")
    print("  • Цепочки вопросов подходят для глубокого изучения темы")
