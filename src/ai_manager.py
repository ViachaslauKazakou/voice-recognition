import os
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ollama import chat
from ollama import ChatResponse
from langchain.chains import LLMChain, ConversationChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from src.utils import timer
from src.simple_rag_manager import SimpleRAG
from src.rag_langchain import AdvancedRAG
from langchain.chains import RetrievalQA
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)
# Add this block to enable console logging
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] - %(levelname)s [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

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


class HFTransformer:
    def __init__(self):
        # Загружаем токенизатор и модель
        self.model_id = AIModels.distilgpt2
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")

    def generate(self, prompt):
        # Создаём pipeline
        generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        # Генерируем текст
        result = generator(
            prompt,
            max_length=200,
            truncation=True,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=10,
            num_return_sequences=5,
        )
        return result[0]["generated_text"]
    

class AiManager:
    """
    Class to manage AI models and their interactions.
    """
    def __init__(self):
        # Initialize Ollama with LangChain
        self.llm = Ollama(model=AIModels.llama_mistral)
        self.memory = ConversationBufferMemory()
        # Create a conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
        # Add simple RAG component
        # self.rag_simple = SimpleRAG 
        # Add advanced RAG
        # self.rag = AdvancedRAG("knowledge_base")
        
        # # Create RAG chain with LangChain
        # if self.rag.retriever:
        #     self.rag_chain = RetrievalQA.from_chain_type(
        #         llm=self.llm,
        #         chain_type="stuff",
        #         retriever=self.rag.retriever,
        #         return_source_documents=True
        #     ) 

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
        logger.info(f"======Asking Ollama: {prompt}")
        role = "You are a Senior software engineer with main skill Python, answer the next question:"
        if not prompt.endswith("?"):
            prompt += "?"
        if translate:
            prompt = f"{prompt} (Ответ должен быть на русском языке.)"

        modified_prompt = f"[INST]{role} {prompt}[/INST]"
        response: ChatResponse = chat(
            model=AIModels.llama_mistral,
            messages=[
                {
                    "role": "user",
                    "content": modified_prompt,
                },
            ],
        )
        print(response["message"]["content"])
        # or access fields directly from the response object
        return response.message.content
    
    @timer
    def ask_ollama_memory(self, prompt, translate=True):
        if not prompt.endswith("?"):
            prompt += "?"
        if translate:
            prompt += " (Ответ должен быть на русском языке.)"
            
        # Use the conversation chain which maintains history
        response = self.conversation.predict(input=prompt)
        print(response)
        return response
        return {"result": response, "execution_time": 0}  # Timer decorator will add actual time
    
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
            follow_up_prompt = f"Based on your previous answer that '{response[:100]}...', can you elaborate further on the most important aspect mentioned?"
            
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
            depth_q = f"For someone at a {level+1}/5 understanding level of {topic}, what should they focus on learning next?"
            explanation = self.ask_ollama_memory(depth_q, translate)
            detailed_explanations.append({"level": level+1, "focus": explanation})
        
        # Final practical application
        apply_q = f"Give me a practical example or exercise to apply what I've learned about {topic}"
        application = self.ask_ollama_memory(apply_q, translate)
        
        return {
            "topic": topic,
            "introduction": intro,
            "core_concepts": concepts,
            "learning_path": detailed_explanations,
            "practical_application": application
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
            options_q = f"Based on the answer '{response[:100]}...', what are two different approaches or perspectives we could explore next?"
            options = self.ask_ollama_memory(options_q, translate)
            
            # Generate specific questions for each path
            path_q = f"Based on these options, formulate two specific questions that would explore each path."
            path_questions_text = self.ask_ollama_memory(path_q, translate)
            
            # For simplicity, we'll assume the model returns reasonably formatted questions
            # In production, you might want to use more structured prompting
            path1_q = f"Let's explore the first approach: {path_questions_text.split('1.')[1].split('2.')[0] if '1.' in path_questions_text else 'Tell me more about the first approach'}"
            path2_q = f"Let's explore the second approach: {path_questions_text.split('2.')[1] if '2.' in path_questions_text else 'Tell me more about the second approach'}"
            
            # Recursively explore both paths
            return {
                "question": question,
                "answer": response,
                "paths": [
                    explore_branch(path1_q, current_depth + 1),
                    explore_branch(path2_q, current_depth + 1)
                ]
            }
        
        # Start with the problem statement
        return explore_branch(problem_statement)

    @timer
    def ask_ollama_with_rag(self, prompt, translate=True, use_rag=True):
        """
        Ask Ollama with RAG support
        """
        logger.info(f"====Processing prompt: {prompt} with simple RAG: {use_rag}")
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
            model=AIModels.llama_mistral,
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
        logger.info("=" * 20)
        logger.info(f"===== Processing prompt: {prompt} with LangChain RAG")
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
        if not hasattr(self, 'rag_chain') or not self.rag_chain:
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


# Создайте папку knowledge_base и добавьте туда .txt файлы
# Например: knowledge_base/python_basics.txt, knowledge_base/algorithms.txt

if __name__ == "__main__":
    ai_manager = AiManager()
    
    # Обычный вопрос
    # response1 = ai_manager.ask_ollama("Что такое декоратор в Python?")
    
    # Вопрос с RAG
    # response2 = ai_manager.ask_ollama_with_rag("Что такое декоратор в Python?", use_rag=True)
    
    # Вопрос с LangChain RAG
    response3 = ai_manager.ask_ollama_langchain_rag("Что такое декоратор в Python? Приведи пример использования.", translate=True)
