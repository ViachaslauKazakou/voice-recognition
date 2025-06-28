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
        self.llm = Ollama(model=AIModels.gemma)
        self.memory = ConversationBufferMemory()
        
        # Create a conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )

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


if __name__ == "__main__":
    sound_manager = SoundManager()
    # sound_manager.record()
    text = sound_manager.recognition()
    text = "Напиши пример декоратора на Python."
    print(text)
    ai_manager = AiManager()
    response = ai_manager.ask_ollama(text)
    print("Use local HF model")
    hf = HFTransformer()
    # prompt = "Какой язык программирования лучше всего подходит для веб-разработки?"
    response = hf.generate(text)
    print("Generated response:")
    print(response)
