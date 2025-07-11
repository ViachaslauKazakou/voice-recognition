from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline


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
