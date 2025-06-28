# Using Hugging Face Models with Ollama

You can integrate models from Hugging Face (HF) into your Ollama setup in a couple of ways. This allows you to leverage a wider range of models, including those not yet officially in the Ollama library or specific GGUF-quantized versions.

## Method 1: Using `ollama pull` (for models in Ollama Library)

Many popular models from Hugging Face are already available directly through the Ollama library. This is the simplest method.

1.  **Check Model Availability:**
    Visit the [Ollama Library](httpsama.com/library) to see if the Hugging Face model you're interested in (or a version of it) is listed. For example, `mistralai/Mistral-7B-Instruct-v0.2` is available as `mistral`.

2.  **Pull the Model:**
    If the model is available, use the `ollama pull` command in your terminal:
    ```bash
    ollama pull <model_name_in_ollama_library>
    ```
    For instance, to get the Mistral model used in `AIModels.llama_mistral` (`"mistral"`):
    ```bash
    ollama pull mistral
    ```
    Ollama will download the model (typically in GGUF format) and make it available for use.

## Method 2: Importing a GGUF File from Hugging Face

If the model isn't in the Ollama library, or you need a specific GGUF version (e.g., a particular quantization), you can download the GGUF file from Hugging Face and import it manually.

1.  **Download the GGUF File:**
    *   Go to the Hugging Face Hub ([huggingface.co/models](https://huggingface.co/models)).
    *   Search for the model you want. Look for GGUF versions, often provided by users like "TheBloke" or sometimes by the official model creators.
    *   Navigate to the "Files and versions" tab of the model repository.
    *   Download the desired `.gguf` file to your local machine.

2.  **Create a `Modelfile`:**
    A `Modelfile` is a blueprint that tells Ollama how to create a model.
    *   Create a new text file in your project directory (e.g., `MyCustomMistral.modelfile` or simply `MyCustomMistral`).
    *   Add the following content, adjusting the `FROM` path and `TEMPLATE` as needed:

    ```Modelfile
    # filepath: ./MyCustomMistral.modelfile
    # Replace with the actual path to your downloaded GGUF file
    FROM /path/to/your/downloaded-model-name.gguf

    # Define the prompt template (this example is for Mistral Instruct)
    # You might need to adjust this based on the model's requirements
    TEMPLATE """[INST] {{ .Prompt }} [/INST]"""

    # Optional: Specify system message
    # SYSTEM """You are a helpful AI assistant."""

    # Optional: Add parameters like stop tokens
    # PARAMETER stop "[INST]"
    # PARAMETER stop "[/INST]"
    # PARAMETER stop "</s>"
    ```
    *   **`FROM`**: Specifies the path to the local GGUF file you downloaded.
    *   **`TEMPLATE`**: Defines how prompts should be formatted for the model. This is crucial and varies between models. Check the model card on Hugging Face for the correct prompt format.
    *   **`SYSTEM`** (Optional): Sets a system message for the model.
    *   **`PARAMETER`** (Optional): Allows you to set various model parameters like stop tokens, temperature, etc.

3.  **Create the Model in Ollama:**
    Open your terminal, navigate to the directory where you saved the `Modelfile`, and run:
    ```bash
    ollama create <your_ollama_model_name> -f ./NameOfYourModelfile
    ```
    For example:
    ```bash
    ollama create my-custom-mistral -f ./MyCustomMistral.modelfile
    ```
    Replace `<your_ollama_model_name>` with the name you want to use to refer to this model in Ollama (e.g., `my-custom-mistral`).
    Replace `./NameOfYourModelfile` with the actual name of your `Modelfile`.

4.  **Use the Model:**
    Once created, you can use your custom model like any other Ollama model:
    *   Via the command line: `ollama run my-custom-mistral`
    *   Via the Ollama API or libraries like LangChain, by specifying `my-custom-mistral` as the model name.

    In your `ai_manager.py`, you would update `AIModels` or the `Ollama` instance initialization:
    ````python
    # filepath: src/ai_manager.py
    # ...existing code...
    class AIModels:
        # ...existing code...
        my_custom_hf_model: str = "my-custom-mistral" # Add your new model name
    # ...existing code...
    class AiManager:
        def __init__(self):
            # Example: Using your custom model
            self.llm = Ollama(model=AIModels.my_custom_hf_model) 
            # Or directly: self.llm = Ollama(model="my-custom-mistral")
    # ...existing code...
    ````

By following these steps, you can effectively use a wide variety of models from Hugging Face with your Ollama setup. Remember to check the model card on Hugging Face for specific instructions on prompt formatting and recommended parameters.

// ...existing code...
    Ollama will download the model (typically in GGUF format) and make it available for use.

## Checking Available Local Models

To see a list of all models that you have already pulled or created locally in Ollama, run the following command in your terminal:

```bash
ollama list
```
This will display details such as the model name, ID, size, and when it was last modified.
// ...existing code...