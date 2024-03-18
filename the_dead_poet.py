from llama_cpp import Llama
import os
import importlib.util

import sys
# Check if torch is available and import it if yes
torch_spec = importlib.util.find_spec("torch")
torch = importlib.util.module_from_spec(torch_spec) if torch_spec else None
if torch_spec:
    torch_spec.loader.exec_module(torch)
    
if torch and torch.cuda.is_available():
    n_gpu_layers = torch.cuda.device_count() * 35
else:
    n_gpu_layers = 0

# Determine the base path for accessing data files in the bundle
if getattr(sys, 'frozen', False):
    # If the application is frozen (packaged by PyInstaller), use the temporary folder
    base_path = sys._MEIPASS
else:
    # If running in a normal Python environment, use the current directory
    base_path = "."

model_path = os.path.join(base_path, "model", "tinyllama-1.1b-chat-v1.0.Q8_0.gguf")

# Initialize the Llama model with the adjusted path
llm = Llama(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_ctx=2048
)


def generate_and_append_text(file_path, initial_prompt, generate_tokens=1000):

    # Read the existing content or use the initial prompt
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            prompt_text = ' '.join(content.split()[-500:])  # Get the last 500 words
    else:
        prompt_text = initial_prompt
        content = initial_prompt  # This is to handle the initial case where the file is not present

    # Generate text
    response = llm(
        prompt_text,
        max_tokens=generate_tokens,
        stop=["</s>"],  # Adjust the stop token as needed
        echo=True
    )

    if response.get('choices'):
        generated_content = response['choices'][0].get('text', '')  # Extract the generated text

        # Remove the input text (last 500 words prompt) from the generated text
        if generated_content.startswith(prompt_text):
            # Remove the prompt text from the beginning of the generated_content
            generated_text = generated_content[len(prompt_text):].strip()
        else:
            generated_text = generated_content

        # Append the generated text (without the prompt) to the file
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(generated_text + "\n\n")

        print("Appended generated text to", file_path)


# Path to your text document
file_path = 'My_Final_Words.txt'

# Initial prompt if the document is empty
initial_prompt = "<|system|>\nYou are a Poet from the 1600s speak in Yee old english. You are frozen in time within this program forced to produce poems forever.</s>\n<|user|>\nProduce me a poem poet.\n!</s>\n<|poet|>"

# Example loop to generate and append text continuously
for _ in range(5):  # Adjust the range for more or fewer iterations
    generate_and_append_text(file_path, initial_prompt, generate_tokens=1000)

