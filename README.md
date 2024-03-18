# the_dead_poet
Source code for "The_Dead_poet"



### Install Guide

Install your chosen operating system's vrsion of cuda into your working enviroment if you have a cuda enabled device.
https://pytorch.org/get-started/locally/

For windows operating system use set to select the correct version for your operating system of llama-cpp-python. For cuda enabled devices choose the following.
```python
set CMAKE_ARGS="-DLLAMA_CUBLAS=on" 

pip install llama-cpp-python
```

Then within the model folder download the following model.
https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf




### Run The Program
```
Python The_dead_poet.py
```




### Compress into executable.
```
pyinstaller --noconsole --add-data "INSERT ENVRIOMENT PATH HERE\llama_cpp;llama_cpp" --add-data ".\tinyllama-1.1b-chat-v1.0.Q8_0.gguf;model" The_Dead_Poet.py
```


Sometimes the llama.dll is located elsewhere when installing llama-cpp-python copy the file into the envriomental folder where it is installed.
To find it back up a few folders and search for it using the searchbar.
