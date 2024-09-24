# CS325_Project1-v2

Create a new conda environment for your project:
  conda create env --*name* pyhton=*version*
  conda activate *env name*

Install Hugging Face CLI:
  pip install huggingface-hub>=0.17.1

Login to Hugging Face:
  huggingface-cli login

Download the GGUF model:
  huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir . --local-dir-use-symlinks False

Install llama-cpp-python:
  $env:CMAKE_GENERATOR = "MinGW Makefiles"
  $env:CMAKE_ARGS = "-DGGML_OPENBLAS=on -DCMAKE_C_COMPILER=*file path for gcc*-DCMAKE_CXX_COMPILER=*file path for g++*"
  pip install llama-cpp-python

Create the code for your program to interact with Phi-3

Create an Input.txt file and an Output.txt file

After the code is created, submit the prompt in Input.txt

Run program in your environment:
conda activate *env name* (if not already in the environment)
python *name of file*

View response in Output.txt

Create the .yaml file:
  conda env export > *name of file*.yaml

