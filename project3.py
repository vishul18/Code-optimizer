import os
import io
import sys
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
import anthropic
from IPython.display import Markdown, display, update_display
import gradio as gr
import subprocess
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')
openai = OpenAI()
claude = anthropic.Anthropic()
OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
system_message = "You are an assistant that reimplements Python code in high performance C++ for an M1 Mac. "
system_message += "Respond only with C++ code; use comments sparingly and do not provide any explanation other than occasional comments. "
system_message += "The C++ response needs to produce an identical output in the fastest possible time. Keep implementations of random number generators identical so that results match exactly."
def user_prompt_for(python):
    user_prompt = "Rewrite this Python code in C++ with the fastest possible implementation that produces identical output in the least time. "
    user_prompt += "Respond only with C++ code; do not explain your work other than a few comments. "
    user_prompt += "Pay attention to number types to ensure no int overflows. Remember to #include all necessary C++ packages such as iomanip.\n\n"
    user_prompt += python
    return user_prompt
def messages_for(python):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt_for(python)}
    ]

def write_output(cpp):
    code = cpp.replace("```cpp","").replace("```","")
    with open("optimized.cpp", "w") as f:
        f.write(code)
def optimize_gpt(python):    
    stream = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages_for(python), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        print(fragment, end='', flush=True)
    write_output(reply)
def optimize_claude(python):
    result = claude.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=system_message,
        messages=[{"role": "user", "content": user_prompt_for(python)}],
    )
    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            print(text, end="", flush=True)
    write_output(reply)

import time 

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations + 1):
        j = i * param1 - param2
        result -= (1 / j)
        j = i * param1 + param2
        result += (1 / j)
    return result

start_time = time.time()
pi = calculate(100_000_000, 4, 1) * 4  
end_time = time.time()

print(f"Result (pi): {pi:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")


exec(pi)
optimize_gpt(pi)
exec(pi)
optimize_claude(pi)
def python_hard():
    import time

    def lcg(seed, a=1664525, c=1013904223, m=2**32):
        value = seed
        while True:
            value = (a * value + c) % m
            yield value

    def max_subarray_sum(n, seed, min_val, max_val):
        lcg_gen = lcg(seed)
        random_numbers = [next(lcg_gen) % (max_val - min_val + 1) + min_val for _ in range(n)]
        max_sum = float('-inf')
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += random_numbers[j]
                if current_sum > max_sum:
                    max_sum = current_sum
        return max_sum

    def total_max_subarray_sum(n, initial_seed, min_val, max_val):
        total_sum = 0
        lcg_gen = lcg(initial_seed)
        for _ in range(20):
            seed = next(lcg_gen)
            total_sum += max_subarray_sum(n, seed, min_val, max_val)
        return total_sum

    
    n = 10000         
    initial_seed = 42 
    min_val = -10     
    max_val = 10      

   
    start_time = time.time()
    result = total_max_subarray_sum(n, initial_seed, min_val, max_val)
    end_time = time.time()

    print("Total Maximum Subarray Sum (20 runs):", result)
    print("Execution Time: {:.6f} seconds".format(end_time - start_time))



python_hard()


exec(python_hard)
optimize_gpt(python_hard)
optimize_claude(python_hard)
def stream_gpt(python):    
    stream = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages_for(python), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        yield reply.replace('```cpp\n','').replace('```','')
def stream_claude(python):
    result = claude.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=system_message,
        messages=[{"role": "user", "content": user_prompt_for(python)}],
    )
    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            yield reply.replace('```cpp\n','').replace('```','')
def optimize(python, model):
    if model=="GPT":
        result = stream_gpt(python)
    elif model=="Claude":
        result = stream_claude(python)
    else:
        raise ValueError("Unknown model")
    for stream_so_far in result:
        yield stream_so_far        
with gr.Blocks() as ui:
    with gr.Row():
        python = gr.Textbox(label="Python code:", lines=10, value=python_hard)
        cpp = gr.Textbox(label="C++ code:", lines=10)
    with gr.Row():
        model = gr.Dropdown(["GPT", "Claude"], label="Select model", value="GPT")
        convert = gr.Button("Convert code")

    convert.click(optimize, inputs=[python, model], outputs=[cpp])

ui.launch(inbrowser=True)
def execute_python(code):
        try:
            output = io.StringIO()
            sys.stdout = output
            exec(code)
        finally:
            sys.stdout = sys.__stdout__
        return output.getvalue()
def execute_cpp(code):
        write_output(code)
        try:
            compile_cmd = ["clang++", "-Ofast", "-std=c++17", "-march=armv8.5-a", "-mtune=apple-m1", "-mcpu=apple-m1", "-o", "optimized", "optimized.cpp"]
            compile_result = subprocess.run(compile_cmd, check=True, text=True, capture_output=True)
            run_cmd = ["./optimized"]
            run_result = subprocess.run(run_cmd, check=True, text=True, capture_output=True)
            return run_result.stdout
        except subprocess.CalledProcessError as e:
            return f"An error occurred:\n{e.stderr}"
def clean_css(css_string):
    import re

    
    lines = css_string.strip().splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if ':' in line and not line.endswith(';'):
            line += ';'
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)
with gr.Blocks(css=css) as ui:
    gr.Markdown("## Convert code from Python to C++")
    with gr.Row():
        python = gr.Textbox(label="Python code:", value=python_hard, lines=10)
        cpp = gr.Textbox(label="C++ code:", lines=10)
    with gr.Row():
        model = gr.Dropdown(["GPT", "Claude"], label="Select model", value="GPT")
    with gr.Row():
        convert = gr.Button("Convert code")
    with gr.Row():
        python_run = gr.Button("Run Python")
        cpp_run = gr.Button("Run C++")
    with gr.Row():
        python_out = gr.TextArea(label="Python result:", elem_classes=["python"])
        cpp_out = gr.TextArea(label="C++ result:", elem_classes=["cpp"])

    convert.click(optimize, inputs=[python, model], outputs=[cpp])
    python_run.click(execute_python, inputs=[python], outputs=[python_out])
    cpp_run.click(execute_cpp, inputs=[cpp], outputs=[cpp_out])

ui.launch(inbrowser=True)
from huggingface_hub import login, InferenceClient
from transformers import AutoTokenizer
hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)
code_qwen = "Qwen/CodeQwen1.5-7B-Chat"
code_gemma = "google/codegemma-7b-it"
CODE_QWEN_URL = "https://h1vdol7jxhje3mpn.us-east-1.aws.endpoints.huggingface.cloud"
CODE_GEMMA_URL = "https://c5hggiyqachmgnqg.us-east-1.aws.endpoints.huggingface.cloud"
tokenizer = AutoTokenizer.from_pretrained(code_qwen)
messages = messages_for(pi)
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
client = InferenceClient(CODE_QWEN_URL, token=hf_token)
stream = client.text_generation(text, stream=True, details=True, max_new_tokens=3000)
for r in stream:
    print(r.token.text, end = "")
def stream_code_qwen(python):
    tokenizer = AutoTokenizer.from_pretrained(code_qwen)
    messages = messages_for(python)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    client = InferenceClient(CODE_QWEN_URL, token=hf_token)
    stream = client.text_generation(text, stream=True, details=True, max_new_tokens=3000)
    result = ""
    for r in stream:
        result += r.token.text
        yield result    
def optimize(python, model):
    if model=="GPT":
        result = stream_gpt(python)
    elif model=="Claude":
        result = stream_claude(python)
    elif model=="CodeQwen":
        result = stream_code_qwen(python)
    else:
        raise ValueError("Unknown model")
    for stream_so_far in result:
        yield stream_so_far    
with gr.Blocks(css=css) as ui:
    gr.Markdown("## Convert code from Python to C++")
    with gr.Row():
        python = gr.Textbox(label="Python code:", value=python_hard, lines=10)
        cpp = gr.Textbox(label="C++ code:", lines=10)
    with gr.Row():
        model = gr.Dropdown(["GPT", "Claude", "CodeQwen"], label="Select model", value="GPT")
    with gr.Row():
        convert = gr.Button("Convert code")
    with gr.Row():
        python_run = gr.Button("Run Python")
        cpp_run = gr.Button("Run C++")
    with gr.Row():
        python_out = gr.TextArea(label="Python result:", elem_classes=["python"])
        cpp_out = gr.TextArea(label="C++ result:", elem_classes=["cpp"])

    convert.click(optimize, inputs=[python, model], outputs=[cpp])
    python_run.click(execute_python, inputs=[python], outputs=[python_out])
    cpp_run.click(execute_cpp, inputs=[cpp], outputs=[cpp_out])

ui.launch(inbrowser=True)
                    
        
                                    