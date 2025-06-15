import os
import sys
import io
import subprocess
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from huggingface_hub import login, InferenceClient
from transformers import AutoTokenizer
import gradio as gr

load_dotenv()
openai = OpenAI()
claude = anthropic.Anthropic()

hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)

OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
CODE_QWEN_URL = "https://h1vdol7jxhje3mpn.us-east-1.aws.endpoints.huggingface.cloud"
code_qwen = "Qwen/CodeQwen1.5-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(code_qwen)

system_message = "You are an assistant that rewrites Python into C++ as efficiently as possible. Output C++ only."

def user_prompt_for(python):
    return f"Convert this Python code to high performance C++:\n\n{python}"

def messages_for(python):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt_for(python)}
    ]

def write_output(cpp_code):
    cpp_clean = cpp_code.replace("```cpp", "").replace("```", "")
    with open("optimized.cpp", "w") as f:
        f.write(cpp_clean)

def stream_gpt(python):
    stream = openai.chat.completions.create(
        model=OPENAI_MODEL, messages=messages_for(python), stream=True
    )
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        yield reply.replace("```cpp\n", "").replace("```", "")

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
            yield reply.replace("```cpp\n", "").replace("```", "")

def stream_code_qwen(python):
    messages = messages_for(python)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    client = InferenceClient(CODE_QWEN_URL, token=hf_token)
    result = ""
    for r in client.text_generation(prompt, stream=True, details=True, max_new_tokens=3000):
        result += r.token.text
        yield result.replace("```cpp\n", "").replace("```", "")

def optimize(python_code, model):
    if model == "GPT":
        stream = stream_gpt(python_code)
    elif model == "Claude":
        stream = stream_claude(python_code)
    elif model == "CodeQwen":
        stream = stream_code_qwen(python_code)
    else:
        raise ValueError("Unsupported model")

    final_code = ""
    for chunk in stream:
        final_code = chunk
        yield chunk
    write_output(final_code)

def execute_python(code):
    try:
        output = io.StringIO()
        sys.stdout = output
        exec(code)
        return output.getvalue()
    finally:
        sys.stdout = sys.__stdout__

def execute_cpp(_):
    try:
        subprocess.run(
            ["clang++", "-Ofast", "-std=c++17", "-o", "optimized", "optimized.cpp"],
            check=True, capture_output=True
        )
        run_result = subprocess.run(["./optimized"], check=True, capture_output=True, text=True)
        return run_result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error:\n{e.stderr}"

with gr.Blocks() as ui:
    gr.Markdown("### Python to C++ Code Optimizer")
    with gr.Row():
        python = gr.Textbox(label="Python Code", lines=10)
        cpp = gr.Textbox(label="C++ Output", lines=10)
    with gr.Row():
        model = gr.Dropdown(["GPT", "Claude", "CodeQwen"], value="GPT", label="Choose Model")
        convert = gr.Button("Convert to C++")
    with gr.Row():
        python_out = gr.TextArea(label="Python Output")
        cpp_out = gr.TextArea(label="C++ Execution Output")
    with gr.Row():
        run_py = gr.Button("Run Python")
        run_cpp = gr.Button("Run C++")

    convert.click(optimize, inputs=[python, model], outputs=[cpp])
    run_py.click(execute_python, inputs=[python], outputs=[python_out])
    run_cpp.click(execute_cpp, inputs=[cpp], outputs=[cpp_out])

ui.launch(inbrowser=True)
