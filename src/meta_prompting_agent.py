import os
import json
from openai import OpenAI
from utils import get_client


class MetaPromptingAgent:
    def __init__(self, model=None):
        self.client = get_client()
        self.model = model or "Llama-4-Maverick-17B-128E-Instruct-FP8"
        self.meta_prompt = ""

    def llama_chat(self, p, system_p=None, use_meta_prompt=False):
        messages = []

        if system_p and use_meta_prompt and self.meta_prompt:
            meta_system = system_p + "\n\n" + self.meta_prompt
            messages.append({"role":"system", "content": meta_system})
        elif use_meta_prompt and self.meta_prompt:
            messages.append({"role":"system", "content": self.meta_prompt})
        if system_p:
            messages.append({"role":"system", "content": system_p})

        messages.append({"role":"user", "content": p})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content

    def handle_feedback(self, user_prompt, assistant_response, user_feedback=None):
        p = f"User prompt: {user_prompt}\n\nAssistant response:{assistant_response}"

        if user_feedback:
            p += f"\n\nUser feedback: {user_feedback}"

        s = """
            You are a assistant tasked with taking in an instance 
            of a user input and corresponding assistant response where the 
            user indicated they didnt like the response. Your job is to think
            about the situation and determine what the problem was, what a better
            resposne would have been, and write an instruction that will result
            in the AI providing a better response in the future.

            IMPORTANT:
            The instruction should be as concice as possible and if possible 
            a simple one liner. If the case requires more detail then provide it,
            hile still being concise.

            Ex.
            User doesnt like a response where the assistant prefers to use Javascript but
            the user prefers Python.  You should write an instruction that says:
            '- User prefers Python over Javascript'

            IMPORTANT:
            The instruction you write will be added to the users meta prompt as
            is, so don't write any extra explanatory text, your output should 
            just be the instruction. Your output should literally start with a '- ' 
            and the instruction and nothing else.

            ONLY OUTPUT THE INSTRUCTION AND NOTHING ELSE.  START WITH THE DASH 
            AND STOP TALKING IMMEDIATELY WHEN THE INSTRUCTION IS WRITTEN.
            """

        instruction = self.llama_chat(p, s, use_meta_prompt=False)
        self.meta_prompt += f"\n{instruction}"

        return instruction

    def save_meta_prompt(self, filepath="meta_prompt.txt"):
        with open(filepath, 'w') as f:
            f.write(self.meta_prompt)

    def load_meta_prompt(self, filepath="meta_prompt.txt"):
        with open(filepath, 'r') as f:
            self.meta_prompt = f.read()

















if __name__ == "__main__":
    prompt = "Hello Llama! Can you give me a quick intro?"
    response = llama_chat(prompt)
    print(response)
