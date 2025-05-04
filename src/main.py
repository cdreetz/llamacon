import os
import json
from meta_prompting_agent import MetaPromptingAgent

def main():
    agent = MetaPromptingAgent()

    try:
        agent.load_meta_prompt()
        print("Loaded existing meta prompt")
    except:
        print("No existing meta prompt found. Starting fresh.")

    system_prompt = "You are a helpful but casual assistant. Please use short responses."

    print("\nWelcome to Meta Prompt Chat")
    print("Type 'exit' to quit, 'feedback' after a response to provide feedback")
    print("=========================")

    last_user_input = None
    last_assistant_response = None

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'exit':
            agent.save_meta_prompt()
            print("Meta prompt saved and chat ended.")
            break

        if user_input.lower() == 'feedback' and last_assistant_response:
            feedback = input("Please provide some feedback about the last response.").strip()

            print("\nGenerating instruction based on your feedback..")
            instruction = agent.handle_feedback(last_user_input, last_assistant_response, feedback)

            print(f"\nAdded instruction to meta prompt: \"{instruction}\"")
            continue

        last_user_input = user_input

        print("\nThinking..")
        response = agent.llama_chat(user_input, system_prompt, use_meta_prompt=True)

        last_assistant_response = response

        print(f"\nAssistant: {response}")
        print("\nType 'feedback' to provide feedback on this repsonse.")


if __name__ == "__main__":
    main()

