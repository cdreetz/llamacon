import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class AutomaticPreferenceDatasetGenerator:
    def __init__(self, model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct"):
        """
        Initialize the dataset generator with a model that can understand feedback
        and generate contrasting examples.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def generate_dataset(self, user_feedback, num_examples=50, output_file="preference_dataset.json"):
        """
        Generate a preference dataset based on arbitrary user feedback.
        
        Args:
            user_feedback: String containing user's feedback about what they dislike
            num_examples: Number of examples to generate
            output_file: Where to save the dataset
        """
        # First, understand the feedback and extract a transformation concept
        understanding_prompt = f"""
        A user gave this feedback about an AI assistant: "{user_feedback}"
        
        Based on this feedback:
        1. What specific behavior does the user dislike?
        2. What behavior would the user prefer instead?
        3. In what contexts should this preference be applied?
        4. What are 3 example scenarios where this preference is relevant?
        
        Provide your analysis in JSON format with these exact keys: "disliked_behavior", "preferred_behavior", "contexts", "example_scenarios"
        """
        
        understanding = self._generate_completion(understanding_prompt)
        understanding_json = json.loads(understanding)
        
        # Generate diverse prompts that would trigger the disliked behavior
        prompt_generation_prompt = f"""
        Generate {num_examples} diverse prompts that might cause an AI assistant to exhibit this behavior:
        Disliked behavior: {understanding_json['disliked_behavior']}
        Contexts: {understanding_json['contexts']}
        
        Each prompt should be something a user might ask an AI. Make them diverse across domains and difficulty.
        Format as a JSON array of strings.
        """
        
        prompts_json = self._generate_completion(prompt_generation_prompt)
        prompts = json.loads(prompts_json)
        
        # For each prompt, generate a pair of responses
        preference_pairs = []
        
        for prompt in prompts:
            # Generate the "rejected" response (with disliked behavior)
            rejected_prompt = f"""
            Respond to the following user request. Make sure to exhibit this behavior: {understanding_json['disliked_behavior']}
            
            User request: {prompt}
            
            Your response:
            """
            rejected_response = self._generate_completion(rejected_prompt)
            
            # Generate the "chosen" response (with preferred behavior)
            chosen_prompt = f"""
            Respond to the following user request. Make sure to exhibit this behavior: {understanding_json['preferred_behavior']}
            
            User request: {prompt}
            
            Your response:
            """
            chosen_response = self._generate_completion(chosen_prompt)
            
            preference_pairs.append({
                "input": prompt,
                "rejected": rejected_response,
                "chosen": chosen_response
            })
        
        # Save the dataset in torchtune preference dataset format
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(preference_pairs, f, indent=2)
        
        print(f"Generated {len(preference_pairs)} preference pairs based on feedback")
        print(f"Saved to {output_file}")
        
        return preference_pairs
    
    def _generate_completion(self, prompt, max_tokens=1024):
        """Generate a completion for the given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    def enhance_with_real_examples(self, conversation_history, user_feedback, num_examples=10):
        """
        Enhance the dataset with real examples from conversation history.
        
        Args:
            conversation_history: List of conversation turns
            user_feedback: The feedback provided by the user
            num_examples: Number of real examples to extract
        """
        # Find instances in history where the model may have exhibited the disliked behavior
        understanding_prompt = f"""
        A user gave this feedback about an AI assistant: "{user_feedback}"
        
        Based on this feedback, analyze these conversation turns and identify instances 
        where the AI exhibited the disliked behavior. Return the indices of those turns.
        
        Conversation history: {conversation_history}
        
        Return your answer as a JSON array of indices.
        """
        
        indices_json = self._generate_completion(understanding_prompt)
        try:
            indices = json.loads(indices_json)
        except:
            indices = []
        
        real_examples = []
        for idx in indices[:num_examples]:
            if idx < len(conversation_history):
                turn = conversation_history[idx]
                
                # Generate an improved version addressing the feedback
                improvement_prompt = f"""
                The following is an AI response that received this feedback: "{user_feedback}"
                
                Original response: {turn['response']}
                
                Create an improved version that addresses the feedback while maintaining
                the same information and helpfulness.
                """
                
                improved_response = self._generate_completion(improvement_prompt)
                
                real_examples.append({
                    "input": turn['prompt'],
                    "rejected": turn['response'],
                    "chosen": improved_response
                })
        
        return real_examples

# Usage:
# generator = AutomaticPreferenceDatasetGenerator()
# generator.generate_dataset("I don't like all the comments you always write in code")
#
#
#
# # Generate the dataset
feedback = "I don't like all the comments you always write in code"
generator = AutomaticPreferenceDatasetGenerator()
dataset = generator.generate_dataset(feedback, output_file="./data/auto_preferences.json")
 
 # Use torchtune with the dataset
 # In your YAML file:
# """
# dataset:
#   _component_: torchtune.datasets.preference_dataset
#   source: "json"
#   data_files: "./data/auto_preferences.json"
#   train_on_input: False
#   split: "train"
# """
