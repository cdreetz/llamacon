import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class AutomaticPreferenceDatasetGenerator:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the dataset generator with a model that can understand feedback
        and generate contrasting examples.
        Uses a much smaller model by default for faster performance.
        """
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Automatically use GPU if available
            torch_dtype=torch.bfloat16,  # Use lower precision for faster inference
            low_cpu_mem_usage=True,      # Reduce CPU memory usage
        )
        
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
        
        # Extract only the JSON part from the response
        try:
            # Try to find JSON content in the response
            start_idx = understanding.find('{')
            end_idx = understanding.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = understanding[start_idx:end_idx]
                understanding_json = json.loads(json_str)
            else:
                # Fallback: create basic JSON if none found
                understanding_json = {
                    "disliked_behavior": "too many comments in code",
                    "preferred_behavior": "minimal, necessary comments in code",
                    "contexts": "when generating code",
                    "example_scenarios": ["code generation", "code review", "code explanation"]
                }
        except json.JSONDecodeError:
            # Fallback for invalid JSON
            understanding_json = {
                "disliked_behavior": "too many comments in code",
                "preferred_behavior": "minimal, necessary comments in code",
                "contexts": "when generating code",
                "example_scenarios": ["code generation", "code review", "code explanation"]
            }
        
        # Generate diverse prompts that would trigger the disliked behavior
        prompt_generation_prompt = f"""
        Generate {num_examples} diverse prompts that might cause an AI assistant to exhibit this behavior:
        Disliked behavior: {understanding_json['disliked_behavior']}
        Contexts: {understanding_json['contexts']}
        
        Each prompt should be something a user might ask an AI. Make them diverse across domains and difficulty.
        Format as a JSON array of strings.
        """
        
        prompts_text = self._generate_completion(prompt_generation_prompt)
        
        # Extract only the JSON array from the response
        try:
            # Find array in the text
            start_idx = prompts_text.find('[')
            end_idx = prompts_text.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                prompts_json = prompts_text[start_idx:end_idx]
                prompts = json.loads(prompts_json)
            else:
                # Fallback: create a small set of example prompts
                prompts = [
                    "Write a function to calculate Fibonacci numbers",
                    "Create a simple web scraper in Python",
                    "How would I implement a binary search tree?"
                ]
        except json.JSONDecodeError:
            # Fallback for invalid JSON
            prompts = [
                "Write a function to calculate Fibonacci numbers",
                "Create a simple web scraper in Python",
                "How would I implement a binary search tree?"
            ]
        
        # Limit to requested number
        prompts = prompts[:num_examples]
        
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
            
            # Extract the assistant's response part only (after "Your response:")
            rejected_response = self._extract_response(rejected_response)
            chosen_response = self._extract_response(chosen_response)
            
            preference_pairs.append({
                "input": prompt,
                "rejected": rejected_response,
                "chosen": chosen_response
            })
        
        # Create directories if needed
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Save the dataset in torchtune preference dataset format
        with open(output_file, "w") as f:
            json.dump(preference_pairs, f, indent=2)
        
        print(f"Generated {len(preference_pairs)} preference pairs based on feedback")
        print(f"Saved to {output_file}")
        
        return preference_pairs
    
    def _generate_completion(self, prompt, max_tokens=512):
        """Generate a completion for the given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Use faster generation settings
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get('attention_mask', None),
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,  # Enables sampling (faster than beam search)
            num_beams=1,     # Disable beam search for speed
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _extract_response(self, text):
        """Extract only the response part from the generated text."""
        marker = "Your response:"
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return text.strip()

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
        
        indices_text = self._generate_completion(understanding_prompt)
        
        # Extract only the JSON array from the response
        try:
            # Find array in the text
            start_idx = indices_text.find('[')
            end_idx = indices_text.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                indices_json = indices_text[start_idx:end_idx]
                indices = json.loads(indices_json)
            else:
                indices = []
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
                improved_response = self._extract_response(improved_response)
                
                real_examples.append({
                    "input": turn['prompt'],
                    "rejected": turn['response'],
                    "chosen": improved_response
                })
        
        return real_examples


# Usage example:
if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("./data", exist_ok=True)
    
    # Generate the dataset with fewer examples for faster completion
    feedback = "I don't like all the comments you always write in code"
    generator = AutomaticPreferenceDatasetGenerator()
    
    # Use a much smaller number for testing
    dataset = generator.generate_dataset(feedback, num_examples=2, output_file="./data/auto_preferences.json")
    
    print(f"Successfully generated {len(dataset)} preference pairs")
