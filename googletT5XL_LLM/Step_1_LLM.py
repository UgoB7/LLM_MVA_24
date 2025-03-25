from transformers import pipeline
import json
import re

generator = pipeline("text2text-generation", model="google/flan-t5-xl", device=0)

def extract_physical_info(user_prompt, show=False):
    prompt = f"""You are a physics expert. Your task is to identify the main object in the given user prompt and provide the physical laws in reality the main object should obey with as much detail as possible in a descriptive way without giving formulas. Please think step by step and provide a thorough explanation, covering at least three distinct physical aspects.
Include at least three detailed points regarding the physical laws involved.
Answer strictly using the following format:
main_object: your answer here | physical_law: your detailed description here

Do NOT add any extra text, punctuation, or formatting besides what is requested.

Example:
Input: "a rubber ball hits the ground and then bounces up"
Output: main_object: rubber ball | physical_law: The primary physical laws that should be obeyed include: 1. Newton's Law of Motion to describe the acceleration and deceleration; 2. Conservation of Energy to explain the energy transfer during the bounce; 3. Gravitational acceleration affecting the fall and rebound.

### Current Task:
User prompt: {user_prompt}
Output:"""

    # Adjusting generation settings: enabling sampling and setting a moderate temperature.
    result = generator(prompt, max_length=768, do_sample=False, temperature=1.0)
    generated_text = result[0]['generated_text'].strip()

    # Split based on the separator "|" #JSON worked
    try:
        main_object_part, physical_law_part = generated_text.split('|')
        main_object = main_object_part.replace('main_object:', '').strip()
        physical_law = physical_law_part.replace('physical_law:', '').strip()

        parsed_output = {
            "main_object": main_object,
            "physical_law": physical_law
        }
        print("Extraction réussie.")
    except Exception as e:
        print("Parsing error:", e)
        parsed_output = None

    if show:
        return {"parsed_output": parsed_output, "full_output": generated_text}
    else:
        return parsed_output


if __name__ == '__main__':
    user_prompt = "A hot cup of coffee is placed on a wooden table near an open window, and steam slowly rises as the cup cools down."
    output = extract_physical_info(user_prompt, show=True)
    print("Extracted Information:")
    print(json.dumps(output, indent=2))
