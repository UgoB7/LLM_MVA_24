# step3_refine_t2v_prompt.py
from transformers import pipeline

def refine_t2v_prompt(original_prompt, physical_info, mismatch_evaluation, show=False):
    """
    Raffine le prompt T2V en intégrant les règles physiques (physical_info) et la résolution du mismatch (mismatch_evaluation)
    via du step-back prompting.
    
    Paramètres:
      - original_prompt: Le prompt T2V initial.
      - physical_info: Un dictionnaire contenant les clés "main_object" et "physical_law" extraites en Step 1.
      - mismatch_evaluation: Le texte généré en Step 2 évaluant le décalage entre le prompt utilisateur et la vidéo.
      - show: Si True, retourne également le texte complet généré par le modèle.
    
    Retourne:
      - refined_prompt: Le prompt T2V raffiné.
    """
    prompt = f"""You are a video generation expert. Your task is to refine a given text-to-video (T2V) prompt by incorporating the physical rules derived from Step 1 and resolving the mismatches identified in Step 2 through a step-back prompting approach.
    
Original T2V Prompt:
{original_prompt}

Physical Rules (Step 1):
Main Object: {physical_info.get("main_object", "N/A")}
Physical Law: {physical_info.get("physical_law", "N/A")}

Mismatch Evaluation (Step 2):
{mismatch_evaluation}

Please refine the original T2V prompt so that it better reflects the intended physical rules and resolves the mismatch issues. Provide a refined prompt that is clear, detailed, and coherent with the physical laws.
Let's think step by step and produce the final refined T2V prompt."""
    
    # Utilisation du pipeline pour générer la réponse
    eval_pipeline = pipeline("text2text-generation", model="google/flan-t5-xl", device=0)
    result = eval_pipeline(prompt, max_length=768, do_sample=False, temperature=1.0)
    generated_text = result[0]['generated_text'].strip()
    
    refined_prompt = generated_text
    if show:
        return {"refined_prompt": refined_prompt, "full_output": generated_text}
    else:
        return refined_prompt

if __name__ == '__main__':
    original_prompt = "Generate a video showing a cup of coffee cooling down on a table with rising steam."
    physical_info = {
        "main_object": "coffee cup",
        "physical_law": "The coffee cup should follow the physical laws of thermodynamics, heat dissipation, and natural convection; steam rising should reflect evaporation and cooling dynamics."
    }
    mismatch_evaluation = ("The original video caption focused solely on a static coffee cup with minimal steam, "
                           "omitting details on heat transfer and evaporation. It did not capture the dynamic cooling process "
                           "and natural convection that the prompt implies.")
    
    refined = refine_t2v_prompt(original_prompt, physical_info, mismatch_evaluation, show=True)
    print("Refined T2V Prompt:")
    print(refined["refined_prompt"])
