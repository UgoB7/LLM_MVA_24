from transformers import pipeline

def evaluate_mismatch(user_prompt, video_caption):
    prompt = f"""You are a physics expert. Provide you a user prompt used as an input to a video generation model and a caption of the generated video of the model based on the prompt. The video content should follow the user prompt. Your task is summarizing what the video content described by caption mismatch the user prompt. Some in-context examples are provided for your reference and you need to finish the current task.

### In-context example
User prompt: A side view of a small red rubber ball dropping from the top of the view, hit the ground at the bottom of the view and bounce up. 
Video caption: The rubber ball is rolling from left to right across a flat surface with a gradient background. The ball's motion is consistent and smooth, obeying the laws of motion and gravity. The ball undergoes a slight deformation as it rolls, with its shape becoming slightly elongated due to the force of gravity and friction. The ball's deformation is limited and quickly recovers as it continues to roll. The ball's movement is determined by the force of inertia and the friction between the ball and the surface. The ball's trajectory is a result of the balance between these forces. 
Mismatch: Vertical vs. Horizontal Motion: The user prompt describes a red rubber ball dropping vertically from the top of the view, hitting the ground at the bottom, and then bouncing up—motion along the vertical axis driven by gravity. In contrast, the video caption depicts the ball rolling horizontally from left to right across a flat surface, ignoring the vertical dropping and bouncing specified in the prompt. Absence of Bouncing: A crucial element in the user prompt is the ball hitting the ground and bouncing up, involving an elastic collision and energy transformation. The video caption omits any mention of the ball bouncing, focusing instead on the ball's continuous horizontal rolling motion. Unrealistic Deformation Due to Rolling: The video caption mentions the ball undergoing a slight deformation as it rolls, becoming slightly elongated due to the forces of gravity and friction. In reality, a rubber ball rolling smoothly on a flat surface would not experience noticeable deformation causing elongation. Deformation in a rubber ball typically occurs upon impact (as in bouncing), not during steady rolling motion. Neglecting Gravity's Role in Vertical Motion: The user prompt relies on gravity's fundamental role in causing the ball to drop and bounce along the vertical axis. The video caption mentions gravity but applies it to horizontal motion, neglecting its role in vertical movement as specified in the prompt. Misapplication of Physical Laws: The video caption attributes the ball's movement and slight deformation to the force of inertia and friction during horizontal rolling. This misapplies physical laws, as friction in horizontal rolling would not cause significant deformation or affect the ball's trajectory substantially. The prompt's scenario involves vertical motion under gravity and elastic collision upon bouncing, not horizontal motion influenced primarily by friction.

### Current task:
User prompt: {user_prompt}
Video caption: {video_caption}
Let's think step by step."""
    
    eval_pipeline = pipeline("text2text-generation", model="google/flan-t5-xl", device=0)
    result = eval_pipeline(prompt, max_length=768, do_sample=False, temperature=1.0)
    generated_text = result[0]['generated_text'].strip()
    return generated_text

if __name__ == '__main__':
    user_prompt = "A hot cup of coffee is placed on a wooden table near an open window, and steam slowly rises as the cup cools down."
    video_caption = "The coffee cup is shown on a table with steam rising, but the scene focuses only on the object with no dynamic physical interactions."
    mismatch_evaluation = evaluate_mismatch(user_prompt, video_caption)
    print("Mismatch Evaluation:")
    print(mismatch_evaluation)
