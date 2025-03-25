# step2_video_captioning.py
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import cv2
from PIL import Image
import torch

def extract_frames(video_path, sample_rate=1):
    """
    Extrait une frame toutes les `sample_rate` secondes de la vidéo.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo.")
        return frames

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Impossible de déterminer le fps de la vidéo.")
        return frames

    frame_interval = int(fps * sample_rate)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            # Conversion de BGR (OpenCV) en RGB (Pillow)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            frames.append(image)
        frame_count += 1

    cap.release()
    return frames

def generate_video_captions(video_path):
    frames = extract_frames(video_path, sample_rate=1)
    if not frames:
        return "Aucune frame extraite."
    
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    captions = []
    for image in frames:
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        captions.append(caption)
    
    return captions

if __name__ == '__main__':
    video_file = "/home/onyxia/work/LLM/A_car_brush_lathers_shampoo_onto_a_dirty_vehicle_1.mp4"
    captions = generate_video_captions(video_file)
    if isinstance(captions, list):
        joined_caption = "\n".join(captions)
    else:
        joined_caption = captions
    print("Video Captions:")
    print(joined_caption)
