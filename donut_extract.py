from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json

def extract_text_donut(image_path):
    image = Image.open(image_path).convert("RGB")

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

    task_prompt = "<s_docvqa><s_question>Extract information</s_question><s_answer>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
    generated = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    with open("donut_output.txt", "w", encoding="utf-8") as f:
        f.write(generated)
    
    return generated

if __name__ == "__main__":
    result = extract_text_donut("hackathon_input_image/input_image.jpeg")
    print("Donut output saved.")
