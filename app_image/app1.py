import gradio as gr
import torch
import whisper
from diffusers import DiffusionPipeline
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

#import os
#MY_SECRET_TOKEN=os.environ.get('HF_TOKEN_SD')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained("whispy/whisper_italian").to(device)
processor = WhisperProcessor.from_pretrained("whispy/whisper_italian")

diffuser_pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    custom_pipeline="speech_to_image_diffusion",
    speech_model=model,
    speech_processor=processor,
    use_auth_token=MY_SECRET_TOKEN,
    revision="fp16",
    torch_dtype=torch.float16,
)

diffuser_pipeline.enable_attention_slicing()
diffuser_pipeline = diffuser_pipeline.to(device)


#————————————————————————————————————————————
# GRADIO SETUP
title = "Speech to Diffusion • Community Pipeline"
description = """
<p style='text-align: center;'>This demo can generate an image from an audio sample using pre-trained OpenAI whisper-small and Stable Diffusion.<br />
Community examples consist of both inference and training examples that have been added by the community.<br />
<a href='https://github.com/huggingface/diffusers/tree/main/examples/community#speech-to-image' target='_blank'> Click here for more information about community pipelines </a>
</p>
"""
article = """
<p style='text-align: center;'>Community pipeline by Mikail Duzenli • Gradio demo by Sylvain Filoni & Ahsen Khaliq<p>
"""
audio_input = gr.Audio(source="microphone", type="filepath")
image_output = gr.Image()

def speech_to_text(audio_sample):
  
  process_audio = whisper.load_audio(audio_sample)
  output = diffuser_pipeline(process_audio)
 
  print(f"""
  ————————
  output: {output}
  ————————
  """)
  
  return output.images[0]

demo = gr.Interface(fn=speech_to_text, inputs=audio_input, outputs=image_output, title=title, description=description, article=article)
demo.launch()