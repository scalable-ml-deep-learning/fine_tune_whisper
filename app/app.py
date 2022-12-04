from transformers import pipeline
import gradio as gr

pipe = pipeline(model="whispy/whisper_italian")

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Whisper Italian",
    description="Realtime demo for Italian speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()