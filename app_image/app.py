import torch
import gradio as gr
import pytube as pt
from transformers import pipeline
from diffusers import StableDiffusionPipeline


MODEL_NAME = "whispy/whisper_italian"

summarizer = pipeline(
    "summarization",
    model="it5/it5-efficient-small-el32-news-summarization",
)

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device="cpu",
)

YOUR_TOKEN="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"
image_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-it-en")

def transcribe(microphone, file_upload):
    warn_output = ""
    if (microphone is not None) and (file_upload is not None):
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )

    elif (microphone is None) and (file_upload is None):
        return "ERROR: You have to either use the microphone or upload an audio file"

    file = microphone if microphone is not None else file_upload

    text = pipe(file)["text"]

    translate = translator(text)
    translate = translate[0]["translation_text"]

    image = image_pipe(translate)["sample"][0]

    return warn_output + text, translate, image


def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str


def yt_transcribe(yt_url):
    yt = pt.YouTube(yt_url)
    html_embed_str = _return_yt_html_embed(yt_url)
    stream = yt.streams.filter(only_audio=True)[0]
    stream.download(filename="audio.mp3")

    text = pipe("audio.mp3")["text"]

    summary = summarizer(text)
    summary = summary[0]["summary_text"]
      
    translate = translator(summary)
    translate = translate[0]["translation_text"]

    return html_embed_str, text, summary, translate

demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath", optional=True),
        gr.inputs.Audio(source="upload", type="filepath", optional=True),
    ],
    outputs=["text", "text", "image"],
    layout="horizontal",
    theme="huggingface",
    title="Whisper Demo: Transcribe Audio",
    description=(
        "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the the fine-tuned"
        f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
        " of arbitrary length."
    ),
    allow_flagging="never",
)

yt_transcribe = gr.Interface(
    fn=yt_transcribe,
    inputs=[gr.inputs.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL")],
    outputs=["html", "text", "text", "text"],
    layout="horizontal",
    theme="huggingface",
    title="Whisper Demo: Transcribe YouTube",
    description=(
        "Transcribe long-form YouTube videos with the click of a button! Demo uses the the fine-tuned checkpoint:"
        f" [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files of"
        " arbitrary length."
    ),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mf_transcribe, yt_transcribe], ["Transcribe Audio", "Transcribe YouTube"])

demo.launch(enable_queue=True)