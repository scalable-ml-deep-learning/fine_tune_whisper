import torch

import gradio as gr
import pytube as pt
from transformers import pipeline

asr = pipeline(
    task="automatic-speech-recognition",
    model="whispy/whisper_hf",
    chunk_length_s=30,
    device="cpu",
)

summarizer = pipeline(
    "summarization",
    model="it5/it5-efficient-small-el32-news-summarization",
)

translator = pipeline(
    "translation", 
    model="Helsinki-NLP/opus-mt-it-en")

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

    text = asr(file)["text"]

    translate = translator(text)
    translate = translate[0]["translation_text"]

    return warn_output + text, translate

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

    text = asr("audio.mp3")["text"]

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
    outputs=[
             gr.Textbox(label="Transcribed text"),
             gr.Textbox(label="Translated text"),
    ],
    layout="horizontal",
    theme="huggingface",
    title="Whisper Demo: Transcribe and Translate Italian Audio",
    description=(
        "Transcribe and Translate long-form microphone or audio inputs with the click of a button! Demo uses the the fine-tuned"
        f" [whispy/whisper_hf](https://huggingface.co/whispy/whisper_hf) and 🤗 Transformers to transcribe audio files"
        " of arbitrary length. It also uses another model for the translation."
    ),
    allow_flagging="never",
)

yt_transcribe = gr.Interface(
    fn=yt_transcribe,
    inputs=[gr.inputs.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL")],
    outputs=["html",
             gr.Textbox(label="Transcribed text"),
             gr.Textbox(label="Summarized text"),
             gr.Textbox(label="Translated text"),
    ],
    layout="horizontal",
    theme="huggingface",
    title="Whisper Demo: Transcribe, Summarize and Translate YouTube",
    description=(
        "Transcribe, Summarize and Translate long-form YouTube videos with the click of a button! Demo uses the the fine-tuned "
        f" [whispy/whisper_hf](https://huggingface.co/whispy/whisper_hf) and 🤗 Transformers to transcribe audio files of"
        " arbitrary length. It also uses other two models to first summarize and then translate the text input. You can try with the following examples: " 
        f" [Video1](https://www.youtube.com/watch?v=xhWhyu8cBTk)"
        f" [Video2](https://www.youtube.com/watch?v=C6Vw_Z3t_2U)"
    ),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mf_transcribe, yt_transcribe], ["Transcribe and Translate Audio", "Transcribe, Summarize and Translate YouTube"])

demo.launch(enable_queue=True)
