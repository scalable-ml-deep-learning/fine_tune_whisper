import torch
import gradio as gr
import pytube as pt
from transformers import pipeline
from diffusers import StableDiffusionPipeline


MODEL_NAME = "whispy/whisper_italian"
YOUR_TOKEN="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"
# whisper model fine-tuned for italian
speech_ppl = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device="cpu"
    )
# model summarizing text
summarizer_ppl = pipeline(
    "summarization",
    model="it5/it5-efficient-small-el32-news-summarization"
    )
# model translating text from Italian to English
translator_ppl = pipeline(
    "translation", 
    model="Helsinki-NLP/opus-mt-it-en"
    )
# model producing an image from text
image_ppl = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)

#def transcribe(microphone, file_upload):
def transcribe(microphone):
    warn_output = ""
#    if (microphone is not None) and (file_upload is not None):
    if (microphone is not None):
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )

#    elif (microphone is None) and (file_upload is None):
    elif (microphone is None):
        return "ERROR: You have to either use the microphone or upload an audio file"

#    file = microphone if microphone is not None else file_upload
    file = microphone

    text = speech_ppl(file)["text"]
    print("Text: ", text)
    translate = translator_ppl(text)
    print("Translate: ", translate)
    translate = translate[0]["translation_text"]
    print("Translate 2: ", translate)
    print("Building image .....")
    #image = image_ppl(translate).images[0]
    #image = image_ppl(translate, num_inference_steps=15)["sample"]
    #prompt = "a photograph of an astronaut riding a horse"
    image = image_ppl(translate, num_inference_steps=15)
    print("Image output: ", image)
    print("Image: ", image.images)
    #image.save("text-to-image.png")

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

#demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath", optional=True),
        #gr.inputs.Audio(source="upload", type="filepath", optional=True),
    ],
    outputs=[gr.Textbox(label="Transcribed text"),
             gr.Textbox(label="Summarized text"),
             gr.Image(type="pil", label="Output image")],
    layout="horizontal",
    theme="huggingface",
    title="Whisper Demo: Transcribe Audio to Image",
    description=(
        "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the the fine-tuned"
        f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
        " of arbitrary length."
    ),
    allow_flagging="never",
)
'''
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
'''
'''
with demo:
    #gr.TabbedInterface([mf_transcribe, yt_transcribe], ["Transcribe Audio", "Transcribe YouTube"])
    gr.TabbedInterface(mf_transcribe, "Transcribe Audio to Image")

demo.launch(enable_queue=True)
'''
mf_transcribe.launch(enable_queue=True)
