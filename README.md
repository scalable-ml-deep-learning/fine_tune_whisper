# Fine-tune Whisper

https://www.youtube.com/watch?v=xhWhyu8cBTk
https://www.youtube.com/watch?v=C6Vw_Z3t_2U

Contributors:
<a href="https://github.com/Bralli99">Brando Chiminelli</a>, 
<a href="https://github.com/boyscout99">Tommaso Praturlon</a>

Course: <a href="https://id2223kth.github.io/">Scalable Machine Learning and Deep Learning</a>, at <a href="https://www.kth.se/en">KTH Royal Institute of Technology</a>

## About

In this project we fine-tune <a href="https://huggingface.co/openai/whisper-small">OpenAI's Whisper model</a> for italian automatic speech recognition (ASR). We do that by using the <a href="https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0">Common Voice dataset</a>, the italian subset and by leveraging Hugging Face 🤗 Transformers. 

Whisper is a transformer based encoder-decoder model...

## Implementation

The implementation consists in a feature engineering pipeline, a training pipeline and an inference program (Hugging Face Space). This is done to run the feature engineering on CPUs and the training pipeline on GPUs.
The training pipeline exploits the free GPU given by Google Colaboratory, the problem with that is double: getting a good GPU is rare and the GPU is given for a maximum of 12 hours (not guaranteed, could be even less).
 
1. We wrote a feature engineering pipeline that loads the datset and takes 10% of the Italian ASR dataset since it's already 16 GB of data and more than that wouldn't fit in the VM's disk on Colab. We take both the train and test split to run some evaluation tests. Then, we load the Whisper Feature Extractor that pads the audio input to 30s and converts the inputs to Log-Mel spectograms. We also load the Whisper Tokenizer which post-processes the model output to text format, our labels.

2. We wrote a training pipeline ...

3. We wrote a Gradio application, our inference program, ...

## Spaces on Hugging Face

### Whisper Italian - Spaces
https://huggingface.co/spaces/whispy/Whisper-New-Model

## Built With

* [Colab](https://colab.research.google.com/)
* [Hugging Face](https://huggingface.co/)

