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
We apply both the feature extractor and the tokenizer to our dataset to prepare for the traning. Finally we create a zip of our dataset and save it on Google Drive, to be able in the next pipeline to load the data ready for the training, without going through all the previous steps.

2. We wrote a training pipeline that loads our dataset, consisting in 15261 rows for the training set and 1500 rows for the testing set, loads a pre-trained Whisper checkpoint <a href="https://huggingface.co/openai/whisper-small">Whisper small</a> and runs the training and evaluations to verify that we have correctly trained it to transcribe speech in Italian.
The evaluation uses the word error rate (WER) metric, the 'de-facto' metric for assessing ASR systems.
The crucial part of the training pipeline is defining the parameters and hyperparameters; our choices and possible improvements are explained in the next section.

3. We wrote a Gradio application, our inference program, 

## Training parameters and possible improvements
-define parameters (+checkpoints on google drive)
-improvements

The most relevant parameters we used are:
- `output_dir="/content/drive/MyDrive/whisper_hf"`: model predictions and checkpoints will be written on Google Drive to be able to recover them and restart the training from the latest checkpoint in case Colab takes away our resources.
- `evaluation_strategy="steps"`: the evaluation is done (and logged) every `eval_steps=100` so that every 100 step of our training (1 steps has a batch size of 16) we compute the WER metric. 
- `save_strategy="steps"`: our checkpoints are also saved every `eval_steps=100` and by setting `save_total_limit=2` we limit the total amount of checkpoints deleting the older checkpoints in our folder on Google Drive.

## Spaces on Hugging Face

### Whisper Italian - Spaces
https://huggingface.co/spaces/whispy/Whisper-New-Model

## Built With

* [Colab](https://colab.research.google.com/)
* [Hugging Face](https://huggingface.co/)

