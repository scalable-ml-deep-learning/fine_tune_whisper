# Fine-tune Whisper

Contributors:
<a href="https://github.com/Bralli99">Brando Chiminelli</a>, 
<a href="https://github.com/boyscout99">Tommaso Praturlon</a>

Course: <a href="https://id2223kth.github.io/">Scalable Machine Learning and Deep Learning</a>, at <a href="https://www.kth.se/en">KTH Royal Institute of Technology</a>

## About

In this project we fine-tune <a href="https://huggingface.co/openai/whisper-small">OpenAI's Whisper model</a> for italian automatic speech recognition (ASR). We do that by using the <a href="https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0">Common Voice dataset</a>, the italian subset and by leveraging Hugging Face 🤗 Transformers. 

Whisper is a pre-trained model for automatic speech recognition (ASR) published in September 2022 by the authors Alec Radford et al. from OpenAI. It is trained on a large dataset of diverse audio, 680,000 hours of labelled audio-transcription, and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification. As a consequence, Whisper requires little additional fine-tuning to yield a performant ASR model.

## Implementation

The implementation consists in a feature engineering pipeline, a training pipeline and an inference program (Hugging Face Space). This is done to run the feature engineering on CPUs and the training pipeline on GPUs.
The training pipeline exploits the free GPU given by Google Colaboratory, the problem with that is double: getting a good GPU is rare and the GPU is given for a maximum of 12 hours (not guaranteed, could be even less).
 
1. We wrote a **feature engineering pipeline** that loads the datset and takes 10% of the Italian ASR dataset since it's already 16 GB of data and more than that wouldn't fit in the VM's disk on Colab. We take both the train and test split to run some evaluation tests. Then, we load the Whisper Feature Extractor that pads the audio input to 30s and converts the inputs to Log-Mel spectograms. We also load the Whisper Tokenizer which post-processes the model output to text format, our labels.
We apply both the feature extractor and the tokenizer to our dataset to prepare for the traning. Finally we create a zip of our dataset and save it on Google Drive, to be able in the next pipeline to load the data ready for the training, without going through all the previous steps.

2. We wrote a **training pipeline** that loads our dataset, consisting in 15261 rows for the training set and 1500 rows for the testing set, loads a pre-trained Whisper checkpoint <a href="https://huggingface.co/openai/whisper-small">Whisper small</a> and runs the training and evaluations to verify that we have correctly trained it to transcribe speech in Italian.
The evaluation uses the word error rate (WER) metric, the 'de-facto' metric for assessing ASR systems.
The crucial part of the training pipeline is defining the parameters and hyperparameters; our choices and possible improvements are explained in the next section.

3. We wrote a Gradio application, our **inference program**, that:
- allows the user to speak into the microphone or upload an audio file and transcribe and translate what he/she says or uploads
- allows the user to paste in the URL to a video, and transcribe, summarize and translate what is spoken in the video

## Training parameters and possible improvements

The most relevant parameters we used are:
- `output_dir="/content/drive/MyDrive/whisper_hf"`: model predictions and checkpoints will be written on Google Drive to be able to recover them and restart the training from the latest checkpoint in case Colab takes away our resources.
- `evaluation_strategy="steps"`: the evaluation is done (and logged) every `eval_steps=100` so that every 100 step of our training (1 steps has a batch size of 16) we compute the WER metric.
- `save_strategy="steps"`: our checkpoints are also saved every `save_steps=100` and by setting `save_total_limit=2` we limit the total amount of checkpoints deleting the older checkpoints in our folder on Google Drive. This was done to be sure that we could save some checkpoints before loosing the GPU.

Possible improvements (**model-centric approach**):
- `num_train_epochs=1`: is the total number of training epochs to perform during training, we set it at 1 to be able to finish our training in reasonable timings (training was about 6hrs long). With low values of epochs there's the risk of underfitting but in our case the final WER is low so it seems that we are not dealing that much with underfitting. We need also to be careful in setting high values of epochs because that might lead to overfitting meaning the model has been overtrained and cannot generalize well to new data.
We trained the model also with `num_train_epochs=2` getting a slightly better performance as we can see on the model card of whisper, we can also notice how the training loss has improved in respect to the model trained with only one epoch.
- `learning_rate=1e-5`: is the value for the initial learning rate for AdamW optimizer. We could try to fine-tune the value around the default one that is 5e-5 to choose the optimum one, being careful that high learning rate almost never gets you to the global minima and a small learning rate can help our neural network converge to the global minima but it takes a huge amount of time.
- Another way to improve our model performance is by selecting a larger pre-trained Whisper checkpoint for our training, like the `openai/whisper-medium` or `openai/whisper-large`. We couldn't test that with Colab because the session keep crushing after using all the available RAM.

Possible improvements (**data-centric approach**): 
- Having more data is always a good approach so we could select a wider set from the Italian ASR dataset, but as mentioned previously this leads to a problem of space on Colab. One free solution to this could be to run the code on your own laptop or use Google Cloud Platform that gives you 300 $ for new users.

## Spaces on Hugging Face

### Whisper Italian ASR - Spaces
https://huggingface.co/spaces/whispy/Italian-ASR


## Models on Hugging Face

### whisper_hf
https://huggingface.co/whispy/whisper_hf

### whisper
https://huggingface.co/spaces/whispy/Italian-ASR

## Built With

* [Colab](https://colab.research.google.com/)
* [Hugging Face](https://huggingface.co/)

