import whisper
import queue
import threading
import torch
import numpy as np
import speech_recognition as sr


def main():
    audio_queue = queue.Queue()
    threading.Thread(target=record, args={audio_queue}).start()
    threading.Thread(target=transcribe, args={audio_queue}).start()


def init_whisper():
    model = whisper.load_model("base")
    return model


def init_recognizer():
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.pause_threshold = 0.8
    r.dynamic_energy_threshold = False
    return r


def record(audio_queue):
    r = init_recognizer()
    print("Say something!")
    while True:
        with sr.Microphone(sample_rate=16000) as source:
            audio = r.listen(source)
            torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
            audio_queue.put_nowait(torch_audio)
            

def transcribe(audio_queue):
    while True:
        audio = audio_queue.get()
        model = init_whisper()
        transcried_audio = model.transcribe(audio)
        ## TODO Translate with a separate library
        translated_audio = model.transcribe(audio, task="translate") 

        print("\n")
        print("Language: " + transcried_audio["language"])
        print("Input: " + transcried_audio["text"])
        print("Translation: " + translated_audio["text"])


main()