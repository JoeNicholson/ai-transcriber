"""
AI Transcriber
==============

Author: Joe Nicholson
Date: 2023-06-09

An application that transcribes audio and allows the transcript to be edited and corrected via AI.

Prerequisites:
1. Install the following Ubuntu packages:
   - sudo apt-get install portaudio19-dev python3-tk ffmpeg

2. Ensure that you have the following Python modules installed:
   - pip install pyaudio numpy pydub openai python-dotenv

3. Set up a file named .env in the same directory as your script, containing your OpenAI API key:
   - OPENAI_API_KEY=<your OpenAI API key>

4. (Optional) Create a 'prompt.txt' file with a custom prompt message for the AI.

The application will transcribe audio and allow the transcript to be edited and corrected via AI. The correction prompt is
loaded from a 'prompt.txt' file if it exists. After transcription, this prompt is used to correct possible misinterpretations 
and grammatical errors in the transcription.

For more details, refer to the README file.

"""
import re
import pyaudio
import numpy as np
from pydub import AudioSegment
import os
import openai
from dotenv import load_dotenv
import threading
from tkinter import *
from tkinter import ttk, Frame, Text, WORD, BOTH, TOP, X, LEFT, YES, DISABLED
from tkinter import messagebox

# Load the .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class AudioRecorderApp:
    def __init__(self, master):
        """
        Initialize the AudioRecorderApp.

        Args:
            master: Root widget for the application.
        """
        # Initialize the UI
        self.master = master
        self.master.title("AI Transcriber - Joe Nicholson")
        self.master.geometry("800x400")
        
        # UI components setup
        self.setup_ui_components()

        # Set initial states
        self.is_recording = False
        self.is_paused = False
        self.audio_frames = []

        # Try to load the prompt from the 'prompt.txt' file.
        try:
            with open('prompt.txt', 'r') as file:
                self.prompt_message = file.read()
        except FileNotFoundError:
            print("Prompt file not found. The application will use the default prompt.")
            self.prompt_message = """
            Given a piece of text that was transcribed using a voice-to-text converter, please perform the following two tasks:
            1. Identify and replace any specific words that the converter may have misinterpreted, some examples are in the list provided below:
            * full stop -> [Blank]
            2. Review and correct any grammatical errors present in the text to ensure it is coherent and accurate.
            Reply only with the corrected text.  Thanks.
            The text to be corrected is as follows:
            """


    def setup_ui_components(self):
        """
        Set up the UI components of the application.
        """
        # Frame for buttons
        button_frame = Frame(self.master)
        button_frame.pack(side=TOP, fill=X)

        # Record, pause, stop and AI call buttons
        self.record_button = ttk.Button(button_frame, text="Record", command=self.start_recording)
        self.record_button.pack(side=LEFT, padx=5, pady=7)

        self.pause_button = ttk.Button(button_frame, text="Pause", command=self.pause_recording, state=DISABLED)
        self.pause_button.pack(side=LEFT, padx=5, pady=7)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_recording, state=DISABLED)
        self.stop_button.pack(side=LEFT, padx=5, pady=7)

        self.ai_button = ttk.Button(button_frame, text="AI", command=self.call_ai)
        self.ai_button.pack(side=LEFT, padx=5, pady=7)

        # Text box for displaying the transcript
        self.transcript = Text(self.master, wrap=WORD, undo=True, maxundo=100)  # Enable the undo and redo feature
        self.transcript.pack(expand=YES, fill=BOTH)

        # Add key bindings for 'Select All', 'Copy', 'Undo' and 'Redo'
        self.transcript.bind("<Control-a>", self.select_all)
        self.transcript.bind("<Control-A>", self.select_all)
        self.transcript.bind("<Control-c>", self.copy)
        self.transcript.bind("<Control-C>", self.copy)
        self.transcript.bind('<Control-z>', self.undo_text_action)
        self.transcript.bind('<Control-Z>', self.undo_text_action)
        self.transcript.bind('<Control-Shift-z>', self.redo_text_action)
        self.transcript.bind('<Control-Shift-Z>', self.redo_text_action)

        self.is_recording = False
        self.is_paused = False
        self.audio_frames = []


    def undo_text_action(self, event):
        """
        Undo the most recent action in the transcript text box. 

        Args:
            event: The event that triggered this method.

        If there are no actions to undo, the method does nothing.
        """
        try:
            self.transcript.edit_undo()
        except TclError:
            pass  # Do nothing if there is nothing to undo


    def redo_text_action(self, event):
        """
        Redo the most recent action in the transcript text box that was undone. 

        Args:
            event: The event that triggered this method.

        If there are no actions to redo, the method does nothing.
        """
        try:
            self.transcript.edit_redo()
        except TclError:
            pass  # Do nothing if there is nothing to redo


    def start_recording(self):
        """
        Start the recording process.
        """
        self.is_recording = True
        self.record_button.config(state=DISABLED)
        self.pause_button.config(state=NORMAL)
        self.stop_button.config(state=NORMAL)

        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()


    def pause_recording(self):
        """
        Pause or resume the recording process.
        """
        if not self.is_paused:
            self.is_paused = True
            self.pause_button.config(text="Resume")
        else:
            self.is_paused = False
            self.pause_button.config(text="Pause")


    def stop_recording(self):
        """
        Stop the recording process.
        """
        self.is_recording = False
        self.record_button.config(state=NORMAL)
        self.pause_button.config(state=DISABLED)
        self.stop_button.config(state=DISABLED)

        self.audio_thread.join()

        audio_data, sample_rate, channels = b''.join(self.audio_frames), self.sample_rate, self.channels
        file_name = "output.mp3"
        self.save_audio_to_mp3(audio_data, sample_rate, channels, file_name)
        transcript = self.call_whisper_ai()

        self.transcript.insert(END, transcript.text + "\n")

        self.audio_frames = []

    
    def call_chatgpt(self, text, prompt_message=None):
        """
        Call the GPT-4 model to correct and edit the transcript.

        Args:
            text: The transcript to be corrected.
            prompt_message: A custom prompt message for the model. If None, the default prompt message is used.

        Returns:
            The corrected transcript from the model.
        """
        MODEL = "gpt-4"

        if prompt_message is None:
            prompt_message = self.prompt_message

        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_message + text}
            ],
            temperature=0,
        )
        print(response)
        return response.choices[0].message.content
    

    def record_audio(self):
        """
        Record the audio input and store the audio data in frames.
        """
        p = pyaudio.PyAudio()
        CHUNK = 1024

        default_device_index = p.get_default_input_device_info()["index"]
        device_info = p.get_device_info_by_index(default_device_index)
        self.sample_rate = int(device_info['defaultSampleRate'])
        self.channels = 1
        self.channels = device_info['maxInputChannels']

        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=None)

        while self.is_recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if not self.is_paused:
                self.audio_frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()


    def save_audio_to_mp3(self, audio_data, sample_rate, channels, file_name="output.mp3"):
        """
        Save the audio data to an mp3 file.

        Args:
            audio_data: The raw audio data.
            sample_rate: The sample rate of the audio data.
            channels: The number of audio channels.
            file_name: The name of the mp3 file.
        """
        audio = AudioSegment(
            audio_data,
            frame_rate=sample_rate,
            sample_width=2,
            channels=1,
        )
        audio.export(file_name, format="mp3")
        print(f"Audio saved to {file_name}")


    def call_whisper_ai(self):
        """
        Transcribe the audio using Whisper.

        Returns:
            The transcript of the audio.
        """
        audio_file = open("output.mp3", "rb")
        print("Transcribing audio...")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript
    

    # Functions to handle 'Select All' and 'Copy' key shortcuts
    def select_all(self, event):
        """
        Select all the text in the transcript text box.

        Args:
            event: The event that triggered this method.

        Returns:
            A string indicating that the event has been handled.
        """
        self.transcript.tag_add(SEL, "1.0", END)
        self.transcript.mark_set(INSERT, "1.0")
        self.transcript.see(INSERT)
        return "break"


    def copy(self, event):
        """
        Copy the selected text in the transcript text box to the clipboard.

        Args:
            event: The event that triggered this method.

        Returns:
            A string indicating that the event has been handled.
        """
        if not self.transcript.tag_ranges(SEL):
            self.transcript.clipboard_clear()
            text = self.transcript.get("1.0", END)
            self.transcript.clipboard_append(text)
        else:
            self.transcript.clipboard_clear()
            text = self.transcript.get(SEL_FIRST, SEL_LAST)
            self.transcript.clipboard_append(text)
        return "break"
    
    
    def call_ai(self):
        """
        Call the AI to correct and edit the selected transcript.
        """
        print("Calling AI...")
        try:
            # Check if any text is selected
            sel_first = self.transcript.index(SEL_FIRST)
            text = self.transcript.get(SEL_FIRST, SEL_LAST)
            sel_start = self.transcript.index(SEL_FIRST)  # Store the SEL_FIRST index
            self.transcript.delete(SEL_FIRST, SEL_LAST)
        except TclError:
            # If no text is selected, use the entire transcript
            text = self.transcript.get("1.0", END)
            sel_start = "1.0"
            self.transcript.delete("1.0", END)
            
        edited_text = self.call_chatgpt(text)
        self.transcript.insert(sel_start, edited_text)


if __name__ == "__main__":
    root = Tk()
    AudioRecorderApp(root)
    root.mainloop()
