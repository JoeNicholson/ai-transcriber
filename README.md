# AI Transcriber

AI Transcriber is an application that uses the power of AI to transcribe audio inputs and correct transcriptions. 
It's built with Python, utilizing OpenAI's GPT-4 model for text correction and OpenAI's Whisper API for audio transcription.

## Prerequisites

You will need an API key from OpenAI in order to use the GPT-4 model. You can get the API key by creating an account at https://beta.openai.com/signup/, and then visiting the API section in the dashboard once your account is set up. 

Once you have the API key, create a .env file in the root directory of the application and add your API key like so:

```
OPENAI_API_KEY=<your OpenAI API key>
```

Make sure to replace `<your OpenAI API key>` with the actual key you received from OpenAI.

## Installation

Ensure that you have Python 3.7 or newer installed.

1. Clone this repository:

```
git clone https://github.com/yourusername/ai-transcriber.git
```

2. Change into the directory:

```
cd ai-transcriber
```

3. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

1. Start the application by running:

```
python transcribe_gui.py
```

2. Use the GUI to record audio, transcribe it, and correct the transcription. You can correct a selected portion of the transcription, or the entire text.

## License

AI Transcriber is released under the [MIT License](LICENSE.txt).

## Contributing

I welcome contributions :-) Feel free to fork the repository and submit pull requests.


## Acknowledgements

- Thanks to OpenAI for their powerful AI models and APIs.
- Thanks to the open-source community for the various packages used in this project.
