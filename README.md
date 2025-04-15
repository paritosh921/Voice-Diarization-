# Voice Diarization & Transcription with WhisperX

This project demonstrates a simple application that performs audio transcription with speaker diarization using [WhisperX](https://github.com/m-bain/whisperx). The script loads an audio file, performs diarization to differentiate between speakers, and produces a structured transcription output.

## Prerequisites

- **Python 3.7+**
- **CUDA-enabled GPU** (optional, but recommended for faster processing; the code is set to use `"cuda"`)
- An audio file (WAV format) named `videoplayback.wav` in your working directory
- A valid Hugging Face authentication token (replace the token in the code if needed)

## Setup and Installation

1. **Clone the Repository or Create a Project Directory**

   If you haven't already, create a new directory and navigate into it. You can use Git to clone the repository if it's hosted, or simply create the necessary files locally.

2. **Install Dependencies**

   The only external package required for this project is `whisperx`, which you can install directly from its GitHub repository. Open your terminal and run:

   ```bash
   pip install --quiet git+https://github.com/m-bain/whisperx.git
   ```

   This command installs the latest version of WhisperX along with its dependencies. Make sure your environment has the necessary CUDA libraries if you're using a GPU.

3. **Configure Your Hugging Face Token**

   The script uses a Hugging Face token for accessing models. In the code, there is a parameter:
   
   ```python
   use_auth_token="H_TOKEN"
   ```

   Replace this string with your own Hugging Face token if necessary:
   
   - Create a token by logging into your Hugging Face account and visiting [settings/tokens](https://huggingface.co/settings/tokens).
   - Update the token in the code or consider storing it in an environment variable for better security.

## How to Run

1. **Prepare Your Audio File**

   Ensure the audio file `videoplayback.wav` is located in the same directory as your script or provide the correct path to your file.

2. **Execute the Script**

   Save your code (provided below) in a file, for example, `transcribe.py`:

   ```python
   import whisperx
   import gc

   # Set device to "cuda" for GPU usage, or change to "cpu" if necessary
   device = "cuda"
   batch_size = 4  # Reduce if you have limited GPU memory
   compute_type = "float16"  # Change to "int8" if low on GPU memory (may reduce accuracy)

   # Initialize the diarization pipeline with your Hugging Face token
   diarize_model = whisperx.DiarizationPipeline(
       use_auth_token="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",
       device=device
   )

   # Path to your audio file (ensure it's in the working directory)
   audio_file = "videoplayback.wav"
   audio = whisperx.load_audio(audio_file)

   # Perform diarization (customize speaker parameters as needed)
   diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)

   # Load a multilingual transcription model
   model = whisperx.load_model("large-v2", device, compute_type=compute_type)

   # Transcribe audio file with batching
   result = model.transcribe(audio, batch_size=batch_size)

   # Assign speaker labels to the transcription output
   result = whisperx.assign_word_speakers(diarize_segments, result)

   # Output the result
   print(result)

   # Free up unused memory (optional)
   gc.collect()
   ```

3. **Run the Script**

   Open a terminal in your project directory and execute:

   ```bash
   python transcribe.py
   ```

   The console will display the transcription with speaker diarization in a structured format.

## Troubleshooting

- **GPU/CPU Selection:**  
  If you do not have a CUDA-enabled GPU, change the `device` parameter from `"cuda"` to `"cpu"` in the script.

- **Hugging Face Token Issues:**  
  Ensure that the Hugging Face token is valid. If the token is incorrect or missing, the pipeline may fail to load the required models.

- **Audio File Path:**  
  Make sure that the audio file path is correct. If your audio file is located elsewhere, adjust the `audio_file` variable accordingly.

