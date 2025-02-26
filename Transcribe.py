import whisper
import os
import warnings
import torch

warnings.filterwarnings("ignore")

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Whisper
print("Loading model......")
model = whisper.load_model("medium", device=device)
print("Model loaded successfully!")

# paths
video_path = r"C:\Users\saimi\Downloads\Vector databases\Vid.mp4"
video_data = r"C:\Users\saimi\Downloads\Vector databases\Video_data"

if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit()

os.makedirs(video_data, exist_ok=True)

# Transcribe the video
print("Transcribing video... (this may take time)")
try:
    result = model.transcribe(video_path, verbose=True, language="english")
    transcription = result["text"]
except Exception as e:
    print(f"Transcription error: {e}")
    exit()

output_file = os.path.join(video_data, "transcript_3.txt")
with open(output_file, "w", encoding="utf-8") as file:
    file.write(transcription)

print(f"Transcription saved as '{output_file}'")


