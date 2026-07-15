import os
import librosa

data_path = "data/ravdess"

# Step 1: List actor folders
actors = os.listdir(data_path)
print("Actor folders found:", actors[:5])

# Step 2: Loop through actors until we find a non-empty folder
sample_file = None
for actor in actors:
    actor_path = os.path.join(data_path, actor)
    files = os.listdir(actor_path)
    if files:  # only proceed if folder is not empty
        sample_file = os.path.join(actor_path, files[0])
        break

if sample_file:
    print("Sample file path:", sample_file)

    # Step 3: Load audio with librosa
    signal, sr = librosa.load(sample_file, sr=22050)
    print(f"Loaded audio file with sample rate {sr} and length {len(signal)} samples")
else:
    print("No audio files found in any actor folder!")
