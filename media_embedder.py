import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
from torchaudio.transforms import Resample
from torchvision.models.video import r3d_18  # Example: using a pretrained R3D model
from torchvision.io import read_video
"""
class MediaEmbedder:
    def __init__(self):
        # Initialize processors and models for audio and video
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # For video, initialize the relevant model and processor (example placeholders)
        self.video_processor = None
        self.video_model = None

    def embed_audio(self, audio_path, chunk_duration, resample_rate=16000):
        batch_waveforms = []
        chunk_size = resample_rate * chunk_duration  # Calculate the chunk size in samples

        waveform, sample_rate = torchaudio.load(audio_path)
        waveform=torch.mean(waveform, dim=0, keepdim=False)
        # Resample if the sample rate is different
        if sample_rate != resample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=resample_rate)
            waveform = resampler(waveform)
         # Segment the waveform into chunks
        total_samples = waveform.size(0)
        num_chunks = total_samples // chunk_size

        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = waveform[start:end]

            # Process audio chunk
            inputs = self.audio_processor(chunk, sampling_rate=resample_rate, return_tensors="pt")
            processed_audio = inputs.input_values.squeeze(0)
            if processed_audio.dim() == 2:
                processed_audio = processed_audio.view(-1, processed_audio.size(-1))  # Flatten to [16000]
            # Aggregate processed audio chunks for the batch
            batch_waveforms.append(processed_audio)

        # Stack all processed waveforms into a single batch tensor
        batch_waveforms_tensor = torch.stack(batch_waveforms)
        # Generate embeddings for the batch
        with torch.no_grad():
            embeddings = self.audio_model(batch_waveforms_tensor).last_hidden_state
        return embeddings

    def embed_video(self, video_path, chunk_duration, fps):
        local_model_path ="/Users/markrademaker/Downloads/Work/Scriptie/r3d_18-b3b3357e.pth"
        # Load the model
        self.model = r3d_18(pretrained=False)  # Set pretrained to False
        self.model.load_state_dict(torch.load(local_model_path))
        self.model.eval()  # Set the model to evaluation mode

        # Load and preprocess video
        video, _, _ = read_video(video_path, start_pts=0, end_pts=None, pts_unit='sec')
        
        # Calculate the number of frames per chunk
        chunk_frame_count = chunk_duration * fps

        # Assuming the video is already at the desired fps
        total_frames = video.shape[0]
        num_chunks = total_frames // chunk_frame_count

        embeddings = []

        for i in range(num_chunks):
            start_frame = i * chunk_frame_count
            end_frame = start_frame + chunk_frame_count
            video_chunk = video[start_frame:end_frame]

            # Preprocess the video chunk as required by your model
            # For example, resizing, normalizing, etc.
            # preprocessed_chunk = preprocess_video_chunk(video_chunk)

            # Generate embedding for the chunk
            with torch.no_grad():
                # Adjust as per your model's input requirements
                chunk_embedding = self.video_model(video_chunk.unsqueeze(0))
                embeddings.append(chunk_embedding)

        # Concatenate all embeddings
        embeddings_tensor = torch.cat(embeddings, dim=0)

        return embeddings_tensor
"""   
import torch
from torch.utils.data import Dataset
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, audio_file_paths, chunk_duration, resample_rate=16000):
        self.audio_file_paths = audio_file_paths
        self.chunk_duration = chunk_duration
        self.resample_rate = resample_rate

    def __len__(self):
        return len(self.audio_file_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_file_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        # Resample waveform if necessary
        if sample_rate != self.resample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.resample_rate)
            waveform = resampler(waveform)
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono
        return waveform, self.resample_rate
class MediaEmbedder:
    def __init__(self):
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    def embed_audio(self, audio_path, chunk_duration, resample_rate=16000):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != resample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=resample_rate)
            waveform = resampler(waveform)
        inputs = self.audio_processor(waveform, sampling_rate=resample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = self.audio_model(**inputs).last_hidden_state
        # Pooling the embeddings could be necessary depending on your application
        embeddings = torch.mean(embeddings, dim=1)  # Simple mean pooling

        return embeddings
    
class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]