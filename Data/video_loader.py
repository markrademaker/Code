from moviepy.editor import VideoFileClip
#from spleeter.spleeter.separator import Separator
class VideoLoader:
    def __init__(self, path):
        self.path = path
        self.video_clip = None

    def separate_audio_sources(audio_path, output_path, separation_type='2stems'):
        """
        Separates audio sources using Spleeter.

        :param audio_path: Path to the input audio file.
        :param output_path: Path to save the separated audio files.
        :param separation_type: Type of separation ('2stems', '4stems', '5stems', etc.).
                                '2stems' separates into vocals and accompaniment.
                                '4stems' and '5stems' provide more granular separation.
        """
        # Create a separator with the specified configuration
        separator = Separator(f'spleeter:{separation_type}')

        # Perform separation and save output
        separator.separate_to_file(audio_path, output_path)

    def load_video(self):
        try:
            self.video_clip = VideoFileClip(self.path)
            print(f"Video loaded successfully: {self.path}")
        except Exception as e:
            print(f"Error loading video: {e}")

    # Add additional methods here as needed
    def split_audio_video(self):
        if self.video_clip is None:
            print("No video loaded. Please load a video first.")
            return

        try:
            # Extract audio
            audio = self.video_clip.audio
            audio_path = self.path.rsplit('.', 1)[0] + '_audio.mp3'
            audio.write_audiofile(audio_path)
            print(f"Audio extracted: {audio_path}")

            # Create video without audio
            video_no_audio = self.video_clip.without_audio()
            video_no_audio_path = self.path.rsplit('.', 1)[0] + '_no_audio.mp4'
            video_no_audio.write_videofile(video_no_audio_path, audio=False)
            print(f"Video without audio created: {video_no_audio_path}")

        except Exception as e:
            print(f"Error splitting audio and video: {e}")

    def extract_subpart(self, start_time, end_time, output_path):
        if self.video_clip is None:
            self.load_video()
        
        # Extract the specified part of the video
        extracted_video = self.video_clip.subclip(start_time, end_time)

        # Write the extracted part to a file
        extracted_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
