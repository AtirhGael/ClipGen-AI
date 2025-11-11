"""
Audio Transcription Service using OpenAI Whisper
"""
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import whisper
import torch
import librosa
import soundfile as sf
import numpy as np
from moviepy import VideoFileClip
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import json

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Custom exception for transcription errors"""
    pass


class AudioTranscriber:
    """Advanced audio transcription with speaker detection and alignment"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize transcriber with Whisper model
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Don't load model at startup - load lazily when needed
        # self._load_model()
    
    def _load_model(self):
        """Load Whisper model (lazy loading)"""
        if self.model is not None:
            return  # Already loaded
            
        try:
            logger.info(f"Loading Whisper model '{self.model_size}' on {self.device}")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise TranscriptionError(f"Failed to load Whisper model: {e}")
    
    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            output_path: Optional output path for audio file
            
        Returns:
            Path to extracted audio file
        """
        try:
            if output_path is None:
                # Create temporary audio file
                temp_dir = Path(tempfile.gettempdir()) / "clipgen_audio"
                temp_dir.mkdir(exist_ok=True)
                video_name = Path(video_path).stem
                output_path = str(temp_dir / f"{video_name}_audio.wav")
            
            logger.info(f"Extracting audio from {video_path}")
            
            # Use moviepy to extract audio
            with VideoFileClip(video_path) as video:
                audio = video.audio
                if audio is None:
                    raise TranscriptionError("No audio track found in video")
                
                # Write audio to file with optimal settings for Whisper
                audio.write_audiofile(
                    output_path,
                    codec='pcm_s16le',  # 16-bit PCM
                    ffmpeg_params=['-ac', '1', '-ar', '16000'],  # Mono, 16kHz
                    verbose=False,
                    logger=None
                )
            
            logger.info(f"Audio extracted to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise TranscriptionError(f"Failed to extract audio: {e}")
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Preprocess audio for optimal Whisper performance
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Load audio with librosa (ensures proper format for Whisper)
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
            # Remove silence (basic)
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
            
            return audio_trimmed
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise TranscriptionError(f"Failed to preprocess audio: {e}")
    
    def transcribe_audio(
        self, 
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        temperature: float = 0.0,
        word_timestamps: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_path: Path to audio file
            language: Language code (auto-detect if None)
            task: 'transcribe' or 'translate'
            temperature: Sampling temperature
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            Transcription result with segments and metadata
        """
        try:
            # Load model lazily when first needed
            if self.model is None:
                self._load_model()
            
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                temperature=temperature,
                word_timestamps=word_timestamps,
                fp16=torch.cuda.is_available()  # Use FP16 on GPU
            )
            
            # Process segments for better structure
            processed_segments = []
            for segment in result.get('segments', []):
                processed_segment = {
                    'id': segment['id'],
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'text': segment['text'].strip(),
                    'confidence': segment.get('avg_logprob', 0.0),
                    'no_speech_prob': segment.get('no_speech_prob', 0.0),
                    'word_count': len(segment['text'].split()),
                    'words': []
                }
                
                # Add word-level timestamps if available
                if 'words' in segment:
                    for word in segment['words']:
                        processed_segment['words'].append({
                            'word': word['word'],
                            'start': word['start'],
                            'end': word['end'],
                            'probability': word.get('probability', 1.0)
                        })
                
                processed_segments.append(processed_segment)
            
            # Compile full result
            transcription_result = {
                'text': result['text'].strip(),
                'language': result.get('language', 'unknown'),
                'segments': processed_segments,
                'duration': processed_segments[-1]['end_time'] if processed_segments else 0.0,
                'total_segments': len(processed_segments),
                'word_count': len(result['text'].split()),
                'model_used': self.model_size,
                'device_used': self.device
            }
            
            logger.info(f"Transcription completed: {len(processed_segments)} segments, {transcription_result['word_count']} words")
            return transcription_result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise TranscriptionError(f"Failed to transcribe audio: {e}")
    
    def detect_speakers(self, audio_path: str, min_segment_length: float = 1.0) -> List[Dict[str, Any]]:
        """
        Simple speaker detection using audio characteristics
        
        Args:
            audio_path: Path to audio file
            min_segment_length: Minimum segment length for speaker detection
            
        Returns:
            List of speaker segments with basic speaker ID
        """
        try:
            logger.info("Performing basic speaker detection")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)
            
            # Extract features for speaker detection
            # Use MFCC features to detect speaker changes
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Calculate frame-to-frame differences
            mfcc_delta = np.diff(mfccs, axis=1)
            change_points = []
            
            # Find significant changes in MFCC features
            window_size = int(sr * 0.5)  # 0.5 second window
            hop_length = int(sr * 0.1)   # 0.1 second hop
            
            for i in range(0, mfcc_delta.shape[1] - window_size, hop_length):
                window1 = mfcc_delta[:, i:i+window_size//2]
                window2 = mfcc_delta[:, i+window_size//2:i+window_size]
                
                # Calculate similarity between windows
                similarity = np.corrcoef(window1.flatten(), window2.flatten())[0, 1]
                
                if similarity < 0.7:  # Threshold for speaker change
                    time_point = (i + window_size//2) * 0.1
                    change_points.append(time_point)
            
            # Create speaker segments
            speaker_segments = []
            current_speaker = 1
            start_time = 0.0
            
            for change_point in change_points:
                if change_point - start_time >= min_segment_length:
                    speaker_segments.append({
                        'speaker_id': current_speaker,
                        'start_time': start_time,
                        'end_time': change_point,
                        'duration': change_point - start_time,
                        'confidence': 0.7  # Basic confidence score
                    })
                    start_time = change_point
                    current_speaker = 2 if current_speaker == 1 else 1
            
            # Add final segment
            audio_duration = len(audio) / sr
            if audio_duration - start_time >= min_segment_length:
                speaker_segments.append({
                    'speaker_id': current_speaker,
                    'start_time': start_time,
                    'end_time': audio_duration,
                    'duration': audio_duration - start_time,
                    'confidence': 0.7
                })
            
            logger.info(f"Detected {len(speaker_segments)} speaker segments")
            return speaker_segments
            
        except Exception as e:
            logger.error(f"Error in speaker detection: {e}")
            return [{'speaker_id': 1, 'start_time': 0.0, 'end_time': 0.0, 'duration': 0.0, 'confidence': 1.0}]
    
    def align_transcription_with_speakers(
        self, 
        transcription: Dict[str, Any], 
        speaker_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Align transcription segments with detected speakers
        
        Args:
            transcription: Transcription result from Whisper
            speaker_segments: Speaker detection results
            
        Returns:
            Aligned segments with speaker information
        """
        try:
            aligned_segments = []
            
            for segment in transcription['segments']:
                segment_start = segment['start_time']
                segment_end = segment['end_time']
                segment_mid = (segment_start + segment_end) / 2
                
                # Find which speaker segment this belongs to
                assigned_speaker = 1  # Default speaker
                for speaker_seg in speaker_segments:
                    if speaker_seg['start_time'] <= segment_mid <= speaker_seg['end_time']:
                        assigned_speaker = speaker_seg['speaker_id']
                        break
                
                aligned_segment = segment.copy()
                aligned_segment['speaker_id'] = assigned_speaker
                aligned_segments.append(aligned_segment)
            
            return aligned_segments
            
        except Exception as e:
            logger.error(f"Error aligning transcription with speakers: {e}")
            return transcription['segments']
    
    def transcribe_video_complete(
        self, 
        video_path: str,
        language: Optional[str] = None,
        detect_speakers: bool = False,
        cleanup_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Complete video transcription pipeline
        
        Args:
            video_path: Path to video file
            language: Language for transcription
            detect_speakers: Whether to perform speaker detection
            cleanup_audio: Whether to clean up temporary audio files
            
        Returns:
            Complete transcription result
        """
        try:
            logger.info(f"Starting complete transcription for: {video_path}")
            
            # Extract audio from video
            audio_path = self.extract_audio_from_video(video_path)
            
            try:
                # Transcribe audio
                transcription = self.transcribe_audio(
                    audio_path, 
                    language=language,
                    word_timestamps=True
                )
                
                # Perform speaker detection if requested
                speaker_segments = []
                if detect_speakers:
                    speaker_segments = self.detect_speakers(audio_path)
                    
                    # Align transcription with speakers
                    transcription['segments'] = self.align_transcription_with_speakers(
                        transcription, 
                        speaker_segments
                    )
                
                # Add speaker information to result
                transcription['speaker_segments'] = speaker_segments
                transcription['speaker_detection_enabled'] = detect_speakers
                
                return transcription
                
            finally:
                # Cleanup temporary audio file
                if cleanup_audio and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                        logger.info("Cleaned up temporary audio file")
                    except Exception as e:
                        logger.warning(f"Could not clean up audio file: {e}")
            
        except Exception as e:
            logger.error(f"Error in complete transcription: {e}")
            raise TranscriptionError(f"Complete transcription failed: {e}")
    
    async def transcribe_video_async(
        self, 
        video_path: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Async wrapper for video transcription"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.transcribe_video_complete,
            video_path,
            **kwargs
        )
    
    def change_model(self, model_size: str):
        """Change Whisper model size"""
        if model_size != self.model_size:
            self.model_size = model_size
            self.model = None
            self._load_model()


# Global transcriber instance
audio_transcriber = AudioTranscriber()