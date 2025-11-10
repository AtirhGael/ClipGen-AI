"""
Video Processing Service - Orchestrates the complete video processing pipeline
"""
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import asyncio
import uuid
from datetime import datetime
from sqlalchemy.orm import Session

from app.services.youtube_downloader import youtube_downloader, YouTubeDownloadError
from app.services.scene_detector import scene_detector, SceneDetectionError
from app.services.audio_transcriber import audio_transcriber, TranscriptionError
from app.models import Video, Scene, Transcript, ProcessingJob
from app.database import SessionLocal

logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass


class VideoProcessor:
    """Main video processing orchestrator"""
    
    def __init__(self):
        self.processing_jobs = {}  # Track active processing jobs
    
    def update_video_progress(self, video_id: str, progress: int, status: str = None):
        """Update video processing progress in database"""
        try:
            db = SessionLocal()
            video = db.query(Video).filter(Video.id == video_id).first()
            if video:
                video.progress = progress
                if status:
                    video.status = status
                video.updated_at = datetime.utcnow()
                db.commit()
            db.close()
        except Exception as e:
            logger.error(f"Error updating video progress: {e}")
    
    def create_processing_job(self, job_type: str, video_id: str, db: Session) -> str:
        """Create a new processing job record"""
        job = ProcessingJob(
            id=str(uuid.uuid4()),
            job_type=job_type,
            reference_id=video_id,
            status="pending"
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id
    
    def update_job_progress(self, job_id: str, progress: int, status: str = None, error: str = None):
        """Update processing job progress"""
        try:
            db = SessionLocal()
            job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if job:
                job.progress = progress
                if status:
                    job.status = status
                if error:
                    job.error_message = error
                job.updated_at = datetime.utcnow()
                if status == "completed":
                    job.completed_at = datetime.utcnow()
                db.commit()
            db.close()
        except Exception as e:
            logger.error(f"Error updating job progress: {e}")
    
    async def download_video(
        self, 
        video_id: str, 
        youtube_url: str, 
        quality: str = "best"
    ) -> Dict[str, Any]:
        """
        Download video from YouTube
        
        Args:
            video_id: Video ID in database
            youtube_url: YouTube URL
            quality: Video quality to download
            
        Returns:
            Dictionary with download results
        """
        db = SessionLocal()
        job_id = self.create_processing_job("download", video_id, db)
        db.close()
        
        try:
            logger.info(f"Starting video download for {video_id}")
            
            self.update_video_progress(video_id, 10, "downloading")
            self.update_job_progress(job_id, 0, "running")
            
            # Progress callback for download
            def progress_callback(progress_data):
                percent = progress_data.get('percent', 0)
                # Map download progress to 10-50% of total progress
                total_progress = 10 + int(percent * 0.4)
                self.update_video_progress(video_id, total_progress)
                self.update_job_progress(job_id, percent)
            
            # Extract video ID for file naming
            _, yt_video_id = youtube_downloader.validate_youtube_url(youtube_url)
            
            # Download video
            download_result = await youtube_downloader.download_video_async(
                youtube_url,
                yt_video_id,
                quality,
                progress_callback
            )
            
            # Update database with file paths
            db = SessionLocal()
            video = db.query(Video).filter(Video.id == video_id).first()
            if video:
                video.file_path = download_result.get('video_file')
                video.thumbnail_path = download_result.get('thumbnail_file')
                db.commit()
            db.close()
            
            self.update_video_progress(video_id, 50, "processing")
            self.update_job_progress(job_id, 100, "completed")
            
            logger.info(f"Video download completed for {video_id}")
            return download_result
            
        except Exception as e:
            logger.error(f"Error downloading video {video_id}: {e}")
            self.update_video_progress(video_id, 0, "failed")
            self.update_job_progress(job_id, 0, "failed", str(e))
            raise VideoProcessingError(f"Download failed: {e}")
    
    async def detect_scenes(
        self, 
        video_id: str, 
        video_file_path: str,
        method: str = "pyscenedetect"
    ) -> List[Dict[str, Any]]:
        """
        Detect scenes in video
        
        Args:
            video_id: Video ID in database
            video_file_path: Path to video file
            method: Scene detection method
            
        Returns:
            List of detected scenes
        """
        db = SessionLocal()
        job_id = self.create_processing_job("scene_detection", video_id, db)
        db.close()
        
        try:
            logger.info(f"Starting scene detection for {video_id}")
            
            self.update_job_progress(job_id, 0, "running")
            
            # Configure detection method
            use_pyscene = method in ["pyscenedetect", "combined"]
            use_opencv = method in ["opencv", "combined"]
            
            # Detect scenes
            scenes = await scene_detector.detect_scenes_async(
                video_file_path,
                use_pyscenedetect=use_pyscene,
                use_opencv=use_opencv
            )
            
            # Store scenes in database
            db = SessionLocal()
            for scene_data in scenes:
                scene = Scene(
                    id=str(uuid.uuid4()),
                    video_id=video_id,
                    start_time=scene_data['start_time'],
                    end_time=scene_data['end_time'],
                    scene_type=scene_data.get('detection_method'),
                    confidence=scene_data.get('confidence'),
                    frame_count=scene_data.get('content_analysis', {}).get('frame_count'),
                    avg_brightness=scene_data.get('content_analysis', {}).get('avg_brightness'),
                    avg_contrast=scene_data.get('content_analysis', {}).get('avg_contrast'),
                    dominant_colors=scene_data.get('content_analysis', {}).get('dominant_colors')
                )
                db.add(scene)
            
            db.commit()
            db.close()
            
            self.update_video_progress(video_id, 70)
            self.update_job_progress(job_id, 100, "completed")
            
            logger.info(f"Scene detection completed for {video_id}: {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            logger.error(f"Error in scene detection for {video_id}: {e}")
            self.update_job_progress(job_id, 0, "failed", str(e))
            raise VideoProcessingError(f"Scene detection failed: {e}")
    
    async def transcribe_audio(
        self, 
        video_id: str, 
        video_file_path: str,
        language: Optional[str] = None,
        detect_speakers: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio from video
        
        Args:
            video_id: Video ID in database
            video_file_path: Path to video file
            language: Language for transcription
            detect_speakers: Whether to detect speakers
            
        Returns:
            Transcription results
        """
        db = SessionLocal()
        job_id = self.create_processing_job("transcription", video_id, db)
        db.close()
        
        try:
            logger.info(f"Starting audio transcription for {video_id}")
            
            self.update_job_progress(job_id, 0, "running")
            
            # Transcribe audio
            transcription = await audio_transcriber.transcribe_video_async(
                video_file_path,
                language=language,
                detect_speakers=detect_speakers
            )
            
            # Store transcripts in database
            db = SessionLocal()
            for segment in transcription['segments']:
                transcript = Transcript(
                    id=str(uuid.uuid4()),
                    video_id=video_id,
                    start_time=segment['start_time'],
                    end_time=segment['end_time'],
                    text=segment['text'],
                    confidence=segment.get('confidence'),
                    speaker_id=segment.get('speaker_id'),
                    language=transcription.get('language'),
                    word_count=segment.get('word_count')
                )
                db.add(transcript)
            
            db.commit()
            db.close()
            
            self.update_video_progress(video_id, 90)
            self.update_job_progress(job_id, 100, "completed")
            
            logger.info(f"Audio transcription completed for {video_id}: {len(transcription['segments'])} segments")
            return transcription
            
        except Exception as e:
            logger.error(f"Error in audio transcription for {video_id}: {e}")
            self.update_job_progress(job_id, 0, "failed", str(e))
            raise VideoProcessingError(f"Audio transcription failed: {e}")
    
    async def process_video_complete(
        self,
        video_id: str,
        youtube_url: str,
        quality: str = "best",
        detect_speakers: bool = False,
        language: Optional[str] = None,
        scene_detection_method: str = "pyscenedetect"
    ) -> Dict[str, Any]:
        """
        Complete video processing pipeline
        
        Args:
            video_id: Video ID in database
            youtube_url: YouTube URL
            quality: Video quality to download
            detect_speakers: Whether to detect speakers
            language: Language for transcription
            scene_detection_method: Scene detection method
            
        Returns:
            Processing results
        """
        try:
            logger.info(f"Starting complete video processing pipeline for {video_id}")
            
            # Step 1: Download video
            download_result = await self.download_video(video_id, youtube_url, quality)
            video_file_path = download_result['video_file']
            
            if not video_file_path or not Path(video_file_path).exists():
                raise VideoProcessingError("Video file not found after download")
            
            # Step 2: Detect scenes (parallel with transcription)
            scene_task = asyncio.create_task(
                self.detect_scenes(video_id, video_file_path, scene_detection_method)
            )
            
            # Step 3: Transcribe audio (parallel with scene detection)
            transcription_task = asyncio.create_task(
                self.transcribe_audio(video_id, video_file_path, language, detect_speakers)
            )
            
            # Wait for both tasks to complete
            scenes, transcription = await asyncio.gather(scene_task, transcription_task)
            
            # Step 4: Mark as completed
            self.update_video_progress(video_id, 100, "completed")
            
            results = {
                'video_id': video_id,
                'download_result': download_result,
                'scenes': scenes,
                'transcription': transcription,
                'processing_time': datetime.utcnow()
            }
            
            logger.info(f"Complete video processing pipeline finished for {video_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error in complete video processing for {video_id}: {e}")
            
            # Update video status to failed
            db = SessionLocal()
            video = db.query(Video).filter(Video.id == video_id).first()
            if video:
                video.status = "failed"
                video.error_message = str(e)
                video.updated_at = datetime.utcnow()
                db.commit()
            db.close()
            
            raise VideoProcessingError(f"Complete processing failed: {e}")


# Global video processor instance
video_processor = VideoProcessor()