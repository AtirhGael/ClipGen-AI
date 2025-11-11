"""
Video Processing API Endpoints
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import uuid

from app.services.youtube_downloader import youtube_downloader, YouTubeDownloadError
from app.services.scene_detector import scene_detector, SceneDetectionError
from app.services.audio_transcriber import audio_transcriber, TranscriptionError
from app.services.video_processor import video_processor, VideoProcessingError
from app.database import get_db
from app.models import Video, Scene, Transcript, ProcessingJob, VIDEO_STATUSES, JOB_STATUSES, JOB_TYPES
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/videos", tags=["video-processing"])


# Pydantic models for API requests/responses
class VideoProcessRequest(BaseModel):
    """Request model for video processing"""
    youtube_url: HttpUrl
    quality: str = "best"
    detect_speakers: bool = False
    language: Optional[str] = None
    scene_detection_method: str = "pyscenedetect"  # pyscenedetect, opencv, combined
    
    class Config:
        json_schema_extra = {
            "example": {
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "quality": "best",
                "detect_speakers": False,
                "language": "en",
                "scene_detection_method": "pyscenedetect"
            }
        }


class VideoResponse(BaseModel):
    """Response model for video information"""
    id: str
    youtube_url: str
    youtube_id: Optional[str]
    title: Optional[str]
    duration: Optional[float]
    status: str
    progress: int
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None


class SceneResponse(BaseModel):
    """Response model for scene information"""
    id: str
    video_id: str
    start_time: float
    end_time: float
    scene_type: Optional[str]
    confidence: Optional[float]
    content_analysis: Optional[Dict[str, Any]] = None


class TranscriptResponse(BaseModel):
    """Response model for transcript information"""
    id: str
    video_id: str
    start_time: float
    end_time: float
    text: str
    confidence: Optional[float]
    speaker_id: Optional[int]


class ProcessingJobResponse(BaseModel):
    """Response model for processing job information"""
    id: str
    job_type: str
    reference_id: str
    status: str
    progress: int
    created_at: datetime
    error_message: Optional[str] = None


@router.post("/process", response_model=VideoResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_video(
    request: VideoProcessRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Submit YouTube URL for processing
    
    This endpoint validates the URL, extracts metadata, and starts background processing
    for video download, scene detection, and transcription.
    """
    try:
        # Validate YouTube URL
        is_valid, video_id = youtube_downloader.validate_youtube_url(str(request.youtube_url))
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid YouTube URL"
            )
        
        # Check if video already exists
        existing_video = db.query(Video).filter(Video.youtube_url == str(request.youtube_url)).first()
        if existing_video:
            return VideoResponse(
                id=existing_video.id,
                youtube_url=existing_video.youtube_url,
                youtube_id=existing_video.youtube_id,
                title=existing_video.title,
                duration=existing_video.duration,
                status=existing_video.status,
                progress=existing_video.progress,
                created_at=existing_video.created_at,
                updated_at=existing_video.updated_at,
                error_message=existing_video.error_message
            )
        
        # Extract video metadata
        try:
            metadata = youtube_downloader.get_video_info(str(request.youtube_url))
        except YouTubeDownloadError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not extract video info: {str(e)}"
            )
        
        # Create video record in database
        new_video = Video(
            id=str(uuid.uuid4()),
            youtube_url=str(request.youtube_url),
            youtube_id=video_id,
            title=metadata.get('title'),
            description=metadata.get('description'),
            duration=metadata.get('duration'),
            status="pending",
            progress=0,
            youtube_metadata=metadata
        )
        
        db.add(new_video)
        db.commit()
        db.refresh(new_video)
        
        # Start background processing
        background_tasks.add_task(
            process_video_pipeline,
            new_video.id,
            str(request.youtube_url),
            request.quality,
            request.detect_speakers,
            request.language,
            request.scene_detection_method
        )
        
        logger.info(f"Started processing for video: {new_video.id}")
        
        return VideoResponse(
            id=new_video.id,
            youtube_url=new_video.youtube_url,
            youtube_id=new_video.youtube_id,
            title=new_video.title,
            duration=new_video.duration,
            status=new_video.status,
            progress=new_video.progress,
            created_at=new_video.created_at,
            updated_at=new_video.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{video_id}", response_model=VideoResponse)
def get_video_status(video_id: str, db: Session = Depends(get_db)):
    """Get video processing status and results"""
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        return VideoResponse(
            id=video.id,
            youtube_url=video.youtube_url,
            youtube_id=video.youtube_id,
            title=video.title,
            duration=video.duration,
            status=video.status,
            progress=video.progress,
            created_at=video.created_at,
            updated_at=video.updated_at,
            error_message=video.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{video_id}/scenes", response_model=List[SceneResponse])
def get_video_scenes(video_id: str, db: Session = Depends(get_db)):
    """Get detected scenes for a video"""
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        scenes = db.query(Scene).filter(Scene.video_id == video_id).all()
        
        return [
            SceneResponse(
                id=scene.id,
                video_id=scene.video_id,
                start_time=scene.start_time,
                end_time=scene.end_time,
                scene_type=scene.scene_type,
                confidence=scene.confidence,
                content_analysis={
                    "avg_brightness": scene.avg_brightness,
                    "avg_contrast": scene.avg_contrast,
                    "dominant_colors": scene.dominant_colors
                }
            )
            for scene in scenes
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video scenes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{video_id}/transcript", response_model=List[TranscriptResponse])
def get_video_transcript(video_id: str, db: Session = Depends(get_db)):
    """Get transcript for a video"""
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        transcripts = db.query(Transcript).filter(Transcript.video_id == video_id).all()
        
        return [
            TranscriptResponse(
                id=transcript.id,
                video_id=transcript.video_id,
                start_time=transcript.start_time,
                end_time=transcript.end_time,
                text=transcript.text,
                confidence=transcript.confidence,
                speaker_id=transcript.speaker_id
            )
            for transcript in transcripts
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video transcript: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/", response_model=List[VideoResponse])
def list_videos(
    skip: int = 0, 
    limit: int = 100, 
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all videos with optional filtering"""
    try:
        query = db.query(Video)
        
        if status_filter and status_filter in VIDEO_STATUSES:
            query = query.filter(Video.status == status_filter)
        
        videos = query.offset(skip).limit(limit).all()
        
        return [
            VideoResponse(
                id=video.id,
                youtube_url=video.youtube_url,
                youtube_id=video.youtube_id,
                title=video.title,
                duration=video.duration,
                status=video.status,
                progress=video.progress,
                created_at=video.created_at,
                updated_at=video.updated_at,
                error_message=video.error_message
            )
            for video in videos
        ]
        
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(video_id: str, db: Session = Depends(get_db)):
    """Delete a video and all related data"""
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        # Delete video and related data (cascading delete handled by SQLAlchemy)
        db.delete(video)
        db.commit()
        
        logger.info(f"Deleted video: {video_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting video: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{video_id}/jobs", response_model=List[ProcessingJobResponse])
def get_video_jobs(video_id: str, db: Session = Depends(get_db)):
    """Get processing jobs for a video"""
    try:
        jobs = db.query(ProcessingJob).filter(ProcessingJob.reference_id == video_id).all()
        
        return [
            ProcessingJobResponse(
                id=job.id,
                job_type=job.job_type,
                reference_id=job.reference_id,
                status=job.status,
                progress=job.progress,
                created_at=job.created_at,
                error_message=job.error_message
            )
            for job in jobs
        ]
        
    except Exception as e:
        logger.error(f"Error getting video jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Background processing function
async def process_video_pipeline(
    video_id: str,
    youtube_url: str,
    quality: str,
    detect_speakers: bool,
    language: Optional[str],
    scene_detection_method: str
):
    """
    Background task for complete video processing pipeline
    """
    logger.info(f"Starting video processing pipeline for {video_id}")
    
    try:
        # Run the complete video processing pipeline
        result = await video_processor.process_video_complete(
            video_id=video_id,
            youtube_url=youtube_url,
            quality=quality,
            detect_speakers=detect_speakers,
            language=language,
            scene_detection_method=scene_detection_method
        )
        
        logger.info(f"Video processing pipeline completed for {video_id}")
        
    except Exception as e:
        logger.error(f"Error in video processing pipeline for {video_id}: {e}")