"""
Database models for ClipGen-AI video processing
"""
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class Video(Base):
    """Videos table for YouTube analysis"""
    __tablename__ = "videos"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    youtube_url = Column(String, nullable=False, unique=True)
    youtube_id = Column(String, nullable=True, index=True)  # Extracted from URL
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    duration = Column(Float, nullable=True)  # Duration in seconds
    status = Column(String, nullable=False, default="pending")  # pending, processing, completed, failed
    progress = Column(Integer, nullable=False, default=0)  # Progress percentage 0-100
    file_path = Column(String, nullable=True)  # Local path to downloaded video
    thumbnail_path = Column(String, nullable=True)  # Local path to thumbnail
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    youtube_metadata = Column(JSON, nullable=True)  # Additional YouTube metadata
    error_message = Column(Text, nullable=True)  # Error details if processing failed
    
    # Relationships
    scenes = relationship("Scene", back_populates="video", cascade="all, delete-orphan")
    transcripts = relationship("Transcript", back_populates="video", cascade="all, delete-orphan")
    highlights = relationship("Highlight", back_populates="video", cascade="all, delete-orphan")


class Scene(Base):
    """Scenes table for detected video scenes"""
    __tablename__ = "scenes"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    start_time = Column(Float, nullable=False)  # Start time in seconds
    end_time = Column(Float, nullable=False)    # End time in seconds
    scene_type = Column(String, nullable=True)  # cut, fade, dissolve, etc.
    confidence = Column(Float, nullable=True)   # Detection confidence score
    frame_count = Column(Integer, nullable=True)  # Number of frames in scene
    avg_brightness = Column(Float, nullable=True)  # Average brightness
    avg_contrast = Column(Float, nullable=True)   # Average contrast
    dominant_colors = Column(JSON, nullable=True)  # RGB values of dominant colors
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    video = relationship("Video", back_populates="scenes")


class Transcript(Base):
    """Transcripts table for audio transcription"""
    __tablename__ = "transcripts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    start_time = Column(Float, nullable=False)  # Start time in seconds
    end_time = Column(Float, nullable=False)    # End time in seconds
    text = Column(Text, nullable=False)         # Transcribed text
    confidence = Column(Float, nullable=True)   # Transcription confidence
    speaker_id = Column(Integer, nullable=True)  # Speaker identification (if multiple)
    language = Column(String, nullable=True)    # Detected language
    word_count = Column(Integer, nullable=True)  # Number of words
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    video = relationship("Video", back_populates="transcripts")


class Highlight(Base):
    """Highlights table for detected video highlights"""
    __tablename__ = "highlights"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    start_time = Column(Float, nullable=False)   # Start time in seconds
    end_time = Column(Float, nullable=False)     # End time in seconds
    score = Column(Float, nullable=False)        # Highlight score (0.0 - 1.0)
    tags = Column(JSON, nullable=True)           # Category tags (action, dialogue, etc.)
    reason = Column(Text, nullable=True)         # Explanation for why this is a highlight
    visual_features = Column(JSON, nullable=True)  # Visual analysis data
    audio_features = Column(JSON, nullable=True)   # Audio analysis data
    text_features = Column(JSON, nullable=True)    # Text sentiment data
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    video = relationship("Video", back_populates="highlights")


class ProcessingJob(Base):
    """Processing jobs table for background tasks"""
    __tablename__ = "processing_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_type = Column(String, nullable=False)     # download, scene_detection, transcription, etc.
    reference_id = Column(String, nullable=False)  # ID of related object (video_id, etc.)
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, failed
    progress = Column(Integer, nullable=False, default=0)  # Progress percentage 0-100
    result_data = Column(JSON, nullable=True)     # Job result data
    error_message = Column(Text, nullable=True)   # Error details if job failed
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Job metadata
    job_metadata = Column(JSON, nullable=True)        # Additional job-specific data


# Define valid statuses for better validation
VIDEO_STATUSES = ["pending", "downloading", "processing", "completed", "failed"]
JOB_STATUSES = ["pending", "running", "completed", "failed", "cancelled"]
JOB_TYPES = ["download", "scene_detection", "transcription", "highlight_detection", "analysis"]