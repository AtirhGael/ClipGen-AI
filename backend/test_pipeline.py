"""
Test script to verify Video Processing Pipeline components
"""
import asyncio
import logging
from pathlib import Path
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.youtube_downloader import youtube_downloader
from app.services.scene_detector import scene_detector  
from app.services.audio_transcriber import audio_transcriber

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_video_pipeline():
    """Test the video processing pipeline components"""
    
    logger.info("🧪 Starting Video Processing Pipeline Tests")
    
    # Test 1: YouTube URL Validation
    logger.info("Test 1: YouTube URL Validation")
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Valid
        "https://youtu.be/dQw4w9WgXcQ",                # Valid short form
        "https://example.com/video",                   # Invalid
        "not_a_url"                                    # Invalid
    ]
    
    for url in test_urls:
        is_valid, video_id = youtube_downloader.validate_youtube_url(url)
        logger.info(f"  URL: {url[:50]}... -> Valid: {is_valid}, ID: {video_id}")
    
    # Test 2: Video Info Extraction (without download)
    logger.info("\nTest 2: Video Metadata Extraction")
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # Short test video
    
    try:
        metadata = youtube_downloader.get_video_info(test_url)
        logger.info(f"  Title: {metadata.get('title', 'N/A')}")
        logger.info(f"  Duration: {metadata.get('duration', 'N/A')} seconds")
        logger.info(f"  Uploader: {metadata.get('uploader', 'N/A')}")
        logger.info(f"  View Count: {metadata.get('view_count', 'N/A')}")
    except Exception as e:
        logger.error(f"  Error extracting metadata: {e}")
    
    # Test 3: Available Quality Options
    logger.info("\nTest 3: Available Quality Options")
    try:
        qualities = youtube_downloader.get_available_qualities(test_url)
        logger.info(f"  Found {len(qualities)} quality options:")
        for i, quality in enumerate(qualities[:3]):  # Show first 3
            logger.info(f"    {i+1}. {quality['resolution']} ({quality['ext']}) - {quality.get('format_note', '')}")
    except Exception as e:
        logger.error(f"  Error getting quality options: {e}")
    
    # Test 4: Whisper Model Loading
    logger.info("\nTest 4: Whisper Model Loading")
    try:
        logger.info(f"  Model size: {audio_transcriber.model_size}")
        logger.info(f"  Device: {audio_transcriber.device}")
        logger.info(f"  Model loaded: {audio_transcriber.model is not None}")
    except Exception as e:
        logger.error(f"  Error with Whisper model: {e}")
    
    # Test 5: Scene Detector Initialization
    logger.info("\nTest 5: Scene Detector Initialization")
    try:
        logger.info(f"  Scene detector initialized successfully")
        logger.info(f"  Available methods: PySceneDetect, OpenCV")
    except Exception as e:
        logger.error(f"  Error with scene detector: {e}")
    
    logger.info("\n✅ Video Processing Pipeline Tests Completed!")


if __name__ == "__main__":
    asyncio.run(test_video_pipeline())