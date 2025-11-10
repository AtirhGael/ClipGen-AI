"""
YouTube Video Download Service using yt-dlp
"""
import os
import re
import logging
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import yt_dlp
from urllib.parse import urlparse, parse_qs
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.config import settings

logger = logging.getLogger(__name__)


class YouTubeDownloadError(Exception):
    """Custom exception for YouTube download errors"""
    pass


class YouTubeDownloader:
    """YouTube video downloader with progress tracking"""
    
    def __init__(self):
        self.download_dir = Path(settings.download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def validate_youtube_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate YouTube URL and extract video ID
        
        Args:
            url: YouTube URL to validate
            
        Returns:
            Tuple of (is_valid, video_id)
        """
        try:
            # Common YouTube URL patterns
            patterns = [
                r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)',
                r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]+)',
                r'(?:https?:\/\/)?youtu\.be\/([a-zA-Z0-9_-]+)',
                r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    video_id = match.group(1)
                    return True, video_id
                    
            return False, None
            
        except Exception as e:
            logger.error(f"Error validating YouTube URL: {e}")
            return False, None
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Extract video metadata without downloading
        
        Args:
            url: YouTube URL
            
        Returns:
            Dictionary with video metadata
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'no_download': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Extract relevant metadata
                metadata = {
                    'id': info.get('id'),
                    'title': info.get('title'),
                    'description': info.get('description'),
                    'duration': info.get('duration'),  # seconds
                    'view_count': info.get('view_count'),
                    'like_count': info.get('like_count'),
                    'upload_date': info.get('upload_date'),
                    'uploader': info.get('uploader'),
                    'uploader_id': info.get('uploader_id'),
                    'thumbnail': info.get('thumbnail'),
                    'categories': info.get('categories', []),
                    'tags': info.get('tags', []),
                    'formats': len(info.get('formats', [])),
                    'resolution': info.get('resolution'),
                    'fps': info.get('fps'),
                    'vcodec': info.get('vcodec'),
                    'acodec': info.get('acodec'),
                }
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting video info: {e}")
            raise YouTubeDownloadError(f"Failed to extract video info: {e}")
    
    def get_available_qualities(self, url: str) -> List[Dict[str, Any]]:
        """
        Get available video quality options
        
        Args:
            url: YouTube URL
            
        Returns:
            List of available quality formats
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'listformats': True,
                'no_download': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                formats = info.get('formats', [])
                
                # Filter and organize quality options
                quality_options = []
                seen_qualities = set()
                
                for fmt in formats:
                    if fmt.get('vcodec') != 'none' and fmt.get('acodec') != 'none':
                        quality = {
                            'format_id': fmt.get('format_id'),
                            'ext': fmt.get('ext'),
                            'resolution': fmt.get('resolution') or f"{fmt.get('width', 'unknown')}x{fmt.get('height', 'unknown')}",
                            'fps': fmt.get('fps'),
                            'filesize': fmt.get('filesize'),
                            'vcodec': fmt.get('vcodec'),
                            'acodec': fmt.get('acodec'),
                            'quality': fmt.get('quality', 0),
                            'format_note': fmt.get('format_note', '')
                        }
                        
                        quality_key = (quality['resolution'], quality['fps'])
                        if quality_key not in seen_qualities:
                            quality_options.append(quality)
                            seen_qualities.add(quality_key)
                
                # Sort by quality (higher is better)
                quality_options.sort(key=lambda x: x.get('quality', 0), reverse=True)
                return quality_options
                
        except Exception as e:
            logger.error(f"Error getting quality options: {e}")
            raise YouTubeDownloadError(f"Failed to get quality options: {e}")
    
    def _progress_hook(self, progress_callback, d):
        """Progress hook for yt-dlp"""
        if d['status'] == 'downloading':
            try:
                if 'total_bytes' in d:
                    percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
                elif 'total_bytes_estimate' in d:
                    percent = (d['downloaded_bytes'] / d['total_bytes_estimate']) * 100
                else:
                    percent = 0
                
                progress_data = {
                    'status': 'downloading',
                    'percent': min(100, max(0, int(percent))),
                    'downloaded_bytes': d.get('downloaded_bytes', 0),
                    'total_bytes': d.get('total_bytes') or d.get('total_bytes_estimate', 0),
                    'speed': d.get('speed', 0),
                    'eta': d.get('eta', 0)
                }
                
                if progress_callback:
                    progress_callback(progress_data)
                    
            except Exception as e:
                logger.error(f"Error in progress hook: {e}")
        
        elif d['status'] == 'finished':
            if progress_callback:
                progress_callback({
                    'status': 'finished',
                    'percent': 100,
                    'filename': d.get('filename')
                })
    
    def download_video(
        self, 
        url: str, 
        video_id: str,
        quality: str = "best",
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Download video with specified quality
        
        Args:
            url: YouTube URL
            video_id: Video ID for file naming
            quality: Quality selector (best, worst, or specific format)
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary with download results
        """
        try:
            output_path = self.download_dir / f"{video_id}.%(ext)s"
            
            ydl_opts = {
                'outtmpl': str(output_path),
                'format': quality,
                'writethumbnail': True,
                'writeinfojson': True,
                'ignoreerrors': False,
                'no_warnings': False,
                'progress_hooks': [lambda d: self._progress_hook(progress_callback, d)],
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # Find the downloaded files
                video_file = None
                thumbnail_file = None
                info_file = None
                
                base_path = self.download_dir / video_id
                for file_path in self.download_dir.glob(f"{video_id}.*"):
                    if file_path.suffix in ['.mp4', '.webm', '.mkv', '.avi']:
                        video_file = str(file_path)
                    elif file_path.suffix in ['.jpg', '.png', '.webp']:
                        thumbnail_file = str(file_path)
                    elif file_path.suffix == '.info.json':
                        info_file = str(file_path)
                
                return {
                    'success': True,
                    'video_file': video_file,
                    'thumbnail_file': thumbnail_file,
                    'info_file': info_file,
                    'title': info.get('title'),
                    'duration': info.get('duration'),
                    'file_size': os.path.getsize(video_file) if video_file and os.path.exists(video_file) else 0
                }
                
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise YouTubeDownloadError(f"Failed to download video: {e}")
    
    async def download_video_async(
        self, 
        url: str, 
        video_id: str,
        quality: str = "best",
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Async wrapper for video download
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.download_video, 
            url, 
            video_id, 
            quality, 
            progress_callback
        )
    
    def cleanup_temp_files(self, video_id: str):
        """Clean up temporary files for a video"""
        try:
            for file_path in self.download_dir.glob(f"{video_id}.*"):
                if file_path.suffix in ['.part', '.tmp', '.temp']:
                    file_path.unlink()
                    logger.info(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
    
    def get_download_progress(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get current download progress for a video"""
        # This would typically be stored in Redis or database
        # For now, we'll return None - implement with actual storage later
        return None


# Global downloader instance
youtube_downloader = YouTubeDownloader()