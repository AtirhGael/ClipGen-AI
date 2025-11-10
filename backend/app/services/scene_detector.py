"""
Scene Detection Service using OpenCV and PySceneDetect
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector, ThresholdDetector
from scenedetect.video_splitter import split_video_ffmpeg

logger = logging.getLogger(__name__)


class SceneDetectionError(Exception):
    """Custom exception for scene detection errors"""
    pass


class SceneDetector:
    """Advanced scene detection with multiple algorithms"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def detect_scenes_pyscenedetect(
        self, 
        video_path: str, 
        threshold: float = 30.0,
        min_scene_len: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Detect scenes using PySceneDetect (content-based detection)
        
        Args:
            video_path: Path to video file
            threshold: Content detection threshold
            min_scene_len: Minimum scene length in seconds
            
        Returns:
            List of scene dictionaries with start/end times
        """
        try:
            # Create video manager and scene manager
            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()
            
            # Add content detector
            scene_manager.add_detector(
                ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
            )
            
            # Set video manager and detect scenes
            video_manager.set_video_stream(0)
            video_manager.start()
            
            scene_manager.detect_scenes(frame_source=video_manager)
            
            # Get scene list
            scene_list = scene_manager.get_scene_list(start_in_scene=True)
            
            scenes = []
            for i, (start_time, end_time) in enumerate(scene_list):
                scene = {
                    'scene_number': i + 1,
                    'start_time': start_time.get_seconds(),
                    'end_time': end_time.get_seconds(),
                    'duration': (end_time - start_time).get_seconds(),
                    'detection_method': 'content_detector',
                    'threshold_used': threshold,
                    'confidence': 1.0  # PySceneDetect doesn't provide confidence
                }
                scenes.append(scene)
            
            video_manager.release()
            
            logger.info(f"Detected {len(scenes)} scenes using PySceneDetect")
            return scenes
            
        except Exception as e:
            logger.error(f"Error in PySceneDetect: {e}")
            raise SceneDetectionError(f"PySceneDetect failed: {e}")
    
    def detect_scenes_opencv(
        self, 
        video_path: str,
        threshold: float = 0.3,
        frame_skip: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Detect scenes using OpenCV frame difference analysis
        
        Args:
            video_path: Path to video file
            threshold: Frame difference threshold (0.0-1.0)
            frame_skip: Number of frames to skip between comparisons
            
        Returns:
            List of scene dictionaries
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise SceneDetectionError(f"Cannot open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            scenes = []
            scene_boundaries = []
            
            prev_frame = None
            frame_count = 0
            
            logger.info(f"Starting OpenCV scene detection on {total_frames} frames at {fps} FPS")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    # Convert to grayscale for comparison
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if prev_frame is not None:
                        # Calculate frame difference
                        diff = cv2.absdiff(prev_frame, gray_frame)
                        diff_norm = np.sum(diff) / (diff.shape[0] * diff.shape[1] * 255.0)
                        
                        # Check if difference exceeds threshold
                        if diff_norm > threshold:
                            scene_boundaries.append(frame_count / fps)
                    
                    prev_frame = gray_frame
                
                frame_count += 1
            
            cap.release()
            
            # Convert boundaries to scenes
            scene_boundaries = [0.0] + scene_boundaries + [total_frames / fps]
            
            for i in range(len(scene_boundaries) - 1):
                scene = {
                    'scene_number': i + 1,
                    'start_time': scene_boundaries[i],
                    'end_time': scene_boundaries[i + 1],
                    'duration': scene_boundaries[i + 1] - scene_boundaries[i],
                    'detection_method': 'opencv_framediff',
                    'threshold_used': threshold,
                    'confidence': 0.8  # Fixed confidence for OpenCV method
                }
                scenes.append(scene)
            
            logger.info(f"Detected {len(scenes)} scenes using OpenCV")
            return scenes
            
        except Exception as e:
            logger.error(f"Error in OpenCV scene detection: {e}")
            raise SceneDetectionError(f"OpenCV scene detection failed: {e}")
    
    def analyze_scene_content(
        self, 
        video_path: str, 
        start_time: float, 
        end_time: float
    ) -> Dict[str, Any]:
        """
        Analyze visual content within a scene
        
        Args:
            video_path: Path to video file
            start_time: Scene start time in seconds
            end_time: Scene end time in seconds
            
        Returns:
            Dictionary with scene analysis data
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            brightness_values = []
            contrast_values = []
            motion_values = []
            color_histograms = []
            
            prev_gray = None
            frame_count = 0
            sample_interval = max(1, (end_frame - start_frame) // 10)  # Sample 10 frames
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_interval == 0:
                    # Convert to different color spaces
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    
                    # Calculate brightness (average pixel value)
                    brightness = np.mean(gray)
                    brightness_values.append(brightness)
                    
                    # Calculate contrast (standard deviation of pixel values)
                    contrast = np.std(gray)
                    contrast_values.append(contrast)
                    
                    # Calculate motion (if not first frame)
                    if prev_gray is not None:
                        motion = np.mean(cv2.absdiff(prev_gray, gray))
                        motion_values.append(motion)
                    
                    # Color histogram
                    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
                    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
                    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
                    
                    color_histograms.append({
                        'hue': hist_h.flatten().tolist(),
                        'saturation': hist_s.flatten().tolist(),
                        'value': hist_v.flatten().tolist()
                    })
                    
                    prev_gray = gray
                
                frame_count += 1
            
            cap.release()
            
            # Calculate statistics
            analysis = {
                'avg_brightness': float(np.mean(brightness_values)) if brightness_values else 0.0,
                'avg_contrast': float(np.mean(contrast_values)) if contrast_values else 0.0,
                'avg_motion': float(np.mean(motion_values)) if motion_values else 0.0,
                'brightness_std': float(np.std(brightness_values)) if brightness_values else 0.0,
                'contrast_std': float(np.std(contrast_values)) if contrast_values else 0.0,
                'motion_std': float(np.std(motion_values)) if motion_values else 0.0,
                'frame_count': len(brightness_values),
                'dominant_colors': self._extract_dominant_colors(color_histograms) if color_histograms else []
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing scene content: {e}")
            return {}
    
    def _extract_dominant_colors(self, color_histograms: List[Dict]) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from color histograms"""
        try:
            if not color_histograms:
                return []
            
            # Average the histograms
            avg_hue = np.mean([h['hue'] for h in color_histograms], axis=0)
            avg_sat = np.mean([h['saturation'] for h in color_histograms], axis=0)
            avg_val = np.mean([h['value'] for h in color_histograms], axis=0)
            
            # Find peaks in hue histogram (dominant colors)
            hue_peaks = []
            for i in range(1, len(avg_hue) - 1):
                if avg_hue[i] > avg_hue[i-1] and avg_hue[i] > avg_hue[i+1]:
                    if avg_hue[i] > np.mean(avg_hue) * 2:  # Significant peak
                        hue_peaks.append(i)
            
            # Convert top 3 hue peaks to RGB
            hue_peaks = sorted(hue_peaks, key=lambda x: avg_hue[x], reverse=True)[:3]
            dominant_colors = []
            
            for hue in hue_peaks:
                # Use average saturation and value
                sat = np.argmax(avg_sat)
                val = np.argmax(avg_val)
                
                # Convert HSV to RGB
                hsv_color = np.uint8([[[hue, sat, val]]])
                rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
                dominant_colors.append(tuple(rgb_color.tolist()))
            
            return dominant_colors
            
        except Exception as e:
            logger.error(f"Error extracting dominant colors: {e}")
            return []
    
    def detect_scenes_combined(
        self, 
        video_path: str,
        use_pyscenedetect: bool = True,
        use_opencv: bool = False,
        pyscene_threshold: float = 30.0,
        opencv_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple scene detection methods
        
        Args:
            video_path: Path to video file
            use_pyscenedetect: Whether to use PySceneDetect
            use_opencv: Whether to use OpenCV method
            pyscene_threshold: PySceneDetect threshold
            opencv_threshold: OpenCV threshold
            
        Returns:
            Combined and refined scene list
        """
        all_scenes = []
        
        try:
            if use_pyscenedetect:
                pyscene_scenes = self.detect_scenes_pyscenedetect(
                    video_path, 
                    threshold=pyscene_threshold
                )
                all_scenes.extend(pyscene_scenes)
            
            if use_opencv:
                opencv_scenes = self.detect_scenes_opencv(
                    video_path,
                    threshold=opencv_threshold
                )
                all_scenes.extend(opencv_scenes)
            
            if not all_scenes:
                raise SceneDetectionError("No scenes detected by any method")
            
            # If using both methods, merge similar scenes
            if use_pyscenedetect and use_opencv:
                all_scenes = self._merge_similar_scenes(all_scenes)
            
            # Add content analysis to each scene
            for scene in all_scenes:
                scene['content_analysis'] = self.analyze_scene_content(
                    video_path,
                    scene['start_time'],
                    scene['end_time']
                )
            
            return sorted(all_scenes, key=lambda x: x['start_time'])
            
        except Exception as e:
            logger.error(f"Error in combined scene detection: {e}")
            raise SceneDetectionError(f"Combined scene detection failed: {e}")
    
    def _merge_similar_scenes(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge scenes that are very similar in timing"""
        if len(scenes) <= 1:
            return scenes
        
        # Sort by start time
        scenes = sorted(scenes, key=lambda x: x['start_time'])
        merged = []
        
        i = 0
        while i < len(scenes):
            current_scene = scenes[i]
            
            # Look for overlapping or very close scenes
            j = i + 1
            while j < len(scenes):
                next_scene = scenes[j]
                
                # Check if scenes overlap or are very close (within 2 seconds)
                if (abs(current_scene['start_time'] - next_scene['start_time']) <= 2.0 or
                    abs(current_scene['end_time'] - next_scene['end_time']) <= 2.0):
                    
                    # Merge scenes - take the wider boundaries
                    current_scene = {
                        'scene_number': len(merged) + 1,
                        'start_time': min(current_scene['start_time'], next_scene['start_time']),
                        'end_time': max(current_scene['end_time'], next_scene['end_time']),
                        'detection_method': f"{current_scene['detection_method']}, {next_scene['detection_method']}",
                        'confidence': (current_scene['confidence'] + next_scene['confidence']) / 2
                    }
                    j += 1
                else:
                    break
            
            current_scene['duration'] = current_scene['end_time'] - current_scene['start_time']
            merged.append(current_scene)
            i = j
        
        # Renumber scenes
        for idx, scene in enumerate(merged):
            scene['scene_number'] = idx + 1
        
        return merged
    
    async def detect_scenes_async(
        self, 
        video_path: str, 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Async wrapper for scene detection"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.detect_scenes_combined,
            video_path,
            **kwargs
        )


# Global scene detector instance
scene_detector = SceneDetector()