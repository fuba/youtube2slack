"""Lightweight scene change detection module."""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class SceneChangeDetector:
    """Lightweight scene change detector using frame difference and histogram comparison."""

    def __init__(self, 
                 diff_threshold: float = 0.3,
                 hist_threshold: float = 0.7, 
                 resize_dims: Tuple[int, int] = (320, 240),
                 check_interval: float = 1.0):
        """Initialize scene detector.
        
        Args:
            diff_threshold: Threshold for pixel difference (0-1)
            hist_threshold: Threshold for histogram similarity (0-1, higher = more similar)
            resize_dims: Dimensions to resize frames for processing
            check_interval: Minimum interval between scene checks in seconds
        """
        self.diff_threshold = diff_threshold
        self.hist_threshold = hist_threshold
        self.resize_dims = resize_dims
        self.check_interval = check_interval
        
        self.last_frame: Optional[np.ndarray] = None
        self.last_hist: Optional[np.ndarray] = None
        self.last_check_time: float = 0.0
        
        logger.info(f"Scene detector initialized: diff_thresh={diff_threshold}, "
                   f"hist_thresh={hist_threshold}, resize={resize_dims}")

    def detect_scene_change(self, frame: np.ndarray) -> bool:
        """Detect if current frame represents a scene change.
        
        Args:
            frame: Current video frame (BGR format)
            
        Returns:
            True if scene change detected
        """
        current_time = time.time()
        
        # Skip if too soon since last check
        if current_time - self.last_check_time < self.check_interval:
            return False
            
        self.last_check_time = current_time
        
        # Resize frame for lightweight processing
        small_frame = cv2.resize(frame, self.resize_dims)
        
        # Convert to RGB for histogram
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # If this is the first frame, store and return False
        if self.last_frame is None:
            self.last_frame = small_frame.copy()
            self.last_hist = self._calculate_histogram(rgb_frame)
            return False
        
        # Calculate frame difference
        frame_diff = self._calculate_frame_difference(small_frame, self.last_frame)
        
        # Calculate histogram similarity
        current_hist = self._calculate_histogram(rgb_frame)
        hist_similarity = self._calculate_histogram_similarity(self.last_hist, current_hist)
        
        # Determine if scene changed
        scene_changed = (frame_diff > self.diff_threshold and 
                        hist_similarity < self.hist_threshold)
        
        if scene_changed:
            logger.info(f"Scene change detected: frame_diff={frame_diff:.3f}, "
                       f"hist_sim={hist_similarity:.3f}")
        
        # Update stored frame and histogram
        self.last_frame = small_frame.copy()
        self.last_hist = current_hist.copy()
        
        return scene_changed

    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate normalized pixel difference between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Normalized difference (0-1)
        """
        # Convert to grayscale for difference calculation
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Calculate mean difference normalized by max pixel value
        mean_diff = np.mean(diff) / 255.0
        
        return mean_diff

    def _calculate_histogram(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Calculate RGB histogram for frame.
        
        Args:
            rgb_frame: Frame in RGB format
            
        Returns:
            Concatenated RGB histogram
        """
        # Calculate histogram for each channel
        hist_r = cv2.calcHist([rgb_frame], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([rgb_frame], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([rgb_frame], [2], None, [32], [0, 256])
        
        # Normalize histograms
        hist_r = hist_r.flatten() / np.sum(hist_r)
        hist_g = hist_g.flatten() / np.sum(hist_g)
        hist_b = hist_b.flatten() / np.sum(hist_b)
        
        # Concatenate all channels
        return np.concatenate([hist_r, hist_g, hist_b])

    def _calculate_histogram_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Calculate histogram similarity using correlation.
        
        Args:
            hist1: First histogram
            hist2: Second histogram
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        # Use correlation coefficient as similarity metric
        correlation = cv2.compareHist(hist1.astype(np.float32), 
                                     hist2.astype(np.float32), 
                                     cv2.HISTCMP_CORREL)
        
        # Ensure value is in [0, 1] range
        return max(0.0, correlation)

    def reset(self) -> None:
        """Reset detector state."""
        self.last_frame = None
        self.last_hist = None
        self.last_check_time = 0.0
        logger.info("Scene detector reset")

    def get_status(self) -> Dict[str, Any]:
        """Get detector status.
        
        Returns:
            Status dictionary
        """
        return {
            'has_reference_frame': self.last_frame is not None,
            'diff_threshold': self.diff_threshold,
            'hist_threshold': self.hist_threshold,
            'resize_dims': self.resize_dims,
            'check_interval': self.check_interval,
            'last_check_time': self.last_check_time
        }


class StreamSceneCapture:
    """Capture frames from video stream for scene change detection."""

    def __init__(self, detector: SceneChangeDetector, output_dir: str = "/tmp"):
        """Initialize stream scene capture.
        
        Args:
            detector: Scene change detector instance
            output_dir: Directory to save captured frames
        """
        self.detector = detector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.frame_count = 0
        self.scene_count = 0
        
        logger.info(f"Stream scene capture initialized, output: {self.output_dir}")

    def process_stream_url(self, stream_url: str, max_scenes: int = 10) -> bool:
        """Process video stream and capture scene changes.
        
        Args:
            stream_url: Video stream URL
            max_scenes: Maximum number of scenes to capture
            
        Returns:
            True if successful
        """
        try:
            # Open video stream
            cap = cv2.VideoCapture(stream_url)
            
            if not cap.isOpened():
                logger.error(f"Failed to open stream: {stream_url}")
                return False
            
            logger.info(f"Processing stream: {stream_url}")
            
            while self.scene_count < max_scenes:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame, stream may have ended")
                    break
                
                self.frame_count += 1
                
                # Check for scene change
                if self.detector.detect_scene_change(frame):
                    # Save current frame
                    frame_path = self.output_dir / f"scene_{self.scene_count:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    self.scene_count += 1
                    
                    logger.info(f"Scene {self.scene_count} captured: {frame_path}")
            
            cap.release()
            logger.info(f"Stream processing completed: {self.scene_count} scenes captured")
            return True
            
        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            return False

    def get_captured_frames(self) -> list[Path]:
        """Get list of captured frame files.
        
        Returns:
            List of captured frame file paths
        """
        return sorted(self.output_dir.glob("scene_*.jpg"))

    def cleanup_frames(self) -> None:
        """Remove all captured frames."""
        for frame_path in self.get_captured_frames():
            try:
                frame_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove {frame_path}: {e}")
        
        logger.info("Captured frames cleaned up")