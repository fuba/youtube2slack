"""Tests for scene detection module."""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from youtube2slack.scene_detector import SceneChangeDetector, StreamSceneCapture


class TestSceneChangeDetector:
    """Test SceneChangeDetector functionality."""

    def test_detector_initialization(self):
        """Test detector initialization with default parameters."""
        detector = SceneChangeDetector()
        
        assert detector.diff_threshold == 0.3
        assert detector.hist_threshold == 0.7
        assert detector.resize_dims == (320, 240)
        assert detector.check_interval == 1.0
        assert detector.last_frame is None

    def test_detector_initialization_custom_params(self):
        """Test detector initialization with custom parameters."""
        detector = SceneChangeDetector(
            diff_threshold=0.5,
            hist_threshold=0.8,
            resize_dims=(640, 480),
            check_interval=2.0
        )
        
        assert detector.diff_threshold == 0.5
        assert detector.hist_threshold == 0.8
        assert detector.resize_dims == (640, 480)
        assert detector.check_interval == 2.0

    def test_first_frame_no_change(self):
        """Test that first frame never triggers scene change."""
        detector = SceneChangeDetector(check_interval=0.0)  # No time delay
        
        # Create a test frame (blue image)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [255, 0, 0]  # Blue frame
        
        # First frame should never trigger change
        result = detector.detect_scene_change(frame)
        assert result == False
        assert detector.last_frame is not None

    def test_identical_frames_no_change(self):
        """Test that identical frames don't trigger scene change."""
        detector = SceneChangeDetector(check_interval=0.0)
        
        # Create identical test frames
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1[:, :] = [255, 0, 0]  # Blue frame
        frame2 = frame1.copy()
        
        # Process first frame
        detector.detect_scene_change(frame1)
        
        # Second identical frame should not trigger change
        result = detector.detect_scene_change(frame2)
        assert result == False

    def test_very_different_frames_trigger_change(self):
        """Test that very different frames trigger scene change."""
        detector = SceneChangeDetector(
            diff_threshold=0.2,
            hist_threshold=0.8,
            check_interval=0.0
        )
        
        # Create very different frames
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1[:, :] = [255, 0, 0]  # Blue frame
        
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[:, :] = [0, 255, 0]  # Green frame
        
        # Process first frame
        detector.detect_scene_change(frame1)
        
        # Very different second frame should trigger change
        result = detector.detect_scene_change(frame2)
        assert result == True

    def test_subtle_changes_no_trigger(self):
        """Test that subtle changes don't trigger scene change."""
        detector = SceneChangeDetector(
            diff_threshold=0.3,
            hist_threshold=0.7,
            check_interval=0.0
        )
        
        # Create slightly different frames
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1[:, :] = [100, 100, 100]  # Gray frame
        
        frame2 = frame1.copy()
        frame2[0:100, 0:100] = [110, 110, 110]  # Slightly lighter corner
        
        # Process first frame
        detector.detect_scene_change(frame1)
        
        # Subtle change should not trigger detection
        result = detector.detect_scene_change(frame2)
        assert result == False

    def test_check_interval_respected(self):
        """Test that check interval is respected."""
        detector = SceneChangeDetector(check_interval=1.0)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [255, 0, 0]
        
        # Process first frame
        result1 = detector.detect_scene_change(frame)
        
        # Immediate second check should be skipped
        result2 = detector.detect_scene_change(frame)
        assert result2 == False  # Should be skipped due to interval

    def test_reset_functionality(self):
        """Test detector reset functionality."""
        detector = SceneChangeDetector(check_interval=0.0)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect_scene_change(frame)
        
        # Verify state is set
        assert detector.last_frame is not None
        assert detector.last_hist is not None
        
        # Reset and verify state is cleared
        detector.reset()
        assert detector.last_frame is None
        assert detector.last_hist is None
        assert detector.last_check_time == 0.0

    def test_get_status(self):
        """Test status reporting."""
        detector = SceneChangeDetector()
        status = detector.get_status()
        
        expected_keys = {
            'has_reference_frame', 'diff_threshold', 'hist_threshold',
            'resize_dims', 'check_interval', 'last_check_time'
        }
        assert set(status.keys()) == expected_keys
        assert status['has_reference_frame'] == False


class TestStreamSceneCapture:
    """Test StreamSceneCapture functionality."""

    def test_initialization(self):
        """Test StreamSceneCapture initialization."""
        detector = SceneChangeDetector()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            capture = StreamSceneCapture(detector, temp_dir)
            
            assert capture.detector == detector
            assert capture.output_dir == Path(temp_dir)
            assert capture.frame_count == 0
            assert capture.scene_count == 0

    def test_get_captured_frames_empty(self):
        """Test getting captured frames from empty directory."""
        detector = SceneChangeDetector()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            capture = StreamSceneCapture(detector, temp_dir)
            frames = capture.get_captured_frames()
            assert len(frames) == 0

    def test_get_captured_frames_with_files(self):
        """Test getting captured frames with existing files."""
        detector = SceneChangeDetector()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            capture = StreamSceneCapture(detector, temp_dir)
            
            # Create test files
            test_files = ["scene_0001.jpg", "scene_0002.jpg", "other.txt"]
            for filename in test_files:
                (Path(temp_dir) / filename).touch()
            
            frames = capture.get_captured_frames()
            assert len(frames) == 2  # Only .jpg files
            assert all(f.suffix == ".jpg" for f in frames)

    def test_cleanup_frames(self):
        """Test cleanup functionality."""
        detector = SceneChangeDetector()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            capture = StreamSceneCapture(detector, temp_dir)
            
            # Create test files
            test_files = ["scene_0001.jpg", "scene_0002.jpg"]
            for filename in test_files:
                (Path(temp_dir) / filename).touch()
            
            # Verify files exist
            assert len(capture.get_captured_frames()) == 2
            
            # Cleanup and verify files are removed
            capture.cleanup_frames()
            assert len(capture.get_captured_frames()) == 0

    @patch('cv2.VideoCapture')
    def test_process_stream_url_open_failure(self, mock_video_capture):
        """Test stream processing with video capture failure."""
        # Mock failed video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        detector = SceneChangeDetector()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            capture = StreamSceneCapture(detector, temp_dir)
            result = capture.process_stream_url("fake_url")
            assert result == False

    @patch('cv2.VideoCapture')
    def test_process_stream_url_success(self, mock_video_capture):
        """Test successful stream processing."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        
        # Create test frames
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1[:, :] = [100, 100, 100]
        
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[:, :] = [200, 200, 200]  # Different frame
        
        # Mock read sequence: success, success, failure (to end)
        mock_cap.read.side_effect = [
            (True, frame1),
            (True, frame2),
            (False, None)  # End of stream
        ]
        
        mock_video_capture.return_value = mock_cap
        
        # Use detector that will detect change
        detector = SceneChangeDetector(
            diff_threshold=0.1,
            hist_threshold=0.9,
            check_interval=0.0
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            capture = StreamSceneCapture(detector, temp_dir)
            
            with patch('cv2.imwrite') as mock_imwrite:
                mock_imwrite.return_value = True
                result = capture.process_stream_url("fake_url", max_scenes=1)
                
                assert result == True
                # Should have processed frames
                assert capture.frame_count > 0


def test_frame_difference_calculation():
    """Test frame difference calculation."""
    detector = SceneChangeDetector()
    
    # Create identical frames
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame1.fill(128)
    frame2 = frame1.copy()
    
    diff = detector._calculate_frame_difference(frame1, frame2)
    assert diff == 0.0
    
    # Create different frames
    frame3 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame3.fill(255)
    
    diff = detector._calculate_frame_difference(frame1, frame3)
    assert diff > 0.4  # Should be significant difference


def test_histogram_calculation():
    """Test histogram calculation."""
    detector = SceneChangeDetector()
    
    # Create test frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = [255, 128, 64]  # RGB values
    
    hist = detector._calculate_histogram(frame)
    
    # Should return concatenated RGB histograms
    assert len(hist) == 96  # 32 bins * 3 channels
    assert np.sum(hist) == pytest.approx(3.0, rel=1e-3)  # Normalized histograms sum to 3


def test_histogram_similarity():
    """Test histogram similarity calculation."""
    detector = SceneChangeDetector()
    
    # Create identical histograms
    hist1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    hist2 = hist1.copy()
    
    similarity = detector._calculate_histogram_similarity(hist1, hist2)
    assert similarity == pytest.approx(1.0, rel=1e-3)  # Perfect similarity
    
    # Create different histograms
    hist3 = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    
    similarity = detector._calculate_histogram_similarity(hist1, hist3)
    assert similarity < 1.0  # Should be less similar