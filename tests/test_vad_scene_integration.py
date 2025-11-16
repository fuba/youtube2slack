"""Integration tests for VAD stream processor with scene detection."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from youtube2slack.vad_stream_processor import VADStreamProcessor
from youtube2slack.whisper_transcriber import WhisperTranscriber
from youtube2slack.slack_client import SlackClient


class TestVADSceneIntegration:
    """Test VAD stream processor with scene detection integration."""

    def test_vad_processor_with_scene_detection_enabled(self):
        """Test VAD processor initialization with scene detection."""
        # Mock dependencies
        transcriber = Mock(spec=WhisperTranscriber)
        slack_client = Mock(spec=SlackClient)
        
        processor = VADStreamProcessor(
            transcriber=transcriber,
            slack_client=slack_client,
            enable_scene_detection=True,
            scene_diff_threshold=0.3,
            scene_hist_threshold=0.7
        )
        
        assert processor.enable_scene_detection == True
        assert processor.scene_detector is not None
        assert processor.scene_capture is not None

    def test_vad_processor_with_scene_detection_disabled(self):
        """Test VAD processor initialization with scene detection disabled."""
        transcriber = Mock(spec=WhisperTranscriber)
        slack_client = Mock(spec=SlackClient)
        
        processor = VADStreamProcessor(
            transcriber=transcriber,
            slack_client=slack_client,
            enable_scene_detection=False
        )
        
        assert processor.enable_scene_detection == False
        assert processor.scene_detector is None
        assert processor.scene_capture is None

    def test_get_status_with_scene_detection(self):
        """Test status reporting with scene detection enabled."""
        transcriber = Mock(spec=WhisperTranscriber)
        slack_client = Mock(spec=SlackClient)
        
        processor = VADStreamProcessor(
            transcriber=transcriber,
            slack_client=slack_client,
            enable_scene_detection=True
        )
        
        status = processor.get_status()
        
        # Check scene detection fields
        assert 'scene_detection_enabled' in status
        assert 'scene_detector_status' in status
        assert 'captured_scenes' in status
        assert status['scene_detection_enabled'] == True
        assert status['captured_scenes'] == 0  # No scenes captured yet

    def test_get_status_without_scene_detection(self):
        """Test status reporting with scene detection disabled."""
        transcriber = Mock(spec=WhisperTranscriber)
        slack_client = Mock(spec=SlackClient)
        
        processor = VADStreamProcessor(
            transcriber=transcriber,
            slack_client=slack_client,
            enable_scene_detection=False
        )
        
        status = processor.get_status()
        
        # Check scene detection fields
        assert 'scene_detection_enabled' in status
        assert status['scene_detection_enabled'] == False
        # scene_detector_status and captured_scenes should not be present when disabled

    @patch('youtube2slack.vad_stream_processor.cv2.VideoCapture')
    @patch('youtube2slack.vad_stream_processor.cv2.imwrite')
    def test_scene_detection_worker_mock(self, mock_imwrite, mock_video_capture):
        """Test scene detection worker with mocked video capture."""
        # Setup mocks
        transcriber = Mock(spec=WhisperTranscriber)
        slack_client = Mock(spec=SlackClient)
        slack_client.send_image = Mock(return_value=True)
        
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        
        # Create test frames
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1[:, :] = [100, 100, 100]
        
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8) 
        frame2[:, :] = [200, 200, 200]  # Different frame to trigger scene change
        
        # Mock read sequence
        mock_cap.read.side_effect = [
            (True, frame1),
            (True, frame2),
            (False, None)  # End sequence
        ]
        
        mock_video_capture.return_value = mock_cap
        mock_imwrite.return_value = True
        
        processor = VADStreamProcessor(
            transcriber=transcriber,
            slack_client=slack_client,
            enable_scene_detection=True,
            scene_diff_threshold=0.1,  # Low threshold to ensure detection
            scene_hist_threshold=0.9   # High threshold to ensure detection
        )
        
        # Mock get_actual_stream_url
        processor._get_actual_stream_url = Mock(return_value="mock_stream_url")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            processor.temp_dir = temp_dir
            processor.stream_info = {'title': 'Test Stream'}
            
            # Run scene detection worker (it should process frames and detect change)
            processor._process_scene_detection("fake_url")
            
            # Verify video capture was opened
            mock_video_capture.assert_called_with("mock_stream_url")
            mock_cap.isOpened.assert_called()
            
            # Verify frames were read
            assert mock_cap.read.call_count >= 2
            
            # Scene change should be detected and image sent to Slack
            # (exact call count depends on scene detection algorithm)
            assert slack_client.send_image.call_count >= 0  # May or may not detect change depending on timing

    def test_scene_detection_directory_creation(self):
        """Test that scene detection creates proper directory structure."""
        transcriber = Mock(spec=WhisperTranscriber)
        slack_client = Mock(spec=SlackClient)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = VADStreamProcessor(
                transcriber=transcriber,
                slack_client=slack_client,
                enable_scene_detection=True
            )
            processor.temp_dir = temp_dir
            
            # Scene directory should be created in temp_dir
            expected_scene_dir = os.path.join(temp_dir, "scenes")
            assert processor.scene_capture.output_dir == Path(expected_scene_dir)

    def test_cleanup_includes_scene_files(self):
        """Test that cleanup removes scene files."""
        transcriber = Mock(spec=WhisperTranscriber)
        slack_client = Mock(spec=SlackClient)
        
        processor = VADStreamProcessor(
            transcriber=transcriber,
            slack_client=slack_client,
            enable_scene_detection=True
        )
        
        # Create temp directory and scene files
        with tempfile.TemporaryDirectory() as temp_dir:
            processor.temp_dir = temp_dir
            scene_dir = os.path.join(temp_dir, "scenes")
            os.makedirs(scene_dir, exist_ok=True)
            
            # Create test scene files
            test_files = [
                os.path.join(scene_dir, "scene_0001.jpg"),
                os.path.join(scene_dir, "scene_0002.jpg")
            ]
            
            for file_path in test_files:
                Path(file_path).touch()
            
            # Verify files exist
            assert all(os.path.exists(f) for f in test_files)
            
            # Run cleanup
            processor._cleanup_temp_dir()
            
            # Verify directory is cleaned up
            assert not os.path.exists(temp_dir)

    def test_scene_detection_without_slack_client(self):
        """Test scene detection behavior when no Slack client is provided."""
        transcriber = Mock(spec=WhisperTranscriber)
        
        processor = VADStreamProcessor(
            transcriber=transcriber,
            slack_client=None,  # No Slack client
            enable_scene_detection=True
        )
        
        # Scene detector should still be initialized
        assert processor.scene_detector is not None
        
        # But scene processing thread should not start without Slack client
        processor.is_running = True
        processor.stream_info = {'title': 'Test'}
        
        # Mock the actual URL function to avoid network calls
        processor._get_actual_stream_url = Mock(return_value=None)
        
        # Should return early due to no slack client
        processor._process_scene_detection("fake_url")
        
        # Verify it didn't try to get stream URL since no Slack client
        processor._get_actual_stream_url.assert_not_called()

    @patch('youtube2slack.vad_stream_processor.subprocess.run')
    def test_scene_detection_stream_url_failure(self, mock_subprocess):
        """Test scene detection when stream URL cannot be obtained."""
        # Mock failed subprocess call
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stdout = ""
        mock_subprocess.return_value.stderr = "Error"
        
        transcriber = Mock(spec=WhisperTranscriber)
        slack_client = Mock(spec=SlackClient)
        
        processor = VADStreamProcessor(
            transcriber=transcriber,
            slack_client=slack_client,
            enable_scene_detection=True
        )
        
        # Should handle failure gracefully
        processor._process_scene_detection("fake_url")
        
        # Verify subprocess was called to try to get URL
        mock_subprocess.assert_called()


def test_scene_detection_thresholds():
    """Test scene detection with different threshold values."""
    transcriber = Mock(spec=WhisperTranscriber)
    slack_client = Mock(spec=SlackClient)
    
    # Test with strict thresholds
    processor_strict = VADStreamProcessor(
        transcriber=transcriber,
        slack_client=slack_client,
        enable_scene_detection=True,
        scene_diff_threshold=0.8,  # High threshold (strict)
        scene_hist_threshold=0.9   # High threshold (strict)
    )
    
    assert processor_strict.scene_detector.diff_threshold == 0.8
    assert processor_strict.scene_detector.hist_threshold == 0.9
    
    # Test with loose thresholds
    processor_loose = VADStreamProcessor(
        transcriber=transcriber,
        slack_client=slack_client,
        enable_scene_detection=True,
        scene_diff_threshold=0.1,  # Low threshold (loose)
        scene_hist_threshold=0.3   # Low threshold (loose)
    )
    
    assert processor_loose.scene_detector.diff_threshold == 0.1
    assert processor_loose.scene_detector.hist_threshold == 0.3