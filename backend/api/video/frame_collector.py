import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    frame_index: int
    frame_data: str
    timestamp: str
    timestamp_seconds: float
    fps: float
    video_duration: float
    is_image: bool = False


@dataclass
class DetectionData:
    frame_index: int
    is_anomaly: bool
    fake_prob: float
    real_prob: float
    confidence: float
    anomaly_type: Optional[str] = None


@dataclass
class MergedFrameData:
    frame_data: FrameData
    detection_data: Optional[DetectionData] = None


class FrameCollector:
    """
    Shadow Frame Collector - Non-blocking, memory-efficient buffer for batch XAI (TimeSHAP).
    
    This collector runs silently in the background of the websocket_task to build
    a batch of frame+detection pairs without blocking the real-time stream.
    
    Key features:
    - asyncio.Lock for thread-safety
    - Dictionary-based storage with frame_index as primary key
    - Fire-and-forget collection using asyncio.create_task
    - Automatic data fusion (frame + detection merge)
    - Deferred XAI execution via Celery
    - Explicit memory cleanup after dispatch
    """
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self._frames: Dict[int, FrameData] = {}
        self._detections: Dict[int, DetectionData] = {}
        self._merged: Dict[int, MergedFrameData] = {}
        self._lock = asyncio.Lock()
        self._is_finalized = False
        self._frame_count = 0
        self._detection_count = 0
        
    async def collect_frame(self, frame_data: FrameData) -> None:
        """
        Store raw frame data from frame_ready event.
        Called asynchronously - does not block the WebSocket send loop.
        """
        async with self._lock:
            self._frames[frame_data.frame_index] = frame_data
            self._frame_count += 1
            
            if frame_data.frame_index not in self._merged:
                self._merged[frame_data.frame_index] = MergedFrameData(
                    frame_data=frame_data
                )
            else:
                self._merged[frame_data.frame_index].frame_data = frame_data
                
        logger.debug(
            f"[FrameCollector] Collected frame {frame_data.frame_index} "
            f"(total frames: {self._frame_count})"
        )
    
    async def collect_detection(self, detection_data: DetectionData) -> None:
        """
        Store detection data from detection_ready event.
        Automatically merges with existing frame data if already collected.
        """
        async with self._lock:
            self._detections[detection_data.frame_index] = detection_data
            self._detection_count += 1
            
            if detection_data.frame_index in self._merged:
                self._merged[detection_data.frame_index].detection_data = detection_data
            else:
                self._merged[detection_data.frame_index] = MergedFrameData(
                    frame_data=FrameData(
                        frame_index=detection_data.frame_index,
                        frame_data="",
                        timestamp="",
                        timestamp_seconds=0.0,
                        fps=30.0,
                        video_duration=0.0
                    ),
                    detection_data=detection_data
                )
                
        logger.debug(
            f"[FrameCollector] Collected detection for frame {detection_data.frame_index} "
            f"(total detections: {self._detection_count})"
        )
    
    async def get_merged_batch(self) -> list:
        """
        Get all merged frame+detection pairs sorted by frame_index.
        Returns list of dicts suitable for run_explainable_ai task.
        """
        async with self._lock:
            merged_list = []
            
            for frame_index in sorted(self._merged.keys()):
                merged = self._merged[frame_index]
                frame_d = merged.frame_data
                detection_d = merged.detection_data
                
                if frame_d.frame_data and detection_d:
                    merged_list.append({
                        "frame_index": frame_index,
                        "frame_data": frame_d.frame_data,
                        "timestamp": frame_d.timestamp,
                        "timestamp_seconds": frame_d.timestamp_seconds,
                        "fps": frame_d.fps,
                        "video_duration": frame_d.video_duration,
                        "is_anomaly": detection_d.is_anomaly,
                        "fake_prob": detection_d.fake_prob,
                        "real_prob": detection_d.real_prob,
                        "confidence": detection_d.confidence,
                        "anomaly_type": detection_d.anomaly_type,
                    })
                    
            return merged_list
    
    def get_stats(self) -> Dict[str, int]:
        """Get collection statistics."""
        return {
            "frames_collected": self._frame_count,
            "detections_collected": self._detection_count,
            "merged_count": len(self._merged),
            "is_finalized": self._is_finalized,
        }
    
    async def finalize(self) -> Optional[list]:
        """
        Finalize collection and prepare batch for XAI.
        
        This method:
        1. Creates the merged batch sorted by frame_index
        2. Clears frame data from memory (base64 strings)
        3. Returns the batch for XAI processing
        
        Note: Memory cleanup is explicit to prevent VRAM/RAM leaks
        on long-duration videos.
        """
        async with self._lock:
            if self._is_finalized:
                logger.warning(
                    f"[FrameCollector] Already finalized task {self.task_id}"
                )
                return None
                
            self._is_finalized = True
            merged_batch = []
            
            for frame_index in sorted(self._merged.keys()):
                merged = self._merged[frame_index]
                frame_d = merged.frame_data
                detection_d = merged.detection_data
                
                if frame_d.frame_data and detection_d:
                    merged_batch.append({
                        "frame_index": frame_index,
                        "frame_data": frame_d.frame_data,
                        "timestamp": frame_d.timestamp,
                        "timestamp_seconds": frame_d.timestamp_seconds,
                        "fps": frame_d.fps,
                        "video_duration": frame_d.video_duration,
                        "is_anomaly": detection_d.is_anomaly,
                        "fake_prob": detection_d.fake_prob,
                        "real_prob": detection_d.real_prob,
                        "confidence": detection_d.confidence,
                        "anomaly_type": detection_d.anomaly_type,
                    })
            
            logger.info(
                f"[FrameCollector] Finalized task {self.task_id}: "
                f"{len(merged_batch)} frames ready for XAI"
            )
            
            return merged_batch
    
    async def clear(self) -> None:
        """
        Explicit memory cleanup.
        
        Clears all base64 strings from memory to prevent
        VRAM/RAM leaks on long-duration videos.
        """
        async with self._lock:
            for frame_index in self._frames:
                if self._frames[frame_index]:
                    self._frames[frame_index].frame_data = ""
                    
            for frame_index in self._merged:
                if self._merged[frame_index].frame_data:
                    self._merged[frame_index].frame_data.frame_data = ""
                    
            self._frames.clear()
            self._detections.clear()
            self._merged.clear()
            
            logger.info(f"[FrameCollector] Memory cleared for task {self.task_id}")


class ShadowCollectorRegistry:
    """
    Registry for FrameCollector instances per task.
    Provides thread-safe access to collectors.
    """
    
    def __init__(self):
        self._collectors: Dict[str, FrameCollector] = {}
        self._lock = asyncio.Lock()
    
    async def create(self, task_id: str) -> FrameCollector:
        """Create a new FrameCollector for a task."""
        async with self._lock:
            if task_id in self._collectors:
                await self._collectors[task_id].clear()
                
            collector = FrameCollector(task_id)
            self._collectors[task_id] = collector
            logger.info(f"[ShadowCollectorRegistry] Created collector for task {task_id}")
            return collector
    
    async def get(self, task_id: str) -> Optional[FrameCollector]:
        """Get existing FrameCollector for a task."""
        async with self._lock:
            return self._collectors.get(task_id)
    
    async def remove(self, task_id: str) -> None:
        """Remove and cleanup FrameCollector for a task."""
        async with self._lock:
            if task_id in self._collectors:
                await self._collectors[task_id].clear()
                del self._collectors[task_id]
                logger.info(f"[ShadowCollectorRegistry] Removed collector for task {task_id}")


shadow_registry = ShadowCollectorRegistry()