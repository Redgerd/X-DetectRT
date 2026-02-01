# backend/core/celery/spatialDetection.py
import cv2
import numpy as np
import base64
import json
import logging
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from typing import List, Dict, Any, Optional
import redis

from config import settings
from .celery_app import celery_app

logger = logging.getLogger(__name__)

# Redis client for publishing detection results
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    encoded_data = base64_str.split(',')[1] if ',' in base64_str else base64_str
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def image_to_base64(img: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buffer).decode("utf-8")


def detect_spatial_anomalies(frame: np.ndarray) -> Dict[str, Any]:
    """
    Run spatial anomaly detection on a frame.
    This is a placeholder for actual spatial detection logic.
    Replace with your actual spatial detection model.
    
    Returns:
        Dict containing detection results
    """
    h, w = frame.shape[:2]
    
    # Placeholder: Simple edge-based analysis
    # In production, replace with your actual spatial detection model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / (h * w)
    
    # Placeholder: Calculate ELA-like score using JPEG compression analysis
    # In production, use actual ELA algorithm
    _, compressed = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    reconstructed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(frame, reconstructed)
    ela_score = float(np.mean(diff) / 255.0)
    
    # Placeholder: Frequency analysis using DCT
    # In production, use proper frequency analysis
    dct = cv2.dct(np.float32(gray))
    high_freq_energy = float(np.sum(np.abs(dct[h//2:, w//2:])) / (h * w))
    normalized_freq = min(high_freq_energy / 50.0, 1.0)
    
    # Determine anomaly based on heuristics
    # In production, use actual model predictions
    is_anomaly = (ela_score > 0.15 or normalized_freq > 0.7 or edge_density > 0.3)
    confidence = min((ela_score * 3 + normalized_freq * 2 + edge_density * 2) / 6 * 100, 99.9)
    
    if is_anomaly:
        anomaly_type = "Spatial Anomaly"
    else:
        anomaly_type = None
        confidence = 100 - confidence
    
    return {
        "is_anomaly": is_anomaly,
        "confidence": round(confidence, 2),
        "anomaly_type": anomaly_type,
        "ela_score": round(ela_score, 4),
        "frequency_spike": round(normalized_freq * 100, 2),
        "edge_density": round(edge_density, 4)
    }


def annotate_frame(frame: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
    """Annotate frame with detection results"""
    annotated = frame.copy()
    
    # Draw bounding box if anomaly detected
    if detection_result.get("is_anomaly"):
        h, w = annotated.shape[:2]
        # Draw rectangle around the frame
        cv2.rectangle(annotated, (10, 10), (w - 10, h - 10), (0, 0, 255), 2)
        
        # Add label
        label = f"ANOMALY: {detection_result.get('anomaly_type', 'Unknown')}"
        cv2.putText(annotated, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)
        
        confidence = detection_result.get("confidence", 0)
        conf_label = f"Confidence: {confidence:.1f}%"
        cv2.putText(annotated, conf_label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 255), 2)
    
    return annotated


@shared_task(name="spatial_detection.analyze_frames", bind=True, max_retries=3, soft_time_limit=120)
def analyze_spatial_detection(self, frames_data: List[Dict[str, Any]], task_id: str = "unknown") -> Dict[str, Any]:
    """
    Celery task for spatial anomaly detection on processed frames.
    
    Args:
        frames_data: List of frame dictionaries with frame_index and frame_data
        task_id: The task identifier for Redis publishing
    
    Returns:
        Dict containing detection results for each frame
    """
    logger.info(f"[SpatialDetection] Starting analysis for {len(frames_data)} frames, task_id: {task_id}")
    
    if not frames_data:
        return {"error": "No frames provided", "task_id": task_id}
    
    results = []
    anomaly_count = 0
    
    for frame_info in frames_data:
        frame_index = frame_info.get("frame_index", 0)
        frame_data = frame_info.get("frame_data", "")
        timestamp = frame_info.get("timestamp", "")
        timestamp_seconds = frame_info.get("timestamp_seconds", 0)
        
        try:
            # Convert base64 to image
            frame = base64_to_image(frame_data)
            if frame is None:
                logger.warning(f"[SpatialDetection] Failed to decode frame {frame_index}")
                continue
            
            # Run spatial detection
            detection_result = detect_spatial_anomalies(frame)
            
            if detection_result.get("is_anomaly"):
                anomaly_count += 1
            
            # Annotate frame if anomaly detected
            annotated_frame = annotate_frame(frame, detection_result)
            annotated_base64 = image_to_base64(annotated_frame)
            
            # Prepare result
            result = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "timestamp_seconds": timestamp_seconds,
                "is_anomaly": detection_result.get("is_anomaly", False),
                "confidence": detection_result.get("confidence", 0),
                "anomaly_type": detection_result.get("anomaly_type"),
                "ela_score": detection_result.get("ela_score", 0),
                "frequency_spike": detection_result.get("frequency_spike", 0),
                "edge_density": detection_result.get("edge_density", 0),
                "annotated_frame_data": annotated_base64,  # Use annotated frame for display
                "original_frame_data": frame_data  # Keep original
            }
            results.append(result)
            
            # Publish detection result to Redis for real-time updates
            try:
                redis_client.publish(
                    f"task_detection:{task_id}",
                    json.dumps({
                        "type": "detection_ready",
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "timestamp_seconds": timestamp_seconds,
                        "frame_data": annotated_base64,  # Send annotated frame
                        "is_anomaly": result["is_anomaly"],
                        "confidence": result["confidence"],
                        "anomaly_type": result["anomaly_type"],
                        "ela_score": result["ela_score"],
                        "frequency_spike": result["frequency_spike"],
                        "task_id": task_id
                    })
                )
            except Exception as redis_err:
                logger.warning(f"[SpatialDetection] Redis publish error for frame {frame_index}: {redis_err}")
            
            logger.debug(f"[SpatialDetection] Analyzed frame {frame_index}: anomaly={result['is_anomaly']}")
            
        except Exception as e:
            logger.error(f"[SpatialDetection] Error analyzing frame {frame_index}: {e}", exc_info=True)
            continue
    
    # Final result
    final_result = {
        "message": "Spatial detection analysis complete",
        "task_id": task_id,
        "total_frames": len(results),
        "anomaly_count": anomaly_count,
        "results": results,
        "anomaly_percentage": round((anomaly_count / len(results) * 100) if results else 0, 2)
    }
    
    # Store final result in Redis
    try:
        redis_client.set(f"detection_result:{task_id}", json.dumps(final_result))
    except Exception as e:
        logger.warning(f"[SpatialDetection] Failed to store final result: {e}")
    
    logger.info(f"[SpatialDetection] Completed analysis: {anomaly_count}/{len(results)} anomalies detected")
    
    return final_result


@celery_app.task(name="spatial_detection.run_chained_detection")
def run_chained_detection(task_id: str, frame_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper task that chains after frame_selection and runs spatial detection.
    This task is designed to be called from the frame_selection task using apply_chain
    or called directly with the results of frame_selection.
    
    Args:
        task_id: The task identifier
        frame_results: The result dict from extract_faces_with_optical_flow
    
    Returns:
        Combined results from frame selection and spatial detection
    """
    logger.info(f"[SpatialDetection] Chained detection started for task_id: {task_id}")
    
    # Extract preview frames from frame_selection result
    preview_frames = frame_results.get("preview_frames", [])
    
    if not preview_frames or all(f == "" for f in preview_frames):
        logger.warning(f"[SpatialDetection] No preview frames found in frame_results")
        return {"error": "No frames to analyze", "task_id": task_id}
    
    # Build frames data list from preview frames
    frames_data = []
    total_frames = frame_results.get("total_frames", len(preview_frames))
    
    for i, frame_b64 in enumerate(preview_frames):
        if frame_b64:
            frames_data.append({
                "frame_index": i,
                "frame_data": frame_b64,
                "timestamp": f"00:00:{i // 30:02d}:{(i % 30) * 2:02d}",
                "timestamp_seconds": i / 30.0 if total_frames > 0 else i
            })
    
    if not frames_data:
        return {"error": "No valid frames to analyze", "task_id": task_id}
    
    # Run spatial detection
    detection_result = analyze_spatial_detection.delay(frames_data, task_id)
    
    # Return immediately with task info - actual results will be available via Redis
    return {
        "message": "Spatial detection task dispatched",
        "task_id": task_id,
        "frames_to_analyze": len(frames_data),
        "detection_task_id": detection_result.id
    }


# Helper function to convert detection result to frame-ready format
def format_detection_for_websocket(detection_result: Dict[str, Any]) -> Dict[str, Any]:
    """Format detection result for WebSocket publishing"""
    return {
        "type": "frame_ready",
        "frame_index": detection_result.get("frame_index", 0),
        "timestamp": detection_result.get("timestamp", ""),
        "timestamp_seconds": detection_result.get("timestamp_seconds", 0),
        "frame_data": detection_result.get("annotated_frame_data", detection_result.get("original_frame_data", "")),
        "is_anomaly": detection_result.get("is_anomaly", False),
        "confidence": detection_result.get("confidence", 0),
        "anomaly_type": detection_result.get("anomaly_type"),
        "ela_score": detection_result.get("ela_score", 0),
        "frequency_spike": detection_result.get("frequency_spike", 0),
        "task_id": detection_result.get("task_id", "unknown")
    }
