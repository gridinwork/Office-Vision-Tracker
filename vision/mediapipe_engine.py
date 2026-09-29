"""MediaPipe Holistic Landmarker: pose, face, and both hands in one pass."""

from __future__ import annotations
import logging,time
import cv2, mediapipe as mp, numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HolisticLandmarker,HolisticLandmarkerOptions,RunningMode
from app.settings import HOLISTIC_MODEL,MIN_FACE_DETECTION_CONFIDENCE,MIN_HAND_DETECTION_CONFIDENCE,MIN_MODEL_BYTES,MIN_POSE_DETECTION_CONFIDENCE
from vision.face_tracker import FaceTracker
from vision.gesture_analyzer import GestureAnalyzer
from vision.hand_tracker import HandTracker
from vision.motion_tracker import MotionTracker
from vision.pose_tracker import PoseTracker,landmarks_to_points
from vision.types import FrameVision,Point

logger=logging.getLogger("office_vision.engine")

def _require_model(path)->None:
    if not path.exists() or path.stat().st_size<MIN_MODEL_BYTES: raise FileNotFoundError("Model file not found.")

def _pixel_points(landmarks,width:int,height:int)->list[Point]:
    return [Point(float(l.x)*width,float(l.y)*height,float(getattr(l,"z",0.0) or 0.0),1.0) for l in landmarks]

class MediaPipeEngine:
    def __init__(self)->None:
        _require_model(HOLISTIC_MODEL); logger.info("Loading model %s",HOLISTIC_MODEL.name); self._holistic=None
        try:self._open()
        except FileNotFoundError:self.close(); raise
        except Exception as exc:self.close(); logger.exception("MediaPipe init failed"); raise RuntimeError("Unable to initialize MediaPipe model.") from exc
        logger.info("MediaPipe model loaded (mediapipe %s)",mp.__version__)
        self._input_size=None; self.pose_tracker=PoseTracker(slots=1); self.face_tracker=FaceTracker(slots=1); self.hand_tracker=HandTracker(); self.motion_tracker=MotionTracker(); self.gesture_analyzer=GestureAnalyzer(); self._timestamp=0; self._logged_detect_error=False
    def _open(self)->None:
        options=HolisticLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(HOLISTIC_MODEL),delegate=BaseOptions.Delegate.CPU),running_mode=RunningMode.VIDEO,min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,min_hand_landmarks_confidence=MIN_HAND_DETECTION_CONFIDENCE,output_face_blendshapes=False,output_segmentation_mask=False)
        self._holistic=HolisticLandmarker.create_from_options(options); self._timestamp=0
    def _ensure_input_size(self,w:int,h:int)->None:
        size=(w,h)
        if self._holistic is not None and (self._input_size is None or self._input_size==size): self._input_size=size; return
        logger.info("Rebuilding MediaPipe graph for %sx%s",w,h); self.close()
        try:self._open()
        except Exception as exc: logger.exception("MediaPipe init failed"); raise RuntimeError("Unable to initialize MediaPipe model.") from exc
        self._input_size=size
    def close(self)->None:
        holistic=getattr(self,"_holistic",None)
        if holistic is None:return
        try:holistic.close()
        except Exception:logger.exception("Failed to close holistic landmarker")
        self._holistic=None
    def reset(self)->None:
        self.pose_tracker.reset(); self.face_tracker.reset(); self.hand_tracker.reset(); self.motion_tracker.reset(); self.gesture_analyzer.reset()
    def _next_timestamp(self)->int:
        timestamp=int(time.monotonic()*1000)
        if timestamp<=self._timestamp:timestamp=self._timestamp+1
        self._timestamp=timestamp; return timestamp
    def process(self,frame_bgr:np.ndarray,scale:float)->FrameVision:
        height,width=frame_bgr.shape[:2]; scale=min(1.0,max(0.3,float(scale)))
        if scale<0.999:
            scaled_w=max(32,int(width*scale)//2*2); scaled_h=max(32,int(height*scale)//2*2); small=cv2.resize(frame_bgr,(scaled_w,scaled_h),interpolation=cv2.INTER_LINEAR)
        else: scaled_w,scaled_h=width,height; small=frame_bgr
        self._ensure_input_size(scaled_w,scaled_h); rgb=np.ascontiguousarray(cv2.cvtColor(small,cv2.COLOR_BGR2RGB)); started=time.perf_counter(); image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        try: result=self._holistic.detect_for_video(image,self._next_timestamp())
        except Exception as exc:
            result=None
            if not self._logged_detect_error:self._logged_detect_error=True; logger.error("MediaPipe error: %s",exc)
        inference_ms=(time.perf_counter()-started)*1000.0
        raw_poses=[landmarks_to_points(result.pose_landmarks,width,height)] if result is not None and getattr(result,"pose_landmarks",None) else []
        raw_faces=[_pixel_points(result.face_landmarks,width,height)] if result is not None and getattr(result,"face_landmarks",None) else []
        raw_hands=[]
        if result is not None:
            left=getattr(result,"left_hand_landmarks",None) or []; right=getattr(result,"right_hand_landmarks",None) or []
            if len(left)>=21:raw_hands.append(("Left",None,_pixel_points(left,width,height)))
            if len(right)>=21:raw_hands.append(("Right",None,_pixel_points(right,width,height)))
        poses=self.pose_tracker.update(raw_poses,width,height); faces=self.face_tracker.update(raw_faces,width,height); hands=self.hand_tracker.update(raw_hands,width,height); primary=poses[0] if poses else None
        return FrameVision(poses=poses,faces=faces,hands=hands,person_count=self.pose_tracker.person_count,person_detected=primary is not None,face_detected=bool(faces),left_hand_detected=any(h.side=="Left" for h in hands),right_hand_detected=any(h.side=="Right" for h in hands),movement=self.motion_tracker.update(primary,fresh=bool(raw_poses),frame_width=width,frame_height=height),gesture=self.gesture_analyzer.analyze(primary),frame_width=width,frame_height=height,timings_ms={"pose":inference_ms,"hands":inference_ms,"face":inference_ms,"inference":inference_ms})
