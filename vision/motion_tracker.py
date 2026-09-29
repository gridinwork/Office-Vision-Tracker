"""Standing / moving state from the body box and shoulder width.

Forward and backward are approximate. A monocular webcam cannot measure
depth directly, so those labels follow a clear change in body size and are
withheld when the signal is mixed or weak.
"""

from __future__ import annotations
import time
from collections import deque
from app.settings import MOTION_DEPTH_THRESHOLD,MOTION_HORIZ_THRESHOLD,MOTION_MIN_SPAN_SEC,MOTION_STABLE_FRAMES,MOTION_VERTICAL_THRESHOLD,MOTION_WINDOW_SEC
from vision.types import PoseInstance

class MotionTracker:
    def __init__(self)->None:
        self._history:deque[tuple[float,float,float,float]]=deque(); self._state="—"; self._candidate="—"; self._candidate_hits=0
    def reset(self)->None:
        self._history.clear(); self._state="—"; self._candidate="—"; self._candidate_hits=0
    def update(self,pose:PoseInstance|None,fresh:bool,frame_width:int,frame_height:int)->str:
        if pose is None or pose.box is None:
            self.reset(); return self._state
        if not fresh: return self._state
        now=time.perf_counter(); box=pose.box
        width_sample=pose.shoulder_width if pose.shoulder_width>1.0 else float(box.w)
        self._history.append((box.cx,box.cy,width_sample,now))
        while self._history and now-self._history[0][3]>MOTION_WINDOW_SEC: self._history.popleft()
        raw=self._classify(frame_width,frame_height)
        if raw==self._candidate: self._candidate_hits+=1
        else: self._candidate=raw; self._candidate_hits=1
        if self._candidate_hits>=MOTION_STABLE_FRAMES: self._state=raw
        return self._state
    def _classify(self,frame_width:int,frame_height:int)->str:
        if len(self._history)<4: return "—"
        span=self._history[-1][3]-self._history[0][3]
        if span<MOTION_MIN_SPAN_SEC: return "—"
        first,last=self._history[0],self._history[-1]
        dx=(last[0]-first[0])/max(1.0,float(frame_width)); dy=(last[1]-first[1])/max(1.0,float(frame_height)); ratio=last[2]/max(1.0,first[2])
        horiz=abs(dx)>=MOTION_HORIZ_THRESHOLD; vertical=abs(dy)>=MOTION_VERTICAL_THRESHOLD; depth=abs(ratio-1.0)>=MOTION_DEPTH_THRESHOLD
        horiz_score=abs(dx)/MOTION_HORIZ_THRESHOLD; depth_score=abs(ratio-1.0)/MOTION_DEPTH_THRESHOLD
        if horiz and depth:
            if horiz_score>=depth_score: return "MOVING RIGHT" if dx>0 else "MOVING LEFT"
            return "MOVING FORWARD" if ratio>1.0 else "MOVING BACKWARD"
        if horiz: return "MOVING RIGHT" if dx>0 else "MOVING LEFT"
        if depth: return "MOVING FORWARD" if ratio>1.0 else "MOVING BACKWARD"
        if vertical: return "MOVING"
        return "STANDING"
