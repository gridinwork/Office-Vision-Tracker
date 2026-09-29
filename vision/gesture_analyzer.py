"""Simple arm gestures based on wrist position relative to the shoulders.

Additional gestures can be added by extending _classify.
"""

from __future__ import annotations
from app.settings import GESTURE_STABLE_FRAMES,POSE_POINT_VISIBILITY
from vision.types import Point,PoseInstance

LEFT_SHOULDER=11; RIGHT_SHOULDER=12; LEFT_WRIST=15; RIGHT_WRIST=16; LEFT_HIP=23; RIGHT_HIP=24

class GestureAnalyzer:
    def __init__(self)->None: self._current="—"; self._candidate="—"; self._hits=0
    def reset(self)->None: self._current="—"; self._candidate="—"; self._hits=0
    def analyze(self,pose:PoseInstance|None)->str:
        raw=self._classify(pose)
        if raw==self._candidate: self._hits+=1
        else: self._candidate=raw; self._hits=1
        if self._hits>=GESTURE_STABLE_FRAMES: self._current=raw
        if pose is None and self._hits>=GESTURE_STABLE_FRAMES: self._current="—"
        return self._current
    def _classify(self,pose:PoseInstance|None)->str:
        if pose is None or len(pose.points)<=RIGHT_WRIST: return "—"
        points=pose.points; ls=points[LEFT_SHOULDER]; rs=points[RIGHT_SHOULDER]; lw=points[LEFT_WRIST]; rw=points[RIGHT_WRIST]
        left_known=self._visible(ls) and self._visible(lw); right_known=self._visible(rs) and self._visible(rw)
        if not left_known and not right_known: return "—"
        margin=self._margin(points,ls,rs)
        left_up=left_known and lw.y<ls.y-margin; right_up=right_known and rw.y<rs.y-margin
        left_down=left_known and lw.y>ls.y+margin; right_down=right_known and rw.y>rs.y+margin
        if left_up and right_up: return "BOTH HANDS UP"
        if left_up: return "LEFT HAND UP"
        if right_up: return "RIGHT HAND UP"
        if left_down and right_down: return "ARMS DOWN"
        return "—"
    @staticmethod
    def _visible(point:Point)->bool: return point.visibility>=POSE_POINT_VISIBILITY
    @staticmethod
    def _margin(points:list[Point],left_shoulder:Point,right_shoulder:Point)->float:
        hips=[]
        if len(points)>RIGHT_HIP and points[LEFT_HIP].visibility>=POSE_POINT_VISIBILITY: hips.append(points[LEFT_HIP])
        if len(points)>RIGHT_HIP and points[RIGHT_HIP].visibility>=POSE_POINT_VISIBILITY: hips.append(points[RIGHT_HIP])
        if hips:
            shoulder_y=(left_shoulder.y+right_shoulder.y)/2.0; hip_y=sum(p.y for p in hips)/len(hips); torso=abs(hip_y-shoulder_y)
            return max(12.0,torso*0.12)
        return 24.0
