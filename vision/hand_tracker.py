"""Two-hand landmarks, boxes, motion state, and short wrist trails."""

from __future__ import annotations
import math,time
from collections import deque
from app.settings import HAND_MOTION_WINDOW_SEC,HAND_SPEED_THRESHOLD,HAND_TRAIL_LENGTH,LOST_FRAMES_TIMEOUT,SMOOTHING_ALPHA,SNAP_FRACTION
from vision.smoothing import BoxSmoother,PointSmoother
from vision.types import Box,HandInstance,Point

def _to_points(landmarks,width:int,height:int)->list[Point]:
    return [Point(float(l.x)*width,float(l.y)*height,float(getattr(l,"z",0.0)),1.0) for l in landmarks]

def _hand_box(points:list[Point],width:int,height:int)->Box|None:
    if len(points)<21:return None
    xs=[p.x for p in points]; ys=[p.y for p in points]; x1,y1,x2,y2=min(xs),min(ys),max(xs),max(ys)
    bw=max(1.0,x2-x1); bh=max(1.0,y2-y1); px=max(8.0,bw*0.12); py=max(8.0,bh*0.12)
    x=int(round(max(0.0,x1-px))); y=int(round(max(0.0,y1-py))); right=int(round(min(width-1,x2+px))); bottom=int(round(min(height-1,y2+py)))
    return Box(x=x,y=y,w=max(1,right-x),h=max(1,bottom-y))

class _HandState:
    def __init__(self,side:str,alpha:float,snap_distance:float)->None:
        self.side=side; self.points=PointSmoother(alpha,snap_distance); self.box=BoxSmoother(alpha); self.lost=0; self.last:HandInstance|None=None
        self.history:deque[tuple[float,float,float]]=deque(); self.trail:deque[tuple[int,int]]=deque(maxlen=HAND_TRAIL_LENGTH); self.moving=False
    def reset(self)->None:
        self.points.reset(); self.box.reset(); self.lost=0; self.last=None; self.history.clear(); self.trail.clear(); self.moving=False
    def _update_motion(self,wrist:Point,frame_width:int)->None:
        now=time.perf_counter(); self.history.append((wrist.x,wrist.y,now))
        while self.history and now-self.history[0][2]>HAND_MOTION_WINDOW_SEC:self.history.popleft()
        self.trail.append((int(round(wrist.x)),int(round(wrist.y))))
        if len(self.history)<3:self.moving=False; return
        x0,y0,t0=self.history[0]; x1,y1,t1=self.history[-1]; speed=math.hypot(x1-x0,y1-y0)/max(1e-3,t1-t0)
        self.moving=speed>HAND_SPEED_THRESHOLD*frame_width
    def update(self,raw:tuple[list[Point],float|None]|None,width:int,height:int)->HandInstance|None:
        if raw is None:
            self.lost+=1
            if self.lost>LOST_FRAMES_TIMEOUT:self.reset(); return None
            return self.last
        points,confidence=raw; self.lost=0; smoothed=self.points.apply(points,use_visibility=False); box=self.box.apply(_hand_box(smoothed,width,height))
        if smoothed:self._update_motion(smoothed[0],width)
        self.last=HandInstance(side=self.side,points=smoothed,box=box,confidence=confidence,moving=self.moving,trail=list(self.trail)); return self.last

class HandTracker:
    def __init__(self)->None:self._states={"Left":_HandState("Left",SMOOTHING_ALPHA,80.0),"Right":_HandState("Right",SMOOTHING_ALPHA,80.0)}
    def reset(self)->None:
        for state in self._states.values():state.reset()
    def update(self,raw_hands:list[tuple[str,float|None,list[Point]]],width:int,height:int)->list[HandInstance]:
        snap=max(30.0,SNAP_FRACTION*width)
        for state in self._states.values():state.points.snap_distance=snap
        best={}
        for side,confidence,points in raw_hands:
            if side not in self._states or len(points)<21:continue
            current=best.get(side); score=confidence if confidence is not None else 0.0; current_score=current[1] if current and current[1] is not None else -1.0
            if current is None or score>=current_score:best[side]=(points,confidence)
        hands=[]
        for side,state in self._states.items():
            hand=state.update(best.get(side),width,height)
            if hand is not None:hands.append(hand)
        return hands

def parse_hand_landmarks(hand_landmarks,width:int,height:int)->list[Point]:
    return _to_points(hand_landmarks,width,height)
