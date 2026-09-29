"""Body pose tracking, bounding boxes, and short-loss persistence."""

from __future__ import annotations
from app.settings import LOST_FRAMES_TIMEOUT, POSE_BOX_VISIBILITY, SMOOTHING_ALPHA, SNAP_FRACTION
from vision.connections import BODY_BOX_INDICES
from vision.smoothing import BoxSmoother, PointSmoother
from vision.types import Box, Point, PoseInstance

def _visibility(landmark)->float:
    value=getattr(landmark,"visibility",None)
    return 1.0 if value is None else float(value)

def landmarks_to_points(landmarks,width:int,height:int)->list[Point]:
    return [Point(float(l.x)*width,float(l.y)*height,float(getattr(l,"z",0.0)),_visibility(l)) for l in landmarks]

def _box_from_points(points:list[Point],width:int,height:int)->tuple[Box|None,float,float]:
    lower={25,26,27,28,31,32}; chosen=[]
    for index in BODY_BOX_INDICES:
        if index>=len(points): continue
        p=points[index]; minimum=0.7 if index in lower else POSE_BOX_VISIBILITY
        if p.visibility>=minimum: chosen.append(p)
    if len(chosen)<4: return None,0.0,0.0
    xs=[p.x for p in chosen]; ys=[p.y for p in chosen]
    x1,y1,x2,y2=min(xs),min(ys),max(xs),max(ys); bw=max(1.0,x2-x1); bh=max(1.0,y2-y1)
    px=max(10.0,bw*0.08); py=max(10.0,bh*0.08)
    x=int(round(max(0.0,x1-px))); y=int(round(max(0.0,y1-py)))
    right=int(round(min(width-1,x2+px))); bottom=int(round(min(height-1,y2+py)))
    confidence=sum(p.visibility for p in chosen)/len(chosen)
    shoulder=0.0
    if len(points)>12 and points[11].visibility>=POSE_BOX_VISIBILITY and points[12].visibility>=POSE_BOX_VISIBILITY:
        shoulder=abs(points[11].x-points[12].x)
    return Box(x,y,max(1,right-x),max(1,bottom-y),confidence),confidence,shoulder

class _PoseSlot:
    def __init__(self,alpha:float,snap_distance:float)->None:
        self.points=PointSmoother(alpha,snap_distance); self.box=BoxSmoother(alpha); self.lost=0; self.last:PoseInstance|None=None
    def reset(self)->None: self.points.reset(); self.box.reset(); self.lost=0; self.last=None
    def update(self,raw:list[Point]|None,width:int,height:int)->PoseInstance|None:
        if not raw:
            self.lost+=1
            if self.lost>LOST_FRAMES_TIMEOUT: self.reset(); return None
            return self.last
        self.lost=0; smoothed=self.points.apply(raw,use_visibility=True); box,confidence,shoulder=_box_from_points(smoothed,width,height)
        if box is None:
            self.lost+=1
            if self.lost>LOST_FRAMES_TIMEOUT: self.reset(); return None
            return self.last
        box=self.box.apply(box)
        self.last=PoseInstance(smoothed,box,confidence,shoulder or float(box.w if box else 0))
        return self.last

class PoseTracker:
    def __init__(self,slots:int=2)->None: self._slots=slots; self._states=[]; self._snap_distance=120.0; self.person_count=0
    def reset(self)->None: self._states=[]; self.person_count=0
    def update(self,raw_poses:list[list[Point]],width:int,height:int)->list[PoseInstance]:
        self._snap_distance=max(40.0,SNAP_FRACTION*width); fresh_count=len(raw_poses)
        if fresh_count: self.person_count=fresh_count
        needed=max(fresh_count,len(self._states),1 if self._states else 0)
        while len(self._states)<min(self._slots,max(needed,fresh_count)):
            self._states.append(_PoseSlot(SMOOTHING_ALPHA,self._snap_distance))
        for state in self._states: state.points.snap_distance=self._snap_distance
        results=[]
        for index,state in enumerate(self._states[:self._slots]):
            raw=raw_poses[index] if index<fresh_count else None
            pose=state.update(raw,width,height)
            if pose is not None: results.append(pose)
        if not results: self.person_count=0
        elif fresh_count==0: self.person_count=max(self.person_count,len(results))
        return results
