"""Face box, head direction, and optional full-mesh points."""

from __future__ import annotations
import math
from app.settings import LOST_FRAMES_TIMEOUT, SMOOTHING_ALPHA, SNAP_FRACTION
from vision.connections import FACE_KEY_INDICES, LEFT_EYE, NOSE_TIP, RIGHT_EYE
from vision.smoothing import BoxSmoother, PointSmoother
from vision.types import Box, FaceInstance, Point

def _to_points(landmarks,width:int,height:int)->list[Point]:
    return [Point(float(l.x)*width,float(l.y)*height,float(getattr(l,"z",0.0)),1.0) for l in landmarks]

def _face_box(points:list[Point],width:int,height:int)->Box|None:
    if len(points)<10:return None
    xs=[min(max(p.x,0.0),width-1) for p in points]; ys=[min(max(p.y,0.0),height-1) for p in points]
    x1,y1,x2,y2=min(xs),min(ys),max(xs),max(ys); bw=max(1.0,x2-x1); bh=max(1.0,y2-y1)
    px=max(6.0,bw*0.06); py=max(6.0,bh*0.08)
    x=int(round(max(0.0,x1-px))); y=int(round(max(0.0,y1-py)))
    right=int(round(min(width-1,x2+px))); bottom=int(round(min(height-1,y2+py)))
    return Box(x=x,y=y,w=max(1,right-x),h=max(1,bottom-y))

def _direction(points:list[Point])->tuple[tuple[int,int]|None,tuple[int,int]|None]:
    if not points:return None,None
    usable=[p for p in points if 0<=p.x and 0<=p.y]
    if not usable:return None,None
    center_x=sum(p.x for p in usable)/len(usable); center_y=sum(p.y for p in usable)/len(usable); center=(int(round(center_x)),int(round(center_y)))
    if len(points)>RIGHT_EYE:
        le,re,nose=points[LEFT_EYE],points[RIGHT_EYE],points[NOSE_TIP]; eye_x=(le.x+re.x)/2; eye_y=(le.y+re.y)/2; vx=nose.x-eye_x; vy=nose.y-eye_y
    else:
        key=[points[i] for i in FACE_KEY_INDICES if i<len(points)]
        if not key:return center,None
        vx=key[0].x-center_x; vy=key[0].y-center_y
    length=math.hypot(vx,vy)
    if length<1.0:vx,vy,length=0.0,1.0,1.0
    eye_span=length
    if len(points)>RIGHT_EYE:
        eye_span=max(length,math.hypot(points[RIGHT_EYE].x-points[LEFT_EYE].x,points[RIGHT_EYE].y-points[LEFT_EYE].y))
    arrow_len=max(24.0,min(90.0,eye_span*1.5))
    return center,(int(round(center_x+vx/length*arrow_len)),int(round(center_y+vy/length*arrow_len)))

class _FaceSlot:
    def __init__(self,alpha:float,snap_distance:float)->None:
        self.points=PointSmoother(alpha,snap_distance); self.box=BoxSmoother(alpha); self.lost=0; self.last:FaceInstance|None=None
    def reset(self)->None:self.points.reset(); self.box.reset(); self.lost=0; self.last=None
    def update(self,raw:list[Point]|None,width:int,height:int)->FaceInstance|None:
        if not raw:
            self.lost+=1
            if self.lost>LOST_FRAMES_TIMEOUT:self.reset(); return None
            return self.last
        self.lost=0; smoothed=self.points.apply(raw,use_visibility=False); box=self.box.apply(_face_box(smoothed,width,height)); center,arrow=_direction(smoothed)
        self.last=FaceInstance(points=smoothed,box=box,center=center,arrow_end=arrow); return self.last

class FaceTracker:
    def __init__(self,slots:int=1)->None:self._slots=slots; self._states:list[_FaceSlot]=[]
    def reset(self)->None:self._states=[]
    def update(self,raw_faces:list[list[Point]],width:int,height:int)->list[FaceInstance]:
        snap=max(30.0,SNAP_FRACTION*width)
        while len(self._states)<self._slots:self._states.append(_FaceSlot(SMOOTHING_ALPHA,snap))
        for state in self._states:state.points.snap_distance=snap
        faces=[]
        for index,state in enumerate(self._states):
            raw=raw_faces[index] if index<len(raw_faces) else None
            face=state.update(raw,width,height)
            if face is not None and face.box is not None:faces.append(face)
        return faces
