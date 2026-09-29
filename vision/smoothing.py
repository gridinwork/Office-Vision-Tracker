"""Exponential moving average for points and boxes."""

from __future__ import annotations
from vision.types import Box, Point

class PointSmoother:
    def __init__(self,alpha:float,snap_distance:float)->None:
        self.alpha=alpha; self.snap_distance=snap_distance; self.prev:list[Point]|None=None
    def reset(self)->None: self.prev=None
    def apply(self,points:list[Point],use_visibility:bool)->list[Point]:
        if not points:
            self.prev=None; return []
        if self.prev is None or len(self.prev)!=len(points):
            self.prev=list(points); return list(points)
        alpha=self.alpha; smoothed=[]
        for current,previous in zip(points,self.prev):
            if use_visibility and current.visibility<0.25:
                smoothed.append(Point(previous.x,previous.y,previous.z,current.visibility)); continue
            if use_visibility and previous.visibility<0.25 and current.visibility>=0.5:
                smoothed.append(current); continue
            jump=abs(current.x-previous.x)+abs(current.y-previous.y)
            if jump>self.snap_distance:
                smoothed.append(current); continue
            visibility=alpha*current.visibility+(1-alpha)*previous.visibility if use_visibility else current.visibility
            smoothed.append(Point(alpha*current.x+(1-alpha)*previous.x,alpha*current.y+(1-alpha)*previous.y,alpha*current.z+(1-alpha)*previous.z,visibility))
        self.prev=smoothed; return smoothed

class BoxSmoother:
    def __init__(self,alpha:float)->None: self.alpha=alpha; self.prev:Box|None=None
    def reset(self)->None: self.prev=None
    def apply(self,box:Box|None)->Box|None:
        if box is None: return None
        if self.prev is None: self.prev=box; return box
        alpha=self.alpha; previous=self.prev
        if abs(box.x-previous.x)+abs(box.y-previous.y)>max(box.w,previous.w):
            self.prev=box; return box
        smoothed=Box(
            x=int(round(alpha*box.x+(1-alpha)*previous.x)),
            y=int(round(alpha*box.y+(1-alpha)*previous.y)),
            w=max(1,int(round(alpha*box.w+(1-alpha)*previous.w))),
            h=max(1,int(round(alpha*box.h+(1-alpha)*previous.h))),
            confidence=box.confidence)
        self.prev=smoothed; return smoothed
