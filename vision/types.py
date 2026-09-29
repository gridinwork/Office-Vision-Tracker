"""Shared vision result types."""

from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    confidence: float | None = None
    @property
    def cx(self) -> float: return self.x + self.w / 2.0
    @property
    def cy(self) -> float: return self.y + self.h / 2.0
    @property
    def area(self) -> float: return float(max(0,self.w)*max(0,self.h))

@dataclass
class PoseInstance:
    points: list[Point]
    box: Box | None
    confidence: float
    shoulder_width: float = 0.0

@dataclass
class FaceInstance:
    points: list[Point]
    box: Box | None
    center: tuple[int,int] | None
    arrow_end: tuple[int,int] | None

@dataclass
class HandInstance:
    side: str
    points: list[Point]
    box: Box | None
    confidence: float | None
    moving: bool
    trail: list[tuple[int,int]] = field(default_factory=list)

@dataclass
class FrameVision:
    poses: list[PoseInstance]=field(default_factory=list)
    faces: list[FaceInstance]=field(default_factory=list)
    hands: list[HandInstance]=field(default_factory=list)
    person_count:int=0
    person_detected:bool=False
    face_detected:bool=False
    left_hand_detected:bool=False
    right_hand_detected:bool=False
    movement:str="—"
    gesture:str="—"
    frame_number:int=0
    frame_width:int=0
    frame_height:int=0
    timings_ms:dict[str,float]=field(default_factory=dict)

def empty_vision(width:int,height:int)->FrameVision:
    return FrameVision(frame_width=width,frame_height=height)
