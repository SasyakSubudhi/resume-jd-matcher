"""
schema.py - Fixed JSON schema for resumes and job descriptions (Pydantic models).

Every resume and every job description is converted into one of these two
fixed structures, so the rest of the system (ranking, skill-gap) can rely on
the same fields always being present.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Experience(BaseModel):
    title: str = Field(description="Job title / role held")
    company: str = Field(default="", description="Company or organisation name")
    years: float = Field(default=0.0, description="Duration of this role in years (e.g. 1.5)")
    description: str = Field(default="", description="One-line summary of the work done")


class ResumeProfile(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: Optional[str] = Field(default=None, description="Email address if present")
    total_experience_years: float = Field(default=0.0, description="Total professional experience in years")
    skills: List[str] = Field(default_factory=list, description="Technical skills, tools, languages, frameworks")
    education: List[str] = Field(default_factory=list, description="Degrees, e.g. 'B.Tech Computer Science, PES University'")
    experience: List[Experience] = Field(default_factory=list, description="Work experience entries")
    projects: List[str] = Field(default_factory=list, description="Short project descriptions")
    certifications: List[str] = Field(default_factory=list, description="Certifications")
    summary: str = Field(default="", description="2-line professional summary of the candidate")


class JobProfile(BaseModel):
    title: str = Field(description="Job title")
    company: str = Field(default="", description="Hiring company")
    min_experience_years: float = Field(default=0.0, description="Minimum years of experience required")
    required_skills: List[str] = Field(default_factory=list, description="Must-have skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Nice-to-have skills")
    education_requirements: List[str] = Field(default_factory=list, description="Required degree(s)")
    responsibilities: List[str] = Field(default_factory=list, description="Key responsibilities")
    summary: str = Field(default="", description="2-line summary of the role")


class SkillGap(BaseModel):
    skill: str = Field(description="Missing or weak skill")
    importance: str = Field(description="'required' or 'preferred'")
    severity: str = Field(description="'high', 'medium' or 'low'")
    recommendation: str = Field(description="Concrete action to close the gap (course, project, certification)")


class SkillGapReport(BaseModel):
    candidate: str
    job_title: str
    fit_score: float = Field(description="Overall fit from 0 to 100")
    verdict: str = Field(description="'Strong fit', 'Moderate fit' or 'Weak fit'")
    strengths: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)
    gaps: List[SkillGap] = Field(default_factory=list)
    experience_gap: str = Field(default="", description="Comment on experience vs requirement")
    learning_plan: List[str] = Field(default_factory=list, description="Ordered 30/60/90-day style steps")
