from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class Example(BaseModel):
    """Example input/output for a coding question."""
    input: str = Field(..., description="Input example")
    output: str = Field(..., description="Expected output")
    explanation: Optional[str] = Field(None, description="Explanation of the example")
    id: Optional[str] = Field(None, alias="_id", description="MongoDB document ID")


class TestCase(BaseModel):
    """Test case for a coding question."""
    input: str = Field(..., description="Test input")
    expectedOutput: str = Field(..., description="Expected output")
    isHidden: bool = Field(default=False, description="Whether this test case is hidden")
    id: Optional[str] = Field(None, alias="_id", description="MongoDB document ID")


class StarterCode(BaseModel):
    """Starter code templates for different languages."""
    python: str = Field(..., description="Python starter code")
    javascript: str = Field(..., description="JavaScript starter code")
    java: str = Field(..., description="Java starter code")
    cpp: str = Field(..., description="C++ starter code")


class DriverCode(BaseModel):
    """Driver code templates for different languages."""
    python: str = Field(..., description="Python driver code")
    javascript: str = Field(..., description="JavaScript driver code")
    java: str = Field(..., description="Java driver code")
    cpp: str = Field(..., description="C++ driver code")


class CodingQuestion(BaseModel):
    """Schema for a single coding question."""
    title: str = Field(..., description="Question title")
    description: str = Field(..., description="Detailed problem description")
    difficulty: str = Field(..., description="Difficulty level: easy, medium, or hard")
    tags: List[str] = Field(default_factory=list, description="Tags/categories for the question")
    functionName: str = Field(..., description="Name of the function to implement")
    functionSignature: str = Field(..., description="Function signature with type hints")
    examples: List[Example] = Field(default_factory=list, description="Example inputs and outputs")
    constraints: List[str] = Field(default_factory=list, description="Problem constraints")
    starterCode: StarterCode = Field(..., description="Starter code for different languages")
    driverCode: DriverCode = Field(..., description="Driver code for different languages")
    testCases: List[TestCase] = Field(default_factory=list, description="Test cases for validation")
    id: Optional[str] = Field(None, alias="_id", description="MongoDB document ID")


class CodingQuestionSet(BaseModel):
    """Schema for a coding question set document."""
    id: Optional[str] = Field(None, alias="_id", description="MongoDB document ID")
    date: datetime = Field(..., description="Date for this question set")
    questions: List[CodingQuestion] = Field(..., description="List of coding questions")
    isActive: bool = Field(default=True, description="Whether this question set is active")
    createdAt: Optional[datetime] = Field(None, description="Creation timestamp")
    updatedAt: Optional[datetime] = Field(None, description="Last update timestamp")
    v: Optional[int] = Field(None, alias="__v", description="MongoDB version key")

