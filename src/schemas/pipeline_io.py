from pydantic import BaseModel, Field
from typing import List, Any

class ObjectCentricInstructionParsingOutput(BaseModel):
    edited_objects: List[str] = Field(
        ..., description="List of objects that are being edited, e.g., ['dog', 'tree']"
    )

class ObjectCentricInstructionParsingReplaceOutput(BaseModel):
    edited_objects: List[str] = Field(
        ..., description="List of objects that are being edited, e.g., ['dog', 'tree']"
    )
    generated_objects: List[str] = Field(
        ..., description="List of new objects that are generated to replace the edited objects, e.g., ['cat', 'flower']"
    )

class HumanCentricInstructionParsingOutput(BaseModel):
    edited_subjects: List[str] = Field(
        ..., description="List of human subjects that are being edited, e.g., ['man in suit']"
    )

class HumanCentricInstructionParsingRubricOutput(BaseModel):
    edited_subjects: str = Field(
        ..., description="Visual description for Grounding model (e.g. 'man in suit')"
    )
    edit_attributes: List[str] = Field(
        ..., description="List of attributes from the pre-defined set (i.e., Face ID, Hair Appearance, and Body Appearance)"
    )

class VLMPairJudgeOutput(BaseModel):
    winner: str = Field(
        ..., description="The winner of the comparison, either 'Image A', 'Image B', or 'Tie'"
    )

class VLMPairJudgeWithReasonOutput(BaseModel):
    winner: str = Field(
        ..., description="The winner of the comparison, either 'Image A', 'Image B', or 'Tie'"
    )
    reasoning: str = Field(
        ..., description="Comparative analysis and rationale behind the judgment"
    )

class UnicEditVIEScoreOutput(BaseModel):
    score: int = Field(
        ..., description="An integer score from 0 to 10."
    )
    reason: str = Field(
        ..., description="A detailed explanation justifying the assigned score"
    )

class EditScoreVIEScoreOutput(BaseModel):
    reasoning: str = Field(
        ..., description="A detailed explanation justifying the assigned score"
    )
    score: List[int] = Field(
        ..., description="[score1, score2] from 0 to 25."
    )

class BasicObjectInstructionParsingInput(BaseModel):
    instruction: str = Field(
        ..., description="Editing Instruction"
    )
    edit_task: str = Field(
        ..., description="Editing Task"
    )

class InstructionParsingInput(BaseModel):
    instruction: str = Field(
        ..., description="Editing Instruction"
    )

class PairJudgeInput(BaseModel):
    instruction: str = Field(
        ..., description="Editing Instruction"
    )
    input_image: Any = Field(description="Input Image")
    edited_images: List[Any] = Field(
        ...,
        min_length=2,
        description="List of edited images to be compared, e.g., [edited_image_A, edited_image_B]",
    )

class VIEScoreInput(BaseModel):
    instruction: str = Field(
        ..., description="Editing Instruction"
    )
    input_image: Any = Field(description="Input Image")
    edited_image: Any = Field(description="Edited Image")