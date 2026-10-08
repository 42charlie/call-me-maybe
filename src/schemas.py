from typing import Literal, Annotated
from pydantic import BaseModel, Field, TypeAdapter

class Parameter(BaseModel):
    type: Literal["string", "number", "boolean"]

InParameterSchema = dict[str, Parameter] # name: type

class TestPrompt(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000, strip_whitespace=True)

class FunctionDefinition(BaseModel):
	name: str = Field(min_length=1, max_length=60, strip_whitespace=True)
	description: str = Field(min_length=10, max_length=500, strip_whitespace=True)
	parameters: InParameterSchema
	returns: Parameter

class FunctionCallResult(BaseModel):
	prompt: str
	name: str
	parameters: dict[str, str | float | bool]

FunctionCallResultList = Annotated[list[FunctionCallResult], Field(min_length=1)]
FunctionCallResultFile = TypeAdapter(FunctionCallResultList)

FunctionDefinitionList = Annotated[list[FunctionDefinition], Field(min_length=1)]
FunctionDefinitionFile = TypeAdapter(FunctionDefinitionList)

TestPromptList = Annotated[list[TestPrompt], Field(min_length=1)]
TestPromptFile = TypeAdapter(TestPromptList)

# ! Invalid data raises ValidationError