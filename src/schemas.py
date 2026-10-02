from typing import Literal, Annotated
from pydantic import BaseModel, Field, TypeAdapter

ParameterType = Literal["string", "number", "boolean"]
ParameterValue = str | int | float | bool

class Parameter(BaseModel):
    type: ParameterType

InParameterSchema = dict[str, Parameter] # name: type
OutParameterSchema = Annotated[dict[str, ParameterValue], Field(min_length=1, max_length=1)]

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
	parameters: OutParameterSchema

FunctionDefinitionFile = TypeAdapter(Annotated[list[FunctionDefinition], Field(min_length=1)])

TestPromptFile = TypeAdapter(Annotated[list[TestPrompt], Field(min_length=1)])

# ! Invalid data raises ValidationError