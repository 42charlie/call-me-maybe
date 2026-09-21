from typing import Literal
from pydantic import BaseModel

ParameterType = Literal["string", "number", "boolean"]
ParameterValue = str | int | float | bool

class Parameter(BaseModel):
    type: ParameterType

InParameterSchema = dict[str, Parameter] # name: type
OutParameterSchema = dict[str, ParameterValue]

class TestPrompt(BaseModel):
    prompt: str

class FunctionDefinition(BaseModel):
	name: str
	description: str
	parameters: InParameterSchema
	returns: Parameter

class FunctionCallResult(BaseModel):
	prompt: str
	name: str
	parameters: OutParameterSchema

# ! Invalid data raises ValidationError