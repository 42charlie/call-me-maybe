from typing import Literal, Annotated
from pydantic import BaseModel, Field, TypeAdapter, StringConstraints


class Parameter(BaseModel):
    type: Literal["string", "number", "boolean"]


InParameterSchema = dict[str, Parameter]  # name: type


# define constrained string types
NonEmptyPrompt = Annotated[
    str, StringConstraints(min_length=1,
                           max_length=2000,
                           strip_whitespace=True)
]
NonEmptyName = Annotated[
    str, StringConstraints(min_length=1,
                           max_length=60,
                           strip_whitespace=True)
]
NonEmptyDescription = Annotated[
    str, StringConstraints(min_length=10,
                           max_length=500,
                           strip_whitespace=True)
]


class TestPrompt(BaseModel):
    prompt: NonEmptyPrompt


class FunctionDefinition(BaseModel):
    name: NonEmptyName
    description: NonEmptyDescription
    parameters: InParameterSchema
    returns: Parameter


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]


FunctionCallResultList = Annotated[list[FunctionCallResult],
                                   Field(min_length=1)]
FunctionCallResultFile = TypeAdapter(FunctionCallResultList)

FunctionDefinitionList = Annotated[list[FunctionDefinition],
                                   Field(min_length=1)]
FunctionDefinitionFile = TypeAdapter(FunctionDefinitionList)

TestPromptList = Annotated[list[TestPrompt], Field(min_length=1)]
TestPromptFile = TypeAdapter(TestPromptList)

# ! Invalid data raises ValidationError
