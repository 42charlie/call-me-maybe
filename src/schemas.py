"""Pydantic schemas and type adapters for the engine's I/O."""

from typing import Literal, Annotated
from pydantic import BaseModel, Field, TypeAdapter, StringConstraints


class EngineConfig(BaseModel):
    """Paths configuration for the engine.

    Attributes:
        funcs_def_path: Path to the functions definition JSON file.
        in_path: Path to the input test prompts JSON file.
        out_path: Path to the output results JSON file.
    """
    funcs_def_path: Annotated[str, StringConstraints(min_length=1,
                                                     strip_whitespace=True)]
    in_path: Annotated[str, StringConstraints(min_length=1,
                                              strip_whitespace=True)]
    out_path: Annotated[str, StringConstraints(min_length=1,
                                               strip_whitespace=True)]


class Parameter(BaseModel):
    """A parameter type descriptor.

    Attributes:
        type: One of "string", "number", or "boolean".
    """
    type: Literal["string", "number", "boolean"]


InParameterSchema = dict[str, Parameter]  # name: type


# define constrained string types
NonEmptyPrompt = Annotated[
    str, StringConstraints(max_length=2000,
                           strip_whitespace=True)
]
NonEmptyName = Annotated[
    str, StringConstraints(max_length=60,
                           strip_whitespace=True)
]
NonEmptyDescription = Annotated[
    str, StringConstraints(max_length=500,
                           strip_whitespace=True)
]


class TestPrompt(BaseModel):
    """A single test prompt.

    Attributes:
        prompt: The user prompt to send to the model.
    """
    prompt: NonEmptyPrompt


class FunctionDefinition(BaseModel):
    """A callable function (tool) definition.

    Attributes:
        name: Function name.
        description: Human-readable description of the function.
        parameters: Mapping of parameter name to its type descriptor.
        returns: Type descriptor for the function's return value.
    """
    name: NonEmptyName
    description: NonEmptyDescription
    parameters: InParameterSchema
    returns: Parameter


class FunctionCallResult(BaseModel):
    """Result of a single function call.

    Attributes:
        prompt: The prompt that triggered the call.
        name: Name of the called function.
        parameters: Argument name to value mapping produced by the model.
    """
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
