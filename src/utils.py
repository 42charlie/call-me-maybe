"""CLI argument parsing and token validation helpers."""

import argparse
from src.schemas import EngineConfig
from pydantic import ValidationError


def parse_args() -> EngineConfig:
    """Parse CLI arguments and build an engine configuration.

    Returns:
        EngineConfig: Validated configuration holding the paths to the
            functions definition, input tests, and output files.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json")
    args = parser.parse_args()
    # validate args using EngineConfig
    try:
        config = EngineConfig(
            funcs_def_path=args.functions_definition,
            in_path=args.input,
            out_path=args.output
        )
    except ValidationError as e:
        raise RuntimeError(
            f"Error in command-line arguments: {format_validation_error(e)}"
        )
    return (config)


def is_valid_string_token(token_str: str) -> bool:
    """Check whether a token contains only printable characters.

    Args:
        token_str: Token string to validate.

    Returns:
        bool: True if no control characters (ASCII < 0x20 or 0x7F) are
            present, False otherwise.
    """
    for ch in token_str:
        # printable = not a control char
        if ord(ch) < 0x20 or ord(ch) == 0x7F:
            return False
    return True


def format_validation_error(err: ValidationError) -> str:
    """Format a Pydantic ValidationError into a clean readable message.

    Args:
        err: The ValidationError to format.

    Returns:
        str: A formatted string containing the validation errors.
    """
    messages: list[str] = []
    for e in err.errors():
        loc = " -> ".join(str(item) for item in e.get("loc", []))
        msg = e.get("msg", "Validation error")
        if loc:
            messages.append(f"Field '{loc}': {msg}")
        else:
            messages.append(msg)
    return "; ".join(messages)
