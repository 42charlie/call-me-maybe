import argparse
from src.schemas import EngineConfig


def parse_args() -> EngineConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    args = parser.parse_args()
    # validate args using EngineConfig
    config = EngineConfig(
        funcs_def_path=args.functions_definition,
        in_path=args.input,
        out_path=args.output
    )
    return (config)


def is_valid_string_token(token_str: str) -> bool:
    for ch in token_str:
        # printable = not a control char
        if ord(ch) < 0x20 or ord(ch) == 0x7F:
            return False
    return True
