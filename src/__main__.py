"""CLI entry point for the function-calling engine.

Parses arguments, initializes the engine, iterates over the test
prompts, and writes function-call results to the output file.
"""

import time
from sys import exit, stderr

from pydantic import ValidationError
from src.utils import parse_args, format_validation_error
from src.Engine import Engine
from src.schemas import FunctionCallResult


def main() -> None:
    """Run the function-calling pipeline on all test prompts."""
    start = time.perf_counter()

    try:
        args = parse_args()
        engine = Engine()
        engine.initialize(args)
    except Exception as e:
        print(f"Error: {e}", file=stderr)
        exit(1)

    prompt = engine.build_outer_prompt()
    output: list[FunctionCallResult] = []

    for test in engine.test_prompts:
        json_start_index = len(prompt)
        input_ids = engine.build_inner_prompt(prompt, test.prompt)
        input_ids = engine.select_function(input_ids)
        input_ids = engine.get_function_params(input_ids)
        func_call = engine.model.decode(input_ids[json_start_index:])
        try:
            vfunc_call = FunctionCallResult.model_validate_json(func_call)
            if engine.current_func:
                for p_name, p_val in vfunc_call.parameters.items():
                    p_spec = engine.current_func.parameters.get(p_name)
                    if p_spec:
                        if (
                            p_spec.type == "integer" and
                            isinstance(p_val, (int, float))
                        ):
                            vfunc_call.parameters[p_name] = int(p_val)
                        elif (
                            p_spec.type == "number" and
                            isinstance(p_val, (int, float))
                        ):
                            vfunc_call.parameters[p_name] = float(p_val)
            output.append(vfunc_call)
        except ValidationError as e:
            print(
                "warning: failed to parse function call for",
                f"`{test.prompt}`: {format_validation_error(e)}", file=stderr
            )
        except Exception as e:
            print(
                "warning: failed to parse function call for",
                f"`{test.prompt}`: {e}", file=stderr
            )
    end = time.perf_counter()
    print(f"Execution time: {end - start:.2f} seconds")

    try:
        engine.save_output(output)
    except Exception as e:
        print(f"Error saving output: {e}", file=stderr)
        exit(1)


if __name__ == "__main__":
    main()
