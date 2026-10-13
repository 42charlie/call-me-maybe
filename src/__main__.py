"""CLI entry point for the function-calling engine.

Parses arguments, initializes the engine, iterates over the test
prompts, and writes function-call results to the output file.
"""

import time
from sys import exit
from src.utils import parse_args
from src.Engine import Engine
from src.schemas import FunctionCallResult


start = time.perf_counter()

try:
    args = parse_args()
    engine = Engine()
    engine.initialize(args)
except Exception as e:
    print(f"Error: {e}")
    exit(1)

prompt = engine.build_outer_prompt()
output: list[FunctionCallResult] = []

for test in engine.test_prompts:
    json_start_index = len(prompt)
    input_ids = engine.build_inner_prompt(prompt, test.prompt)
    input_ids = engine.select_function(input_ids)
    input_ids = engine.get_function_params(input_ids)
    func_call = engine.model.decode(input_ids[json_start_index:])
    vfunc_call = FunctionCallResult.model_validate_json(func_call)
    output.append(vfunc_call)
end = time.perf_counter()
print(f"Execution time: {end - start:.2f} seconds")

engine.save_output(output)
