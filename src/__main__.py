from src.utils import parse_args
from src.Engine import Engine
from src.schemas import FunctionCallResult

args = parse_args()
engine = Engine(args)

# build outer prompt & encode it
prompt = engine.build_outer_prompt()
output = []
i = 0
for test in engine.test_prompts:
    # build inner prompt & encode it
    json_start_index = len(prompt)
    input_ids = engine.build_inner_prompt(prompt, test.prompt)
    input_ids = engine.select_function(input_ids)
    input_ids = engine.get_function_params(input_ids)
    func_call = engine.model.decode(input_ids[json_start_index:])
    vfunc_call = FunctionCallResult.model_validate_json(func_call)
    output.append(vfunc_call)
    if i == 0:
        break
    i += 1
engine.save_output(output)
