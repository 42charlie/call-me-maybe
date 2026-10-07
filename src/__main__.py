from src.utils import parse_args
from src.Engine import Engine

args = parse_args()
engine = Engine(args)

#build outer prompt & encode it
prompt = engine.build_outer_prompt()
output = ""
for test in engine.test_prompts:
	#build inner prompt & encode it
	json_start_index = len(prompt)
	input_ids = engine.build_inner_prompt(prompt, test.prompt)
	input_ids = engine.select_function(input_ids)
	input_ids = engine.get_function_params(input_ids)
	output += engine.model.decode(input_ids[json_start_index:])
	break
engine.save_output(output)