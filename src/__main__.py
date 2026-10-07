from src.utils import parse_args
from src.Engine import Engine

args = parse_args()
engine = Engine(args)

#build outer prompt & encode it
prompt = engine.build_outer_prompt()
for test in engine.test_prompts:
	#build inner prompt & encode it
	json_start = len(prompt)
	input_ids = engine.build_inner_prompt(prompt, test.prompt)
	while True:
		input_ids = engine.select_function(input_ids)
		input_ids = engine.get_function_params(input_ids)
		output = engine.model.decode(input_ids[json_start:])
		break
		
	break