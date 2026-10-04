from src.utils import parse_args
from src.Engine import Engine

args = parse_args()
engine = Engine(args)

#build outer prompt & encode it
engine.build_outer_prompt()
for test in engine.test_prompts:
	#build inner prompt & encode it
	input_ids = engine.build_inner_prompt(test.prompt)
	while True:
		#get function name
		#get functions paramteres
		pass