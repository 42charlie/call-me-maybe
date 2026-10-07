import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
	parser.add_argument("--input", default="data/input/function_calling_tests.json")
	parser.add_argument("--output", default="data/output/function_calls.json")
	return(parser.parse_args())

def is_valid_string_token(token_str):
    for ch in token_str:
        # printable = not a control char
        if ord(ch) < 0x20 or ord(ch) == 0x7F:
            return False
    return True