import argparse
import json
from src.schemas import TestPromptFile, FunctionDefinitionFile

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
	parser.add_argument("--input", default="data/input/function_calling_tests.json")
	parser.add_argument("--output", default="data/output/function_calls.json")
	return(parser.parse_args())

def load_files(args: argparse.Namespace):
	#load functions definitions
	try:
		with open(args.functions_definition, "r") as file:
			functions_def = json.loads(file.read())
		FunctionDefinitionFile.validate_python(functions_def)
		print(functions_def)
	except Exception as e:
		print("Failed to load function definitions:", e)

	#load input tests
	try:
		with open(args.input, "r") as file:
			inputs = json.loads(file.read())
		TestPromptFile.validate_python(inputs)
		print(inputs)
	except Exception as e:
		print("Failed to load input tests :", e)
