import json
import argparse
from sys import stderr
from llm_sdk import Small_LLM_Model
from numpy import inner
from src.schemas import TestPromptFile, FunctionDefinitionFile

class Engine:
	def __init__(self, args):
		self.tool_defs = {}
		self.test_prompts = {}
		self.id_to_token = {}
		self.token_to_id = {}
		self.numeric_ids = []
		self.functions_token_tree = {}
		self.model = Small_LLM_Model()

		#current ids sequence
		#selected function

		self.load_files(args)
		self.load_vocab(self.model)
	def __str__(self):
		out = ""
		out += f"{len(self.tool_defs)} tool defs\n" 
		out += f"{len(self.test_prompts)} test prompts\n" 
		out += f"{len(self.id_to_token)} id to token\n" 
		out += f"{len(self.token_to_id)} token to id\n" 
		out += f"{len(self.numeric_ids)} numeric ids\n" 
		return (out)

	def load_files(self, args: argparse.Namespace):
		#load functions definitions
		try:
			with open(args.functions_definition, "r") as file:
				self.tool_defs = json.loads(file.read())
			self.tool_defs = FunctionDefinitionFile.validate_python(self.tool_defs)
		except Exception as e:
			print("Failed to load function definitions:", e, file=stderr)
		#load input tests
		try:
			with open(args.input, "r") as file:
				self.test_prompts = json.loads(file.read())
			self.test_prompts = TestPromptFile.validate_python(self.test_prompts)
		except Exception as e:
			print("Failed to load input tests :", e, file=stderr)

	def load_vocab(self, model: Small_LLM_Model):
		ALLOWED_CHARS = set("0123456789.-")
		vocab_path = model.get_path_to_vocab_file()
		with open(vocab_path, "r") as file:
			vocab = json.load(file)
		for _id in vocab.values():
			token = model.decode(_id)
			self.id_to_token[_id] = token
			self.token_to_id[token] = _id
			if token in ALLOWED_CHARS:
				self.numeric_ids.append(_id)
		for function in self.tool_defs:
			sequence = self.model.encode(function.name).flatten().tolist()
			node = self.functions_token_tree
			for item in sequence:
				if node.get(item) == None:
					node[item] = {}
				node = node[item]
			node['is_end'] = True

	def build_outer_prompt(self):
		tools_str = FunctionDefinitionFile.dump_json(self.tool_defs).decode()
		prompt = f"""
You are an expert function-calling agent. Your task is to analyze the user request and select the single most appropriate tool from the available tools to satisfy it.

Rules:
1. You must respond ONLY with a valid function call.
2. Select the function whose description and parameters best match the user's intent.
3. Extract all required arguments from the user input and ensure their types match the parameter definitions exactly.
4. Do not include any explanations, greetings, comments, or extra text. Output must strictly adhere to the expected format.

Available Tools:
{tools_str}

Output:
"""
		self.prompt = self.model.encode(prompt).flatten().tolist()

	def build_inner_prompt(self, test_prompt):
		inner_prompt = '''{
"prompt": "''' + test_prompt + '''",
"name": "'''
		return (self.prompt + self.model.encode(inner_prompt).flatten().tolist())
