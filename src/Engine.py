from os import makedirs
import json
import argparse
from sys import stderr
from llm_sdk import Small_LLM_Model
from src.utils import is_valid_string_token
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
		self.current_func = {}

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
		print("loading files...")
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
			print("Failed to load input tests:", e, file=stderr)
		#create output folder
		try:
			makedirs(args.output[:args.output.rfind("/")])
			with open(args.output, "w") as file:
				pass
		except Exception as e:
			print("Failed to create output file:", e, file=stderr)
		self.outfile = args.output

	def load_vocab(self, model: Small_LLM_Model):
		print("loading vocab...")
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

		#maybe using token_to_id is better
		self.comma_token_id = self.model.encode(",").flatten().tolist()[-1]
		self.brace_token_id = self.model.encode("}").flatten().tolist()[-1]

		for function in self.tool_defs:
			sequence = self.model.encode(function.name + '"').flatten().tolist()
			node = self.functions_token_tree
			for item in sequence:
				if node.get(item) == None:
					node[item] = {}
				node = node[item]
			node['func_def'] = function

	def build_outer_prompt(self):
		print("building outer prompt...")
		tools_str = FunctionDefinitionFile.dump_json(self.tool_defs).decode()
		prompt = """
You are an expert function-calling agent. Your task is to analyze the user request and select the single most appropriate tool from the available tools to satisfy it.

Rules:
1. You must respond ONLY with a valid function call.
2. Select the function whose description and parameters best match the user's intent.
3. Extract all required arguments from the user input and ensure their types match the parameter definitions exactly.
4. Do not include any explanations, greetings, comments, or extra text. Output must strictly adhere to the expected format.

Available Tools:
""" + tools_str + """

Output:
"""
		return (self.model.encode(prompt).flatten().tolist())

	def build_inner_prompt(self, prompt, test_prompt):
		print("building inner prompt...")
		inner_prompt = '''{
"prompt": "''' + test_prompt + '''",
"name": "'''
		return (prompt + self.model.encode(inner_prompt).flatten().tolist())

	def select_function(self, input_ids):
		print("selecting function ...")
		tree = self.functions_token_tree
		while True:
			if 'func_def' in tree:
				self.current_func = tree['func_def']
				input_ids += self.model.encode(''',
"parameters": {''').flatten().tolist()
				return input_ids

			valid_tokens = [k for k in tree.keys() if isinstance(k, int)]
			logits = self.model.get_logits_from_input_ids(input_ids)

			for _id in range(len(logits)):
				if _id not in valid_tokens:
					logits[_id] = float('-inf')

			chosen = max(range(len(logits)), key=logits.__getitem__)
			tree = tree[chosen]
			input_ids.append(chosen)

	def get_function_params(self, input_ids):
		print("getting function params...")
		params_list = list(self.current_func.parameters.items())
		
		for idx, (param_name, param_info) in enumerate(params_list):
			is_last = (idx == len(params_list) - 1)
			sep_id = self.brace_token_id if is_last else self.comma_token_id

			# inject deterministic parameter name
			prefix = f'"{param_name}": "' if param_info.type == "string" else f'"{param_name}": '
			input_ids += self.model.encode(prefix).flatten().tolist()

			subtoken = 0
			while True:
				logits = self.model.get_logits_from_input_ids(input_ids)

				for _id in range(len(logits)):
					if param_info.type == "string":
						#valid string token
						if not is_valid_string_token(self.id_to_token[_id]):
							logits[_id] = float('-inf')
					elif param_info.type == "number":
						#first digit: only numeric characters
						# after first digit: allow numeric characters or delimiter
						valid = (_id in self.numeric_ids) or (subtoken > 0 and _id == sep_id)
						if not valid:
							logits[_id] = float('-inf')

				chosen = max(range(len(logits)), key=logits.__getitem__)
				token_str = self.id_to_token[chosen]

				#string end case with: "
				if param_info.type == "string" and '"' in token_str:
					input_ids += self.model.encode('"' + ("}" if is_last else ", ")).flatten().tolist()
					break

				#number end case with: , or }
				if param_info.type == "number" and chosen == sep_id:
					input_ids.append(chosen)
					if not is_last:
						input_ids += self.model.encode(" ").flatten().tolist()
					break

				input_ids.append(chosen)
				subtoken += 1

		# close json payload
		input_ids += self.model.encode("\n}").flatten().tolist()
		return input_ids

	def save_output(self, output):
		try:
			with open(self.outfile) as file:
				file.write(output)
		except Exception as e:
			print("Failed to save output:", e, file=stderr)