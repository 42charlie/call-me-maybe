import json
import argparse
from sys import stderr
from llm_sdk import Small_LLM_Model
from src.schemas import TestPromptFile, FunctionDefinitionFile

class Engine:
	def __init__(self, args):
		self.tool_defs = {}
		self.test_prompts = {}
		self.id_to_token = {}
		self.token_to_id = {}
		self.numeric_ids = []
		self.model = Small_LLM_Model()

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
			FunctionDefinitionFile.validate_python(self.tool_defs)
		except Exception as e:
			print("Failed to load function definitions:", e, file=stderr)
		#load input tests
		try:
			with open(args.input, "r") as file:
				self.test_prompts = json.loads(file.read())
			TestPromptFile.validate_python(self.test_prompts)
		except Exception as e:
			print("Failed to load input tests :", e, file=stderr)

	def load_vocab(self, model: Small_LLM_Model):
		vocab_path = model.get_path_to_vocab_file()
		with open(vocab_path, "r") as file:
			vocab = json.load(file)
		for _id in vocab.values():
			token = model.decode(_id)
			self.id_to_token[_id] = token
			self.token_to_id[token] = _id
			if token in "-0123456789.":
				self.numeric_ids.append(_id)
		#Pre-calculate Function Name Token Sequences