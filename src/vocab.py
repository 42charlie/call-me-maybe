import json
from llm_sdk import Small_LLM_Model

def load_vocab(model: Small_LLM_Model):
	id_to_token = {}
	token_to_id = {}
	numeric_ids = []

	vocab_path = model.get_path_to_vocab_file()
	with open(vocab_path, "r") as file:
		vocab = json.load(file)
	for v in vocab.values():
		value = model.decode(v)
		id_to_token[v] = value
		token_to_id[value] = v
		if value in "-0123456789.":
			numeric_ids.append(v)

	#test
	for i in numeric_ids:
		print(f"{i}: {model.decode(i)}")

# pre-calculate function name token sequences