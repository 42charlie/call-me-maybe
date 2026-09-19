from llm_sdk import Small_LLM_Model
import json

text = "Hello world!"

model = Small_LLM_Model()

inputs_ids = model.encode(text).tolist()[0]
logits = model.get_logits_from_input_ids(inputs_ids)
vocab_file = model.get_path_to_vocab_file()
with open(vocab_file, "r") as file:
    vocab = json.load(file)
print(list(vocab.items())[:10])
print(logits[:10])
