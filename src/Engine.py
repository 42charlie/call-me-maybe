import json
from sys import stderr
from os import makedirs
from typing import Any
from argparse import Namespace
from llm_sdk import Small_LLM_Model  # type: ignore
from src.utils import is_valid_string_token
from src.schemas import (FunctionCallResult, TestPromptFile, TestPromptList,
                         FunctionDefinitionFile, FunctionDefinitionList,
                         FunctionCallResultFile, FunctionDefinition)


class Engine:
    def __init__(self, args: Namespace):
        self.tool_defs: FunctionDefinitionList = []
        self.test_prompts: TestPromptList = []
        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self.numeric_ids: list[int] = []
        self.functions_token_tree: dict[int | str, Any] = {}
        self.model = Small_LLM_Model()
        self.current_func: FunctionDefinition | None = None

        self.load_files(args)
        self.load_vocab(self.model)

    def __str__(self) -> str:
        out = ""
        out += f"{len(self.tool_defs)} tool defs\n"
        out += f"{len(self.test_prompts)} test prompts\n"
        out += f"{len(self.id_to_token)} id to token\n"
        out += f"{len(self.token_to_id)} token to id\n"
        out += f"{len(self.numeric_ids)} numeric ids\n"
        return (out)

    def load_files(self, args: Namespace) -> None:
        print("loading files...")
        # load functions definitions
        try:
            with open(args.functions_definition, "r") as file:
                defs = json.loads(file.read())
            self.tool_defs = FunctionDefinitionFile.validate_python(defs)
        except Exception as e:
            print("Failed to load function definitions:", e, file=stderr)
        # load input tests
        try:
            with open(args.input, "r") as file:
                prompts = json.loads(file.read())
            self.test_prompts = TestPromptFile.validate_python(prompts)
        except Exception as e:
            print("Failed to load input tests:", e, file=stderr)
        # create output folder
        try:
            makedirs(args.output[:args.output.rfind("/")], exist_ok=True)
            with open(args.output, "w") as file:
                pass
        except Exception as e:
            print("Failed to create output file:", e, file=stderr)
        self.outfile = args.output

    def load_vocab(self, model: Small_LLM_Model) -> None:
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

        # maybe using token_to_id is better
        self.comma_token_id = self.model.encode(",").flatten().tolist()[-1]
        self.brace_token_id = self.model.encode("}").flatten().tolist()[-1]
        PARAMS = ',\n"parameters": {'
        self.tokens_param_open = self.model.encode(PARAMS).flatten().tolist()
        self.tokens_json_close = self.model.encode("\n}").flatten().tolist()
        self.tokens_comma_space = self.model.encode(", ").flatten().tolist()

        for function in self.tool_defs:
            seq = self.model.encode(function.name + '"').flatten().tolist()
            node = self.functions_token_tree
            for item in seq:
                if node.get(item) is None:
                    node[item] = {}
                node = node[item]
            node['func_def'] = function

    def build_outer_prompt(self) -> list[int]:
        print("building outer prompt...")
        tools_str = FunctionDefinitionFile.dump_json(self.tool_defs).decode()
        prompt = """
Tools:
""" + tools_str + """

Output:
"""
        prompt_ids: list[int] = self.model.encode(prompt).flatten().tolist()
        return (prompt_ids)

    def build_inner_prompt(
            self, prompt: list[int], test_prompt: str) -> list[int]:
        print("building inner prompt...")
        inner_prompt = '''{
"prompt": "''' + test_prompt + '''",
"name": "'''
        ids: list[int] = self.model.encode(inner_prompt).flatten().tolist()
        return (prompt + ids)

    def select_function(self, input_ids: list[int]) -> list[int]:
        print("selecting function ...")
        tree = self.functions_token_tree
        while True:
            if 'func_def' in tree:
                self.current_func = tree['func_def']
                input_ids += self.tokens_param_open
                return input_ids

            valid_tokens = [k for k in tree.keys() if isinstance(k, int)]
            logits = self.model.get_logits_from_input_ids(input_ids)

            for _id in range(len(logits)):
                if _id not in valid_tokens:
                    logits[_id] = float('-inf')

            chosen = max(range(len(logits)), key=logits.__getitem__)
            tree = tree[chosen]
            input_ids.append(chosen)

    def get_function_params(self, input_ids: list[int]) -> list[int]:
        print("getting function params...")
        if self.current_func is None:
            return input_ids
        params_list = list(self.current_func.parameters.items())

        for idx, (param_name, param_info) in enumerate(params_list):
            is_last = (idx == len(params_list) - 1)
            sep_id = self.brace_token_id if is_last else self.comma_token_id

            # inject deterministic parameter name
            if param_info.type == "string":
                prefix = f'"{param_name}": "'
            else:
                prefix = f'"{param_name}": '
            input_ids += self.model.encode(prefix).flatten().tolist()

            subtoken = 0
            while True:
                logits = self.model.get_logits_from_input_ids(input_ids)

                for _id in range(len(logits)):
                    if param_info.type == "string":
                        # valid string token
                        if not is_valid_string_token(self.id_to_token[_id]):
                            logits[_id] = float('-inf')
                    elif param_info.type == "number":
                        # beginning: only numeric characters
                        # mid: allow numeric characters or delimiter
                        is_numeric = _id in self.numeric_ids
                        is_terminator = subtoken > 0 and _id == sep_id
                        valid = is_numeric or is_terminator

                        if not valid:
                            logits[_id] = float('-inf')

                chosen = max(range(len(logits)), key=logits.__getitem__)
                token_str = self.id_to_token[chosen]

                # string end case with: "
                if param_info.type == "string" and '"' in token_str:
                    input_ids.append(self.token_to_id['"'])
                    if is_last:
                        input_ids.append(self.brace_token_id)
                    else:
                        input_ids += self.tokens_comma_space
                    break

                # number end case with: , or }
                if param_info.type == "number" and chosen == sep_id:
                    input_ids.append(chosen)
                    if not is_last:
                        input_ids.append(self.token_to_id[' '])
                    break

                input_ids.append(chosen)
                subtoken += 1

        # close json payload
        input_ids += self.tokens_json_close
        return input_ids

    def save_output(self, output: list[FunctionCallResult]) -> None:
        try:
            json_bytes = FunctionCallResultFile.dump_json(output, indent=2)
            with open(self.outfile, "wb") as file:
                file.write(json_bytes)
        except Exception as e:
            print("Failed to save output:", e, file=stderr)
