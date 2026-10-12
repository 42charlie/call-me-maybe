"""Function-calling engine: loads files/vocab and generates tool calls."""

import json
from sys import stderr
from os import makedirs, path
from typing import Any
from llm_sdk import Small_LLM_Model  # type: ignore
from src.utils import is_valid_string_token
from src.schemas import (FunctionCallResult, TestPromptFile, TestPromptList,
                         FunctionDefinitionFile, FunctionDefinitionList,
                         FunctionCallResultFile, FunctionDefinition,
                         EngineConfig)


class Engine:
    """Drive tool selection and argument generation via a small LLM.

    Attributes:
        tool_defs: Loaded function definitions.
        test_prompts: Loaded test prompts.
        id_to_token: Vocabulary id to token mapping.
        token_to_id: Token to vocabulary id mapping.
        numeric_ids: Token ids representing numeric characters.
        functions_token_tree: Trie of function-name token sequences.
        model: Underlying small LLM model.
        current_func: Function selected for the current call, if any.
    """
    def __init__(self, args: EngineConfig):
        """Initialize the engine and load files and vocabulary.

        Args:
            args: Engine configuration with input/output paths.

        Raises:
            Exception: If loading files or vocabulary fails.
        """
        self.tool_defs: FunctionDefinitionList = []
        self.test_prompts: TestPromptList = []
        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self.numeric_ids: list[int] = []
        self.functions_token_tree: dict[int | str, Any] = {}
        self.model = Small_LLM_Model()
        self.current_func: FunctionDefinition | None = None

        try:
            self.load_files(args)
            self.load_vocab(self.model)
        except Exception as e:
            print("Failed to initialize engine:", e, file=stderr)
            raise e

    def load_files(self, args: EngineConfig) -> None:
        """Load tool definitions and test prompts, then prepare output file.

        Reads the function definitions and input test prompts from the paths
        in ``args``, validates them against the corresponding schemas, and
        creates (or truncates) the output file.

        Args:
            args: Engine configuration containing the input/output paths.

        Raises:
            Exception: If any file cannot be read, parsed, or validated.
        """
        print("loading files...")
        # load functions definitions
        try:
            with open(args.funcs_def_path, "r") as file:
                defs = json.loads(file.read())
            self.tool_defs = FunctionDefinitionFile.validate_python(defs)
        except Exception as e:
            print("Failed to load function definitions:", e, file=stderr)
            raise e
        # load input tests
        try:
            with open(args.in_path, "r") as file:
                prompts = json.loads(file.read())
            self.test_prompts = TestPromptFile.validate_python(prompts)
        except Exception as e:
            print("Failed to load input tests:", e, file=stderr)
            raise e
        # create output folder
        try:
            normalized = path.normpath(args.out_path)
            dirname = path.dirname(normalized)
            if dirname:
                makedirs(dirname, exist_ok=True)
            with open(normalized, "w") as file:
                pass
        except Exception as e:
            print("Failed to create output file:", e, file=stderr)
            raise e
        self.outfile = normalized

    def load_vocab(self, model: Small_LLM_Model) -> None:
        """Load the model vocabulary and build token lookup structures.

        Populates the id-to-token and token-to-id maps, collects numeric
        token ids, caches frequently used token sequences, and builds the
        prefix tree mapping function names to their definitions.

        Args:
            model: The LLM model providing the vocab file and tokenizer.
        """
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
        """Build and encode the shared outer prompt.

        Serializes the tool definitions and wraps them in the fixed
        "Tools:"/"Output:" template used as the prefix for every request.

        Returns:
            list[int]: Encoded token ids of the outer prompt.
        """
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
        """Build and encode the per-test inner prompt.

        Appends the JSON scaffolding for the current test prompt onto the
        outer prompt, stopping right before the function name.

        Args:
            prompt: Encoded outer prompt token ids.
            test_prompt: Raw user prompt to embed in the JSON body.

        Returns:
            list[int]: Encoded token ids of the combined prompt.
        """
        print("building inner prompt...")
        inner_prompt = '''{
"prompt": "''' + test_prompt + '''",
"name": "'''
        ids: list[int] = self.model.encode(inner_prompt).flatten().tolist()
        return (prompt + ids)

    def select_function(self, input_ids: list[int]) -> list[int]:
        """Select a function by walking the function-name token tree.

        At each step, constrains the logits to valid child tokens of the
        current tree node and greedily picks the best one, until a node
        holding a function definition is reached. Opens the parameters
        block and stores the match in ``self.current_func``.

        Args:
            input_ids: Token ids generated so far.

        Returns:
            list[int]: ``input_ids`` extended with the function name and
                the parameter-block prefix.
        """
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

    def _decode_boolean_param(self, input_ids: list[int]) -> list[int]:
        """Decode a boolean value.

        Constrains the next-token logits to ``true``/``false`` starts and
        appends the chosen literal, including any continuation subtokens.

        Args:
            input_ids: Token ids generated so far.

        Returns:
            list[int]: ``input_ids`` extended with the boolean literal.
        """
        true_ids: list[int] = self.model.encode("true").flatten().tolist()
        false_ids: list[int] = self.model.encode("false").flatten().tolist()

        valid_tokens = [true_ids[0], false_ids[0]]
        logits = self.model.get_logits_from_input_ids(input_ids)

        for _id in range(len(logits)):
            if _id not in valid_tokens:
                logits[_id] = float("-inf")

        chosen = max(range(len(logits)), key=logits.__getitem__)
        input_ids.append(chosen)

        # Append continuation tokens if the literal was split
        if chosen == true_ids[0] and len(true_ids) > 1:
            input_ids.extend(true_ids[1:])
        elif chosen == false_ids[0] and len(false_ids) > 1:
            input_ids.extend(false_ids[1:])

        return input_ids

    def _decode_number_param(
        self, input_ids: list[int], sep_token_id: int
    ) -> list[int]:
        """Decode a numeric value.

        Repeatedly samples from numeric-only tokens until the separator
        token is chosen, which terminates the number.

        Args:
            input_ids: Token ids generated so far.
            sep_token_id: Token id that terminates the number.

        Returns:
            list[int]: ``input_ids`` extended with the numeric tokens.
        """
        subtoken = 0
        while True:
            logits = self.model.get_logits_from_input_ids(input_ids)

            for _id in range(len(logits)):
                is_numeric = _id in self.numeric_ids
                is_terminator = subtoken > 0 and _id == sep_token_id
                if not (is_numeric or is_terminator):
                    logits[_id] = float("-inf")

            chosen = max(range(len(logits)), key=logits.__getitem__)
            if chosen == sep_token_id:
                break

            input_ids.append(chosen)
            subtoken += 1

        return input_ids

    def _decode_string_param(self, input_ids: list[int]) -> list[int]:
        """Decode string content.

        Samples only printable tokens until a token containing a double
        quote is reached, preserving any characters before the quote, then
        appends the closing quote token.

        Args:
            input_ids: Token ids generated so far.

        Returns:
            list[int]: ``input_ids`` extended with the string content.
        """
        while True:
            logits = self.model.get_logits_from_input_ids(input_ids)

            for _id in range(len(logits)):
                if not is_valid_string_token(self.id_to_token[_id]):
                    logits[_id] = float("-inf")

            chosen = max(range(len(logits)), key=logits.__getitem__)
            token_str = self.id_to_token[chosen]

            if '"' in token_str:
                quote_idx = token_str.index('"')
                prefix = token_str[:quote_idx]
                # Preserve characters that appeared before the quote
                if prefix:
                    input_ids.extend(
                        self.model.encode(prefix).flatten().tolist()
                    )
                break

            input_ids.append(chosen)

        # Force the exact closing double-quote token
        input_ids.append(self.token_to_id['"'])
        return input_ids

    def get_function_params(self, input_ids: list[int]) -> list[int]:
        """Dispatch parameter generation by type.

        For each parameter of ``self.current_func``, appends the
        deterministic key prefix and delegates value decoding based on the
        parameter type (boolean, number, or string), inserting separators
        between params and closing the JSON payload.

        Args:
            input_ids: Token ids generated so far.

        Returns:
            list[int]: ``input_ids`` extended with the full parameter
                payload.
        """
        print("getting function params...")
        if self.current_func is None:
            return input_ids

        params_list = list(self.current_func.parameters.items())

        for idx, (param_name, param_info) in enumerate(params_list):
            is_last = (idx == len(params_list) - 1)
            sep_id = self.brace_token_id if is_last else self.comma_token_id

            # Deterministic parameter key prefix
            if param_info.type == "string":
                prefix = f'"{param_name}": "'
            else:
                prefix = f'"{param_name}": '
            input_ids.extend(self.model.encode(prefix).flatten().tolist())

            # Type dispatch
            if param_info.type == "boolean":
                input_ids = self._decode_boolean_param(input_ids)
            elif param_info.type == "number":
                input_ids = self._decode_number_param(input_ids, sep_id)
            elif param_info.type == "string":
                input_ids = self._decode_string_param(input_ids)

            # Deterministic parameter separator
            if is_last:
                input_ids.append(self.brace_token_id)
            else:
                input_ids.extend(self.tokens_comma_space)

        # Deterministic payload closure
        input_ids.extend(self.tokens_json_close)
        return input_ids

    def save_output(self, output: list[FunctionCallResult]) -> None:
        """Serialize function call results and write them to the output file.

        Args:
            output: Function call results to persist.

        Raises:
            Exception: If serialization or writing to ``self.outfile`` fails.
        """
        try:
            json_bytes = FunctionCallResultFile.dump_json(output, indent=2)
            with open(self.outfile, "wb") as file:
                file.write(json_bytes)
        except Exception as e:
            print("Failed to save output:", e, file=stderr)
            raise e
