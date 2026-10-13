"""Function-calling engine: loads files/vocab and generates tool calls."""

import json
from sys import stderr
from os import makedirs, path
from src.utils import format_validation_error
from typing import Any
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from llm_sdk import Small_LLM_Model  # type: ignore
from src.utils import is_valid_string_token
from src.schemas import (FunctionCallResult, TestPromptFile, TestPromptList,
                         FunctionDefinitionFile, FunctionDefinitionList,
                         FunctionCallResultFile, FunctionDefinition,
                         EngineConfig)


class Engine(BaseModel):
    """Drive tool selection and argument generation via a small LLM.

    Attributes:
        tool_defs: Loaded function definitions.
        test_prompts: Loaded test prompts.
        id_to_token: Vocabulary id to token mapping.
        token_to_id: Token to vocabulary id mapping.
        numeric_ids: Token ids representing numeric characters.
        valid_string_token_ids: Token ids of valid printable string characters.
        functions_token_tree: Trie of function-name token sequences.
        model: Underlying small LLM model.
        current_func: Function selected for the current call, if any.
        outfile: Path to the output file.
        comma_token_id: Encoded token id for a comma.
        brace_token_id: Encoded token id for a closing brace.
        tokens_param_open: Encoded prefix opening parameters block.
        tokens_json_close: Encoded suffix closing the JSON object.
        tokens_comma_space: Encoded token sequence for comma separator.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_defs: FunctionDefinitionList = Field(default_factory=list)
    test_prompts: TestPromptList = Field(default_factory=list)
    id_to_token: dict[int, str] = Field(default_factory=dict)
    token_to_id: dict[str, int] = Field(default_factory=dict)
    numeric_ids: list[int] = Field(default_factory=list)
    valid_string_token_ids: list[int] = Field(default_factory=list)
    functions_token_tree: dict[int | str, Any] = Field(default_factory=dict)
    model: Small_LLM_Model = Field(default_factory=Small_LLM_Model)
    current_func: FunctionDefinition | None = None
    outfile: str = ""
    comma_token_id: int = 0
    brace_token_id: int = 0
    tokens_param_open: list[int] = Field(default_factory=list)
    tokens_json_close: list[int] = Field(default_factory=list)
    tokens_comma_space: list[int] = Field(default_factory=list)

    def initialize(self, args: EngineConfig) -> None:
        """Load external files, output path, and tokenizer vocab.

        Args:
            args: Engine configuration containing the input/output paths.

        Raises:
            Exception: If loading files or vocabulary fails.
        """
        try:
            self.load_files(args)
            self.load_vocab(self.model)
        except Exception as e:
            raise e

    def load_files(self, args: EngineConfig) -> None:
        """Load tool definitions and test prompts, then set output file path.

        Args:
            args: Engine configuration containing the input/output paths.

        Raises:
            Exception: If any file cannot be read, parsed, or validated.
        """
        print("loading files...")
        try:
            with open(args.funcs_def_path, "r", encoding="utf-8") as file:
                defs = json.load(file)
            self.tool_defs = FunctionDefinitionFile.validate_python(defs)
        except FileNotFoundError:
            raise RuntimeError(
                f"Function definitions file not found: '{args.funcs_def_path}'"
            )
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Invalid JSON in function definitions file"+
                f"('{args.funcs_def_path}', line {e.lineno}): {e.msg}"
            )
        except ValidationError as e:
            raise RuntimeError(
                "Schema error in function definitions"+
                f"('{args.funcs_def_path}'): {format_validation_error(e)}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load function definitions file '{args.funcs_def_path}': {e}"
            )

        # 2. Input test prompts file
        try:
            with open(args.in_path, "r", encoding="utf-8") as file:
                prompts = json.load(file)
            self.test_prompts = TestPromptFile.validate_python(prompts)
        except FileNotFoundError:
            raise RuntimeError(
                f"Test prompts file not found: '{args.in_path}'"
            )
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Invalid JSON in test prompts file"+
                f"('{args.in_path}', line {e.lineno}): {e.msg}"
            )
        except ValidationError as e:
            raise RuntimeError(
                "Schema error in test prompts"+
                f"('{args.in_path}'): {format_validation_error(e)}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load test prompts file '{args.in_path}': {e}"
            )

        # Store normalized path without truncating file prematurely
        self.outfile = path.normpath(args.out_path)

    def load_vocab(self, model: Small_LLM_Model) -> None:
        """Load the model vocabulary and build token lookup structures.

        Args:
            model: The LLM model providing the vocab file and tokenizer.
        """
        print("loading vocab...")
        allowed_chars = set("0123456789.-")
        vocab_path = model.get_path_to_vocab_file()
        try:
            with open(vocab_path, "r") as file:
                vocab = json.load(file)
        except FileNotFoundError:
            raise RuntimeError(
                f"Vocabulary file not found at '{vocab_path}'"
            )
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Vocabulary file '{vocab_path}' is corrupt: {e.msg}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load vocabulary file '{vocab_path}': {e}"
            )

        for _id in vocab.values():
            token = model.decode(_id)
            self.id_to_token[_id] = token
            self.token_to_id[token] = _id
            if token in allowed_chars:
                self.numeric_ids.append(_id)
            if is_valid_string_token(token):
                self.valid_string_token_ids.append(_id)

        self.comma_token_id = int(
            self.model.encode(",").flatten().tolist()[-1])
        self.brace_token_id = int(
            self.model.encode("}").flatten().tolist()[-1])
        params_str = ',\n"parameters": {'
        self.tokens_param_open = [
            int(t) for t in self.model.encode(params_str).flatten().tolist()]
        self.tokens_json_close = [
            int(t) for t in self.model.encode("\n}").flatten().tolist()]
        self.tokens_comma_space = [
            int(t) for t in self.model.encode(", ").flatten().tolist()]

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

        Returns:
            list[int]: Encoded token ids of the outer prompt.
        """
        print("building outer prompt...")
        tools_str = FunctionDefinitionFile.dump_json(self.tool_defs).decode()
        prompt = f"\nTools:\n{tools_str}\n\nOutput:\n"
        prompt_ids: list[int] = [
            int(t) for t in self.model.encode(prompt).flatten().tolist()
        ]
        return prompt_ids

    def build_inner_prompt(
            self, prompt: list[int], test_prompt: str) -> list[int]:
        """Build and encode the per-test inner prompt.

        Args:
            prompt: Encoded outer prompt token ids.
            test_prompt: Raw user prompt to embed in the JSON body.

        Returns:
            list[int]: Encoded token ids of the combined prompt.
        """
        print("building inner prompt...")
        escaped_prompt = json.dumps(test_prompt)
        inner_prompt = f'{{\n"prompt": {escaped_prompt},\n"name": "'
        ids: list[int] = [
            int(t) for t in self.model.encode(inner_prompt).flatten().tolist()
        ]
        return prompt + ids

    def select_function(self, input_ids: list[int]) -> list[int]:
        """Select a tool by traversing the token tree with masked logits.

        Args:
            input_ids: Token sequence generated so far.

        Returns:
            list[int]: Updated input IDs with selected function name.
        """
        print("selecting function...")
        tree = self.functions_token_tree
        while True:
            if 'func_def' in tree:
                self.current_func = tree['func_def']
                input_ids.extend(self.tokens_param_open)
                return input_ids

            valid_tokens = [k for k in tree.keys() if isinstance(k, int)]
            logits = self.model.get_logits_from_input_ids(input_ids)

            # Set all invalid tokens to -inf
            logits_arr = np.array(logits, dtype=np.float32)
            masked_logits = np.full_like(logits_arr, -np.inf)
            masked_logits[valid_tokens] = logits_arr[valid_tokens]

            chosen = int(np.argmax(masked_logits))
            tree = tree[chosen]
            input_ids.append(chosen)

    def _decode_boolean_param(self, input_ids: list[int]) -> list[int]:
        """Decode a boolean value.

        Args:
            input_ids: Token ids generated so far.

        Returns:
            list[int]: Extended input_ids with boolean literal.
        """
        true_ids: list[int] = [
            int(t) for t in self.model.encode("true").flatten().tolist()
        ]
        false_ids: list[int] = [
            int(t) for t in self.model.encode("false").flatten().tolist()
        ]

        valid_tokens = [true_ids[0], false_ids[0]]
        logits = self.model.get_logits_from_input_ids(input_ids)

        for _id in range(len(logits)):
            if _id not in valid_tokens:
                logits[_id] = float("-inf")

        chosen = max(range(len(logits)), key=logits.__getitem__)
        input_ids.append(chosen)

        if chosen == true_ids[0] and len(true_ids) > 1:
            input_ids.extend(true_ids[1:])
        elif chosen == false_ids[0] and len(false_ids) > 1:
            input_ids.extend(false_ids[1:])

        return input_ids

    def _decode_number_param(
            self,
            input_ids: list[int],
            sep_token_id: int,
            max_tokens: int = 16) -> list[int]:
        """Sample a numeric value by masking logits until a separator.

        Args:
            input_ids: Token sequence generated so far.
            sep_token_id: Token ID indicating end of numeric input.

        Returns:
            list[int]: Updated input IDs with numeric tokens.
        """
        subtoken = 0
        current_num_str = ""

        while subtoken < max_tokens:
            logits = self.model.get_logits_from_input_ids(input_ids)
            logits_arr = np.array(logits, dtype=np.float32)

            has_dot = "." in current_num_str
            has_digit = any(c.isdigit() for c in current_num_str)

            step_valid_tokens = []
            for _id in self.numeric_ids:
                t_str = self.id_to_token.get(_id, "")
                stripped = t_str.strip()

                # '-' is only valid at the start (subtoken == 0)
                if "-" in stripped:
                    if subtoken > 0 or not stripped.startswith("-"):
                        continue

                # '.' can only appear once
                if "." in stripped and has_dot:
                    continue

                step_valid_tokens.append(_id)

            # Allow terminator once we have at least one digit
            if has_digit:
                step_valid_tokens.append(sep_token_id)

            if not step_valid_tokens:
                break

            masked_logits = np.full_like(logits_arr, -np.inf)
            masked_logits[step_valid_tokens] = logits_arr[step_valid_tokens]

            chosen = int(np.argmax(masked_logits))
            if chosen == sep_token_id:
                return input_ids

            input_ids.append(chosen)
            current_num_str += self.id_to_token.get(chosen, "")
            subtoken += 1

        return input_ids

    def _decode_string_param(self,
                             input_ids: list[int],
                             max_tokens: int = 64) -> list[int]:
        """Sample string content by masking logits to printable tokens.

        Args:
            input_ids: Token sequence generated so far.

        Returns:
            list[int]: Updated input IDs with the string content.
        """
        subtoken = 0
        quote_id = self.token_to_id['"']

        while subtoken < max_tokens:
            logits = self.model.get_logits_from_input_ids(input_ids)
            logits_arr = np.array(logits, dtype=np.float32)

            valid_tokens = list(self.valid_string_token_ids)
            if subtoken > 0:
                valid_tokens.append(quote_id)

            # Set all non-valid string tokens to -inf
            masked_logits = np.full_like(logits_arr, -np.inf)
            masked_logits[valid_tokens] = logits_arr[valid_tokens]

            chosen = int(np.argmax(masked_logits))
            token_str = self.id_to_token.get(chosen, "")

            if '"' in token_str:
                quote_idx = token_str.index('"')
                prefix = token_str[:quote_idx]
                if prefix:
                    prefix_ids = [
                        int(t) for t in self.model.
                        encode(prefix).flatten().tolist()
                    ]
                    input_ids.extend(prefix_ids)
                break

            input_ids.append(chosen)
            subtoken += 1

        input_ids.append(quote_id)
        return input_ids

    def get_function_params(self, input_ids: list[int]) -> list[int]:
        """Dispatch parameter generation by type.

        Args:
            input_ids: Token ids generated so far.

        Returns:
            list[int]: Extended input_ids with parameters block.
        """
        print("getting function params...")
        if self.current_func is None:
            return input_ids

        params_list = list(self.current_func.parameters.items())

        # If the function takes no arguments, close parameters '{}' and finish
        if not params_list:
            input_ids.append(self.brace_token_id)
            input_ids.extend(self.tokens_json_close)
            return input_ids
        for idx, (param_name, param_info) in enumerate(params_list):
            is_last = (idx == len(params_list) - 1)
            sep_id = self.brace_token_id if is_last else self.comma_token_id

            if param_info.type == "string":
                prefix = f'"{param_name}": "'
            else:
                prefix = f'"{param_name}":'
            prefix_ids = [
                int(t) for t in self.model.encode(prefix).flatten().tolist()
            ]
            input_ids.extend(prefix_ids)

            if param_info.type == "boolean":
                input_ids = self._decode_boolean_param(input_ids)
            elif param_info.type == "number":
                input_ids = self._decode_number_param(input_ids, sep_id)
            elif param_info.type == "string":
                input_ids = self._decode_string_param(input_ids)

            if is_last:
                input_ids.append(self.brace_token_id)
            else:
                input_ids.extend(self.tokens_comma_space)

        input_ids.extend(self.tokens_json_close)
        return input_ids

    def save_output(self, output: list[FunctionCallResult]) -> None:
        """Serialize function call results and write them to disk.

        Args:
            output: Function call results to persist.

        Raises:
            Exception: If saving fails.
        """
        dirname = path.dirname(self.outfile)
        if dirname:
            try:
                makedirs(dirname, exist_ok=True)
            except OSError as e:
                raise RuntimeError(
                    f"Cannot create directory '{dirname}': {e.strerror}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create directory '{dirname}': {e}"
                )
        try:
            json_bytes = FunctionCallResultFile.dump_json(output, indent=2)
            with open(self.outfile, "wb") as file:
                file.write(json_bytes)
        except OSError as e:
            raise RuntimeError(
                f"Failed to save output file '{self.outfile}': {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error while saving output file '{self.outfile}': {e}"
            )
