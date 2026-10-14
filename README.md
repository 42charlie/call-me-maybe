*This project has been created as part of the 42 curriculum by ahsadik.*

# Call Me Maybe - Constrained Decoding Function Calling Engine

## Description
Language models generate human-readable text by sampling tokens sequentially from predicted probability distributions. However, small-scale language models (such as the 0.6B parameter Qwen model) struggle to reliably output structured, syntactically valid JSON conforming to strict schemas without external guidance.

The goal of this project is to build an autonomous function-calling engine that translates natural language prompts into machine-executable tool invocations. Instead of relying on open-ended prompt engineering or speculative prompting, the engine uses **constrained decoding** (logit masking at generation time). By intervening on the next-token logits at every generation step, the engine enforces strict JSON syntax, tool name validity, parameter names, and argument typing (string, number, boolean) with 100% structural predictability.

---

## Instructions

### Requirements & Setup
The project runs on Python 3.10+ and relies on `uv` for dependency management.

1. **Install dependencies and sync environment:**
```bash
make install
# or directly: uv sync
```

2. **Execute the default pipeline:**
```bash
make run
# or: uv run python -m src
```

3. **Run with custom file paths:**
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

4. **Code Quality and Linters:**
```bash
make lint
make lint-strict
```

5. **Debugging & Cleanup:**
```bash
make debug
make clean
```

---

## Algorithm Explanation

The generation pipeline enforces structure by controlling the vocabulary candidates available to the model at each decoding stage:

1. **Prompt Scaffolding:**
   The engine builds an outer prompt encoding the available tools definition list. For each input prompt, it prepends the outer prompt and the opening JSON structure:
```json
{
"prompt": "<escaped_user_prompt>",
"name": "
```

2. **Trie-Based Function Selection:**
   All function names from the tool definitions are converted into token sequences and indexed in a prefix tree (trie). During function selection, next-token logits are extracted from the LLM via `get_logits_from_input_ids`. Any token ID not matching an active branch in the trie at the current depth is masked out to negative infinity (`-inf`). The token with the highest logit (`np.argmax`) is greedily selected. Once a leaf node is reached, the exact function definition is locked.

3. **Structured Parameter Scaffolding:**
   The parameter keys and structural delimiters (`,\n"parameters": {`, `"param_name":`, etc.) are appended deterministically according to the tool's schema, eliminating any chance of missing or misspelled keys.

4. **Type-Specific Constrained Value Decoding:**
   - **Boolean:** Logits are masked to only permit the opening tokens for `true` and `false`. The chosen literal is appended directly.
   - **Numeric:** Candidates are restricted to valid numeric characters (`0-9`, `-`, `.`) and tokens with leading spaces. A state tracker enforces numerical syntax: `-` is only allowed as the first token, at most one `.` is allowed once digits exist, and the separator token (`,` or `}`) is only unmasked once at least one digit is present.
   - **String:** Logits are restricted to printable character tokens. Decoding continues until a closing double-quote token or delimiter is encountered.
   - **Zero-Argument Functions:** If a function specifies no parameters, the parameters block is closed immediately as `{}`.

5. **Payload Closure:**
   Deterministic closing braces (`}`) and newlines are appended to close the object, guaranteeing 100% valid JSON matching the schema.

---

## Design Decisions

- **Vectorized Logit Masking via NumPy:** Iterating through the model's entire vocabulary (~152,000 tokens) in pure Python introduces significant overhead per decoding step. Using NumPy boolean masks and array slicing ensures masking runs in milliseconds, keeping the entire test suite run time well under the 5-minute requirement.
- **Pydantic Validation at Boundaries:** All configuration objects, tool schemas, input prompts, and generated results are validated using Pydantic models. This ensures strict type enforcement and schema compliance while isolating business logic.
- **BPE Token Space Alignment:** Byte-Pair Encoding models attach leading spaces to value tokens (e.g., `' -'`, `' 12'`). Stripping whitespace before parameter values in the deterministic prefix (`"param":`) ensures that the model can select tokens with natural leading whitespace without causing low-probability double spaces.
- **Fault-Tolerant Batch Execution:** The CLI execution loop isolates individual prompt generation steps in guarded blocks. A malformed prompt or single parsing anomaly logs an informative warning to `stderr` rather than terminating the entire batch job.

---

## Performance Analysis

- **JSON Validity:** 100% parseable JSON output across all tests. Because delimiters and keys are generated deterministically and values are strictly masked, syntax errors and missing keys are structurally impossible.
- **Accuracy:** Function selection accuracy exceeds 90% on benchmark prompts by narrowing the generation space strictly to valid function identifiers.
- **Runtime:** Processing standard test batches takes under 40 seconds on standard hardware, comfortably below the 5-minute limit specified by the subject.

---

## Challenges Faced

- **Negative Number Suppression:** Initial implementations masked out compound numeric tokens with leading spaces (e.g., token ID for `' -'`), causing the model to skip negative signs and output positive counterparts (e.g., `45` instead of `-45`). Resolving token-space boundaries and admitting whitespace-prefixed numeric tokens restored accurate negative number extraction.
- **Infinite Decoding Loops on Edge Cases:** On empty or ambiguous input prompts, unconstrained string and numeric decoders could loop indefinitely without selecting a separator. Introducing hard token safety caps and state-aware separator unmasking resolved hanging issues across all edge cases.
- **Zero-Parameter Handling:** Functions without parameters initially failed outer JSON closure due to skipped parameter loops. Adding an explicit branch for empty schemas ensured consistent JSON syntax.

---

## Testing Strategy

The engine was evaluated against multiple test configurations:
- **Baseline Functionality:** Standard arithmetic and text transformation prompts testing string, number, and boolean arguments.
- **Edge-Case Prompts:** Empty prompts (`""`), numbers with negative values and decimals, strings containing single quotes and symbols, and prompts with no matching tool definitions.
- **Schema Variations:** Multi-parameter tools with mixed types, tools with zero parameters, and tools with overlapping name prefixes.
- **Error Handling & CLI Flags:** Tested with missing input files, invalid JSON syntax, and whitespace-only arguments to verify that clear, single-line error messages are printed to `stderr` and the program exits cleanly with code `1`.
- **Static Analysis:** Verified using `flake8` for PEP 8 compliance and `mypy` with `--warn-return-any`, `--warn-unused-ignores`, `--ignore-missing-imports`, `--disallow-untyped-defs`, and `--check-untyped-defs`.

---

## Example Usage

Run the engine on the standard input files:
```bash
uv run python -m src
```

Sample output generated in `data/output/function_calling_results.json`:
```json
[
  {
    "prompt": "Calculate the sum of -45 and 12.5",
    "name": "fn_add_numbers",
    "parameters": {
      "a": -45,
      "b": 12.5
    }
  },
  {
    "prompt": "Turn the living room light on",
    "name": "fn_set_light_state",
    "parameters": {
      "is_on": true
    }
  },
  {
    "prompt": "Reverse the string 'hello world'",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "hello world"
    }
  }
]
```

---

## Resources & AI Usage

### References
- Qwen Technical Report and Tokenization documentation.
- JSON Schema Specification and Pydantic V2 Documentation.
- Fast and Constrained Decoding in Language Models literature (Outlines, Guidance, and Trie-based sampling).

### AI Usage Disclosure
In accordance with the 42 AI guidelines:
- **Prompt Scaffolding & Brainstorming:** AI was consulted to explore edge-case failure modes in BPE tokenizers, specifically regarding leading whitespace handling (`Ġ` token spaces) and token trie construction.
- **Refactoring & Lint Compliance:** AI was used to format repetitive Pydantic error mappings into clean human-readable strings and verify line-length constraints for `flake8`.
- **Peer Review & Verification:** All logic, trie traversal algorithms, logit masking routines, and state constraints were verified, debugged, and tested directly against the subject requirements.