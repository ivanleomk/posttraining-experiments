"""Contract tests for static extraction. Run: python -m unittest discover -s code-sft-data/tests -v"""

import ast
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_splits import extract


class ExtractTests(unittest.TestCase):
    def row(self, code: str, instruction: str = "Return the input.") -> dict[str, str]:
        return {"instruction": instruction, "response": f"```python\n{code}\n```"}

    def test_builds_prompt_and_answer_from_source_fields(self):
        row = self.row(
            'def identity(value):\n    """Old teacher docstring."""\n    return value'
        )
        row.update(
            seed="SEED_MUST_NOT_APPEAR", prompt="GENERATOR_PROMPT_MUST_NOT_APPEAR"
        )
        row["response"] = (
            "Teacher explanation.\n" + row["response"] + "\nMore explanation."
        )

        prompt, completion, fingerprint = extract(row)

        self.assertEqual(prompt, "def identity(value):\n    'Return the input.'\n")
        self.assertEqual(completion, "    return value\n")
        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")
        self.assertEqual(
            ast.get_docstring(ast.parse(prompt + completion).body[0]),
            row["instruction"],
        )

    def test_accepts_supported_fences(self):
        for language in ("python", "py", ""):
            with self.subTest(language=language):
                row = {
                    "instruction": "Return one.",
                    "response": f"```{language}\ndef f():\n    return 1\n```",
                }
                self.assertEqual(extract(row)[1], "    return 1\n")

    def test_preserves_signature_instruction_and_control_flow(self):
        instruction = 'Return "yes" for positive values.\nOtherwise return None. Use \\ carefully.'
        code = 'def check(value: int, /, *, minimum=0) -> str | None:\n    if value > minimum:\n        return "yes"\n    return None'
        prompt, completion, _ = extract(self.row(code, instruction))
        function = ast.parse(prompt + completion).body[0]

        self.assertEqual(ast.get_docstring(function), instruction)
        self.assertEqual(function.args.posonlyargs[0].arg, "value")
        self.assertEqual(function.args.kwonlyargs[0].arg, "minimum")
        self.assertIsInstance(function.body[1], ast.If)
        original = ast.parse(code).body[0]
        self.assertEqual(
            [ast.dump(node) for node in function.body[1:]],
            [ast.dump(node) for node in original.body],
        )

    def test_body_fingerprint_ignores_formatting_and_docstrings(self):
        first = self.row('def f(x):\n    """First docstring."""\n    return x+1')
        second = self.row(
            "def f(x):\n    # a comment\n    return x + 1", "Different instruction."
        )
        changed = self.row("def f(x):\n    return x - 1")
        self.assertEqual(extract(first)[2], extract(second)[2])
        self.assertNotEqual(extract(first)[2], extract(changed)[2])

    def test_rejects_missing_multiple_and_invalid_code_blocks(self):
        cases = [
            ("def f():\n    return 1", "not_one_code_block"),
            ("```javascript\nfunction f() {}\n```", "not_one_code_block"),
            (
                "```python\ndef f():\n    return 1\n```\n```py\ndef g():\n    return 2\n```",
                "not_one_code_block",
            ),
            ("```python\ndef broken(:\n```", "syntax_error"),
        ]
        for response, reason in cases:
            with self.subTest(reason=reason, response=response):
                with self.assertRaisesRegex(ValueError, f"^{reason}$"):
                    extract({"instruction": "Example", "response": response})

    def test_rejects_unsupported_structure_and_dependencies(self):
        cases = [
            (
                "def f():\n    return 1\n\ndef g():\n    return 2",
                "not_one_standalone_function",
            ),
            ("import math\ndef f():\n    return 1", "not_one_standalone_function"),
            ("class Example:\n    pass", "not_one_standalone_function"),
            ("async def f():\n    return 1", "not_one_standalone_function"),
            ("@decorator\ndef f():\n    return 1", "decorated_function"),
            (
                "def f():\n    import math\n    return math.pi",
                "complex_or_external_dependencies",
            ),
            (
                "def f():\n    from math import pi\n    return pi",
                "complex_or_external_dependencies",
            ),
            (
                "def f():\n    global x\n    return x",
                "complex_or_external_dependencies",
            ),
            (
                "def f():\n    class Nested:\n        pass\n    return Nested",
                "complex_or_external_dependencies",
            ),
            ('def f():\n    """Only a docstring."""', "empty_function"),
        ]
        for code, reason in cases:
            with self.subTest(code=code):
                with self.assertRaisesRegex(ValueError, f"^{reason}$"):
                    extract(self.row(code))

    def test_rejects_selected_io_and_dynamic_calls(self):
        for name in (
            "open",
            "exec",
            "eval",
            "compile",
            "__import__",
            "input",
            "globals",
            "locals",
        ):
            with self.subTest(call=name):
                with self.assertRaisesRegex(ValueError, "^io_or_dynamic_execution$"):
                    extract(self.row(f"def f():\n    return {name}()"))

    def test_never_executes_defaults_or_function_body(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must-not-exist"
            code = f"def f(value=1 / 0):\n    marker = {str(marker)!r}\n    raise RuntimeError(marker)"
            prompt, completion, _ = extract(self.row(code))
            self.assertIn("1 / 0", prompt)
            self.assertIn("raise RuntimeError", completion)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
