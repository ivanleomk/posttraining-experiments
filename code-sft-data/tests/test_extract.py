"""Contract tests for static extraction. Run: python -m unittest discover -s code-sft-data/tests -v"""

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_splits import extract


class ExtractTests(unittest.TestCase):
    def row(self, code: str, instruction: str = "Return the input.") -> dict[str, str]:
        return {
            "instruction": instruction,
            "response": f"""```python
{code}
```""",
        }

    def test_builds_prompt_and_answer_from_source_fields(self):
        cases = [
            (
                "code only",
                dedent('''
                    ```python
                    def identity(value):
                        """Old teacher docstring."""
                        return value
                    ```
                ''').strip(),
            ),
            (
                "code surrounded by explanations",
                dedent('''
                    Teacher explanation.
                    ```python
                    def identity(value):
                        """Old teacher docstring."""
                        return value
                    ```
                    More explanation.
                ''').strip(),
            ),
        ]
        for name, response in cases:
            with self.subTest(case=name):
                row = {
                    "instruction": "Return the input.",
                    "seed": "SEED_MUST_NOT_APPEAR",
                    "prompt": "GENERATOR_PROMPT_MUST_NOT_APPEAR",
                    "response": response,
                }

                prompt, completion, fingerprint = extract(row)

                self.assertEqual(
                    prompt, "def identity(value):\n    'Return the input.'\n"
                )
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
                    "response": dedent(f"""
                        ```{language}
                        def f():
                            return 1
                        ```
                    """).strip(),
                }
                self.assertEqual(extract(row)[1], "    return 1\n")

    def test_preserves_signature_instruction_and_control_flow(self):
        instruction = dedent(
            """
            Return "yes" for positive values.
            Otherwise return None. Use \\ carefully.
            """
        ).strip()
        code = dedent(
            """
            def check(value: int, /, *, minimum=0) -> str | None:
                if value > minimum:
                    return "yes"
                return None
            """
        ).strip()
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
        first = self.row(
            dedent(
                '''
            def f(x):
                """First docstring."""
                return x+1
            '''
            ).strip()
        )
        second = self.row(
            dedent(
                """
                def f(x):
                    # a comment
                    return x + 1
                """
            ).strip(),
            "Different instruction.",
        )
        changed = self.row(
            dedent(
                """
            def f(x):
                return x - 1
            """
            ).strip()
        )
        self.assertEqual(extract(first)[2], extract(second)[2])
        self.assertNotEqual(extract(first)[2], extract(changed)[2])

    def test_rejects_missing_multiple_and_invalid_code_blocks(self):
        cases = [
            (
                dedent(
                    """
                def f():
                    return 1
                """
                ).strip(),
                "not_one_code_block",
            ),
            (
                dedent(
                    """
                ```javascript
                function f() {}
                ```
                """
                ).strip(),
                "not_one_code_block",
            ),
            (
                dedent(
                    """
                    ```python
                    def f():
                        return 1
                    ```
                    ```py
                    def g():
                        return 2
                    ```
                    """
                ).strip(),
                "not_one_code_block",
            ),
            (
                dedent(
                    """
                ```python
                def broken(:
                ```
                """
                ).strip(),
                "syntax_error",
            ),
        ]
        for response, reason in cases:
            with self.subTest(reason=reason, response=response):
                with self.assertRaisesRegex(ValueError, f"^{reason}$"):
                    extract({"instruction": "Example", "response": response})

    def test_rejects_unsupported_structure_and_dependencies(self):
        cases = [
            (
                dedent(
                    """
                    def f():
                        return 1

                    def g():
                        return 2
                    """
                ).strip(),
                "not_one_standalone_function",
            ),
            (
                dedent(
                    """
                import math
                def f():
                    return 1
                """
                ).strip(),
                "not_one_standalone_function",
            ),
            (
                dedent(
                    """
                class Example:
                    pass
                """
                ).strip(),
                "not_one_standalone_function",
            ),
            (
                dedent(
                    """
                async def f():
                    return 1
                """
                ).strip(),
                "not_one_standalone_function",
            ),
            (
                dedent(
                    """
                @decorator
                def f():
                    return 1
                """
                ).strip(),
                "decorated_function",
            ),
            (
                dedent(
                    """
                    def f():
                        import math
                        return math.pi
                    """
                ).strip(),
                "complex_or_external_dependencies",
            ),
            (
                dedent(
                    """
                    def f():
                        from math import pi
                        return pi
                    """
                ).strip(),
                "complex_or_external_dependencies",
            ),
            (
                dedent(
                    """
                    def f():
                        global x
                        return x
                    """
                ).strip(),
                "complex_or_external_dependencies",
            ),
            (
                dedent(
                    """
                    def f():
                        class Nested:
                            pass
                        return Nested
                    """
                ).strip(),
                "complex_or_external_dependencies",
            ),
            (
                dedent(
                    '''
                def f():
                    """Only a docstring."""
                '''
                ).strip(),
                "empty_function",
            ),
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
                    extract(
                        self.row(
                            dedent(f"""
                        def f():
                            return {name}()
                    """).strip()
                        )
                    )

    def test_never_executes_defaults_or_function_body(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must-not-exist"
            code = dedent(f"""
                def f(value=1 / 0):
                    marker = {str(marker)!r}
                    raise RuntimeError(marker)
            """).strip()
            prompt, completion, _ = extract(self.row(code))
            self.assertIn("1 / 0", prompt)
            self.assertIn("raise RuntimeError", completion)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
