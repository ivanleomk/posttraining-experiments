"""Contract tests for static extraction. Run: python -m unittest discover -s code-sft-data/tests -v"""

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_repo_warmup import extract_repo_function
from generate_splits import choose, extract
from generate_tests import build_mutants, extract_check_function
from launch_ablation import paired_bootstrap_ci, pass_at_k
from nextlat import build_nextlat_mask
from train_lora import parse_args as parse_train_args


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

    def test_choose_keeps_train_test_val_families_disjoint_and_preserves_prefix(self):
        rows = [
            {
                "source_id": i,
                "source_row": i,
                "source_seed_sha1": f"seed-{i}",
                "canonical_body_sha256": f"body-{i}",
                "instruction_sha256": f"task-{i}",
            }
            for i in range(10)
        ]
        train_a, test_a, val_a, _ = choose(rows, train_size=4, test_size=2, seed=42, val_size=0)
        train_b, test_b, val_b, _ = choose(rows, train_size=4, test_size=2, seed=42, val_size=3)
        self.assertEqual(val_a, [])
        self.assertEqual([r["source_id"] for r in train_a], [r["source_id"] for r in train_b])
        self.assertEqual([r["source_id"] for r in test_a], [r["source_id"] for r in test_b])
        self.assertEqual(len(val_b), 3)
        families = [
            {r["family_id"] for r in train_b},
            {r["family_id"] for r in test_b},
            {r["family_id"] for r in val_b},
        ]
        self.assertFalse(families[0] & families[1])
        self.assertFalse(families[0] & families[2])
        self.assertFalse(families[1] & families[2])

    def test_extract_check_function_and_mutants(self):
        valid = dedent(
            """
            ```python
            def check(candidate):
                assert candidate(2) == 4
                assert candidate(0) == 0
            ```
            """
        ).strip()
        parsed = extract_check_function(valid)
        self.assertIn("def check(candidate):", parsed)
        for bad, reason in (
            ("```python\ndef wrong(candidate):\n    assert candidate(1) == 1\n    assert candidate(2) == 2\n```", "invalid_check_signature"),
            ("```python\ndef check(candidate):\n    assert candidate(1) == 1\n```", "insufficient_assertions"),
            ("```python\ndef check(candidate):\n    import os\n    assert candidate(1) == 1\n    assert candidate(2) == 2\n```", "complex_or_external_dependencies"),
        ):
            with self.assertRaisesRegex(ValueError, f"^{reason}$"):
                extract_check_function(bad)
        mutants = build_mutants("def double(x):\n    'Double x.'\n", "    return x * 2\n")
        self.assertIn("identity_first_arg", mutants["generic"])
        self.assertTrue(any(k.startswith("ast_") for k in mutants["ast"]))

    def test_train_config_and_paired_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_file = root / "train.jsonl"
            val_file = root / "val.jsonl"
            train_file.write_text('{"split": "train"}\n')
            val_file.write_text('{"split": "val"}\n')
            cfg = parse_train_args(
                [
                    "--mode",
                    "full",
                    "--train",
                    str(train_file),
                    "--val",
                    str(val_file),
                    "--output",
                    str(root / "out"),
                    "--eval-benchmark",
                    "humaneval",
                ]
            )
            self.assertEqual(cfg.mode, "full")
            self.assertEqual(cfg.eval_benchmarks, ["humaneval"])

            base_res = root / "base_results.jsonl"
            arm_res = root / "arm_results.jsonl"
            base_res.write_text(
                '{"task_id": "t1", "passed": false}\n{"task_id": "t2", "passed": true}\n'
            )
            arm_res.write_text(
                '{"task_id": "t1", "passed": true}\n{"task_id": "t2", "passed": true}\n'
            )
            self.assertAlmostEqual(pass_at_k(20, 0, 1), 0.0)
            self.assertAlmostEqual(pass_at_k(20, 20, 10), 1.0)
            ci = paired_bootstrap_ci(base_res, arm_res, seed=42, resamples=500)
            self.assertEqual(ci["tasks"], 2)
            self.assertAlmostEqual(ci["pass@1"]["delta"], 0.5)

    def test_nextlat_mask_contract(self):
        ids = [10, 11, 20, 21, 22, 0]
        labels = [-100, -100, 20, 21, 22, 0]
        mask = build_nextlat_mask(ids, labels, eos_token_id=0)
        self.assertEqual(mask, [False, True, True, True])
        self.assertEqual(build_nextlat_mask([1, 0], [-100, 0], eos_token_id=0), [])

    def test_repo_warmup_extract_and_warmup_args(self):
        fn_node = ast.parse(
            dedent(
                '''
                def clamp_value(x: int, low: int, high: int) -> int:
                    """Clamp x into the inclusive range [low, high]."""
                    if x < low:
                        return low
                    if x > high:
                        return high
                    return x
                '''
            ).strip()
        ).body[0]
        assert isinstance(fn_node, ast.FunctionDef)
        extracted = extract_repo_function(fn_node)
        self.assertIsNotNone(extracted)
        prompt, completion, doc, body_sha = extracted
        self.assertIn("def clamp_value", prompt)
        self.assertIn("return low", completion)
        self.assertEqual(doc, "Clamp x into the inclusive range [low, high].")
        self.assertRegex(body_sha, r"^[0-9a-f]{64}$")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_file = root / "train.jsonl"
            warmup_file = root / "warmup.jsonl"
            train_file.write_text('{"split": "train"}\n')
            warmup_file.write_text('{"split": "train"}\n')
            cfg = parse_train_args(
                [
                    "--mode",
                    "full",
                    "--objective",
                    "nextlat",
                    "--train",
                    str(train_file),
                    "--warmup-train",
                    str(warmup_file),
                    "--warmup-epochs",
                    "1",
                    "--cpt-epochs",
                    "1",
                    "--zero-init-dynamics",
                    "--lambda-mse",
                    "0.2",
                    "--lambda-kl",
                    "0.1",
                    "--output",
                    str(root / "out"),
                ]
            )
            self.assertTrue(cfg.zero_init_dynamics)
            self.assertEqual(cfg.warmup_epochs, 1)
            self.assertEqual(cfg.cpt_epochs, 1)
            self.assertAlmostEqual(cfg.lambda_mse, 0.2)
            self.assertAlmostEqual(cfg.lambda_kl, 0.1)


if __name__ == "__main__":
    unittest.main()
