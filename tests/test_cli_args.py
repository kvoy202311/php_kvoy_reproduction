from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


_MODULE_PATH = Path(__file__).parents[1] / "scripts/rsl_rl/cli_args.py"
_SPEC = importlib.util.spec_from_file_location("php_kvoy_reproduction_cli_args", _MODULE_PATH)
cli_args = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(cli_args)


class RslRlCliArgsTest(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        cli_args.add_rsl_rl_args(self.parser)

    def test_resume_boolean_flags_are_unambiguous(self):
        self.assertIs(self.parser.parse_args(["--resume"]).resume, True)
        self.assertIs(self.parser.parse_args(["--no-resume"]).resume, False)
        self.assertIsNone(self.parser.parse_args([]).resume)

    def test_warm_start_and_resume_are_mutually_exclusive(self):
        parsed = self.parser.parse_args(["--warm_start"])
        self.assertTrue(parsed.warm_start)
        self.assertIsNone(parsed.resume)
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--resume", "--warm_start"])

    def test_warm_start_disables_configured_full_resume(self):
        parsed = self.parser.parse_args(["--warm_start"])
        parsed.seed = None
        agent_cfg = SimpleNamespace(
            seed=42,
            experiment_name="elf3_climb",
            resume=True,
            load_run=".*",
            load_checkpoint="model_.*.pt",
            run_name="",
            logger="tensorboard",
            wandb_project="isaaclab",
            neptune_project="isaaclab",
        )

        updated = cli_args.update_rsl_rl_cfg(agent_cfg, parsed)

        self.assertFalse(updated.resume)

    def test_experiment_name_override_is_applied(self):
        parsed = self.parser.parse_args(["--experiment_name", "elf3_climb_audit"])
        parsed.seed = None
        agent_cfg = SimpleNamespace(
            seed=42,
            experiment_name="elf3_climb",
            resume=False,
            load_run=".*",
            load_checkpoint="model_.*.pt",
            run_name="",
            logger="tensorboard",
            wandb_project="isaaclab",
            neptune_project="isaaclab",
        )

        updated = cli_args.update_rsl_rl_cfg(agent_cfg, parsed)

        self.assertEqual(updated.experiment_name, "elf3_climb_audit")


if __name__ == "__main__":
    unittest.main()
