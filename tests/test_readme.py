import argparse
import importlib
import shlex
import sys

import pytest

from pafbench.config import REPO


def readme_block(heading):
    """Lines of the first bash block under a README heading, inline comments dropped."""
    section = (REPO / "README.md").read_text().split(f"\n## {heading}\n", 1)[1]
    block = section.split("```bash\n", 1)[1].split("```", 1)[0]
    return [shlex.split(line, comments=True) for line in block.splitlines() if line.strip()]


class Parsed(Exception):
    pass


@pytest.mark.parametrize("cmd", readme_block("Running the pipeline"), ids=lambda c: " ".join(c[2:]))
def test_readme_pipeline_command_parses(cmd, monkeypatch):
    # parse the README arguments with the script's own parser, then stop before the stage touches any data
    assert cmd[:2] == ["python", "-m"]
    module, args = cmd[2], cmd[3:]
    assert (REPO / f"{module.replace('.', '/')}.py").exists(), module
    mod = importlib.import_module(module)
    if not hasattr(mod, "argparse"):
        assert args == [], f"{module} takes no arguments"
        return
    real = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, *a, **kw):
        real(self, *a, **kw)
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    monkeypatch.setattr(sys, "argv", [module, *args])
    with pytest.raises(Parsed):
        mod.main()
