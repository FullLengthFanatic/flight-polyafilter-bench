"""Stage 91: write the internal-priming definition (pafbench.definition) to docs/definition.json.

tecap and scNoiseMeter copy this file verbatim. Run as a module from the repository root, so that the version comes from
this checkout's pyproject.toml.
"""
import json

from pafbench.config import REPO
from pafbench.definition import definition


def main():
    out = REPO / "docs" / "definition.json"
    out.write_text(json.dumps(definition(), indent=2) + "\n")
    print(json.dumps({"wrote": str(out)}))


if __name__ == "__main__":
    main()
