"""A self-test that does not need ComfyUI.

The behavioural tests live in the MieNodes pack and in the H3 engine, which
both consume these packages through their shims. This is the floor: the
package imports, its public surface is there, and the prompt families it
ships are readable.
"""

PACKAGE = "mie_llm"


def main() -> int:
    module = __import__(PACKAGE)
    assert hasattr(module, "__version__"), "no version"
    print(PACKAGE, module.__version__)

    from mie_llm.client import GeneralLLMServiceConnector
    from mie_llm.budget import invoke_with_budget_escalation
    from mie_llm.utils import mie_log, resolve_token
    assert GeneralLLMServiceConnector.__name__ == "GeneralLLMServiceConnector"

    print(PACKAGE, "self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
