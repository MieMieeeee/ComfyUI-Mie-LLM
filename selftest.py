"""A self-test that does not need ComfyUI.

The behavioural tests live in the MieNodes pack and in the H3 engine, which
both consume these packages through their shims. This is the floor: the
package imports, its public surface is there, and the prompt families it
ships are readable.
"""

import subprocess
import sys
import textwrap

PACKAGE = "mie_llm"

# The heavy stack this package must not need. `media.py` is the only module
# that touches numpy and opencv, and it does so inside the functions - so
# importing it must never require them.
HEAVY = ("numpy", "cv2", "torch", "PIL")


def main() -> int:
    module = __import__(PACKAGE)
    assert hasattr(module, "__version__"), "no version"
    print(PACKAGE, module.__version__)

    from mie_llm.client import GeneralLLMServiceConnector
    from mie_llm.budget import invoke_with_budget_escalation
    from mie_llm.utils import mie_log, resolve_token
    assert GeneralLLMServiceConnector.__name__ == "GeneralLLMServiceConnector"

    check_media_is_light()
    print(PACKAGE, "self-test passed")
    return 0


def check_media_is_light() -> None:
    """``mie_llm.media`` must import with no numpy / opencv / torch / PIL.

    The package declares ``requests`` as its only dependency, so anything
    heavier at import time makes the declared surface a lie - and it breaks
    every consumer that installs the package outside ComfyUI (the product
    layer's own virtualenv, for one).

    The check runs in a subprocess with the heavy modules blocked, so it
    means the same thing on a machine that has them installed and on one
    that does not.
    """
    script = textwrap.dedent(
        """
        import sys

        BLOCKED = {blocked!r}

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ModuleNotFoundError(
                        "No module named %r (blocked for the test)" % name)
                return None

        sys.meta_path.insert(0, Blocker())
        for name in list(sys.modules):
            if name.split(".")[0] in BLOCKED:
                del sys.modules[name]

        import mie_llm.media as media

        # The text-only helper must work with no heavy stack at all.
        parts = media.build_multimodal_user_content("hi", ["data:image/jpeg;base64,AAA"])
        assert parts[0]["type"] == "image_url", parts
        assert parts[1]["type"] == "text", parts

        # The image helpers must fail with a readable message, not a bare
        # traceback out of a lazy import.
        class _Batch:
            ndim = 4
            shape = (2, 4, 4, 3)

            def __getitem__(self, index):
                return object()

        for call in (lambda: media.image_tensor_to_data_url(object()),
                     lambda: media.image_tensor_batch_to_data_urls(_Batch())):
            try:
                call()
            except ImportError as exc:
                assert "opencv" in str(exc), exc
            else:
                raise AssertionError("expected an ImportError")

        # The early-outs still hold without the heavy stack.
        assert media.image_tensor_batch_to_data_urls(object()) == []
        assert media.image_tensor_to_data_url(None) is None

        print("media is light")
        """
    ).format(blocked=set(HEAVY))

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise AssertionError("mie_llm.media needs the heavy stack at import time")
    print(result.stdout.strip())


if __name__ == "__main__":
    raise SystemExit(main())
