import pathlib

from csc import Script

self_path = pathlib.Path(__file__).resolve().parent


def test_args():
    script = Script(self_path / "script_env.py", args=["--value", "42"])
    script["Parse args"].run()

    assert script.ns.args.value == 42


def test_cwd(tmp_path):
    script = Script(self_path / "script_env.py", cwd=tmp_path)
    script["Write file"].run()

    assert (tmp_path / "foo.txt").exists()
