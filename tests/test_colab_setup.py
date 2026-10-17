import ast
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from scripts import colab_setup
from tools import build_colab_notebook
from tools.build_colab_notebook import ROOT, SETUP


def test_existing_pip_does_not_bootstrap(monkeypatch, tmp_path):
    monkeypatch.setattr(colab_setup, "pip_available", lambda _: True)
    run = Mock()
    monkeypatch.setattr(colab_setup.subprocess, "run", run)
    colab_setup.ensure_pip(tmp_path / "bin/python", tmp_path)
    run.assert_not_called()


def test_missing_pip_is_seeded_with_virtualenv_and_system_packages(monkeypatch, tmp_path):
    monkeypatch.setattr(colab_setup, "pip_available", Mock(side_effect=[False, True]))
    monkeypatch.setattr(colab_setup.importlib.util, "find_spec", lambda _: None)
    run = Mock()
    monkeypatch.setattr(colab_setup.subprocess, "run", run)
    colab_setup.ensure_pip(tmp_path / "bin/python", tmp_path)
    assert run.call_args_list[0].args[0][-4:] == ["pip", "install", "-q", "virtualenv"]
    assert run.call_args_list[1].args[0][-3:] == [
        "virtualenv",
        "--system-site-packages",
        str(tmp_path),
    ]


@pytest.mark.parametrize("virtualenv_fails", [False, True])
def test_get_pip_repairs_environment_when_virtualenv_does_not(
    monkeypatch, tmp_path, virtualenv_fails
):
    python = tmp_path / "bin/python"
    monkeypatch.setattr(
        colab_setup, "pip_available", Mock(side_effect=[False, False, True])
    )
    requests, bootstrapped = [], []

    def run(command, **kwargs):
        if command[0] == str(python):
            bootstrapped.append(Path(command[1]).read_bytes())
        elif virtualenv_fails:
            raise subprocess.CalledProcessError(1, command)

    def download(url, **kwargs):
        requests.append(url)
        return io.BytesIO(b"# official get-pip fixture")

    monkeypatch.setattr(colab_setup.subprocess, "run", run)
    monkeypatch.setattr(colab_setup, "urlopen", download)
    colab_setup.ensure_pip(python, tmp_path)
    assert requests == ["https://bootstrap.pypa.io/get-pip.py"]
    assert bootstrapped == [b"# official get-pip fixture"]


def test_prepare_environment_handles_existing_venv_without_pip(monkeypatch, tmp_path):
    python = tmp_path / (
        "Scripts/python.exe" if colab_setup.os.name == "nt" else "bin/python"
    )
    python.parent.mkdir()
    python.touch()
    ensure = Mock()
    builder = Mock()
    monkeypatch.setattr(colab_setup, "ensure_pip", ensure)
    monkeypatch.setattr(colab_setup.venv, "EnvBuilder", builder)
    assert colab_setup.prepare_environment(tmp_path) == python
    builder.assert_not_called()
    ensure.assert_called_once_with(python, tmp_path)


def test_prepare_environment_does_not_require_ensurepip(monkeypatch, tmp_path):
    builder = Mock()
    ensure = Mock()
    virtualenv = Mock()
    monkeypatch.setattr(colab_setup.venv, "EnvBuilder", builder)
    monkeypatch.setattr(colab_setup, "ensure_pip", ensure)
    monkeypatch.setattr(colab_setup, "create_virtualenv", virtualenv)
    colab_setup.prepare_environment(tmp_path)
    virtualenv.assert_called_once_with(tmp_path)
    builder.assert_not_called()
    ensure.assert_called_once()


def test_new_environment_falls_back_without_ensurepip_or_retrying_virtualenv(
    monkeypatch, tmp_path
):
    builder = Mock()
    ensure = Mock()
    virtualenv = Mock(side_effect=subprocess.CalledProcessError(1, "virtualenv"))
    monkeypatch.setattr(colab_setup.venv, "EnvBuilder", builder)
    monkeypatch.setattr(colab_setup, "ensure_pip", ensure)
    monkeypatch.setattr(colab_setup, "create_virtualenv", virtualenv)
    python = colab_setup.prepare_environment(tmp_path)
    virtualenv.assert_called_once_with(tmp_path)
    builder.assert_called_once_with(with_pip=False, system_site_packages=True)
    builder.return_value.create.assert_called_once_with(tmp_path)
    ensure.assert_called_once_with(python, tmp_path, try_virtualenv=False)


def test_installed_virtualenv_does_not_need_another_pip_install(monkeypatch, tmp_path):
    monkeypatch.setattr(colab_setup.importlib.util, "find_spec", lambda _: object())
    run = Mock()
    monkeypatch.setattr(colab_setup.subprocess, "run", run)
    colab_setup.create_virtualenv(tmp_path)
    run.assert_called_once_with(
        [
            colab_setup.sys.executable,
            "-m",
            "virtualenv",
            "--system-site-packages",
            str(tmp_path),
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "layouts,expected",
    [
        (["SpanishBCBL"], "SpanishBCBL"),
        (["SpanishBCBL_Full"], "SpanishBCBL_Full"),
        (["SpanishBCBL", "SpanishBCBL_Full"], "SpanishBCBL"),
    ],
)
def test_notebook_setup_selects_raw_data_and_never_links_on_drive(
    tmp_path, monkeypatch, layouts, expected
):
    content = tmp_path / "content"
    workspace = content / "drive/MyDrive/Brain2Qwerty_Workspace"
    for name in layouts:
        (workspace / "data" / name).mkdir(parents=True)
    links = content / "studies_links"
    created_links = []

    def local_path(value):
        return content / Path(value).relative_to(Path("/content"))

    def symlink(path, target, **kwargs):
        if not path.is_relative_to(links):
            raise OSError(95, "Drive symlinks are unsupported")
        created_links.append((path, target))

    monkeypatch.setattr(Path, "symlink_to", symlink)
    environment = {}
    namespace = {
        "Path": local_path,
        "os": SimpleNamespace(environ=environment, chdir=Mock()),
        "subprocess": SimpleNamespace(run=Mock()),
        "sys": SimpleNamespace(),
        "drive": SimpleNamespace(mount=Mock()),
    }
    tree = ast.parse(SETUP)
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    exec(compile(tree, "colab_setup_cell", "exec"), namespace)
    raw = (workspace / "data" / expected).resolve()
    assert created_links == [(links / "Pinet2024Eeg", raw), (links / "Pinet2024Meg", raw)]
    assert environment["BRAIN2QWERTY_STUDIES"] == str(links)
    assert environment["BRAIN2QWERTY_CACHE"] == "/content/local_cache"
    assert not (workspace / "data/Pinet2024Eeg").exists()


def test_both_notebooks_share_identical_setup_and_local_cache():
    cpu = json.loads(
        (ROOT / "notebooks/colab_cpu_cache.ipynb").read_text(encoding="utf-8")
    )
    gpu = json.loads(
        (ROOT / "notebooks/colab_gpu_train.ipynb").read_text(encoding="utf-8")
    )
    assert cpu["cells"][1]["source"] == gpu["cells"][1]["source"]
    assert "".join(cpu["cells"][1]["source"]).strip() == SETUP.strip()
    for notebook in (cpu, gpu):
        for entry in notebook["cells"]:
            if entry["cell_type"] == "code":
                ast.parse("".join(entry["source"]))
        assert "cache_persisted" not in json.dumps(notebook)
    assert '"pack"' in "".join(cpu["cells"][-1]["source"])
    assert "archives/features.tar" in "".join(cpu["cells"][-1]["source"])
    assert "archives/features.tar" in "".join(gpu["cells"][3]["source"])
    for notebook, mode in ((cpu, "cache"), (gpu, "train")):
        source = "\n".join("".join(entry["source"]) for entry in notebook["cells"])
        assert f'run_live([PYTHON, "-u", "-m", "brain2qwerty_v1", "{mode}"' in source
        assert "rm -rf" not in source and "sed -i" not in source


@pytest.mark.parametrize("returncode", [0, 7])
def test_live_commands_stream_logs_and_propagate_failure(monkeypatch, capsys, returncode):
    process = MagicMock()
    process.__enter__.return_value = process
    process.stdout = iter(["Epoch 1\n", "loss=1.0\n"])
    process.wait.return_value = process.returncode = returncode
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=process))
    helper = next(
        node
        for node in ast.parse(SETUP).body
        if isinstance(node, ast.FunctionDef) and node.name == "run_live"
    )
    namespace = {"subprocess": subprocess}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "run_live", "exec"), namespace
    )
    if returncode:
        with pytest.raises(subprocess.CalledProcessError):
            namespace["run_live"](["training-command"])
    else:
        namespace["run_live"](["training-command"])
    assert capsys.readouterr().out == "Epoch 1\nloss=1.0\n"


def test_real_child_traceback_is_visible_in_notebook_output(capsys):
    helper = next(
        node
        for node in ast.parse(SETUP).body
        if isinstance(node, ast.FunctionDef) and node.name == "run_live"
    )
    namespace = {"subprocess": subprocess}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "run_live", "exec"), namespace
    )
    with pytest.raises(subprocess.CalledProcessError):
        namespace["run_live"](
            [
                sys.executable,
                "-u",
                "-c",
                "import sys; print('cache progress', flush=True); print('stderr detail', file=sys.stderr); raise RuntimeError('child failed')",
            ]
        )
    output = capsys.readouterr().out
    assert "cache progress" in output and "stderr detail" in output
    assert "Traceback (most recent call last)" in output
    assert "RuntimeError: child failed" in output


def test_regeneration_preserves_colab_hardware_settings_and_is_repeatable(
    monkeypatch, tmp_path
):
    destination = tmp_path / "notebooks"
    destination.mkdir()
    metadata = {
        "colab_cpu_cache.ipynb": {"colab": {"machine_shape": "hm"}},
        "colab_gpu_train.ipynb": {"accelerator": "GPU", "colab": {"gpuType": "T4"}},
    }
    for name, settings in metadata.items():
        (destination / name).write_text(
            json.dumps({"metadata": settings}), encoding="utf-8"
        )
    monkeypatch.setattr(build_colab_notebook, "ROOT", tmp_path)
    build_colab_notebook.build()
    generated = {name: (destination / name).read_bytes() for name in metadata}
    for name, settings in metadata.items():
        result = json.loads(generated[name])
        assert all(result["metadata"][key] == value for key, value in settings.items())
    build_colab_notebook.build()
    assert all(
        (destination / name).read_bytes() == content
        for name, content in generated.items()
    )
