# SPDX-License-Identifier: Apache-2.0
"""
Tests for the /tag-and-rerun-xpu-ci slash command.

Two things are pinned here. The behavioural tests drive main() and assert on
which labels were applied and which workflow runs were restarted, because the
command's whole purpose is to start the XPU lane without starting a CUDA one.
The contract tests pin the strings the handler and the workflows share, which
are checked by neither side at runtime: a rename on one side leaves the command
silently doing nothing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
HANDLER_WORKFLOW = REPO_ROOT / ".github/workflows/slash-command-handler.yml"
XPU_WORKFLOW = REPO_ROOT / ".github/workflows/omni-xpu-ci.yaml"
HANDLER_SCRIPT = REPO_ROOT / "scripts/ci/utils/slash_command_handler.py"

COMMAND = "/tag-and-rerun-xpu-ci"
PR_NUMBER = 4321
COMMENT_ID = 99
MAINTAINER = "maintainer-with-tag-rights"
OUTSIDER = "contributor-without-tag-rights"
# Two workflows, so a test can tell "restarted the XPU lane" apart from
# "restarted everything".
WORKFLOW_IDS = {"omni-xpu-ci.yaml": 7, "omni-ci.yaml": 8}


def _workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


class _FakeGithubException(Exception):
    pass


def _load_handler():
    """
    Imports the handler with a stub github module.

    The neighbouring TTS test parses this file with ast instead of importing it,
    because PyGithub exists only on the slash-command runner. Stubbing the
    module keeps that property while allowing the handler to actually run.
    """
    if "github" not in sys.modules:
        stub = types.ModuleType("github")
        stub.Auth = types.SimpleNamespace(Token=lambda token: token)
        stub.Github = object
        stub.GithubException = _FakeGithubException
        exceptions = types.ModuleType("github.GithubException")
        exceptions.GithubException = _FakeGithubException
        sys.modules["github"] = stub
        sys.modules["github.GithubException"] = exceptions

    spec = importlib.util.spec_from_file_location(
        "slash_command_handler", HANDLER_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRun:
    def __init__(self, name, workflow_id, created_at, event="pull_request"):
        self.name = name
        self.workflow_id = workflow_id
        self.created_at = created_at
        self.event = event
        self.id = workflow_id * 1000
        self.status = "completed"
        self.conclusion = "failure"
        self.reran = False

    def rerun(self):
        self.reran = True
        return True

    def rerun_failed_jobs(self):
        self.reran = True
        return True


class _FakeComment:
    def __init__(self):
        self.reactions = []

    def create_reaction(self, content):
        self.reactions.append(content)


class _FakePR:
    def __init__(self, author):
        self.head = types.SimpleNamespace(sha="c0ffee")
        self.user = types.SimpleNamespace(login=author)
        self.labels = []

    def add_to_labels(self, label):
        self.labels.append(label)

    def get_labels(self):
        return [types.SimpleNamespace(name=label) for label in self.labels]

    def remove_from_labels(self, label):
        self.labels = [existing for existing in self.labels if existing != label]


class _FakeRepo:
    def __init__(self, pr, comment, existing_labels=()):
        self.pr = pr
        self.comment = comment
        self.existing_labels = set(existing_labels)
        self.created_labels = []
        now = datetime(2026, 9, 5, tzinfo=timezone.utc)
        self.runs = [
            _FakeRun("XPU CI", 7, now),
            _FakeRun("Omni CI", 8, now - timedelta(minutes=1)),
        ]

    def run_named(self, name):
        return next(run for run in self.runs if run.name == name)

    def get_pull(self, number):
        assert number == PR_NUMBER
        return self.pr

    def get_issue(self, number):
        assert number == PR_NUMBER
        return types.SimpleNamespace(get_comment=lambda _id: self.comment)

    def get_workflow(self, file_name):
        if file_name not in WORKFLOW_IDS:
            raise _FakeGithubException(f"no such workflow: {file_name}")
        return types.SimpleNamespace(id=WORKFLOW_IDS[file_name])

    def get_workflow_runs(self, head_sha=None):
        assert head_sha == self.pr.head.sha
        return list(self.runs)

    def get_label(self, name):
        if name not in self.existing_labels:
            raise _FakeGithubException(f"no such label: {name}")
        return name

    def create_label(self, name, color):
        self.created_labels.append(name)
        self.existing_labels.add(name)


@pytest.fixture
def run_command(monkeypatch, tmp_path):
    """
    Returns a callable that runs main() for one comment body and hands back the
    fake PR / comment / repo so a test can assert what the handler did.
    """
    module = _load_handler()
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    permissions = tmp_path / "ci_permissions.json"
    permissions.write_text(
        json.dumps(
            {MAINTAINER: {"can_tag_run_ci_label": True, "can_rerun_failed_ci": True}}
        )
    )
    monkeypatch.setattr(module, "PERMISSIONS_FILE_PATH", str(permissions))

    def run(body, user=MAINTAINER, pr_author="someone-else", existing_labels=()):
        pr = _FakePR(pr_author)
        comment = _FakeComment()
        repo = _FakeRepo(pr, comment, existing_labels)
        monkeypatch.setattr(
            module,
            "Github",
            lambda **_kwargs: types.SimpleNamespace(get_repo=lambda _name: repo),
        )
        for name, value in {
            "GITHUB_TOKEN": "token",
            "REPO_FULL_NAME": "sgl-project/sglang-omni",
            "PR_NUMBER": str(PR_NUMBER),
            "COMMENT_ID": str(COMMENT_ID),
            "COMMENT_BODY": body,
            "USER_LOGIN": user,
        }.items():
            monkeypatch.setenv(name, value)

        module.main()
        return pr, comment, repo

    return run


def test_xpu_command_reruns_the_xpu_lane_and_leaves_the_cuda_lane_alone(run_command):
    pr, comment, repo = run_command(COMMAND)

    assert pr.labels == ["run-xpu-ci"]
    assert repo.run_named("XPU CI").reran is True
    assert repo.run_named("Omni CI").reran is False
    assert comment.reactions == ["+1"]


def test_xpu_command_creates_the_label_when_the_repository_lacks_it(run_command):
    _, _, repo = run_command(COMMAND)

    assert repo.created_labels == ["run-xpu-ci"]


def test_xpu_command_reuses_an_existing_label(run_command):
    _, _, repo = run_command(COMMAND, existing_labels=["run-xpu-ci"])

    assert repo.created_labels == []


def test_xpu_command_needs_tag_rights_not_just_pr_authorship(run_command):
    """
    main() grants an unlisted PR author can_rerun_failed_ci. That must not be
    enough here: the command puts PR code on a self-hosted Intel runner.
    """
    pr, comment, repo = run_command(COMMAND, user=OUTSIDER, pr_author=OUTSIDER)

    assert pr.labels == []
    assert repo.run_named("XPU CI").reran is False
    assert comment.reactions == ["confused"]


@pytest.mark.parametrize(
    "body",
    [
        "/tag-and-rerun-ci xpu",
        "/tag-and-rerun-ci XPU",
        "/tag-and-rerun-ci-xpu",
        "/tag-run-ci-label xpu",
    ],
)
def test_xpu_is_refused_as_a_selector_of_the_all_lane_commands(run_command, body):
    """
    Every spelling here parses as an all-lane command, which would tag run-ci
    and occupy the CUDA runners -- the opposite of the request. Refusing costs
    the author a re-comment; accepting costs an H100 suite.
    """
    pr, comment, repo = run_command(body)

    assert pr.labels == []
    assert all(not run.reran for run in repo.runs)
    assert comment.reactions == ["confused"]


def test_base_command_still_selects_a_tts_model(run_command):
    """
    parse_model_targets is shared, and the xpu refusal above runs inside it.
    """
    pr, comment, repo = run_command("/tag-and-rerun-ci moss")

    assert sorted(pr.labels) == ["run-ci", "run-moss"]
    assert repo.run_named("Omni CI").reran is True
    assert comment.reactions == ["+1"]


def test_handler_resolves_a_workflow_file_that_exists(run_command):
    module = _load_handler()

    assert (
        REPO_ROOT / ".github/workflows" / module.XPU_CI_WORKFLOW_FILE
    ).is_file(), "the handler resolves the XPU workflow by filename"


def test_trigger_workflow_reaches_the_handler_for_the_xpu_command():
    job = _workflow(HANDLER_WORKFLOW)["jobs"]["slash_command"]

    assert COMMAND in job["if"]
    assert "slash_command_handler.py" in _step(job, "Handle Slash Command")["run"]


def test_xpu_workflow_accepts_the_label_and_overrides_the_paths_filter():
    jobs = _workflow(XPU_WORKFLOW)["jobs"]

    decide = _step(jobs["check-changes"], "Decide whether to run the XPU lane")
    assert (
        decide["env"]["LABEL_REQUESTED"] == "${{ steps.xpu_label.outputs.requested }}"
    )
    assert 'LABEL_REQUESTED}" == "true"' in decide["run"]
    assert (
        "run-xpu-ci"
        in _step(jobs["check-changes"], "Read run-xpu-ci label")["with"]["script"]
    )


def test_xpu_workflow_gate_accepts_either_label():
    gate = _step(
        _workflow(XPU_WORKFLOW)["jobs"]["preflight"],
        "Require run-ci or run-xpu-ci label",
    )

    assert "'run-xpu-ci'" in gate["env"]["HAS_RUN_XPU_CI"]
    assert 'HAS_RUN_CI}" == "false" && "${HAS_RUN_XPU_CI}" == "false"' in gate["run"]
