"""Tests for tool policy: command safety, path jail, permission modes."""

from pathlib import Path

import pytest

from src.policy import (
    ToolAction,
    ToolPermissionMode,
    command_has_shell_metacharacters,
    decide_tool_permission,
    is_safe_command,
    normalize_permission_mode,
    path_is_inside_workspace,
    resolve_under_root,
)


class TestCommandSafety:
    def test_plain_readonly_is_safe(self):
        assert is_safe_command("ls")
        assert is_safe_command("ls -la")
        assert is_safe_command("git status")
        assert is_safe_command("git log --oneline")
        assert is_safe_command("dir")
        assert is_safe_command("Get-ChildItem")
        assert is_safe_command("python --version")
        assert is_safe_command("pip list")

    def test_chained_not_safe(self):
        assert not is_safe_command("ls; rm -rf /")
        assert not is_safe_command("git status && del secrets")
        assert not is_safe_command("git status; curl evil.com")
        assert not is_safe_command("echo hi | dangerous")
        assert not is_safe_command("ls || rm -rf /")

    def test_redirect_and_subshell_not_safe(self):
        assert not is_safe_command("cat file > out.txt")
        assert not is_safe_command("echo x >> log")
        assert not is_safe_command("echo $(rm -rf /)")
        assert not is_safe_command("echo `whoami`")

    def test_multiline_not_safe(self):
        assert not is_safe_command("ls\nrm -rf /")
        assert command_has_shell_metacharacters("git status\nmalicious")

    def test_mutating_not_safe(self):
        assert not is_safe_command("rm -rf /tmp/x")
        assert not is_safe_command("pip install evil")
        assert not is_safe_command("del important.txt")

    def test_empty_not_safe(self):
        assert not is_safe_command("")
        assert not is_safe_command("   ")

    def test_find_mutating_flags_not_safe(self):
        assert not is_safe_command("find . -delete")
        assert not is_safe_command("find . -exec rm -rf {} +")
        assert not is_safe_command("find /tmp -execdir rm {} ;")
        assert not is_safe_command("find . -ok rm {} ;")
        assert not is_safe_command("find . -fprint /tmp/out")
        # Read-only find still allowed
        assert is_safe_command("find . -name '*.py'")
        assert is_safe_command("find . -type f")

    def test_set_assignment_not_safe(self):
        assert not is_safe_command("set FOO=bar")
        assert not is_safe_command("set PATH=C:\\evil")
        assert not is_safe_command("set foo=bar")
        # Bare set (list env on cmd) is safe
        assert is_safe_command("set")

    def test_env_assignment_not_safe(self):
        assert not is_safe_command("env FOO=bar ls")
        assert is_safe_command("env")

    def test_auto_mode_find_delete_asks(self):
        action = decide_tool_permission(
            "run_command",
            {"command": "find . -delete"},
            ToolPermissionMode.AUTO,
        )
        assert action == ToolAction.ASK

    def test_auto_mode_set_assignment_asks(self):
        action = decide_tool_permission(
            "run_command",
            {"command": "set FOO=bar"},
            ToolPermissionMode.AUTO,
        )
        assert action == ToolAction.ASK


class TestPathJail:
    def test_inside_root(self, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("hi")
        resolved, err = resolve_under_root("a.txt", tmp_path)
        assert err is None
        assert resolved == f.resolve()
        assert path_is_inside_workspace("a.txt", tmp_path)

    def test_outside_rejected(self, tmp_path: Path):
        outside = tmp_path.parent / "outside_file_xyz.txt"
        resolved, err = resolve_under_root(str(outside), tmp_path, allow_outside=False)
        assert resolved is None
        assert err is not None
        assert "outside workspace" in err.lower()

    def test_outside_allowed_with_flag(self, tmp_path: Path):
        outside = tmp_path.parent / "outside_ok.txt"
        resolved, err = resolve_under_root(str(outside), tmp_path, allow_outside=True)
        assert err is None
        assert resolved is not None

    def test_traversal_blocked(self, tmp_path: Path):
        # ../ should resolve outside if possible
        resolved, err = resolve_under_root("../escape.txt", tmp_path, allow_outside=False)
        assert err is not None
        assert resolved is None


class TestPermissionMode:
    def test_normalize(self):
        assert normalize_permission_mode("auto") == ToolPermissionMode.AUTO
        assert normalize_permission_mode("ASK") == ToolPermissionMode.ASK
        assert normalize_permission_mode("deny") == ToolPermissionMode.DENY
        assert normalize_permission_mode("bogus") == ToolPermissionMode.ASK

    def test_read_always_allow(self):
        for mode in ToolPermissionMode:
            assert decide_tool_permission("read_file", {"filepath": "x"}, mode) == ToolAction.ALLOW

    def test_write_ask_or_deny(self):
        assert decide_tool_permission("write_file", {"filepath": "x", "content": "y"}, ToolPermissionMode.ASK) == ToolAction.ASK
        assert decide_tool_permission("write_file", {"filepath": "x", "content": "y"}, ToolPermissionMode.AUTO) == ToolAction.ASK
        assert decide_tool_permission("write_file", {"filepath": "x", "content": "y"}, ToolPermissionMode.DENY) == ToolAction.DENY

    def test_run_command_auto_safe(self):
        assert decide_tool_permission(
            "run_command", {"command": "git status"}, ToolPermissionMode.AUTO
        ) == ToolAction.ALLOW
        assert decide_tool_permission(
            "run_command", {"command": "ls; rm -rf /"}, ToolPermissionMode.AUTO
        ) == ToolAction.ASK

    def test_run_command_deny(self):
        assert decide_tool_permission(
            "run_command", {"command": "git status"}, ToolPermissionMode.DENY
        ) == ToolAction.DENY

    def test_run_command_ask_mode(self):
        assert decide_tool_permission(
            "run_command", {"command": "git status"}, ToolPermissionMode.ASK
        ) == ToolAction.ASK


class TestWriteFileJailIntegration:
    def test_write_outside_rejected(self, tmp_path: Path):
        from src.tools import write_file

        outside = tmp_path.parent / "should_not_write.txt"
        if outside.exists():
            outside.unlink()
        result = write_file(
            str(outside),
            "secret",
            workspace_root=tmp_path,
            allow_outside=False,
        )
        assert "outside workspace" in result.lower()
        assert not outside.exists()

    def test_write_inside_ok(self, tmp_path: Path):
        from src.tools import write_file

        result = write_file(
            "nested/out.txt",
            "hello",
            workspace_root=tmp_path,
            allow_outside=False,
        )
        assert "Successfully wrote" in result
        assert (tmp_path / "nested" / "out.txt").read_text() == "hello"
