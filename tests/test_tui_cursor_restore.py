from __future__ import annotations

from io import StringIO

from simple.evals import tui


class _FakeConsole:
    def __init__(self) -> None:
        self.file = StringIO()
        self.cursor_visibility: list[bool] = []

    def show_cursor(self, visible: bool) -> None:
        self.cursor_visibility.append(visible)


def test_restore_cursor_resets_style_and_shows_cursor(monkeypatch) -> None:
    console = _FakeConsole()
    stderr = StringIO()
    stdout = StringIO()
    tty_writes: list[tuple[int, bytes]] = []

    monkeypatch.setattr(tui.sys, "stderr", stderr)
    monkeypatch.setattr(tui.sys, "__stderr__", stderr)
    monkeypatch.setattr(tui.sys, "stdout", stdout)
    monkeypatch.setattr(tui.sys, "__stdout__", stdout)
    monkeypatch.setattr(tui.os, "open", lambda *_args: 17)
    monkeypatch.setattr(tui.os, "write", lambda fd, data: tty_writes.append((fd, data)))
    monkeypatch.setattr(tui.os, "close", lambda _fd: None)

    tui.restore_cursor(console)

    expected = "\x1b[0m\x1b[?25h"
    assert console.cursor_visibility == [True]
    assert console.file.getvalue() == expected
    assert stderr.getvalue() == expected
    assert stdout.getvalue() == expected
    assert tty_writes == [(17, expected.encode())]


def test_register_cursor_restore_uses_atexit(monkeypatch) -> None:
    console = _FakeConsole()
    registrations = []
    monkeypatch.setattr(
        tui.atexit,
        "register",
        lambda callback, *args: registrations.append((callback, args)),
    )

    tui.register_cursor_restore(console)

    assert registrations == [(tui.restore_cursor, (console,))]
