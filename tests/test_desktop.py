"""Desktop smoke and queue integration checks, skipped without a display."""
import threading
import time
import logging

import numpy as np
import pytest


def _make_app(monkeypatch, tmp_path):
    tk = pytest.importorskip("tkinter")
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No graphical display available")
    root.withdraw()
    from ssgs import desktop
    app = desktop.SpectroApp(root)
    app.captured_errors = []
    monkeypatch.setattr(desktop.messagebox, "showerror", lambda *args, **kwargs: app.captured_errors.append(args))
    monkeypatch.setattr(desktop.messagebox, "askyesno", lambda *args, **kwargs: True)
    return root, app


def _wait(root, app):
    deadline = time.monotonic() + 10
    while app.busy and time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)
    assert not app.busy, "Desktop worker did not complete"


def test_window_initializes_without_a_model(monkeypatch, tmp_path):
    root, app = _make_app(monkeypatch, tmp_path)
    try:
        root.update_idletasks()
        assert app.generate_button.instate(["disabled"])
        assert app.save_button.instate(["disabled"])
        assert app.model is None
    finally:
        app.close()


def test_worker_completion_returns_to_ui_thread(monkeypatch, tmp_path):
    root, app = _make_app(monkeypatch, tmp_path)
    main_thread = threading.get_ident()
    completed = []
    try:
        def work(progress, cancel):
            assert threading.get_ident() != main_thread
            progress("Half way", 0.5)
            return "finished"
        def complete(result):
            completed.append((result, threading.get_ident()))
        app._start("Test job", work, complete)
        _wait(root, app)
        assert completed == [("finished", main_thread)]
        assert app.status.get() == "Ready"
    finally:
        app.close()


def test_cancellation_restores_controls_without_modal_error(monkeypatch, tmp_path):
    root, app = _make_app(monkeypatch, tmp_path)
    try:
        from ssgs.workflows import WorkflowCancelled
        from ssgs import desktop
        errors = []
        monkeypatch.setattr(desktop.messagebox, "showerror", lambda *args, **kwargs: errors.append(args))
        def work(progress, cancel):
            cancel.wait(2)
            raise WorkflowCancelled("cancelled")
        app._start("Cancellable job", work, lambda _: None)
        app.cancel()
        _wait(root, app)
        assert errors == []
        assert app.status.get() == "Cancelled"
        assert app.cancel_button.instate(["disabled"])
    finally:
        app.close()


def test_short_audio_plot_and_preferences(monkeypatch, tmp_path):
    root, app = _make_app(monkeypatch, tmp_path)
    try:
        app._plot_audio(np.sin(np.arange(128) * 0.1), 16000)
        root.update_idletasks()
        assert app.wave_axis.lines
        assert app.spectrum_axis.collections
        app.states.set("34")
    finally:
        app.close()
    assert '"states": "34"' in (tmp_path / "config" / "spectro" / "preferences.json").read_text()


def test_full_threaded_demo_prepares_model_and_preview(monkeypatch, tmp_path):
    root, app = _make_app(monkeypatch, tmp_path)
    try:
        app.demo()
        _wait(root, app)
        assert app.captured_errors == []
        assert app.model is not None
        assert app.model_path.is_file()
        assert app.output_path.is_relative_to(tmp_path / "data")
        assert app.output_path.is_file()
        assert app.play_button.instate(["!disabled"])
        assert app.wave_axis.lines and app.spectrum_axis.collections
    finally:
        app.close()


def test_worker_error_restores_controls_and_logger(monkeypatch, tmp_path):
    original_level = logging.getLogger("ssgs").level
    original_handlers = list(logging.getLogger("ssgs").handlers)
    root, app = _make_app(monkeypatch, tmp_path)
    try:
        def fail(*_):
            raise OSError("simulated read failure")
        app._start("Failing task", fail, lambda _: None)
        _wait(root, app)
        assert "simulated read failure" in app.captured_errors[0][1]
        assert app.cancel_button.instate(["disabled"])
        assert not app.busy
    finally:
        app.close()
    assert logging.getLogger("ssgs").level == original_level
    assert logging.getLogger("ssgs").handlers == original_handlers
