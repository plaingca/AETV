from PySide6.QtWidgets import QApplication

from aetv.gui.clip_editor import TrimSlider


_APP = QApplication.instance() or QApplication([])


def test_trim_slider_pips_clamp_to_minimum_selected_range(monkeypatch):
    slider = TrimSlider()
    slider.set_trim(100, 900, minimum_gap=200)
    changes = []
    slider.trimChanged.connect(lambda start, end: changes.append((start, end)))
    monkeypatch.setattr(slider, "_x_to_value", lambda _x: 850)

    slider._dragging = "in"
    slider._move_pip(0)

    assert slider.trim() == (700, 900)
    assert changes[-1] == (700, 900)


def test_trim_slider_out_pip_cannot_cross_in_pip(monkeypatch):
    slider = TrimSlider()
    slider.set_trim(250, 750, minimum_gap=100)
    monkeypatch.setattr(slider, "_x_to_value", lambda _x: 100)

    slider._dragging = "out"
    slider._move_pip(0)

    assert slider.trim() == (250, 350)
