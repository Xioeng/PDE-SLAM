"""
test_interactive.py
===================
Unit tests for InteractiveWaypointPicker and pick_waypoints_gui.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Headless test backend

import numpy as np

from pde_slam.viz.interactive import InteractiveWaypointPicker, pick_waypoints_gui


def test_interactive_waypoint_picker_initialization():
    poly = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]])
    defaults = np.array([[10.0, 10.0], [40.0, 40.0]])

    picker = InteractiveWaypointPicker(
        polygon_enu=poly,
        default_waypoints=defaults,
    )

    assert len(picker.waypoints) == 2
    assert picker.waypoints[0] == [10.0, 10.0]
    assert picker.waypoints[1] == [40.0, 40.0]


def test_interactive_waypoint_picker_callbacks():
    poly = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]])
    picker = InteractiveWaypointPicker(polygon_enu=poly)

    # Simulate Left click event (add point)
    class FakeMouseEvent:
        def __init__(self, xdata, ydata, button, inaxes):
            self.xdata = xdata
            self.ydata = ydata
            self.button = button
            self.inaxes = inaxes

    class FakeKeyEvent:
        def __init__(self, key):
            self.key = key

    event1 = FakeMouseEvent(15.0, 25.0, 1, picker.ax)
    picker._on_click(event1)
    assert len(picker.waypoints) == 1
    assert picker.waypoints[-1] == [15.0, 25.0]

    event2 = FakeMouseEvent(30.0, 45.0, 1, picker.ax)
    picker._on_click(event2)
    assert len(picker.waypoints) == 2

    # Simulate Right click event (remove last point)
    event3 = FakeMouseEvent(0.0, 0.0, 3, picker.ax)
    picker._on_click(event3)
    assert len(picker.waypoints) == 1
    assert picker.waypoints[-1] == [15.0, 25.0]

    # Simulate 'c' key event (clear all)
    key_c = FakeKeyEvent("c")
    picker._on_key(key_c)
    assert len(picker.waypoints) == 0

    # Simulate 'enter' key event (confirm)
    key_enter = FakeKeyEvent("enter")
    picker._on_key(key_enter)
    assert picker.confirmed is True


def test_pick_waypoints_gui_fallback():
    poly = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]])
    defaults = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])

    # In Agg backend, plt.show() does not block
    pts = pick_waypoints_gui(polygon_enu=poly, default_waypoints=defaults)
    assert len(pts) == 3
    np.testing.assert_allclose(pts, defaults)
