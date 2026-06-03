"""
tests/test_utils.py
===================
Unit tests for utils.py — find_nearest_free_slot, astar, build_obstacle_grid.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import SlotConfig
from utils import (
    GRID_CELL,
    astar,
    build_obstacle_grid,
    find_nearest_free_slot,
)


# ─────────────────────────────────────────────
# find_nearest_free_slot
# ─────────────────────────────────────────────

class TestFindNearestFreeSlot:
    def test_returns_none_when_all_occupied(self):
        cfg = SlotConfig(
            rect_w=100,
            rect_h=33,
            slots=[(0, 0), (200, 0), (400, 0)],
        )
        statuses = [False, False, False]  # all occupied
        result = find_nearest_free_slot((0, 0), cfg, statuses)
        assert result is None

    def test_returns_index_of_nearest_free(self):
        cfg = SlotConfig(
            rect_w=100,
            rect_h=33,
            slots=[(0, 0), (500, 0), (1000, 0)],
        )
        statuses = [False, True, True]  # slot 1 and 2 are free
        # Entry at (500, 0) — slot 1 center is at (550, 16), slot 2 center at (1050, 16)
        result = find_nearest_free_slot((500, 0), cfg, statuses)
        assert result == 1  # slot index 1 is nearest

    def test_returns_none_when_no_slots(self):
        cfg = SlotConfig(rect_w=100, rect_h=33, slots=[])
        result = find_nearest_free_slot((0, 0), cfg, [])
        assert result is None

    def test_returns_only_free_slot(self):
        cfg = SlotConfig(
            rect_w=100,
            rect_h=33,
            slots=[(0, 0), (200, 0)],
        )
        statuses = [False, True]  # only slot 1 is free
        result = find_nearest_free_slot((0, 0), cfg, statuses)
        assert result == 1


# ─────────────────────────────────────────────
# astar
# ─────────────────────────────────────────────

class TestAstar:
    def test_returns_none_when_no_path_exists(self):
        """
        Goal is completely surrounded by obstacles (all cells = 1) with no
        free neighbour within the 3-cell search radius, so astar returns None.
        """
        # Create a 20x20 grid (all obstacles)
        rows, cols = 20, 20
        grid = np.ones((rows, cols), dtype=np.uint8)

        # Leave only the start cell free so the search can begin
        grid[0, 0] = 0

        # Start at pixel (0, 0) → grid cell (0, 0)
        # Goal at pixel (19*GRID_CELL, 19*GRID_CELL) → grid cell (19, 19)
        # The goal cell and all its neighbours within 3 cells are obstacles,
        # and the rest of the grid is also obstacles, so no path exists.
        start_px = (0, 0)
        goal_px = (19 * GRID_CELL, 19 * GRID_CELL)

        result = astar(grid, start_px, goal_px)
        assert result is None

    def test_returns_path_when_clear_grid(self):
        """A* should find a path on a fully clear grid."""
        rows, cols = 10, 10
        grid = np.zeros((rows, cols), dtype=np.uint8)
        start_px = (0, 0)
        goal_px = (5 * GRID_CELL, 5 * GRID_CELL)
        result = astar(grid, start_px, goal_px)
        assert result is not None
        assert len(result) >= 1

    def test_start_equals_goal_returns_path(self):
        """When start and goal map to the same grid cell, a path is returned."""
        rows, cols = 10, 10
        grid = np.zeros((rows, cols), dtype=np.uint8)
        # Both start and goal map to grid cell (2, 2)
        px = (2 * GRID_CELL + 5, 2 * GRID_CELL + 5)
        result = astar(grid, px, px)
        assert result is not None


# ─────────────────────────────────────────────
# build_obstacle_grid
# ─────────────────────────────────────────────

class TestBuildObstacleGrid:
    def test_occupied_slot_cells_marked_as_1(self):
        """Cells covered by an occupied slot should be 1 in the grid."""
        frame_shape = (200, 200, 3)
        cfg = SlotConfig(
            rect_w=40,
            rect_h=20,
            slots=[(0, 0)],  # slot at top-left
        )
        statuses = [False]  # occupied

        grid = build_obstacle_grid(frame_shape, cfg, statuses)

        # The slot covers x=[0,40), y=[0,20)
        # In grid coords: gx1=0, gy1=0, gx2=40//GRID_CELL=2, gy2=20//GRID_CELL=1
        # So grid[0:2, 0:3] should be 1
        assert grid[0, 0] == 1
        assert grid[1, 0] == 1
        assert grid[0, 1] == 1

    def test_free_slot_cells_not_marked(self):
        """Cells covered by a free slot should remain 0."""
        frame_shape = (200, 200, 3)
        cfg = SlotConfig(
            rect_w=40,
            rect_h=20,
            slots=[(0, 0)],
        )
        statuses = [True]  # free

        grid = build_obstacle_grid(frame_shape, cfg, statuses)

        assert grid[0, 0] == 0

    def test_empty_slots_returns_zero_grid(self):
        """No slots → grid should be all zeros."""
        frame_shape = (100, 100, 3)
        cfg = SlotConfig(rect_w=40, rect_h=20, slots=[])
        grid = build_obstacle_grid(frame_shape, cfg, [])
        assert np.all(grid == 0)

    def test_multiple_slots_mixed_statuses(self):
        """Only occupied slots should mark their cells as 1."""
        frame_shape = (200, 400, 3)
        cfg = SlotConfig(
            rect_w=40,
            rect_h=20,
            slots=[(0, 0), (200, 0)],
        )
        statuses = [False, True]  # first occupied, second free

        grid = build_obstacle_grid(frame_shape, cfg, statuses)

        # First slot (occupied) at x=0, y=0 → grid cell (0,0) should be 1
        assert grid[0, 0] == 1
        # Second slot (free) at x=200, y=0 → grid cell (0, 10) should be 0
        assert grid[0, 200 // GRID_CELL] == 0
