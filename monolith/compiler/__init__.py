"""The compiler: passes over the IR (canonicalize → fuse → select packs → partition → barriers → memory plan → emit)
and the coverage guard. Passes arrive with plan M4; the guard is here from the start because every registry consumer
depends on it."""

from .coverage import CoverageError, check_coverage

__all__ = ["CoverageError", "check_coverage"]
