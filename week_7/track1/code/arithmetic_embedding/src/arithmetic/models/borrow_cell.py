"""Shared local controller for one subtraction limb and its borrow state."""

from arithmetic.models.carry_cell import CarryCell, CarryCellOutput


class BorrowCell(CarryCell):
    """Same latent transition shape as CarryCell, with borrow-state semantics."""


BorrowCellOutput = CarryCellOutput
