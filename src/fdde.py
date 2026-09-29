# Fixed Depth Differential Encoder
# Author: Christopher J. Cole

from dataclasses import dataclass
from typing import Union, List, Sequence
import math

# dx = Size of each list for averaging/comparing
# If it's one, we just use each value and compare it to the previous one.
# If it's >1, we take the mean of the list of that size and use that.
# 0 means either downwards or no change.
# At depth = 2, that's roughly:
# Prev down, curr down: 00
# Prev down, curr up: 01
# Prev up, curr down: 10
# Prev up, curr up: 11

@dataclass
class FDDE:
    """
    FDDE Settings and Functions
    """
    dx: int # Step size (size of list at each step-- the size of the list whose values we take the mean of). 
    depth: int # Depth -- typically 2. Can be anything, though.
    size: int # The number of residuals to concat to make 1 FDDE residual. 4 is a good starting place.

    def __post_init__(self):
        if self.dx < 1 or self.depth < 1 or self.size < 1:
            raise ValueError("dx, depth and size must all be at least 1.")

    # Mean of a slice
    def mean(self, data: Sequence[float]) -> float:
        # Check size (if larger than dx regect)
        if len(data) > self.dx:
            raise ValueError("List is too large to mean. Adjust dx.")
        if len(data) < 1:
            raise ValueError("Cannot take the mean of an empty list.")
        # Do mean
        return math.fsum(data) / len(data)

    # Slice data into size dx lists.
    def slice_data(self, data: Sequence[float]) -> List[Union[List[float], float]]:
        if len(data) < 1:
            raise ValueError("No data given to the slicer.")
        # If dx is 1, it's just the list of values
        if self.dx == 1:
            return list(data)
        # Slice here (the final slice may be shorter than dx)
        return [list(data[i:i + self.dx]) for i in range(0, len(data), self.dx)]

    def list_means(self, slices: Sequence[Union[Sequence[float], float]]) -> List[float]:
        """
        Give a list of lists, convert it into a list of means.
        If a plain list of floats, return the list of floats, making sure dx = 1 beforehand.
        """
        if slices and not isinstance(slices[0], (list, tuple)):
            if self.dx != 1:
                raise ValueError("A plain list of floats requires dx = 1.")
            return [float(v) for v in slices]
        return [self.mean(s) for s in slices]

    # Encode the data using FDDE.
    def encode(self, data: Sequence[float]) -> List[int]:
        """
        Returns the list of FDDE residuals as integers.
        Each step's residual is a depth-bit shift register of up/down bits (newest bit last).
        Every `size` step residuals are concatenated into one FDDE residual; a short final
        group is zero-padded at the back to a full size * depth bits.
        """
        means = self.list_means(self.slice_data(data))
        mask = (1 << self.depth) - 1
        # The first mean has nothing to compare against, so it produces no residual (hidden).
        steps: List[int] = []
        state = 0
        for prev, curr in zip(means, means[1:]):
            state = ((state << 1) | (1 if curr > prev else 0)) & mask
            steps.append(state)

        residuals: List[int] = []
        for i in range(0, len(steps), self.size):
            group = steps[i:i + self.size]
            value = 0
            for step in group:
                value = (value << self.depth) | step
            # Pad a short final group with zeroes at the back.
            value <<= self.depth * (self.size - len(group))
            residuals.append(value)
        return residuals



# Rows Encode the depth
# Interpreting FDDE columnwise reconstructs the curve (sorta).
# Example
# -----------------------------------------------------------------
# Slice Index  | Slice Mean (dx = 1)   | Residual (depth = 4)
# ---------------------------------------------
# 0            |0 (Hidden)    | (0) Hidden
# 1            | 0.5538       | 0001 (because 0.6 is greater than 0)
# 2            | -0.3698      | 0010
# 3            | -0.9534      | 0100
# 4            | -0.6604      | 1001
# 5            | 0.2397       | 0011
# 6            | 0.9195       | 0111
# 7            | 0.7539       | 1110
# 8            | -0.1048      | 1100
# 9            | -0.8672      | 1000
# 10           | -0.9589      | 0000
# -----------------------------------------------------------------
# fdde.encode(...) is the function which does this.
# If the size here was equal to 5, it would concat 5 residuals at a time to produce 2 FDDE residuals.
# If it doesn't divide evenly, the number is packed with zeroes at the back.
# At size = 4, the first residual is 0001001001001001, the second is 0011011111101100 , and the third is
# 1000000000000000 

if __name__ == "__main__":
    # Demonstration of FDDE encoding, printing the FDDE residuals in hex.
    signal = [math.sin(0.9 * i) for i in range(256)]
    fdde = FDDE(dx=1, depth=4, size=8)
    print([f"{r:0{fdde.depth * fdde.size // 4}x}" for r in fdde.encode(signal)])