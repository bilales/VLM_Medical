import numpy as np

class SimpleRegistrationModel:
    def register(self, fixed, moving):
        return np.roll(moving, shift=3, axis=1)
