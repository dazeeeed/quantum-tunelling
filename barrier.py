import numpy as np


class Barrier:
    def __init__(self, x, x_0, value=0, width=0, barrier_type='square'):
        '''Potential barrier array creation. x_0 is a position of the center of the barrier.
           Value is a height of this barrier. Width is a width of the barrier.'''
        self.x = x
        self.x_0 = x_0
        self.value = value
        self.width = width
        self.barrier_type = barrier_type

        self.barrier = np.empty(len(self.x))

    def get_barrier(self):
        if self.barrier_type == 'square':
            for ix in range(len(self.x)):
                if self.x_0 - self.width/2 <= self.x[ix] <= self.x_0 + self.width/2:
                    self.barrier[ix] = self.value
                else:
                    self.barrier[ix] = 0

            return self.barrier

        elif self.barrier_type == 'smooth':
            for ix in range(len(self.x)):
                # x to calculate smooth step
                smooth_x = self.x_0 - self.width/2 - 1
                smooth_x2 = self.x_0 + self.width/2 + 1

                # left slope
                if self.x_0 - self.width/2 - 1 <= self.x[ix] <= self.x_0 - self.width/2:
                    self.barrier[ix] = (3 * (self.x[ix]-smooth_x)**2 - 2 * (self.x[ix]-smooth_x)**3) * self.value

                # center
                elif self.x_0 - self.width/2 <= self.x[ix] <= self.x_0 + self.width/2:
                    self.barrier[ix] = self.value

                # right slope
                elif self.x_0 + self.width / 2 <= self.x[ix] <= self.x_0 + self.width/2 + 1:
                    self.barrier[ix] = (3 * (smooth_x2 - self.x[ix]) ** 2 - 2 * (smooth_x2 - self.x[ix]) ** 3) * self.value
                else:
                    self.barrier[ix] = 0

            return self.barrier
        else:
            return self.barrier
