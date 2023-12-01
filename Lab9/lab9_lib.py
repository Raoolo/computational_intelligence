# Copyright © 2023 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free for personal or classroom use; see 'LICENSE.md' for details.

from abc import abstractmethod


class AbstractProblem:
    def __init__(self):
        self._calls = 0

    @property
    @abstractmethod
    def x(self):    # method to be redifined when calling the class
        pass

    @property
    def calls(self):
        return self._calls

    @staticmethod
    def onemax(genome): # computes the onemax of a genome (a list)
        return sum(bool(g) for g in genome)

    def __call__(self, genome): # genome is a list btw
        '''updates num of calls, calculates fitness of a genome'''
        self._calls += 1
        fitnesses = sorted((AbstractProblem.onemax(genome[s :: self.x]) for s in range(self.x)), reverse=True)
        val = sum(f for f in fitnesses if f == fitnesses[0]) - sum(
            f * (0.1 ** (k + 1)) for k, f in enumerate(f for f in fitnesses if f < fitnesses[0])
        )   
        return val / len(genome)    # this is the actual fitness of the genome


def make_problem(a):
    class Problem(AbstractProblem):
        @property
        @abstractmethod
        def x(self):    # redefine x method to return a, 
            return a

    return Problem()
