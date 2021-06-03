#!/usr/bin/env  python3
"""
Modelize population growth using different models, by solving differential equations.
"""
"""
- Malthus model: simple modelisation
- Verhulst model: modelisation with a maximum capacity
- Lotka Volterra: modelisation with prey / predator interactions
"""


from diff_equation_resol import meth_epsilon, step_Euler, step_Heun, step_Runge_Kutta, step_point_milieu
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import Callable, List
import argparse
import warnings
warnings.simplefilter('ignore') # Disable overflow warnings


class Model:
    """
    A class representing a model

    Parameters:
        - name: model name
        - function: function computing the model
        - params: parameters to give to the model function
        - dimension: model dimension
    """

    def __init__(self, name: str, function: Callable, params: dict, dimension: int) -> None:
        self.name = name
        self.function = function
        self.params = params
        self.dimension = dimension
        self.solutions, self.times = function(*self.params.values())

    def plot_parameters(self) -> None:
        """
        Plot the model parameters in a text box
        """
        txtparams = []
        for item in self.params.items():
            key = item[0]
            value = item[1].__name__ if callable(item[1]) else item[1]
            txtparams.append(f'{key} = {value}')

        txtparams = '\n'.join(txtparams)
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        plt.gcf().text(0.14, .86, txtparams, verticalalignment='top', bbox=props, fontsize=8)
        print(f'\nPlotted {self.name} model with parameters:\n{txtparams}')

    def plot(self, name: str = None) -> None:
        """
        Plot the current model plot
        """
        self.plot_parameters()
        if name:
            plt.savefig(name)
        else:
            plt.show()

        plt.clf()

    def plot_solutions(self, name: str = None) -> None:
        """
        Plot the model solutions
        """
        plt.plot(self.times, self.solutions)
        plt.xlabel('Temps (arbitraire)')
        plt.ylabel('Population')

        if self.dimension == 2:
            plt.legend(['Lapins', 'Renards'], loc='upper right')

        self.plot(name)

    def plot_variations(self, name: str = None) -> None:
        """
        Plot the model variation and print its period
        """
        if self.dimension != 2:
            raise ValueError(
                f'Invalid method plot_variations for dimension other than 2 ({self.dimension})')

        prey = [solution[0] for solution in self.solutions]
        predator = [solution[1] for solution in self.solutions]
        difference = [prey[i] - predator[i]
                      for i in range(len(self.solutions))]

        plt.plot(self.times, difference)
        self.plot(name)
        print(f'Plotted couple variations for {self.name}')
        period = find_period(difference)
        print(
            f'Period: {period * self.params["time"] / len(self.solutions) if period else None}')

    def plot_solutions_around_point(self, n: int, name: str = None) -> None:

        if self.dimension != 2:
            raise ValueError(
                f'Invalid method plot_solution_around_point for dimension other than 2 ({self.dimension})')

        params = self.params.copy()
        point = [params['init1'], params['init2']]
        initials = [point] + [np.array([abs(point[0] + i), abs(point[1] + i)])
                              for i in range(-n // 2, n // 2 + 1)]
        for initial in initials:
            params['init1'] = initial[0]
            params['init2'] = initial[1]
            solutions, times = self.function(*params.values())
            prey = [solution[0] for solution in solutions]
            predator = [solution[1] for solution in solutions]
            plt.plot(prey, predator,
                     color='red' if initial[0] == point[0] else 'steelblue')
            if initial[0] == point[0]:
                plt.legend(['y0'], loc='best')

        plt.scatter(*point, color='red')
        plt.xlabel('N(t)')
        plt.ylabel('P(t)')
        self.plot(name)

    def singular_points(self) -> None:
        """
        Print the singular points of the Lotka Volterra model
        """
        if self.function.__name__ != 'lotka_volterra':
            raise ValueError(
                f'Invalid method plot_singular_point for model other than Lotka Volterra ({self.function.__name__})')

        a, b, c, d = self.params['a'], self.params['b'], self.params['c'], self.params['d']
        print(f'{self.name} singular points are {(0, 0)}, {(d / c, a / b)}')


def find_period(data: List):
    """
    Find the approximate period of a periodic-function given its values
    by searching for peaks

    Parameters:
        - data: function values

    Warning: may not work if a single period have multiple peaks
    """
    # Get all the peaks in data
    peaks = find_peaks(data)[0]

    if len(peaks) < 2:
        print('ERROR: Function seems to be non-periodic')
        return None

    if not np.allclose([data[peak] for peak in peaks], data[peaks[0]], atol=1):
        print('ERROR: Could not find period, single period seems to have multiple peaks')
        return None

    # Get the difference between the peaks and average it to get the period
    period = sum([peaks[i + 1] - peaks[i]
                  for i in range(len(peaks) - 1)]) / (len(peaks) - 1)

    return period


def malthus(initial: int, birth: float, death: float,
            time: int, method: Callable = None) -> None:
    """
    Modelize population growth with the Malthus model

    Parameters:
        - initial: initial population
        - b: birth rate
        - d: death rate
        - method: differential equation resolution method
        - time: duration modelized
    """
    def population(y, t):
        return (birth - death) * y

    return meth_epsilon(np.array([initial]), 0, time, .1, population, method)


def verhulst(initial: int, birth: float, death: float,
             maxsize: int, time: int, method: Callable) -> None:
    """
    Modelize population growth with the Malthus model

    Parameters:
        - initial: initial population
        - b: birth rate
        - d: death rate
        - maxsize: maximum population size
        - method: differential equation resolution method
        - time: duration modelized
    """
    def population(y, t):
        return (birth - death) * y * (1 - y / maxsize)

    return meth_epsilon(np.array([initial]), 0, time, .1, population, method)


def lotka_volterra(preyinit: int, predatorinit: int, preybirth: float, preydeath: float,
                   predatorbirth: float, predatordeath: float, time, method: Callable) -> None:
    """
    Modelize population growth with the Lokta-Volterra model

    Parameters:
        - preyinit: initial prey population
        - predatorinit: initial predator population
        - preybirth: inherent prey birth rate
        - preydeath: prey death rate due to predator threat
        - predatorbirth: predator birth due to prey eaten
        - predatordeath: inherent predator death
        - method: differential equation resolution method
        - time: duration modelized
    """
    def population(y, t):
        return np.array([
            y[0] * (preybirth - preydeath * y[1]),
            y[1] * (predatorbirth * y[0] - predatordeath)
        ])

    return meth_epsilon(
        np.array([preyinit, predatorinit]), 0, time, .1, population, method)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['euler', 'e', 'milieu', 'm', 'runge', 'r', 'heun', 'h'],
                        help='differential equation solving method', default='heun')
    args = parser.parse_args()

    method = step_Euler if args.method == 'euler' or args.method == 'e' \
        else step_point_milieu if args.method == 'milieu' or args.method == 'm' \
        else step_Runge_Kutta if args.method == 'runge' or args.method == 'r' \
        else step_Heun

    ## Compute examples
    # Malthus model
    M = {'init': 50, 'a': .2, 'b': .13, 'time': 100, 'method': method}
    # Verhulst model
    V = {'init': 50, 'a': .2, 'b': .13, 'k': 2000, 'time': 200, 'method': method}
    # Lokta-Volterra model with initial non-singular point
    LVNS = {'init1': 50, 'init2': 50, 'a': .08, 'b': .004,
           'c': .002, 'd': .06, 'time': 365, 'method': method}
    # Lokta-Volterra model with initial singular point
    LVS = {'init1': 30, 'init2': 20, 'a': .08, 'b': .004,
           'c': .002, 'd': .06, 'time': 160, 'method': method}

    Malthus = Model('Malthus', malthus, M, 1)
    Verhulst = Model('Verhulst', verhulst, V, 1)
    LotkaVolterra = Model('LotkaVolterra', lotka_volterra, LVNS, 2)

    Malthus.plot_solutions('malthus')
    Verhulst.plot_solutions('verhulst')
    LotkaVolterra.plot_solutions('loktavolterra')
    LotkaVolterra.singular_points()
    LotkaVolterra.plot_variations()
