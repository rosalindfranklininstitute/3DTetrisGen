#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:25:48 2021

@author: anna
"""

import random
import numpy as np
from pathlib import Path
from scipy import ndimage

import napari
import mrcfile


class Brick:

    def __init__(self, pos, density):
        self.pos = pos
        self.density = density

    def __repr__(self):
        return f'{self.__class__.__name__}({self.pos})'


class Pokemino:

    def __init__(self, seed, size, volume, dim, positioning='central', density=1, algorithm=False, brick_coords=None):

        self.density = density
        self.size = size
        self.seed = seed
        self.dim = dim
        self.bricks = np.empty(size, dtype=np.object)
        self.density = density
        self.algorithm = algorithm

        if not self.algorithm:
            assert len(brick_coords) == self.size, "Size declared not matched with len(brick_pos_list)!"
            assert all([isinstance(i, list) for i in brick_coords]), "Brick positions passed in an incorrect format!"
            assert all([len(i) == self.dim for i in
                        brick_coords]), "Brick coordinates don't match the declared dimensionality!"
            self.brick_coords = brick_coords

        elif self.algorithm == "clumped":
            brick_coords = self.create_coords_for_clumped_pokemino()

        elif self.algorithm == "extended":
            brick_coords = self.create_coords_for_extended_pokemino()

        for i, brick in enumerate(brick_coords):
            self.bricks[i] = Brick(brick, self.density)

        self.make_coords_relative_to_centre_of_mass()
        self.brick_coords = [brick.pos for brick in self.bricks]

        self.max_radius = Pokemino.find_max_radius(self.brick_coords)

        if positioning == 'central':
            self.positioning = np.array(tuple(map(lambda x: int(round(x / 2)), volume.shape)))
        else:
            self.positioning = np.array(positioning)

        self.r = int(self.max_radius)
        self.poke_array = np.zeros((2 * self.r + 1,) * self.dim)

        for brick in self.bricks:
            placement_coords = tuple(x + y for (x, y) in zip((self.r,) * self.dim, brick.pos))
            self.poke_array[placement_coords] = brick.density

        volume.creatures = np.hstack([volume.creatures, self])
        volume.n_creatures += 1


    def create_coords_for_clumped_pokemino(self):

        random.seed(self.seed)

        brick_coords = [[0] * self.dim]
        if self.dim == 2:
            steps = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
        elif self.dim == 3:
            steps = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])

        while len(brick_coords) < self.size:
            random_step = random.choice(steps)
            new_brick = list(random.choice(brick_coords) + random_step)
            if new_brick not in brick_coords:
                brick_coords.append(new_brick)

        return brick_coords


    def create_coords_for_extended_pokemino(self):

        random.seed(self.seed)

        brick_coords = [[0] * self.dim]
        if self.dim == 2:
            steps = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
        elif self.dim == 3:
            steps = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])

        while len(brick_coords) < self.size:
            potential_step_positions = []
            for brick in brick_coords:
                for step in steps:
                    if tuple(np.array(brick) + np.array(step)) not in [brick_coords, potential_step_positions]:
                        potential_step_positions.append(list(np.array(brick) + np.array(step)))
            new_brick = random.choice(potential_step_positions)

            if new_brick not in brick_coords:
                brick_coords.append(new_brick)
        return brick_coords


    def make_coords_relative_to_centre_of_mass(self):

        weighted_coords = np.array([np.array(brick.pos) * brick.density for brick in self.bricks])
        com_pos = np.mean(weighted_coords, axis=0)
        new_com = self.bricks[np.argmin(np.sum(np.square(weighted_coords - com_pos), axis=1))].pos
        for i, brick in enumerate(self.bricks):
            brick.pos = [i - j for (i, j) in zip(brick.pos, new_com)]
        self.brick_coords = [brick.pos for brick in self.bricks]


    @staticmethod
    def find_max_radius(all_positions):
        """Finds the Euclidean distance to the block furthest from the centre of mass"""
        return np.sqrt(np.max(np.sum(np.array(all_positions) ** 2, axis=1)))


    def upscale_pokemino(self, scale_factor):

        if self.dim == 3:
            self.poke_array = self.poke_array.repeat(scale_factor, axis=0).repeat(scale_factor, axis=1).repeat(
                scale_factor, axis=2).astype(np.float32)
        elif self.dim == 2:
            self.poke_array = self.poke_array.repeat(scale_factor, axis=0).repeat(scale_factor, axis=1).astype(
                np.float32)

        self.max_radius *= scale_factor


    def move_pokemino_in_volume(self, vector):

        self.positioning = self.positioning + np.array(vector)


    def visualise_in_napari(self, display_window_size):

        display_window = np.zeros(tuple([display_window_size] * self.dim))
        central_pos = tuple(map(lambda x: round(x / 2), display_window.shape))
        for brick in self.bricks:
            display_window[tuple(x + y for (x, y) in zip(central_pos, brick.pos))] = 1
        napari.view_image(display_window)


class Pokemino2D(Pokemino):

    def __init__(self, seed, size, volume, dim=2, positioning="central", density=1, algorithm=False,
                 brick_pos_list=None):
        self.dim = dim
        super().__init__(seed, size, volume, dim, positioning, density, algorithm, brick_pos_list)
        random.seed()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.seed, self.size})'

    def rotate_the_pokemino(self, theta=None, order=1):
        """Rotates the Pokemino using scipy.ndimage.rotate"""

        if not theta:
            random.seed()
            theta = random.choice([i for i in range(0, 360)])
        else:
            assert (isinstance(theta, int)) and theta in range(0, 360), \
                'Error: Pokemino3D.rotate_the_block requires the value for theta in range <0, 360>.'
        self.poke_array = ndimage.rotate(self.poke_array, angle=theta, order=order, reshape=True)


class Pokemino3D(Pokemino):

    def __init__(self, seed, size, volume, dim=3, positioning="central", density=1, algorithm=False,
                 brick_pos_list=None):
        self.dim = 3
        super().__init__(seed, size, volume, dim, positioning, density, algorithm, brick_pos_list)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.seed, self.size})'

    def rotate_the_pokemino_1_axis(self, axes=(0, 1), theta=None, order=1, s=None):
        """Rotates the Pokemino using scipy.ndimage.rotate"""

        assert type(axes) == tuple \
               and len(axes) == 2 \
               and all([type(i) == int for i in axes]) \
               and all([i in range(0, 3) for i in axes]), \
               "Incorrect axes parameter: pass a tuple of 2 axes."

        if not theta:
            np.random.seed(s)
            theta = np.random.choice([i for i in range(0, 360)])
        else:
            assert (isinstance(theta, int)) and theta in range(0, 360), \
                'Error: Pokemino3D.rotate_the_block requires the value for theta in range <0, 360>.'
        self.poke_array = ndimage.rotate(self.poke_array, angle=theta, axes=axes, order=order, reshape=True)
        print(theta)
        # viewer = napari.view_image(self.poke_array)


    def rotate_the_pokemino_3_axes(self, theta_x=None, theta_y=None, theta_z=None):

        self.rotate_the_pokemino_1_axis(axes=(1, 0), theta=theta_x)
        self.rotate_the_pokemino_1_axis(axes=(2, 1), theta=theta_y)
        self.rotate_the_pokemino_1_axis(axes=(0, 2), theta=theta_z)


class Volume:

    def __init__(self, shape):
        self.dim = len(shape)
        self.shape = shape
        self.array = np.zeros(shape, dtype=np.float32)
        self.n_creatures = 0
        self.creatures = np.empty(0, dtype=np.object)


    def fit_pokemino(self, pokemino):

        """Fir pokemino.poke_array into volume (centred at pokemino.positioning)"""
        slices = [np.s_[i - r: i + r + 1] for i, r in zip(pokemino.positioning, [pokemino.r, ] * pokemino.dim)]
        self.array[slices] += pokemino.poke_array


    def check_for_overlap(self, pokemino1, pokemino2):
        """For two Pokeminos, check if spheres centred at their centres of mass and of their associated max_radia
        overlap. """
        distance_between_centres_of_mass = np.sqrt(np.sum(np.square(pokemino1.positioning - pokemino2.positioning)))
        sum_of_max_radii = (
                    np.sqrt(np.sum(np.square(pokemino1.max_radius))) + np.sqrt(np.sum(np.square(pokemino2.max_radius))))

        if sum_of_max_radii >= distance_between_centres_of_mass + 1:
            return True


    def move_overlapping_apart(self, pokemino1, pokemino2):
        """Move a pair of potentially overlapping Pokeminos apart, one step at a time.

           This works by determining the vector connecting pokemino1.positioning and pokemino2.positioning
           creating a list of all unit vector along x, y and z (lottery_machine) and randomly choosing one of these
           unit vectors to move Pokeminos apart."""

        lottery_machine = []

        # print("Before moving apart:", pokemino1.positioning, np.array(pokemino2.positioning))

        for i, coord in enumerate(pokemino1.positioning - pokemino2.positioning):
            empty = np.array([0] * pokemino1.dim)
            if coord != 0:
                empty[i] = coord / abs(coord)
            lottery_machine.extend([empty] * abs(coord))

        if all(pokemino1.positioning == pokemino2.positioning):
            lottery_machine = [[0] * pokemino1.dim]
            lottery_machine[0][0] = 1

        random.shuffle(lottery_machine)

        i = 0
        while self.check_for_overlap(pokemino1, pokemino2):
            pokemino_to_move = random.choice([pokemino1, pokemino2])
            vector = np.array(lottery_machine[i % len(lottery_machine)])
            if pokemino_to_move is pokemino1:
                pokemino_to_move.move_pokemino_in_volume(vector)
            elif pokemino_to_move is pokemino2:
                pokemino_to_move.move_pokemino_in_volume(-vector)
            i += 1

        # print("After moving apart:", pokemino1.positioning, pokemino2.positioning)


    def check_for_pairwise_overlap(self):
        """For any two Pokeminos in volume, check for potential overlap - if it might occur, move the Pokeminos apart.
           Repeat until no Pokeminos in volume overlap."""
        overlap = True
        while overlap:
            overlap = False
            for i, pokemino_i in enumerate(self.creatures):
                for j, pokemino_j in enumerate(self.creatures):
                    if i < j:
                        if self.check_for_overlap(pokemino_i, pokemino_j):
                            # print("There's overlap between", i+1, j+1)
                            self.move_overlapping_apart(pokemino_i, pokemino_j)
                            overlap = True


    def fit_all_pokeminos(self):

        """First find the coordinates of all top-left and bottom-right corners of all Pokeminos in volume."""
        top_left_corners = np.zeros((self.n_creatures, self.dim))
        bottom_right_corners = np.zeros((self.n_creatures, self.dim))
        for i, poke in enumerate(self.creatures):
            top_left_corners[i] = poke.positioning - np.array([poke.r, ] * poke.dim)
            bottom_right_corners[i] = poke.positioning + np.array([poke.r, ] * poke.dim)

        """Find the lowest and highest coordinates occupied by any of poke_arrays in volume. """
        tl, br = np.min(top_left_corners, axis=0), np.max(bottom_right_corners, axis=0)

        """Extend volume.array in all dimensions in which Pokeminos stick out."""
        negative_extensions = np.where(tl < 0, -tl, 0).astype(np.int32)
        positive_extensions = np.where(br > np.array(self.shape) - 1, br - (np.array(self.shape) - 1), 0).astype(
            np.int16)
        self.array = np.zeros(positive_extensions + negative_extensions + np.array(self.shape))

        """Correct Pokemino positionings for negative extensions of the volume."""
        for poke in self.creatures:
            poke.positioning += negative_extensions

        """Only now fit all Pokeminos."""
        for pokemino in self.creatures:
            self.fit_pokemino(pokemino)


    def save_as_mrcfile(self, output_path: Path):
        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(self.array)


    def display_in_napari(self):
        napari.view_image(self.array)