#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:25:48 2021

@author: anna
"""

import random
import numpy as np

import napari
import mrcfile
from pathlib import Path

from scipy import ndimage

THETA_EACH_90 = [0, 90, 180, 270]

class Brick:

    def __init__(self, pos, density):
        self.pos = pos
        self.density = density

    def __repr__(self):
        return f'{self.__class__.__name__}({self.pos})'


class Pokemino:

    def __init__(self, seed, size, volume, dim, positioning='central', density = 1, algorithm=False, brick_pos_list=None):

        self.density = density
        self.size = size
        self.seed = seed
        self.dim = dim
        self.bricks = np.empty(size, dtype=np.object)
        self.n_bricks = 0
        # self.density = 1
        self.algorithm = algorithm

        if not self.algorithm:
            assert len(brick_pos_list) == self.size, "Size declared not matched with len(brick_pos_list)!"
            assert all([isinstance(i, list) for i in brick_pos_list]), "Brick positions passed in an incorrect format!"
            assert all([len(i) == self.dim for i in
                        brick_pos_list]), "Brick coordinates don't match the declared dimensionality!"
            brick_coords = brick_pos_list

        elif self.algorithm == "clumped":
            brick_coords = self.create_coords_for_clumped_pokemino()

        elif self.algorithm == "extended":
            brick_coords = self.create_coords_for_extended_pokemino()

        for brick in brick_coords:
            self.bricks[self.n_bricks] = Brick(brick, self.density)
            self.n_bricks += 1

        self.make_coords_relative_to_centre_of_mass()

        if positioning == 'central':
            self.positioning = np.array(tuple(map(lambda x: int(round(x / 2)), volume.shape)))
        else:
            self.positioning = np.array(positioning)

        self.find_max_radius()

        r = int(self.max_radius)
        self.poke_array = np.zeros((2 * r + 1,) * self.dim)

        for brick in self.bricks:
            placement_coords = tuple(x + y for (x, y) in zip((r,) * self.dim, brick.pos))
            self.poke_array[placement_coords] = brick.density

        self.excluded_pos = []

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

    # TODO this is outdated since self.poke_array was introduced
    def create_coords_for_extended_pokemino(self):

        random.seed(self.seed)

        brick_coords = [[0] * self.dim]
        if self.dim == 2:
            steps = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
        elif self.dim == 3:
            steps = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])

        # TODO make this faster
        while len(brick_coords) < self.size:
            potential_step_positions = []
            for brick in brick_coords:
                for step in steps:
                    if tuple(np.array(brick) + np.array(step)) not in [brick_coords, potential_step_positions]:
                        potential_step_positions.append(tuple(np.array(brick) + np.array(step)))
            new_brick = random.choice(potential_step_positions)

            if new_brick not in brick_coords:
                brick_coords.append(new_brick)

        return brick_coords

    def make_coords_relative_to_centre_of_mass(self):

        weighted_coords = np.zeros((self.n_bricks, self.dim))
        for i, brick in enumerate(self.bricks):
            weighted_coords[i] = [coord * brick.density for coord in brick.pos]
        com_pos = np.mean(weighted_coords, axis=0)
        new_com = self.bricks[np.argmin(np.sum(np.square(weighted_coords - com_pos), axis=1))].pos
        for i, brick in enumerate(self.bricks):
            brick.pos = [i - j for (i, j) in zip(brick.pos, new_com)]

    def find_max_radius(self):
        """Finds the Euclidean distance to the block furthest from the centre of mass"""
        all_positions = np.array([brick.pos for brick in self.bricks])
        self.max_radius = np.sqrt(np.max(np.sum(all_positions ** 2, axis=1)))

    def upscale_pokemino (self, scale_factor):

        if self.dim == 3:
            self.poke_array = self.poke_array.repeat(scale_factor, axis=0).repeat(scale_factor, axis=1).repeat(
            scale_factor, axis=2).astype(np.float32)
        elif self.dim == 2:
            self.poke_array = self.poke_array.repeat(scale_factor, axis=0).repeat(scale_factor, axis=1).astype(np.float32)

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

    # TODO implement rotation using scipy.ndimage.rotate
    def rotate_the_pokemino(self, theta=None, randomise=True):

        if randomise:
            random.seed()
            theta = random.choice(THETA_EACH_90) / 180 * np.pi

        else:
            assert (isinstance(theta, (int, float))), 'Error: Pokemino2D.rotate_the_block requires the value for ' \
                                                      'theta that is int or float. '
            theta = theta / 180 * np.pi

        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        for brick in self.bricks:
            brick.pos = np.rint(np.matmul(R, brick.pos)).astype(int)

class Pokemino3D(Pokemino):

    def __init__(self, seed, size, volume, dim=3, positioning="central", density=1, algorithm=False, brick_pos_list=None):
        self.dim = 3
        super().__init__(seed, size, volume, dim, positioning, density, algorithm, brick_pos_list)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.seed, self.size})'

    def rotate_the_pokemino_1_axis(self, axes=(0,1), theta=None, order=1):
        """Rotates the Pokemino using scipy.ndimage.rotate"""

        assert type(axes) == tuple \
               and len(axes) == 2 \
               and all([type(i) == int for i in axes]) \
               and all([i in range(0, 3) for i in axes]), \
                "Incorrect axes parameter: pass a tuple of 2 axes."

        if not theta:
            random.seed()
            theta = random.choice([i for i in range(0, 360)])
        else:
            assert (isinstance(theta, int)) and theta in range(0, 360), \
                    'Error: Pokemino3D.rotate_the_block requires the value for theta in range <0, 360>.'
        self.poke_array = ndimage.rotate(self.poke_array, angle=theta, axes=axes, order=1, reshape=True)
        # viewer = napari.view_image(self.poke_array)

    def rotate_the_pokemino_3_axes(self, theta_x=None, theta_y=None, theta_z=None):

        self.rotate_the_pokemino_1_axis(axes=(1,0), theta=theta_x)
        self.rotate_the_pokemino_1_axis(axes=(2,1), theta=theta_y)
        self.rotate_the_pokemino_1_axis(axes=(0,2), theta=theta_z)


class Volume:

    def __init__(self, shape):
        self.shape = shape
        self.array = np.zeros(shape, dtype=np.float32)
        self.n_creatures = 0
        self.creatures = np.empty(0, dtype=np.object)
        self.subpixel_resolution_array = None
        self.new_image = None

    def fit_pokemino(self, pokemino):
        middle_x, middle_y, middle_z = int((pokemino.poke_array.shape[0] - 1) / 2), int((pokemino.poke_array.shape[1] - 1) / 2), int((pokemino.poke_array.shape[2] - 1) / 2)
        for i in range(-middle_x, middle_x + 1):
            for j in range(-middle_y, middle_y + 1):
                for k in range(-middle_z, middle_z + 1):
                    if pokemino.positioning[0] + i >= 0 and pokemino.positioning[1] + j >= 0 and pokemino.positioning[
                        2] + k >= 0:
                        try:
                            self.array[pokemino.positioning[0] + i, pokemino.positioning[1] + j, pokemino.positioning[2] + k] += pokemino.poke_array[middle_x + i, middle_y + j, middle_z + k]
                        except IndexError:
                            pass

    def fit_excluded_volume(self, pokemino):
        for excluded_voxel in pokemino.excluded_pos:
            placement_coords = tuple(x + y for (x, y) in zip(pokemino.positioning, excluded_voxel))
            if all([(coord >= 0) and (coord < self.shape[i]) for i, coord in enumerate(placement_coords)]):
                self.array[placement_coords] = 0.5  # called padding (adding a padding around)

    def check_for_overlap (self, pokemino1, pokemino2):

        distance_between_centres_of_mass = np.sqrt(np.sum(np.square(pokemino1.positioning - pokemino2.positioning)))
        sum_of_max_radii = (np.sqrt(np.sum(np.square(pokemino1.max_radius))) + np.sqrt(np.sum(np.square(pokemino2.max_radius))))

        if sum_of_max_radii >= distance_between_centres_of_mass + 1:
            return True

    def check_for_pairwise_overlap(self):
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

    def move_overlapping_apart(self, pokemino1, pokemino2):
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
            vector = lottery_machine[i % len(lottery_machine)]
            if pokemino_to_move is pokemino1:
                pokemino_to_move.move_pokemino_in_volume(vector)
            elif pokemino_to_move is pokemino2:
                pokemino_to_move.move_pokemino_in_volume(-vector)
            i += 1

        # print("After moving apart:", pokemino1.positioning, pokemino2.positioning)

    def fit_all_pokeminos(self, fit_excluded_volume=False):
        if fit_excluded_volume:
            for pokemino in self.creatures:
                self.fit_excluded_volume(pokemino)

        for pokemino in self.creatures:
            self.fit_pokemino(pokemino)

    # TODO: build something that does it but is compatible with poke_array
    def add_extra_volume_to_remove_cuts(self):
        all_positions = np.array([0, 0, 0])
        for creature in self.creatures:
            all_positions = np.vstack(
                [all_positions, np.array([brick.pos for brick in creature.bricks]) + creature.positioning])
        neg_ext = abs(np.min(all_positions, axis=0))  # extensions to negative
        new_positions = all_positions + neg_ext
        pos_ext = np.max(all_positions, axis=0) - (np.array(self.shape) - 1)  # extensions to positive
        self.new_image = np.zeros((np.array(self.shape) + neg_ext + pos_ext))
        # TODO: get density from brick.density and don't assume uniform
        for placement_coord in new_positions:
            self.new_image[tuple(placement_coord)] = 1

    def create_subpixel_resolution_array(self, subpixels):

        if self.new_image is None:
            self.subpixel_resolution_array = self.array.repeat(subpixels, axis=0).repeat(subpixels, axis=1).repeat(
                subpixels, axis=2).astype(np.float32)
        else:
            self.subpixel_resolution_array = self.new_image.repeat(subpixels, axis=0).repeat(subpixels, axis=1).repeat(
                subpixels, axis=2).astype(np.float32)

    def save_as_mrcfile(self, output_path: Path):
        mrc = mrcfile.new(output_path, overwrite=True)
        if self.subpixel_resolution_array is None:
            mrc.set_data(self.array)
        else:
            mrc.set_data(self.subpixel_resolution_array)
        mrc.close()

    def display_in_napari(self):
        if self.subpixel_resolution_array is None:
            napari.view_image(self.array)
        else:
            napari.view_image(self.subpixel_resolution_array)
