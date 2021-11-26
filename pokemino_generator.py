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

THETA_EACH_90 = [0, 90, 180, 270]


class Brick:

    def __init__(self, pos, density):
        self.pos = pos
        self.density = density

    def __repr__(self):
        return f'{self.__class__.__name__}({self.pos})'


class Pokemino:

    def __init__(self, seed, size, volume, dim, positioning='central', density=1, algorithm=False, brick_pos_list=None):

        self.density = density
        self.size = size
        self.seed = seed
        self.dim = dim
        self.bricks = np.empty(size, dtype=np.object)
        self.n_bricks = 0
        self.density = 1
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

        # Find the Euclidean distance to the block furthest from the centre of mass
        all_positions = np.array([brick.pos for brick in self.bricks])
        self.max_radius = np.sqrt(np.max(np.sum(all_positions ** 2, axis=1)))

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

        brick_positions = []
        for brick in self.bricks:
            brick_positions.append(brick.pos)

    def upscale_pokemino (self, scale_factor):

        self.n_bricks=0
        brick_positions = []
        for brick in self.bricks:
            brick_positions.append(brick.pos)

        new_n_bricks = self.size * scale_factor ** self.dim
        self.bricks = np.empty(new_n_bricks, dtype=np.object)

        for (x0, y0, z0) in brick_positions:
            for x in range(scale_factor):
                for y in range(scale_factor):
                    for z in range(scale_factor):
                        self.bricks[self.n_bricks] = Brick([x0+x, y0+y, z0+z], self.density)
                        self.n_bricks += 1

        self.size = new_n_bricks

    def move_pokemino_in_volume(self, vector):
        self.positioning = self.positioning + np.array(vector)

    def visualise_in_napari(self, display_window_size):

        display_window = np.zeros(tuple([display_window_size] * self.dim))
        central_pos = tuple(map(lambda x: round(x / 2), display_window.shape))
        for brick in self.bricks:
            display_window[tuple(x + y for (x, y) in zip(central_pos, brick.pos))] = 1
        napari.view_image(display_window)


class Pokemino2D(Pokemino):

    def __init__(self, seed, size, volume, dim=2, positioning="central", density=1, algorithm="clumped",
                 brick_pos_list=None):
        self.dim = dim
        super().__init__(seed, size, volume, dim, positioning, density, algorithm, brick_pos_list)
        random.seed()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.seed, self.size})'

    def rotate_the_block(self, theta=None, randomise=True):

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

    def __init__(self, seed, size, volume, dim=3, positioning="central", density=1, algorithm="clumped",
                 brick_pos_list=None):
        self.dim = 3
        super().__init__(seed, size, volume, dim, positioning, density, algorithm, brick_pos_list)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.seed, self.size})'

    def rotate_the_block_1_axis(self, axis=None, theta=None, randomise=True):

        assert (axis in range(0, 3)), 'Error: Pokemino3D.rotate_the_block requires to specify the axis as 0, 1, or 2.'

        if randomise:
            random.seed()
            theta = random.choice(THETA_EACH_90) / 180 * np.pi
        else:
            assert (isinstance(theta, (
                int, float))), 'Error: Pokemino3D.rotate_the_block requires the value for theta that is int or float.'
            theta = theta / 180 * np.pi

        c, s = np.cos(theta), np.sin(theta)

        if axis == 0:
            R = np.array(((1, 0, 0), (0, c, -s), (0, s, c)))
        elif axis == 1:
            R = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))
        elif axis == 2:
            R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

        for brick in self.bricks:
            brick.pos = np.rint(np.matmul(R, brick.pos)).astype(int)

        for i, pos in enumerate(self.excluded_pos):
            self.excluded_pos[i] = np.rint(np.matmul(R, np.array(pos))).astype(int)

    def rotate_the_block_3_axes(self):

        self.rotate_the_block_1_axis(axis=0)
        self.rotate_the_block_1_axis(axis=1)
        self.rotate_the_block_1_axis(axis=2)


class Volume:

    def __init__(self, shape):
        self.shape = shape
        self.array = np.zeros(shape, dtype=np.int8)
        self.n_creatures = 0
        self.creatures = np.empty(0, dtype=np.object)
        self.subpixel_resolution_array = None
        self.new_image = None

    def fit_pokemino(self, pokemino):
        for brick in pokemino.bricks:
            placement_coords = tuple(x + y for (x, y) in zip(pokemino.positioning, brick.pos))
            if all([(coord >= 0) and (coord < self.shape[i]) for i, coord in enumerate(placement_coords)]):
                self.array[placement_coords] = brick.density

    def fit_excluded_volume(self, pokemino):
        for excluded_voxel in pokemino.excluded_pos:
            placement_coords = tuple(x + y for (x, y) in zip(pokemino.positioning, excluded_voxel))
            if all([(coord >= 0) and (coord < self.shape[i]) for i, coord in enumerate(placement_coords)]):
                self.array[placement_coords] = 0.5  # called padding (adding a padding around)

    def check_for_overlap (self, pokemino1, pokemino2):

        distance_between_centres_of_mass = np.sum(np.square(pokemino1.positioning - pokemino2.positioning))
        sum_of_max_radii = (np.sum(np.square(pokemino1.max_radius)) + np.sum(np.square(pokemino2.max_radius))).astype(
            int)

        if sum_of_max_radii >= distance_between_centres_of_mass:

            occupied_coords_pokemino1 = []
            for brick in pokemino1.bricks:
                occupied_coords_pokemino1.append([x + y for (x, y) in zip(pokemino1.positioning, brick.pos)])
            for excluded_pos in pokemino1.excluded_pos:
                occupied_coords_pokemino1.append([x + y for (x, y) in zip(pokemino1.positioning, excluded_pos)])

            occupied_coords_pokemino2 = []
            for brick in pokemino2.bricks:
                occupied_coords_pokemino2.append([x + y for (x, y) in zip(pokemino2.positioning, brick.pos)])

            for coords in occupied_coords_pokemino2:
                if coords in occupied_coords_pokemino1:
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

        print("Before moving apart:", pokemino1.positioning, np.array(pokemino2.positioning))

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
            pokemino_to_move.move_pokemino_in_volume(vector)
            i += 1

        print("After moving apart:", pokemino1.positioning, pokemino2.positioning)

    def fit_all_pokeminos(self, fit_excluded_volume=False):
        if fit_excluded_volume:
            for pokemino in self.creatures:
                self.fit_excluded_volume(pokemino)

        for pokemino in self.creatures:
            self.fit_pokemino(pokemino)

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
                subpixels, axis=2).astype(np.int8)
        else:
            self.subpixel_resolution_array = self.new_image.repeat(subpixels, axis=0).repeat(subpixels, axis=1).repeat(
                subpixels, axis=2).astype(np.int8)

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
