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

STEPS_2D = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
STEPS_3D = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
THETA_EACH_90 = [0, 90, 180, 270]


class Brick:

    def __init__(self, pos, density):
        self.pos = pos
        self.density = density

    def __repr__(self):
        return f'{self.__class__.__name__}({self.pos})'

class Pokemino:

    def __init__(self, seed, size, volume, positioning='central', density=1, algorithm='clumped'):

        random.seed(seed)

        self.size = size
        self.seed = seed
        self.bricks = np.empty(size, dtype=np.object)
        self.bricks[0] = Brick([0] * self.dim, density)
        i = 1
        if positioning == 'central':
            self.positioning = tuple(map(lambda x: int(round(x / 2)), volume.shape))
        else:
            self.positioning = list(positioning)
        self.excluded_pos = []

        if self.dim == 2:
            STEPS = STEPS_2D
        elif self.dim == 3:
            STEPS = STEPS_3D

        if algorithm == 'clumped':
            while np.count_nonzero(self.bricks) < self.size:
                bricks_pos = [block.pos for block in self.bricks[self.bricks != None]]
                random_step = random.choice(STEPS)

                new_brick = list(random.choice(bricks_pos) + random_step)

                if new_brick not in bricks_pos:
                    self.bricks[i] = Brick(new_brick, density)
                    i += 1

        elif algorithm == 'extended':
            while np.count_nonzero(self.bricks) < self.size:
                bricks_pos = [block.pos for block in self.bricks[self.bricks != None]]
                excluded_pos = []
                for brick in bricks_pos:
                    for step in STEPS:
                        if tuple(np.array(brick) + np.array(step)) not in [bricks_pos, excluded_pos]:
                                excluded_pos.append(tuple(np.array(brick) + np.array(step)))
                next_brick = random.choice(excluded_pos)
                self.bricks[i] = Brick(next_brick, density)
                i += 1

        # Correct the coordinates to be relative to the centre of mass
        weighted_coords = np.zeros((self.size, self.dim))
        for i, brick in enumerate(self.bricks):
            weighted_coords[i] = [coord * brick.density for coord in brick.pos]
        com_pos = np.mean(weighted_coords, axis=0)
        new_com = self.bricks[np.argmin(np.sum(np.square(weighted_coords - com_pos), axis=1))].pos
        for i, brick in enumerate(self.bricks):
            brick.pos = [i - j for (i, j) in zip(brick.pos, new_com)]

        brick_positions = []
        for brick in self.bricks:
            brick_positions.append(brick.pos)

        for brick in brick_positions:
            for step in STEPS:
                if list(np.array(brick) + np.array(step)) not in brick_positions:
                    if tuple(np.array(brick) + np.array(step)) not in self.excluded_pos:
                        self.excluded_pos.append(tuple(np.array(brick) + np.array(step)))

        volume.creatures = np.hstack([volume.creatures, self])
        volume.n_creatures += 1

    def visualise_in_napari(self, display_window_size):

        display_window = np.zeros(tuple([display_window_size] * self.dim))
        central_pos = tuple(map(lambda x: round(x / 2), display_window.shape))
        for brick in self.bricks:
            display_window[tuple(x + y for (x, y) in zip(central_pos, brick.pos))] = 1
        napari.view_image(display_window)


class Pokemino2D(Pokemino):

    def __init__(self, seed, size, volume, positioning="central", density=1, algorithm="clumped"):
        self.dim = 2
        super().__init__(seed, size, volume, positioning, density, algorithm)
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

    def __init__(self, seed, size, volume, positioning="central", density=1, algorithm="clumped"):
        self.dim = 3
        super().__init__(seed, size, volume, positioning, density, algorithm)

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

    def check_for_overlap(self, pokemino1, pokemino2):
        occupied_coords_pokemino1 = []
        for brick in pokemino1.bricks:
            occupied_coords_pokemino1.append([x + y for (x, y) in zip(pokemino1.positioning, brick.pos)])
        for excluded_pos in pokemino1.excluded_pos:
            occupied_coords_pokemino1.append([x + y for (x, y) in zip(pokemino1.positioning, excluded_pos)])

        occupied_coords_pokemino2 = []
        for brick in pokemino2.bricks:
            occupied_coords_pokemino2.append([x + y for (x, y) in zip(pokemino2.positioning, brick.pos)])
        for excluded_pos in pokemino2.excluded_pos:
            occupied_coords_pokemino2.append([x + y for (x, y) in zip(pokemino2.positioning, excluded_pos)])

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

        # print ("Before moving apart:", np.array(pokemino1.positioning), np.array(pokemino2.positioning))

        for i, coord in enumerate(np.array(pokemino1.positioning) - np.array(pokemino2.positioning)):
            empty = np.array([0] * pokemino1.dim)
            if coord != 0:
                empty[i] = coord / abs(coord)
            lottery_machine.extend([empty] * abs(coord))

        random.shuffle(lottery_machine)

        i = 0
        while self.check_for_overlap(pokemino1, pokemino2):
            pokemino1.positioning = tuple(np.array(pokemino1.positioning) + lottery_machine[i % len(lottery_machine)])
            i += 1

        # print ("After moving apart:", np.array(pokemino1.positioning), np.array(pokemino2.positioning))

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
            self.subpixel_resolution_array = self.array.repeat(subpixels, axis=0).repeat(subpixels, axis=1).repeat(subpixels, axis=2).astype(np.int8)
        else:
            self.subpixel_resolution_array = self.new_image.repeat(subpixels, axis=0).repeat(subpixels, axis=1).repeat(subpixels, axis=2).astype(np.int8)

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
