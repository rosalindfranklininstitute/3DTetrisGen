#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:25:48 2021

@author: anna
"""

import random
import numpy as np
from pathlib import Path
from icecream import ic
from scipy import ndimage
from itertools import product

import napari
import mrcfile
from tqdm import tqdm

class Brick:

    def __init__(self, pos, density=1, stick_prob=np.inf):
        self.pos = pos
        self.density = density
        self.stick_prob = stick_prob

    def __repr__(self):
        return f'{self.__class__.__name__}({self.pos})'


class Pokemino:

    def __init__(self,
                 size,
                 volume,
                 dim,
                 seed,
                 crowd_factor,
                 target_ratio,
                 positioning,
                 algorithm,
                 brick_coords,
                 scale_factor,
                 density,
                 max_ratio):

        """
        Initialising a Pokemino object

        Args:
        size (int)           :: Target number of blocks
        volume (Volume)      :: Volume in which the Pokemino will live in
        dim (n)              :: Dimensionality of the Pokemino (2 or 3)
        seed (str)           :: Random seed for the algorithms generating Pokeminos (Pokemon name)
        crowd_factor (float) :: Factor determining how crowded the core of cluster would be
        target_ratio (list)  :: Extended ratio of sizes in x, y and z directions
        positioning (list)   :: Set of coordinates at which the Pokemino will be placed or "central" for fitting at volume centre
        algorithm            :: 'biased' when using Neville's biased generator or False when providing brick_coords
        brick_coords (list)  :: list of brick coordinates if not using biased algorithm
        scale_factor (int)   :: a scale factor for upscaling user-defined pokemino coords
        density (int)        :: at the moment always == 1
        max_ratio (int)      ::
        """

        self.size = size
        self.volume = volume
        self.dim = dim # inherited from Pokemino2D or Pokemino3D
        self.cf = crowd_factor ** 2.5
        self.density = density
        self.algorithm = algorithm
        self.scale_factor = scale_factor

        if self.algorithm == "biased":

            # Set seeds for random and numpy modules
            self.seed = 'pikachu' if seed is None else seed
            self._np_seed = np.dot(np.array([ord(x) for x in self.seed]), np.arange(len(self.seed)))
            random.seed(self.seed)
            self._np_rng = np.random.default_rng(self._np_seed)

            self.max_ratio = 50 if max_ratio is None else max_ratio

            self.bricks = list([Brick((0, 0, 0)),
                                Brick((1, 0, 0)),
                                Brick((0, 1, 0)),
                                Brick((0, 0, 1))])
            self.neighbours = list([Brick((1, 0, 1)),
                                    Brick((0, 1, 1)),
                                    Brick((0, 0, 2)),
                                    Brick((0, 0, -1)),
                                    Brick((0, 0, -1)),
                                    Brick((0, 0, -1)),
                                    Brick((-1, 0, 0)),
                                    Brick((-1, 1, 0)),
                                    Brick((-1, 0, 1)),
                                    Brick((0, -1, 0)),
                                    Brick((1, -1, 0)),
                                    Brick((0, -1, 1)),
                                    Brick((2, 0, 0)),
                                    Brick((0, 2, 0)),
                                    Brick((1, 1, 0))])

            if target_ratio is None:
                self.target_ratio = np.array([1., 1., 1.], dtype=float)
            self.target_ratio = np.array([self.target_ratio[1] / self.target_ratio[0],  # y/x
                                          self.target_ratio[2] / self.target_ratio[1],  # z/y
                                          self.target_ratio[2] / self.target_ratio[0],  # z/x
                                          self.target_ratio[0] / self.target_ratio[1],  # x/y
                                          self.target_ratio[1] / self.target_ratio[2],  # y/z
                                          self.target_ratio[0] / self.target_ratio[2],  # x/z
                                          ])

            for brick in tqdm(range(self.size)):
                # Pick a brick then generate a random number
                new_brick = random.choice(self.neighbours)
                new_value = self._np_rng.random()
                while new_value > new_brick.stick_prob:
                    new_brick = random.choice(self.neighbours)
                    new_value = self._np_rng.random()
                self._add_single_brick(new_brick.pos)

        elif not self.algorithm:
            assert len(brick_coords) == self.size, "Size declared not matched with len(brick_pos_list)!"
            assert all([isinstance(i, list) for i in brick_coords]), "Brick positions passed in an incorrect format!"
            assert all([len(i) == self.dim for i in
                        brick_coords]), "Brick coordinates don't match the declared dimensionality!"
            self.bricks = list()
            for brick_pos in brick_coords:
                upscaled_coords = list(product(*[[coord * scale_factor + i for i in range(scale_factor)] for coord in brick_pos]))
                for i in upscaled_coords:
                    self.bricks.append(Brick(i))

        self.make_coords_relative_to_centre_of_mass()

        self.max_radius = self.find_max_radius([brick.pos for brick in self.bricks])

        self.ratio = self._calc_ratios([x.pos for x in self.bricks], final=True)
        #self.colour = self._calc_colour()
        #self.fractal_dim = self._calc_fractal_dim([x.pos for x in self.bricks])

        #ic(self.ratio, self.colour)
        #ic(self.fractal_dim)

        if positioning == 'central':
            self.positioning = np.array(tuple(map(lambda x: int(round(x / 2)), volume.shape)))
        else:
            self.positioning = np.array(positioning)

        self.poke_array = np.zeros((2 * int(self.max_radius) + 1,) * self.dim)

        # TODO: deal with reading scaling_factor and upscaling a biased pokemino
        for brick in self.bricks:
            placement_coords = tuple(x + y for (x, y) in zip((int(self.max_radius),) * self.dim, brick.pos))
            self.poke_array[placement_coords] = brick.density

        volume.creatures = np.hstack([volume.creatures, self])
        volume.n_creatures += 1

    def _add_single_brick(self, coord):
        """
        Method to add a brick to a Pokemino object and update neighbour list
        """

        # Check if brick is available in neighbour list
        neighbours_coords = [b.pos for b in self.neighbours]
        assert (coord in neighbours_coords), \
            "Error in Pokemino.add_brick: Input coordinate not eligible."

        # If brick available in neighbour list, find its location and pop into bricks list
        for index, item in enumerate(self.neighbours):
            if item.pos == coord:
                nb_index = index

        self.bricks.append(self.neighbours.pop(nb_index))

        # Add its neighbours to the neighbour list
        if self.dim == 2:
            dirs = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
        elif self.dim == 3:
            dirs = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])

        brick_coords = [x.pos for x in self.bricks]
        my_neighbours_coords = [tuple(x) for x in list(coord) + dirs]
        neighbours_coords = list((set(neighbours_coords) | set(my_neighbours_coords)) - set(brick_coords))

        # Update neighbour brick objects
        self.neighbours = [Brick(x, stick_prob=self.cf) for x in neighbours_coords]

        # Update neighbour Brick object sticking probability
        for brick in self.neighbours:
            brick.stick_prob += self._calc_prob_change(brick_coords, brick.pos, self.target_ratio)

    def _calc_prob_change(self, curr_brick_coords, extra_brick, target_ratio):
        """
        Method to evaluate a Brick's sticking probability CHANGE

        Args:
        curr_brick_coords (list)   :: List storing current Brick coordinates
        extra_brick (tuple)   :: Coordinates of the extra Brick
        target_ratio (ndarray) :: Array storing target dimensionality ratio of cluster

        Returns:
        float
        """
        # Calculate L2-norm of distance between current ratio and target ratio
        curr_ratio = self._calc_ratios(curr_brick_coords)
        curr_dist = np.linalg.norm(curr_ratio - target_ratio)

        # Calculate L2-norm of distance between new ratio and target ratio
        new_ratio = self._calc_ratios([*curr_brick_coords, extra_brick])
        new_dist = np.linalg.norm(new_ratio - target_ratio)

        # Define probability change
        dist_change = new_dist ** 2 - curr_dist ** 2
        prob_change = 0.6 * np.log(len(curr_brick_coords)) * (np.exp(-np.sign(dist_change) * (np.abs(dist_change) ** 0.7) - 1))

        return prob_change

    @staticmethod
    def _calc_ratios(coords_list, final=False):
        """
        Method to calculate x-y-z ratios of Pokemino dimensions
        Args:
        coords_list (ndarray) :: Array storing Block coordinates
        Returns:
        ndarray
        """
        coord_list = np.array([list(x) for x in coords_list])

        evalues, evec = np.linalg.eig(np.cov(coord_list.T))

        # Swap the eigenvalues and eigenvectors if eigenvectors are represented in left-hand order
        if np.dot(np.cross(evec[0], evec[1]), evec[2]) < 0:
            evalues[1], evalues[2] = evalues[2], evalues[1]

        if not final:
            xyz_ratio = np.array([evalues[1]/evalues[0],
                                  evalues[2]/evalues[1],
                                  evalues[2]/evalues[0],
                                  evalues[0]/evalues[1],
                                  evalues[1]/evalues[2],
                                  evalues[0]/evalues[2]
            ])

        else:
            xyz_ratio = evalues / np.min(evalues)

        return xyz_ratio

    def _calc_colour(self):
        """
        Method to calculate colour values (HSV)
        Returns:
        ndarray
        """
        colour_H = 90 + 270 * self.ratio[1] / self.max_ratio
        colour_S = 1.
        colour_V = 0.5 + 0.5 * min(self.ratio[2] / self.max_ratio, 1)

        return np.array([colour_H, colour_S, colour_V])

    @staticmethod
    def _calc_fractal_dim(coords_list, start=5):
        """
        Method to estimate fractal dimension of a given ensemble
        Args:
        coords_list (list) :: List containing the coordinates of the constituent blocks
        start (int)        :: Starting point for estimating fractal dimensions
        Returns:
        float
        """
        coord_list = np.array([list(x) for x in coords_list])

        values_list = []
        for num_points in range(start, len(coord_list)):
            points = coord_list[:num_points]

            # Calculate radius of gyration
            centroid = np.mean(points, axis=0)
            dist_sq = np.linalg.norm(points - centroid, axis=0) ** 2
            r_gyration = np.sqrt(np.sum(dist_sq) / num_points)

            values_list.append([np.log(num_points), np.log(r_gyration)])

        values_list = np.array(values_list)
        fractal_dim = np.polyfit(x=np.array(values_list[:, 1]),
                                 y=np.array(values_list[:, 0]),
                                 deg=1)

        return fractal_dim[0]

    def make_coords_relative_to_centre_of_mass(self):

        weighted_coords = np.array([np.array(brick.pos) * brick.density for brick in self.bricks])
        com_pos = np.mean(weighted_coords, axis=0)
        new_com = self.bricks[np.argmin(np.sum(np.square(weighted_coords - com_pos), axis=1))].pos
        for i, brick in enumerate(self.bricks):
            brick.pos = [i - j for (i, j) in zip(brick.pos, new_com)]

    def find_max_radius(self, all_positions):
        """Finds the Euclidean distance to the brick furthest from the centre of mass"""
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
        napari.view_image(display_window_size)


class Pokemino2D(Pokemino):

    def __init__(self, size, volume, dim=2, seed=None, crowd_factor=0.5, target_ratio=None, positioning='central',
                 algorithm="biased", brick_coords=None, scale_factor=1, density=1, max_ratio=None):
        super().__init__(size, volume, dim, seed, crowd_factor, target_ratio, positioning, algorithm, brick_coords,
                         scale_factor, density, max_ratio)
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
                'Error: Pokemino3D.rotate_the_brick requires the value for theta in range <0, 360>.'
        self.poke_array = ndimage.rotate(self.poke_array, angle=theta, order=order, reshape=False)


class Pokemino3D(Pokemino):

    def __init__(self, size, volume, dim=3, seed=None, crowd_factor=0.5, target_ratio=None, positioning='central',
                 algorithm="biased", brick_coords=None, scale_factor=1, density=1, max_ratio=None):
        super().__init__(size, volume, dim, seed, crowd_factor, target_ratio, positioning, algorithm, brick_coords,
                         scale_factor, density, max_ratio)
        random.seed()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.seed, self.size})'

    def rotate_the_pokemino_1_axis(self, axes=(0, 1), theta=None, order=1, s=None):
        """Rotates the Pokemino using scipy.ndimage.rotate"""

        assert type(axes) == tuple \
               and len(axes) == 2 \
               and all([type(i) == int for i in axes]) \
               and all([i in range(0, 3) for i in axes]), "Incorrect axes parameter: pass a tuple of 2 axes."

        if not theta:
            np.random.seed(s)
            theta = np.random.choice([i for i in range(0, 360)])
        else:
            assert (isinstance(theta, int)) and theta in range(0, 360), \
                'Error: Pokemino3D.rotate_the_brick requires the value for theta in range <0, 360>.'
        self.poke_array = ndimage.rotate(self.poke_array, angle=theta, axes=axes, order=order, reshape=False)

        return theta
        # viewer = napari.view_image(self.poke_array)

    def rotate_the_pokemino_3_axes(self, theta_x=None, theta_y=None, theta_z=None):

        z_rot = self.rotate_the_pokemino_1_axis(axes=(1, 0), theta=theta_x)
        x_rot = self.rotate_the_pokemino_1_axis(axes=(2, 1), theta=theta_y)
        y_rot = self.rotate_the_pokemino_1_axis(axes=(0, 2), theta=theta_z)

        return x_rot, y_rot, z_rot


class Volume:

    def __init__(self, shape):
        self.dim = len(shape)
        self.shape = shape
        self.array = np.zeros(shape, dtype=np.float32)
        self.n_creatures = 0
        self.creatures = np.empty(0, dtype=np.object)

    def fit_pokemino(self, pokemino):

        """Fit pokemino.poke_array into volume (centred at pokemino.positioning)"""
        slices_boundaries = np.array([[i - a, i + b] for i, a, b in zip(pokemino.positioning,
                                                                        np.floor(np.array(pokemino.poke_array.shape) / 2).astype(np.int32),
                                                                        pokemino.poke_array.shape - np.floor(np.array(pokemino.poke_array.shape) / 2).astype(np.int32))])

        extensions = np.array([(min(0, i), max(0, j-(self.shape[0]-1))) for i, j in slices_boundaries])

        array_boundaries = np.array([[0, pokemino.poke_array.shape[i]] for i in range(3)])
        array_boundaries[:,1] -= 1
        slices_boundaries[:, 1] -= 1


        pst = [(i, j) for [i, j] in array_boundaries-extensions]
        to_pst = [(i, j) for [i, j] in slices_boundaries-extensions]

        #print(extensions)
        #print(pst, to_pst)

        try:
            self.array[to_pst[0][0]:to_pst[0][1], to_pst[1][0]:to_pst[1][1], to_pst[2][0]:to_pst[2][1]] += pokemino.poke_array[pst[0][0]:pst[0][1], pst[1][0]:pst[1][1], pst[2][0]:pst[2][1]]
        except ValueError:
            print("Something went wrong this time!")

    @staticmethod
    def check_for_overlap(pokemino1, pokemino2):
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
            top_left_corners[i] = poke.positioning - np.floor(np.array(poke.poke_array.shape) / 2).astype(np.int32)
            bottom_right_corners[i] = poke.positioning + poke.poke_array.shape - np.floor(
                np.array(poke.poke_array.shape) / 2).astype(np.int32) - 1

        """Find the lowest and highest coordinates occupied by any of poke_arrays in volume. """
        tl, br = np.min(top_left_corners, axis=0), np.max(bottom_right_corners, axis=0)

        """Extend volume.array in all dimensions in which Pokeminos stick out."""
        negative_extensions = np.where(tl < 0, -tl, 0).astype(np.int32)
        print("tl, br:", tl, br)
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
