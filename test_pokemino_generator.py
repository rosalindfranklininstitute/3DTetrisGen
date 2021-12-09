import random

import pytest
import numpy as np

import pokemino_generator as poke


@pytest.fixture
def volume2D():
    volume2D = poke.Volume((20, 20))
    return volume2D


@pytest.fixture
def ex_handpicked_poke2D(volume2D):
    ex_poke2D = poke.Pokemino2D(seed="ivysaur",
                                size=5,
                                volume=volume2D,
                                positioning="central",
                                density=1,
                                algorithm=False,
                                brick_pos_list=[[1, 1], [2, 1], [2, 2], [2, 3], [3, 1]])
    return ex_poke2D


@pytest.fixture
def ex_clumped_poke2D(volume2D):
    ex_poke2D = poke.Pokemino2D(seed="ivysaur",
                                size=5,
                                volume=volume2D,
                                positioning="central",
                                density=1,
                                algorithm='clumped')
    return ex_poke2D


@pytest.fixture
def ex_extended_poke2D(volume2D):
    ex_poke2D = poke.Pokemino2D(seed="ivysaur",
                                size=5,
                                volume=volume2D,
                                positioning="central",
                                density=1,
                                algorithm='extended')
    return ex_poke2D


@pytest.fixture
def volume3D():
    volume3D = poke.Volume((20, 20, 20))
    return volume3D


@pytest.fixture
def ex_handpicked_poke3D(volume3D):
    ex_poke3D = poke.Pokemino3D(seed="ivysaur",
                                size=5,
                                volume=volume3D,
                                positioning="central",
                                density=1,
                                algorithm=False,
                                brick_pos_list=[[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2], [1, 1, 3]])
    return ex_poke3D


@pytest.fixture
def ex_clumped_poke3D(volume3D):
    ex_poke3D = poke.Pokemino3D(seed="ivysaur",
                                size=5,
                                volume=volume3D,
                                positioning="central",
                                density=1,
                                algorithm='clumped')
    return ex_poke3D


@pytest.fixture
def ex_extended_poke3D(volume3D):
    ex_poke3D = poke.Pokemino3D(seed="ivysaur",
                                size=5,
                                volume=volume3D,
                                positioning="central",
                                density=1,
                                algorithm='extended')
    return ex_poke3D


def test_creating_Pokemino2D_by_specifying_coords(ex_handpicked_poke2D):
    """Tests if Pokemino3D clumped algorithm reliably returns the same brick_coords in 2D
           for given seed and size."""
    poke2D = ex_handpicked_poke2D
    assert poke2D.size == 5 and poke2D.dim == 2


def test_creating_a_clumped_Pokemino2D(ex_clumped_poke2D):
    """Tests if the clumped algorithm reliably returns the same brick_coords in 2D
       for given seed and size."""
    poke2D = ex_clumped_poke2D
    assert poke2D.brick_coords == [[0, 0], [-1, 0], [1, 0], [0, -1], [-1, 1]]


def test_creating_an_extended_Pokemino2D(ex_extended_poke2D):
    """Tests if the clumped algorithm reliably returns the same brick_coords in 2D
       for given seed and size."""
    poke2D = ex_extended_poke2D
    assert poke2D.brick_coords == [[1, 0], [0, 0], [0, 1], [1, -1], [0, -1]]


def test_creating_Pokemino3D_by_specifying_coords(ex_handpicked_poke3D):
    poke3D = ex_handpicked_poke3D
    assert poke3D.size == 5 and poke3D.dim == 3


def test_creating_a_clumped_Pokemino3D(ex_clumped_poke3D):
    """Tests if the clumped algorithm reliably returns the same brick_coords in 3D
       for given seed and size."""
    poke3D = ex_clumped_poke3D
    assert poke3D.brick_coords == [[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [-1, 1, 0]]


def test_creating_a_extended_Pokemino3D(ex_extended_poke3D):
    """Tests if the clumped algorithm reliably returns the same brick_coords in 3D
       for given seed and size."""
    poke3D = ex_extended_poke3D
    assert poke3D.brick_coords == [[1, 0, 0], [0, 0, 0], [-1, 0, 0], [1, -1, 0], [-1, 1, 0]]


def test_finding_max_radius_of_Pokemino2D():
    assert poke.Pokemino.find_max_radius([[-1, -1], [0, -1], [0, 0], [0, 1], [1, -1]]) == 2 ** (0.5)


def test_finding_max_radius_of_Pokemino3D():
    assert poke.Pokemino.find_max_radius([[-1, -1, 0], [0, -1, 0], [0, 0, 0], [0, 1, 0], [1, -1, 1]]) == 3 ** (0.5)


def test_creation_of_poke_array_for_Pokemino2D(ex_handpicked_poke2D):
    poke2D = ex_handpicked_poke2D
    assert np.all(poke2D.poke_array == np.array([[1., 0., 0.],
                                                 [1., 1., 1.],
                                                 [1., 0., 0.]]))


def test_creation_of_poke_array_for_Pokemino3D(ex_handpicked_poke3D):
    poke3D = ex_handpicked_poke3D
    assert np.all(poke3D.poke_array == np.array([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                                 [[0., 0., 0.], [1., 1., 1.], [1., 0., 0.]],
                                                 [[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]]))


def test_upscaling_Pokemino2D(ex_handpicked_poke2D):
    poke2D = ex_handpicked_poke2D
    poke2D.upscale_pokemino(2)
    assert np.all(poke2D.poke_array == np.array([[1., 1., 0., 0., 0., 0.],
                                                 [1., 1., 0., 0., 0., 0.],
                                                 [1., 1., 1., 1., 1., 1.],
                                                 [1., 1., 1., 1., 1., 1.],
                                                 [1., 1., 0., 0., 0., 0.],
                                                 [1., 1., 0., 0., 0., 0.]]))


def test_upscaling_Pokemino3D(ex_handpicked_poke3D):
    poke3D = ex_handpicked_poke3D
    poke3D.upscale_pokemino(2)
    assert np.all(poke3D.poke_array == np.array([[[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]],
                                                 [[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]],
                                                 [[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.],
                                                  [1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1.],
                                                  [1., 1., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0.]],
                                                 [[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.],
                                                  [1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1.],
                                                  [1., 1., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0.]],
                                                 [[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.],
                                                  [1., 1., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]],
                                                 [[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.],
                                                  [1., 1., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]]]))


def test_translating_Pokemino2D(ex_handpicked_poke2D):
    poke2D = ex_handpicked_poke2D
    initial_position = poke2D.positioning
    poke2D.move_pokemino_in_volume([1, 2])
    final_position = np.array(poke2D.positioning)
    assert list(final_position - initial_position) == [1, 2]


def test_translating_Pokemino3D(ex_handpicked_poke3D):
    poke3D = ex_handpicked_poke3D
    initial_position = poke3D.positioning
    poke3D.move_pokemino_in_volume([1, 2, 3])
    final_position = np.array(poke3D.positioning)
    assert list(final_position - initial_position) == [1, 2, 3]


def test_rotating_Pokemino2D_by_specified_angle(ex_handpicked_poke2D):
    poke2D = ex_handpicked_poke2D
    poke2D.rotate_the_pokemino(theta=30, order=1)
    assert np.all(np.around(poke2D.poke_array, 5) == np.around([[0.       , 0.       , 0.       , 0.       ],
                                                               [0.       , 0.4419873, 0.8169873, 0.       ],
                                                               [0.       , 0.9419873, 0.3169873, 0.       ],
                                                               [0.       , 0.       , 0.       , 0.       ]], 5))

def test_rotating_Pokemino3D_by_specified_angle_around_1_axis(ex_handpicked_poke3D):
    poke3D = ex_handpicked_poke3D
    poke3D.rotate_the_pokemino_1_axis(axes=(0, 1), theta=30, order=1)
    assert np.all(np.around(poke3D.poke_array, 5) == np.around([[[0.       , 0.       , 0.       ],
                                                                 [0.       , 0.       , 0.       ],
                                                                 [0.       , 0.       , 0.       ],
                                                                 [0.       , 0.       , 0.       ]],

                                                                [[0.       , 0.       , 0.       ],
                                                                 [0.2589746, 0.2589746, 0.2589746],
                                                                 [0.8169873, 0.2589746, 0.2589746],
                                                                 [0.       , 0.       , 0.       ]],

                                                                [[0.       , 0.       , 0.       ],
                                                                 [0.3169873, 0.2589746, 0.2589746],
                                                                 [0.875    , 0.2589746, 0.2589746],
                                                                 [0.       , 0.       , 0.       ]],

                                                                [[0.       , 0.       , 0.       ],
                                                                 [0.       , 0.       , 0.       ],
                                                                 [0.       , 0.       , 0.       ],
                                                                 [0.       , 0.       , 0.       ]]], 5))


def test_rotating_Pokemino3D_by_random_angle_around_1_axis(ex_handpicked_poke3D):
    poke3D = ex_handpicked_poke3D
    poke3D.rotate_the_pokemino_1_axis(axes=(0, 1), order=1, s=0)
    assert np.all(np.around(poke3D.poke_array, 5) == np.around([[[0.        , 0.        , 0.        ],
                                                                [0.86218132, 0.00837751, 0.00837751],
                                                                [0.        , 0.        , 0.        ]],

                                                               [[0.8608269 , 0.00837751, 0.00837751],
                                                                [1.        , 1.        , 1.        ],
                                                                [0.00973193, 0.00837751, 0.00837751]],

                                                               [[0.        , 0.        , 0.        ],
                                                                [0.00837751, 0.00837751, 0.00837751],
                                                                [0.        , 0.        , 0.        ]]], 5))
@pytest.mark.parametrize("positioning",
                         [(10, 10), (-1, 1), (21, 20)],
                         ids=['central_positioning', 'indices_negative', 'indices_out_of_range'])
def test_fitting_Pokemino2D(ex_handpicked_poke2D, volume2D, positioning):
    poke2D = ex_handpicked_poke2D
    poke2D.positioning = positioning
    volume = volume2D
    volume2D.fit_all_pokeminos()


@pytest.mark.parametrize("positioning",
                         [(10, 10, 10), (-1, 1, 0), (21, 20, 22)],
                         ids=['central_positioning', 'indices_negative', 'indices_out_of_range'])
def test_fitting_Pokemino3D(ex_handpicked_poke3D, volume3D, positioning):
    poke3D = ex_handpicked_poke3D
    poke3D.positioning = positioning
    volume = volume3D
    volume3D.fit_all_pokeminos()


@pytest.mark.parametrize("pos1, pos2",
                         [(np.array([10, 10, 10]), np.array([10, 10, 10])),
                           (np.array([9, 10, 10]), np.array([10, 10, 10])),
                           (np.array([5, 5,  5]) , np.array([15, 15, 15]))],
                         ids=['same_positioning', 'overlapping_different_positioning', 'not_overlapping'])
def test_moving_overlapping_Pokeminos3D_apart(ex_clumped_poke3D, ex_extended_poke3D, volume3D, pos1, pos2):
    volume = volume3D
    pokeminko1 = ex_clumped_poke3D
    pokeminko1.positioning = pos1
    pokeminko2 = ex_extended_poke3D
    pokeminko2.positioning = pos2
    volume.check_for_pairwise_overlap()
    volume.fit_all_pokeminos()


@pytest.mark.parametrize("pos, expected_volume_shape",
                         [(np.array([10, 10, 10]), (20, 20, 20)),
                           (np.array([21, 19, 20]),  (23, 21, 22)),
                           (np.array([-2, 0, 3]) , (23, 21, 20))],
                         ids=['no_extension_needed', 'positive_extensions', 'negative_extensions'])
def test_extending_volume_to_remove_cuts(ex_clumped_poke3D, volume3D, pos, expected_volume_shape):
    volume = volume3D
    pokeminko = ex_clumped_poke3D
    pokeminko.positioning = pos
    volume.fit_all_pokeminos()
    assert volume.array.shape == expected_volume_shape
