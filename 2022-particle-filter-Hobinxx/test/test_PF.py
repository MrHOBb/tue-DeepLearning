import sys, os
from importing import find_notebook
from importing import NotebookLoader
from importing import NotebookFinder
# os.chdir('..')
sys.meta_path.append(NotebookFinder())
print("current meta path: ", sys.meta_path)
import ParticleFilter as PF

import pytest
import numpy as np

os.chdir(os.getcwd()+"/test")

with open('Matrices.npy', 'rb') as m:

    in_forward = np.load(m, )
    in_turn = np.load(m)
    in_mu = np.load(m)
    in_sigma_forward = np.load(m)
    in_sigma_turn = np.load(m)
    in_distance_forward = np.load(m)
    in_distance_turn = np.load(m)
    in_particle1 = np.load(m)
    in_particle2 = np.load(m)
    in_landmarks = np.load(m)
    in_measurements = np.load(m)
    in_sigma = np.load(m)
    in_distance_pl = np.load(m)
    in_probability = np.load(m)
    in_p_likelihood = np.load(m)
    in_i = np.load(m)
    in_w = np.load(m)
    in_particles = np.load(m)

    out_distance_turn = np.load(m)
    out_distance_forward = np.load(m)
    out_particle = np.load(m)
    out_distance_pl = np.load(m)
    out_probability = np.load(m)
    out_p_likelihood = np.load(m)
    out_particles = np.load(m)


test_distance_forward, test_distance_turn = PF.calculate_motion_distance(in_forward, in_turn, in_mu,
                                                                         in_sigma_forward, in_sigma_turn)

test_particle = PF.update_particle(in_particle1, in_distance_forward, in_distance_turn)


def test_calculate_motion_distance_1():
    # print("test_distance_forward: ", test_distance_forward)
    # print("out_distance_forward: ", out_distance_forward)
    assert np.allclose(test_distance_forward, out_distance_forward, rtol=1e-10, atol=1e-10) == True, "distance_forward is incorrect"


def test_calculate_motion_distance_2():
    # print("test_distance_forward: ", test_distance_turn)
    # print("out_distance_forward: ", out_distance_turn)
    assert np.allclose(test_distance_turn, out_distance_turn, rtol=1e-10, atol=1e-10) == True, "distance_turn is incorrect"


def test_update_particle_0():
    # print("test_particle[0]: ", test_particle[0])
    # print("out_particle[0]: ", out_particle[0])
    assert np.allclose(test_particle[0], out_particle[0], rtol=1e-10, atol=1e-10) == True, "particle[0] is incorrect"


def test_update_particle_1():
    # print("test_particle[1]: ", test_particle[1])
    # print("out_particle[1]: ", out_particle[1])
    assert np.allclose(test_particle[1], out_particle[1], rtol=1e-10, atol=1e-10) == True, "particle[1] is incorrect"


def test_update_particle_2():
    # print("test_particle[2]: ", test_particle[2])
    # print("out_particle[2]: ", out_particle[2])
    assert np.allclose(test_particle[2], out_particle[2], rtol=1e-10, atol=1e-10) == True, "particle[2] is incorrect"


def test_calculate_distance_pl():
    test_distance_pl = PF.calculate_distance_pl(in_particle2, in_landmarks, in_measurements, in_sigma, in_distance_pl, in_probability, in_p_likelihood, in_i)
    # print("test_distance_pl: ", test_distance_pl)
    # print("out_distance_pl: ", out_distance_pl)
    assert np.allclose(test_distance_pl, out_distance_pl, rtol=1e-10, atol=1e-10) == True, "distance_pl is incorrect"


def test_calculate_probability():
    test_probability = PF.calculate_probability(in_particle2, in_landmarks, in_measurements, in_sigma, in_distance_pl, in_probability, in_p_likelihood, in_i)
    # print("test_probability: ", test_probability)
    # print("out_probability: ", out_probability)
    assert np.allclose(test_probability, out_probability, rtol=1e-10, atol=1e-10) == True, "probability is incorrect"


def test_calculate_likelihood():
    test_p_likelihood = PF.calculate_likelihood(in_particle2, in_landmarks, in_measurements, in_sigma, in_distance_pl, in_probability, in_p_likelihood, in_i)
    # print("test_p_likelihood: ", test_p_likelihood)
    # print("out_p_likelihood: ", out_p_likelihood)
    assert np.allclose(test_p_likelihood, out_p_likelihood, rtol=1e-10, atol=1e-10) == True, "p_likelihood is incorrect"


def test_resampling():
    test_particles, test_w = PF.resampling(in_particles, in_w)
    # print("test_particles: ", test_particles)
    # print("out_particles: ", out_particles)
    assert np.allclose(test_particles, out_particles, rtol=1e-10, atol=1e-10) == True, "particles is incorrect"


pytest.main(["--tb=line"])
