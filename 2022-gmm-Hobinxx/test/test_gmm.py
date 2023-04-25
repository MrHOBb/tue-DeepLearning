import sys, os
from importing import NotebookFinder
sys.meta_path.append(NotebookFinder())
import assignment_gmm as assignment
import numpy as np
import pytest

@pytest.fixture
def data_generation():
    # Create sample data
    np.random.seed(30)

    n_clusters = 10
    n_samples = 500-1 # with seed 30 generate_dataset returns 499 samples when 500 requested
    
    samples = assignment.generate_dataset(n_samples, n_clusters)

    # Set the mean for each cluster equal to the position of a random sample
    rand_idx = np.random.randint(n_samples, size=(n_clusters))
    mean = samples[:, rand_idx].T

    # Initialize the covariance matrix
    covariance = 10 * np.tile(np.eye(2, 2), (n_clusters, 1, 1))

    # Initialize the priors
    priors = np.ones((n_clusters, 1))/n_clusters

    # Initialize the membership probabilities
    membership_prob = np.ones((n_clusters, n_samples))*(1/n_clusters)

    return  [samples, mean, covariance, priors, membership_prob, n_clusters, n_samples]


@pytest.fixture
def data_generation_classification():
    # Create sample data
    np.random.seed(30)
    
    # How many clusters per class
    n_clusters = 10

    # Generate a random dataset
    samples_c1, samples_c2 = assignment.generate_classification_dataset()
    n_samples = samples_c1.shape[1]

    # Choose initial values
    rand_idx = np.random.randint(n_samples, size=(n_clusters))

    # For class 1
    mean_c1 = samples_c1[:, rand_idx].T
    covariance_c1 = 2 * np.tile(np.eye(2, 2), (n_clusters, 1, 1))
    priors_c1 = np.ones((n_clusters, 1))/n_clusters
    membership_prob_c1 = np.concatenate((np.ones((1, n_samples)), np.zeros((n_clusters-1, n_samples))))

    # For class 2
    mean_c2 = samples_c2[:, rand_idx].T
    covariance_c2 = 2 * np.tile(np.eye(2, 2), (n_clusters, 1, 1))
    priors_c2 = np.ones((n_clusters, 1))/n_clusters
    membership_prob_c2 = np.concatenate((np.ones((1, n_samples)), np.zeros((n_clusters-1, n_samples))))

    return  [samples_c1, mean_c1, covariance_c1, priors_c1, membership_prob_c1, \
        samples_c2, mean_c2, covariance_c2, priors_c2, membership_prob_c2, n_clusters, n_samples]


def test_expectation_step(data_generation):
    samples, mean, covariance, priors, membership_prob, n_clusters, _ = data_generation
    print(samples.shape)
    # Extract implementation from student solution
    answer = assignment.expectation(samples, mean, covariance, priors, membership_prob, n_clusters)
    
    # Load correct solution
    solution = np.load('test/expectation_solution.npy')

    assert np.allclose(answer, solution), "Expectation not calculated correctly"


def test_prior_calculation(data_generation):
    _, _, _, priors, membership_prob, n_clusters, n_samples = data_generation

    # Extract implementation from student solution
    answer = assignment.compute_priors(priors, membership_prob, n_clusters, n_samples)

    # Load correct solution
    solution = np.load('test/prior_solution.npy') 
    
    assert np.allclose(answer, solution), "Priors not calculated correctly"


def test_mean_calculation(data_generation):
    samples, mean, _, priors, membership_prob, n_clusters, n_samples = data_generation

    # Extract implementation from student solution
    answer = assignment.compute_mean(samples, mean, priors, membership_prob, n_clusters, n_samples)
    
    # Load correct solution
    solution = np.load('test/mean_solution.npy')
    
    assert np.allclose(answer, solution), "Mean not calculated correctly"


def test_covariance_calculation(data_generation):
    samples, mean, covariance, priors, membership_prob, n_clusters, n_samples = data_generation

    # Extract implementation from student solution
    answer = assignment.compute_covariance(samples, mean, covariance, priors, membership_prob, n_clusters, n_samples)
    
    # Load correct solution
    solution = np.load('test/covariance_solution.npy')
    
    assert np.allclose(answer, solution), "Covariance not calculated correctly"


def test_expectation_maximization(data_generation):
    samples, mean, covariance, priors, membership_prob, n_clusters, n_samples = data_generation

    # Extract implementation from student solution
    for i in range(10):
        covariance, priors, mean, membership_prob = \
            assignment.expectation_maximization(samples, mean, covariance, priors, membership_prob, n_clusters, n_samples)
    
    # Load correct solution
    solution = np.load('test/EM_solution.npz')

    assert np.allclose(covariance, solution['covariance']), "Covariance not calculated correctly"
    assert np.allclose(priors, solution['priors']), "Priors not calculated correctly"
    assert np.allclose(mean, solution['mean']), "Mean not calculated correctly"
    assert np.allclose(membership_prob, solution['membership_prob']), "Membership probability not calculated correctly"


def test_classification(data_generation_classification):
    samples_c1, mean_c1, covariance_c1, priors_c1, membership_prob_c1, \
        samples_c2, mean_c2, covariance_c2, priors_c2, membership_prob_c2, n_clusters, n_samples = data_generation_classification

    # Extract implementation from student solution
    for i in range(30):
        covariance_c1, priors_c1, mean_c1, membership_prob_c1 = \
            assignment.expectation_maximization(samples_c1, mean_c1, covariance_c1, priors_c1, membership_prob_c1, n_clusters, n_samples)
        covariance_c2, priors_c2, mean_c2, membership_prob_c2 = \
            assignment.expectation_maximization(samples_c2, mean_c2, covariance_c2, priors_c2, membership_prob_c2, n_clusters, n_samples)
    
    x, y = np.meshgrid(np.arange(-15, 10, 0.25), np.arange(-8, 12, 0.25))
    x = x.flatten()
    y = y.flatten()
    samples = np.array([x, y])

    n_samples = samples.shape[1]

    w_c1 = np.zeros((n_clusters, n_samples))
    w_c2 = np.zeros((n_clusters, n_samples))

    # Extract implementation from student solution
    w_c1 = assignment.compute_probability(samples, mean_c1, covariance_c1, priors_c1, w_c1, n_clusters)
    w_c2 = assignment.compute_probability(samples, mean_c2, covariance_c2, priors_c2, w_c2, n_clusters)

    # Load correct solution
    solution = np.load('test/Classification_solution.npz')

    assert np.allclose(w_c1, solution['w_c1'])
    assert np.allclose(w_c2, solution['w_c2'])
