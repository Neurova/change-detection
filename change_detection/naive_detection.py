
import jax
from jax import jit
from jax.tree_util import register_pytree_node_class, tree_flatten, tree_unflatten
import jax.numpy as jnp
import numpy as np

import numpyro
import numpyro.distributions as dist

from typing import Union


@register_pytree_node_class
class NaiveDetection:

    """
    A class for performing naive detection of change points in data using Bayesian inference.

    Parameters:
    x (Union[jnp.ndarray, np.ndarray, list]): The input data as a 1D numpy array, JAX array, or list representing time points.
    y (Union[jnp.ndarray, np.ndarray, list]): The observed data as a 1D numpy array, JAX array, or list.

    Attributes:
    x (Union[jnp.ndarray, np.ndarray, list]): The input data as a 1D numpy array, JAX array, or list representing time points.
    y (Union[jnp.ndarray, np.ndarray, list]): The observed data as a 1D numpy array, JAX array, or list.
    samples (dict): Dictionary to store posterior samples from Bayesian inference.
    N (int): The number of data points in the input data.
    iter (int): The number of iterations for the inference.
    prior_mu (numpyro.distributions.Distribution): Prior distribution for mean values.
    prior_var (numpyro.distributions.Distribution): Prior distribution for standard deviations.
    pred (numpy.ndarray): Array to store the predicted values.
    upper (numpy.ndarray): Array to store upper bounds of prediction intervals.
    lower (numpy.ndarray): Array to store lower bounds of prediction intervals.

    Methods:
    model(x, y):
        Define the Bayesian model for inference.

    run_inference(iter, warmup, rng_key):
        Run Bayesian inference using MCMC to obtain posterior samples.

    mean_pred(ix):
        Calculate the mean prediction for a given index.

    var_pred(ix):
        Calculate the variance prediction for a given index.

    predict():
        Perform predictions for change points and store results in pred, upper, and lower arrays.
    """
        
    def __init__(
            self, 
            x : Union[jnp.ndarray, np.ndarray, list], 
            y : Union[jnp.ndarray, np.ndarray, list],
            prior_mu : numpyro.distributions.Distribution, 
            prior_var : numpyro.distributions.Distribution,
            iterations: int = 1500,
            warmup: int = 500,
            pred: jnp.ndarray = None,
            upper: jnp.ndarray = None,
            lower: jnp.ndarray = None,
    ):
        """
        Initialize a NaiveDetection object.

        Args:
        x (Union[jnp.ndarray, np.ndarray, list]): The input data as a 1D numpy array, JAX array, or list representing time points.
        y (Union[jnp.ndarray, np.ndarray, list]): The observed data as a 1D numpy array, JAX array, or list.
        prior_mu (numpyro.distributions.Distribution): Prior distribution for mean values.
        prior_var (numpyro.distributions.Distribution): Prior distribution for standard deviations.
        """

        self.x = x
        self.y = y
        self.n = len(x)
        self.prior_mu = prior_mu
        self.prior_var = prior_var
        self.iterations = iterations
        self.warmup = warmup
        self.samples = None
        self.pred = jnp.zeros(shape=(self.n), dtype=jnp.float32)
        self.upper = jnp.zeros(shape=(self.n), dtype=jnp.float32)
        self.lower = jnp.zeros(shape=(self.n), dtype=jnp.float32)

    def model(self,x,y):
        """
        Define the Bayesian model for inference.

        Args:
        x (Union[jnp.ndarray, np.ndarray, list]): The input data as a 1D numpy array, JAX array, or list representing time points.
        y (Union[jnp.ndarray, np.ndarray, list]): The observed data as a 1D numpy array, JAX array, or list.

        Returns:
        None
        """

        # Priors for the mean skill value
        avg_before = numpyro.sample('avg_before', self.prior_mu)
        avg_after = numpyro.sample('avg_after', self.prior_mu)

        # Prior for the change point (assumed to be uniform over the days)
        change_point = numpyro.sample('change_point', dist.Uniform(0, self.n-1))

        # Optionally, priors for the standard deviations
        stdv_before = numpyro.sample('stdv_before', self.prior_var)
        stdv_after = numpyro.sample('stdv_after', self.prior_var)

        mean = avg_before * (x < change_point) + avg_after * (x >= change_point)
        sigma = stdv_before * (x < change_point) + stdv_after * (x >= change_point)
        numpyro.sample('expected', dist.Normal(mean, sigma), obs=y)

    def run_inference(self,rng_key):
        """
        Run Bayesian inference using MCMC to obtain posterior samples.

        Args:
        rng_key (jax.random.PRNGKey): The random number generator key.

        Returns:
        None
        """

        try:
            kernel = numpyro.infer.NUTS(self.model)
            mcmc = numpyro.infer.MCMC(kernel, num_warmup=self.warmup, num_samples=self.iterations)
            mcmc.run(rng_key, self.x, self.y)
            self.samples = mcmc.get_samples()
            print("Samples successfully retrieved.")
            return mcmc
        except TypeError as e:
            print(f"An error occurred: {e}")
    
    @jit
    def mean_pred(self, ix):
        """
        Calculate the mean prediction for a given index.

        Args:
        ix (numpy.ndarray): Boolean index for data points.

        Returns:
        float: Mean prediction.
        """
        before = jnp.sum(jnp.where(ix, self.samples["avg_before"], 0), axis=0)
        after = jnp.sum(jnp.where(~ix, self.samples["avg_after"], 0), axis=0)
        return (before + after) / self.iterations
    
    @jit
    def var_pred(self, ix):
        """
        Calculate the variance prediction for a given index.

        Args:
        ix (numpy.ndarray): Boolean index for data points.

        Returns:
        float: Variance prediction.
        """
        before = jnp.sum(jnp.where(ix, self.samples["stdv_before"], 0), axis=0)
        after = jnp.sum(jnp.where(~ix, self.samples["stdv_after"], 0), axis=0)
        return (before + after) / self.iterations
    
    def predict(self):
        """
        Perform predictions for change points and store results in pred, upper, and lower arrays.

        Returns:
        None
        """
        change_point = jnp.array(self.samples["change_point"])
        for i,d in enumerate(self.x[:-1],1):
            ix = d < change_point
            self.pred = self.pred.at[i].set(self.mean_pred(ix))
            self.upper = self.upper.at[i].set(self.pred.at[i].get() + self.var_pred(ix))
            self.lower = self.lower.at[i].set(self.pred.at[i].get() - self.var_pred(ix))
    


    def tree_flatten(self):
        """
        Flatten the NaiveDetection object to a tuple of its components for PyTree compatibility.

        Returns:
        Tuple: A tuple containing the flattened components of the NaiveDetection object.
        """
            
        # Flatten the samples dictionary if it exists
        if self.samples is not None:
            samples_flat, samples_tree_def = tree_flatten(self.samples)
        else:
            samples_flat, samples_tree_def = None, None

        children = (self.x, self.y, self.prior_mu, self.prior_var, self.pred, self.upper, self.lower, samples_flat)  # arrays / dynamic values
        aux_data = (self.iterations, self.warmup, samples_tree_def)  # static values and samples tree definition
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a NaiveDetection object from its flattened components.

        Args:
        aux_data (tuple): Tuple containing static values and samples tree definition.
        children (tuple): Tuple containing the flattened components of a NaiveDetection object.

        Returns:
        NaiveDetection: A reconstructed NaiveDetection object.
        """
            
        x, y, prior_mu, prior_var, pred, upper, lower, samples_flat = children
        iterations, warmup, samples_tree_def = aux_data

        # Reconstruct the samples dictionary if samples_flat is not None
        samples = tree_unflatten(samples_tree_def, samples_flat) if samples_flat is not None else None

        # Create the class instance without the samples argument
        instance = cls(x=x, y=y, prior_mu=prior_mu, prior_var=prior_var, pred=pred, upper=upper, lower=lower, iterations=iterations, warmup=warmup)
        
        # Set the samples attribute after creating the instance
        instance.samples = samples

        return instance