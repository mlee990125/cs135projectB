'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!

import autograd.numpy as ag_np

import autograd.numpy as ag_np

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        random_state = self.random_state
        
        # Initialize mu to global mean of training ratings
        user_id_N, item_id_N, y_N = train_tuple
        global_mean = ag_np.mean(y_N)
        
        self.param_dict = dict(
            # Initialize mu to global mean
            mu=ag_np.array([global_mean]),
            # Initialize biases to small random values
            b_per_user=0.1 * random_state.randn(n_users),
            c_per_item=0.1 * random_state.randn(n_items),
            # Initialize factors to small random values
            U=0.1 * random_state.randn(n_users, self.n_factors),
            V=0.1 * random_state.randn(n_items, self.n_factors)
        )
    # def init_parameter_dict(self, n_users, n_items, train_tuple):
    #     ''' Initialize parameter dictionary attribute for this instance. '''
    #     random_state = self.random_state  # Inherited RandomState for reproducibility

    #     self.param_dict = {
    #         'mu': ag_np.ones(1) * train_tuple[2].mean(),
    #         'b_per_user': ag_np.zeros(n_users),
    #         'c_per_item': ag_np.zeros(n_items),
    #         'U': 0.001 * random_state.randn(n_users, self.n_factors),
    #         'V': 0.001 * random_state.randn(n_items, self.n_factors)
    #     }


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        """Predict ratings for given user-item pairs."""
        # Cast indices for array indexing
        user_id_N = user_id_N.astype(ag_np.int32)
        item_id_N = item_id_N.astype(ag_np.int32)
        
        # Start with global mean
        yhat_N = ag_np.zeros_like(user_id_N, dtype=ag_np.float64) + mu[0]
        
        # Add user and item biases
        yhat_N = yhat_N + b_per_user[user_id_N]
        yhat_N = yhat_N + c_per_item[item_id_N]
        
        # Add matrix factorization term
        # Get relevant vectors
        U_N = U[user_id_N]  # Shape: (n_examples, n_factors)
        V_N = V[item_id_N]  # Shape: (n_examples, n_factors)
        
        # Compute dot product for each user-item pair
        yhat_N = yhat_N + ag_np.sum(U_N * V_N, axis=1)
        
        return yhat_N
    # def predict(self, user_id_N, item_id_N, mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
    #     ''' Predict ratings at specific user_id, item_id pairs. '''
    #     # Use provided parameters or default to the instance's param_dict
    #     if mu is None: mu = self.param_dict['mu']
    #     if b_per_user is None: b_per_user = self.param_dict['b_per_user']
    #     if c_per_item is None: c_per_item = self.param_dict['c_per_item']
    #     if U is None: U = self.param_dict['U']
    #     if V is None: V = self.param_dict['V']

    #     user_bias = b_per_user[user_id_N]
    #     item_bias = c_per_item[item_id_N]
    #     interaction = ag_np.sum(U[user_id_N] * V[item_id_N], axis=1)
    #     return mu + user_bias + item_bias + interaction


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        """Compute loss function."""
        user_id_N, item_id_N, y_N = data_tuple
        
        # Get predicted ratings
        yhat_N = self.predict(user_id_N, item_id_N, **param_dict)
        
        # Compute MSE loss
        N = ag_np.float64(user_id_N.shape[0])
        err_N = yhat_N - y_N
        mse_loss = ag_np.sum(err_N * err_N) / (2.0 * N)  # Include 1/2 factor for cleaner gradients
        
        # Add L2 regularization on U and V matrices
        reg_loss = (self.alpha / 2.0) * (
            ag_np.sum(param_dict['U']**2) + 
            ag_np.sum(param_dict['V']**2) +
            ag_np.sum(param_dict['b_per_user']**2) +
            ag_np.sum(param_dict['c_per_item']**2)
        ) / N
        
        total_loss = mse_loss + reg_loss
        return total_loss

    # def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
    #     ''' Compute loss at given parameters. '''
    #     user_id_N, item_id_N, y_N = data_tuple
    #     yhat_N = self.predict(user_id_N, item_id_N, **param_dict)
    #     error = y_N - yhat_N
    #     mse_loss = ag_np.mean(error**2)

    #     reg_loss = self.alpha * (
    #         ag_np.sum(param_dict['U']**2) + ag_np.sum(param_dict['V']**2)
    #     )
    #     return mse_loss + reg_loss


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    model = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=10000, step_size=0.1,
        n_factors=2, alpha=0.0)
    model.init_parameter_dict(n_users, n_items, train_tuple)

    # Fit the model with SGD
    model.fit(train_tuple, valid_tuple)