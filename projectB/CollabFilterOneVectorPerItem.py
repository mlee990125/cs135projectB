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

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        # Unpack training data
        user_ids_train = train_tuple[0]  # array of user_ids
        item_ids_train = train_tuple[1]  # array of item_ids
        ratings_train = train_tuple[2]   # array of ratings

        # Compute global mean rating
        mu = ag_np.array([ag_np.mean(ratings_train)], dtype=ag_np.float)

        # Initialize biases per user and per item to zeros
        b_per_user = ag_np.zeros(n_users, dtype=ag_np.float)
        c_per_item = ag_np.zeros(n_items, dtype=ag_np.float)

        # Initialize latent factors U and V to small random numbers
        U = 0.01 * random_state.randn(n_users, self.n_factors)
        V = 0.01 * random_state.randn(n_items, self.n_factors)

        # Store parameters in param_dict
        self.param_dict = dict(
            mu=mu,
            b_per_user=b_per_user,
            c_per_item=c_per_item,
            U=U,
            V=V,
        )

    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        # TODO: Update with actual prediction logic
        if mu is None:
            mu = self.param_dict['mu']
        if b_per_user is None:
            b_per_user = self.param_dict['b_per_user']
        if c_per_item is None:
            c_per_item = self.param_dict['c_per_item']
        if U is None:
            U = self.param_dict['U']
        if V is None:
            V = self.param_dict['V']

        # Get biases and latent factors for the specified user_ids and item_ids
        b_u = b_per_user[user_id_N]      # User biases, shape (N,)
        c_i = c_per_item[item_id_N]      # Item biases, shape (N,)
        U_u = U[user_id_N, :]            # User latent factors, shape (N, K)
        V_i = V[item_id_N, :]            # Item latent factors, shape (N, K)

        # Compute dot product between user and item latent factors
        dot_product = ag_np.sum(U_u * V_i, axis=1)  # shape (N,)

        # Compute predictions
        yhat_N = mu + b_u + c_i + dot_product

        return yhat_N

    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength
        # Unpack data
        user_id_N = data_tuple[0]
        item_id_N = data_tuple[1]
        y_N = data_tuple[2]  # true ratings

        # Compute predictions
        yhat_N = self.predict(user_id_N, item_id_N, **param_dict)

        # Compute squared error loss
        errors = y_N - yhat_N
        squared_error_loss = ag_np.sum(errors ** 2)

        # Compute L2 regularization term
        # Regularize U and V only
        U = param_dict['U']
        V = param_dict['V']

        reg_loss = self.alpha * (ag_np.sum(U ** 2) + ag_np.sum(V ** 2))

        # Total loss
        loss_total = squared_error_loss + reg_loss

        return loss_total  


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