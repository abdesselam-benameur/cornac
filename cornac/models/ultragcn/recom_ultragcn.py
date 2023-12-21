# TODO: add the license and copy right
from ..recommender import Recommender
from ..recommender import ANNMixin, MEASURE_DOT
from ...exception import ScoreException

from tqdm.auto import tqdm, trange


class UltraGCN(Recommender, ANNMixin):
    """
    UltraGCN

    Parameters
    ----------
    name: string, default: 'UltraGCN'
        The name of the recommender model.

    emb_size: int, default: 64
        Size of the node embeddings.

    num_epochs: int, default: 1000
        Maximum number of iterations or the number of epochs.

    learning_rate: float, default: 0.001
        The learning rate that determines the step size at each iteration

    batch_size: int, default: 1024
        Mini-batch size used for train set

    num_layers: int, default: 3
        Number of UltraGCN Layers

    early_stopping: {min_delta: float, patience: int}, optional, default: None
        If `None`, no early stopping. Meaning of the arguments:

        - `min_delta`:  the minimum increase in monitored value on validation
                        set to be considered as improvement,
                        i.e. an increment of less than min_delta will count as
                        no improvement.

        - `patience`:   number of epochs with no improvement after which
                        training should be stopped.

    lambda_reg: float, default: 1e-4
        Weight decay for the L2 normalization

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model
        is already pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: 2020
        Random seed for parameters initialization.

    References
    ----------
    *   TODO: add the names of the authors
        UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation.
    """
    # TODO: check the list of parameters in docstring

    def __init__(
        self,
        name="UltraGCN",
        emb_size=64,
        num_epochs=1000,
        learning_rate=0.001,
        batch_size=1024,
        num_layers=3,
        early_stopping=None,
        lambda_reg=1e-4,
        trainable=True,
        verbose=False,
        seed=2020,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.emb_size = emb_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.early_stopping = early_stopping
        self.lambda_reg = lambda_reg
        self.seed = seed

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        if not self.trainable:
            return self

        # model setup
        import torch
        from .ultragcn import UltraGCN, train, data_param_prepare

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # TODO: add mps support
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        # params = {
        #     "user_num": self.train_set.num_users,
        #     "item_num": self.train_set.num_items,
        #     "lr": self.learning_rate,
        #     "embedding_dim": self.emb_size,
        #     "w1": 
        #     "w2": self.lambda_reg,
        #     "w3": self.lambda_reg,
        #     "w4": self.lambda_reg,
        #     "negative_weight": 1.0,
        #     "gamma": self.lambda_reg,
        #     "lambda": 1.0,
        #     "initial_weight": 0.01
        # }
                
        params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items = data_param_prepare('')

        params = {'embedding_dim': 64, 'ii_neighbor_num': 10, 'model_save_path': './ultragcn_amazon.pt',
                  'max_epoch': 2000, 'enable_tensorboard': True, 'initial_weight': 0.0001, 'dataset': 'amazon',
                  'gpu': '0', 'lr': 0.001, 'batch_size': 1024, 'early_stop_epoch': 15,
                  'w1': 1e-08, 'w2': 1.0, 'w3': 1.0, 'w4': 1e-08, 'negative_num': 500, 'negative_weight': 500.0,
                  'gamma': 0.0001, 'lambda': 2.75, 'sampling_sift_pos': False, 'test_batch_size': 2048, 'topk': 20,
                  'user_num': 52643, 'item_num': 91599}
        
        ultragcn = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
        ultragcn = ultragcn.to(self.device)
        optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])

        train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params)

    def monitor_value(self, train_set, val_set):
        """Calculating monitored value used for early stopping on validation set (`val_set`).
        This function will be called by `early_stop()` function.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        res : float
            Monitored value on validation set.
            Return `None` if `val_set` is `None`.
        """
        if val_set is None:
            return None

        from ...metrics import Recall
        from ...eval_methods import ranking_eval

        recall_20 = ranking_eval(
            model=self,
            metrics=[Recall(k=20)],
            train_set=train_set,
            test_set=val_set,
        )[0][0]

        return recall_20  # Section 4.1.2 in the paper, same strategy as NGCF.

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if item_idx is None:
            if not self.knows_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )
            known_item_scores = self.V.dot(self.U[user_idx, :])
            return known_item_scores
        else:
            if not (self.knows_user(user_idx) and self.knows_item(item_idx)):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            return self.V[item_idx, :].dot(self.U[user_idx, :])

    def get_vector_measure(self):
        """Getting a valid choice of vector measurement in ANNMixin._measures.

        Returns
        -------
        measure: MEASURE_DOT
            Dot product aka. inner product
        """
        return MEASURE_DOT

    def get_user_vectors(self):
        """Getting a matrix of user vectors serving as query for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of user vectors for all users available in the model.
        """
        return self.U

    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of item vectors for all items available in the model.
        """
        return self.V
