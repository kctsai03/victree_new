import unittest

import networkx as nx
import torch

from utils.config import Config, set_seed
from variational_distributions.var_dists import qEpsilonMulti, qT, qEpsilon, qZ, qMuTau, qC, qPi, qCMultiChrom
from variational_distributions.joint_dists import VarTreeJointDist


class qCTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # build config
        set_seed(101)
        self.config = Config(chromosome_indexes=[])
        # self.qc = qC(self.config)
        self.qc = qCMultiChrom(self.config)
        self.qt = qT(self.config)
        self.qeps = qEpsilonMulti(self.config, 2, 5)  # skewed towards 0
        self.qz = qZ(self.config)
        self.qmt = qMuTau(self.config)

        self.qc.initialize()
        self.qt.initialize()
        self.qeps.initialize()
        self.qz.initialize()
        self.qmt.initialize(loc=100, precision_factor=.1, shape=5, rate=5)

        self.obs = torch.randint(low=50, high=150,
                                 size=(self.config.chain_length, self.config.n_cells))

    def test_filtering_probs_update(self):

        pass

    def test_expectation_size(self):
        for qc in self.qc.qC_list:
            chain_length = qc.config.chain_length
            tree = nx.random_tree(self.config.n_nodes, create_using=nx.DiGraph)
            exp_alpha1, exp_alpha2 = qc._exp_alpha(tree, self.qeps)
            self.assertEqual(exp_alpha1.shape, (self.config.n_nodes, self.config.n_states))
            self.assertEqual(exp_alpha2.shape,
                             (self.config.n_nodes, chain_length-1, self.config.n_states, self.config.n_states))

            exp_eta1, exp_eta2 = qc._exp_eta(self.obs, tree, self.qeps, self.qz, self.qmt)
            self.assertEqual(exp_eta1.shape, (self.config.n_nodes, self.config.n_states))
            self.assertEqual(exp_eta2.shape,
                             (self.config.n_nodes, chain_length-1, self.config.n_states, self.config.n_states))

    def test_ELBO(self):
        L_list = [1, 2, 5, 10, 20]
        trees = []
        weights = []
        L_prev = 0
        for L in L_list:
            new_trees, new_weights = self.qt.get_trees_sample(sample_size=(L - L_prev))
            trees = trees + new_trees
            weights = weights + new_weights
            res_1 = self.qc.compute_elbo(trees, weights, self.qeps)
            print(f" {res_1}")
            L_prev = L

    def test_entropy_higher_for_random_transitions_than_deterministic_transitions(self):
        K = 3
        M = 5
        N = 2
        A = 7
        config_1 = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M)
        qc_1 = qC(config_1)
        qc_1.initialize()
        entropy_rand = qc_1.marginal_entropy()
        print(f"Random: {entropy_rand}")
        qc_2 = qC(config_1)
        for k in range(K):
            qc_2.eta1 = torch.log(torch.zeros(K, A) + 0.001)
            qc_2.eta1[2] = 0.
            for m in range(M-1):
                qc_2.eta2[k, m] = torch.log(torch.diag(torch.ones(A, )) + 0.001)

        qc_2.compute_filtering_probs()
        entropy_deterministic = qc_2.marginal_entropy()
        print(f"Deterministic: {entropy_deterministic}")
        self.assertGreater(entropy_rand, entropy_deterministic)

    def test_entropy_lower_for_random_transitions_than_uniform_transitions(self):
        rand_res = self.qc.get_entropy()
        qc_1 = qCMultiChrom(self.config)
        qc_1.initialize(method='uniform')
        uniform_res = qc_1.get_entropy()
        print(f"Entropy random eta_2: {rand_res} - uniform eta_2: {uniform_res}")
        self.assertLess(rand_res, uniform_res)

    @unittest.skip("wrong code")
    def test_cross_entropy_lower_for_random_transitions_than_no_transitions_for_small_epsilon(self):
        K = 3
        M = 5
        N = 2
        A = 7
        config_1 = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M)
        qc_1 = qC(config_1)
        qc_1.initialize()
        q_eps = qEpsilonMulti(config_1).initialize('non-mutation')
        cross_entropy_rand = qc_1.neg_cross_entropy_arc(q_eps, 0, 1)
        print(f"Random: {cross_entropy_rand}")
        qc_2 = qC(config_1).initialize()
        # FIXME: m goes over two cycles but only used once
        for k in range(K):
            for m in range(M - 1):
                qc_2.eta1 = torch.log(torch.zeros(K, A) + 0.001)
                qc_2.eta1[2] = 0.
                for m in range(M - 1):
                    qc_2.eta2[k, m] = torch.log(torch.diag(torch.ones(A, )) + 0.001)

        qc_2.compute_filtering_probs()
        cross_entropy_deterministic = qc_2.neg_cross_entropy_arc(q_eps, 0, 1)
        print(f"Deterministic: {cross_entropy_deterministic}")
        self.assertLess(cross_entropy_rand, cross_entropy_deterministic)

    def test_baum_welch_init(self):
        torch.manual_seed(0)
        K = 1
        M = 20
        N = 200
        A = 7
        config_1 = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M)
        qc_1 = qC(config_1)


        N_K1 = 5
        N_K2 = 12
        N_K3 = 3
        mu_0 = 10.
        tau_0 = 1.
        mu_n_dist = torch.distributions.Normal(mu_0, tau_0)
        obs = torch.ones((M, N))
        C_probs = torch.zeros(A) + 0.1
        C_probs[2] = 1
        C_probs[4] = 1
        C_K1_dist = torch.distributions.Categorical(C_probs)
        C_K1 = C_K1_dist.sample([M])
        C_K1_one_hot = torch.nn.functional.one_hot(C_K1, num_classes=A)
        for n in range(N):
            mu_n = mu_n_dist.sample()
            mu_n_c = mu_n * C_K1
            obs[:, n] = torch.distributions.Normal(mu_n_c, tau_0).sample()

        qmt = qMuTau(config_1)
        qmt.initialize(loc=mu_0, precision_factor=tau_0, shape=1., rate=1.)

        qc_1._baum_welch_init(obs=obs, qmt=qmt)
        q_eps = qEpsilonMulti(config_1)
#        cross_entropy_BW = qc_1.cross_entropy_arc(q_eps, 0, 1)
#        print(f"CE baum-welch init: {cross_entropy_BW}")
        print(f"True: {C_K1[0:5]}")
        print(f"q(C) after baum-welch init: {qc_1.single_filtering_probs[0, 0:5]}")

