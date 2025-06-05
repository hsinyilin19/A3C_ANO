# 2025 06 04
# MiniGrid version, with classical Encoder

# 2025 06 04
# Sam modified

# 2024 10 09
# Use our own custom torch layer, not the qml.qnn.TorchLayer
# pip install gym-minigrid==1.0.3
# pip install gym==0.13.1
# # PennyLane 0.35.1

# 2023 10 06
# Replace MetaQuantum with qml.qnn.TorchLayer


# 2022 09 11
# VQC A3C for QAS two qubits

import numpy as np
from gymnasium.vector import AsyncVectorEnv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse

from utils import v_wrap, set_init, push_and_pull, record
from plot_functions import full_plotting


from shared_adam import SharedAdam

# import gym
import gymnasium as gym
import minigrid
from MiniGridWrappers import ImgObsFlatWrapper
# import gym_cartpolemod
# import gym_twoqubit
import os
from os.path import abspath, dirname, join
import pickle

import pennylane as qml


#####

torch.manual_seed(0)

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 50000

# env = gym.make('CartPole-v1')
# target = np.asarray([0.70710678+0.j,0.        +0.j,0.        +0.j, 0.70710678+0.j])
# env = gym.make('BasicTwoQubit-v0', target = target)

# env = gym.make('MiniGrid-SimpleCrossingS9N1-v0')
# env = ImgObsFlatWrapper(env)

# ENV_NAME = 'MiniGrid-Empty-5x5-v0'
# env = gym.make(ENV_NAME)
# env = ImgObsFlatWrapper(env)


# env = gym.make('CartPole-v1')
ENV_NAME = 'MiniGrid-Empty-6x6-v0'
env = gym.make(ENV_NAME)
env = ImgObsFlatWrapper(env)

# Define VQC

# latent_dim = 8
# input_dim = latent_dim
# output_dim = latent_dim
# num_qubit = latent_dim
# num_vqc_layers = 2
# dev = dev = qml.device("default.qubit", wires = num_qubit)
# hadamard_gate = True
# more_entangle = True


N_S = env.observation_space.shape[0]
N_A = env.action_space.n

n_qubits = 4
vqc_depth = 2
n_local = 3
# qnn_depth = 10 # Original: 10

# n_qubits = latent_dim
# q_depth = num_vqc_layers
# n_class = latent_dim

def H_layer(nqubits):
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def normalize_state(s):
    # MountainCar state bounds: [position, velocity]
    # Official bounds: position [-1.2, 0.6], velocity [-0.07, 0.07]
    s_min = torch.tensor([-1.2, -0.07], device=s.device)
    s_max = torch.tensor([0.6, 0.07], device=s.device)
    return 2.0 * (s - s_min) / (s_max - s_min) - 1.0


def create_Hermitian(N, A, B, D):
    h = torch.zeros((N, N), dtype=torch.complex128)
    count = 0
    for i in range(1, N):
        h[i - 1, i - 1] = D[i].clone()
        for j in range(i):
            h[i, j] = A[count + j].clone() + 1j * B[count + j].clone()
        count += i
    return h + h.conj().T


def entangling_layer(nqubits):
    """ Layer of CNOTs followed by another shifted layer of CNOT."""
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: epoch=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  epoch=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


# Define actual circuit architecture
def quantum_net(X, θ, H):
    """ The variational quantum circuit. """
    # Reshape weights
    # θ = θ.reshape(vqc_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    # print("q input features: {}".format(q_input_features))
    RY_layer(X)

    # Sequence of trainable variational layers
    for k in range(vqc_depth):
        entangling_layer(n_qubits)
        RY_layer(θ[k])

    # Compute Expectation values (for multi-class prediction) using n_local Hermitians
    exp_vals = [qml.expval(qml.Hermitian(H[q], wires=(np.arange(q, q + n_local) % n_qubits).tolist())) for q in
                range(n_qubits)]
    return exp_vals


def standard_quantum_net(X, θ):
    """ The variational quantum circuit. """
    # Reshape weights
    # θ = θ.reshape(vqc_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    # print("q input features: {}".format(q_input_features))
    RY_layer(X)

    # Sequence of trainable variational layers
    for k in range(vqc_depth):
        entangling_layer(n_qubits)
        RY_layer(θ[k])

    # Compute Expectation values (for multi-class prediction) using n_local Hermitians
    exp_vals = [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
    return exp_vals


class Standard_VQC_Model(nn.Module):
    '''Standard VQC model with only rotation'''

    def __init__(self):
        super(Standard_VQC_Model, self).__init__()
        self.θ = nn.Parameter(0.01 * torch.randn(vqc_depth, n_qubits))  # VQC rotation params

        self.dev = qml.device("default.qubit",
                              wires=n_qubits)  # Can use different simulation backend or quantum computers.
        self.VQC = qml.QNode(standard_quantum_net, self.dev, interface="torch")

    def forward(self, X):
        z1 = X.reshape(-1, n_qubits)
        q_out = torch.stack([torch.stack(self.VQC(z, self.θ)).float() for z in z1])

        return q_out


def quantum_circuit(X, H, dim_A, n_qubits):
    return H_net(X, H, dim_A, n_qubits)


def H_net(X, H, dim_A, n_qubits):
    """ only measure dim_A outputs """
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(X)

    # Compute Expectation values (for multi-class prediction) using n_local Hermitians
    # n_local = int(np.log2(H[0].shape[0]))
    exp_vals = [qml.expval(qml.Hermitian(H[_], wires=range(n_qubits))) for _ in range(dim_A)]
    return exp_vals


class Measurement_Model(nn.Module):
    '''Model with only measurement.'''

    def __init__(self, n_qubits, n_local, dim_S, dim_A):
        super(Measurement_Model, self).__init__()
        # self.θ = nn.Parameter(0.01 * torch.randn(args.vqc_depth, n_qubits))  # measure rotation params

        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit",
                              wires=n_qubits)  # Can use different simulation backend or quantum computers.
        # self.measure = qml.QNode(lambda X, H, dim_A: H_net(X, H, dim_A, self.n_qubits), self.dev, interface="torch")
        self.measure = qml.QNode(
            functools.partial(quantum_circuit, n_qubits=self.n_qubits),
            self.dev, interface="torch"
        )
        self.N = 2 ** n_local
        self.dim_S, self.dim_A = dim_S, dim_A

        self.A = nn.ParameterList([nn.Parameter(torch.empty((self.N * (self.N - 1)) // 2)) for _ in
                                   range(n_qubits)])  # off-diagonal entries (real part)
        self.B = nn.ParameterList([nn.Parameter(torch.empty((self.N * (self.N - 1)) // 2)) for _ in
                                   range(n_qubits)])  # off-diagonal entries (imag part)
        self.D = nn.ParameterList(
            [nn.Parameter(torch.empty(self.N)) for _ in range(n_qubits)])  # diagonal elements (all real)

        for q in range(n_qubits):
            nn.init.normal_(self.A[q], std=2.)
            nn.init.normal_(self.B[q], std=2.)
            nn.init.normal_(self.D[q], std=2.)

    def forward(self, s):
        X = torch.tile(s, (1, self.dim_A))
        # create observable H here
        self.H = [create_Hermitian(self.N, self.A[q], self.B[q], self.D[q]) for q in
                  range(self.dim_A)]  # only measure dim_A outputs
        q_out = torch.stack([torch.stack(self.measure(x, self.H, dim_A=self.dim_A)).float() for x in X])
        return q_out



class Nonlocal_VQC_Model(nn.Module):
    '''VQC model with rotation and measurement'''

    def __init__(self, output_features=None):
        super(Nonlocal_VQC_Model, self).__init__()
        self.θ = nn.Parameter(0.01 * torch.randn(vqc_depth, n_qubits))

        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.VQC = qml.QNode(quantum_net, self.dev, interface="torch")

        self.N = 2 ** n_local
        self.A = nn.ParameterList([nn.Parameter(torch.empty((self.N * (self.N - 1)) // 2)) for _ in range(n_qubits)])
        self.B = nn.ParameterList([nn.Parameter(torch.empty((self.N * (self.N - 1)) // 2)) for _ in range(n_qubits)])
        self.D = nn.ParameterList([nn.Parameter(torch.empty(self.N)) for _ in range(n_qubits)])

        for q in range(n_qubits):
            nn.init.normal_(self.A[q], std=2.)
            nn.init.normal_(self.B[q], std=2.)
            nn.init.normal_(self.D[q], std=2.)

        self.output_features = output_features or n_qubits

    def forward(self, X):
        # Pad input to match n_qubits if necessary
        batch_size = X.shape[0]
        if X.shape[1] < n_qubits:
            # Pad with zeros or repeat features
            padding = torch.zeros(batch_size, n_qubits - X.shape[1], device=X.device)
            X = torch.cat([X, padding], dim=1)
        elif X.shape[1] > n_qubits:
            # Truncate if input is larger
            X = X[:, :n_qubits]

        z1 = X.reshape(-1, n_qubits)

        # Create observable H
        self.H = [create_Hermitian(self.N, self.A[q], self.B[q], self.D[q]) for q in range(n_qubits)]
        q_out = torch.stack([torch.stack(self.VQC(z, self.θ, self.H)).float() for z in z1])

        # Return all n_qubits features (let the linear layers handle dimensionality)
        return q_out



class Net(nn.Module):
	def __init__(self, s_dim, a_dim):
		super(Net, self).__init__()
		self.s_dim = s_dim
		self.a_dim = a_dim

		# Classical Encoder for processing (used in MiniGrid)
		self.encoder = nn.Linear(self.s_dim, n_qubits)

		# Separate VQC models for policy and value functions
		self.pi_func = Standard_VQC_Model()
		self.v_func = Standard_VQC_Model()

		# Linear layers to map VQC outputs to correct dimensions
		self.pi_linear = nn.Linear(n_qubits, a_dim)
		self.v_linear = nn.Linear(n_qubits, 1)

		self.distribution = torch.distributions.Categorical

	def forward(self, x):

		x = self.encoder(x)
		
		# Policy network
		pi_features = self.pi_func(x)

		# USE the linear layer to map to correct action dimension
		logits = self.pi_linear(pi_features)

		# Value network
		v_features = self.v_func(x)
		values = self.v_linear(v_features)

		return logits, values

	def choose_action(self, s):
		self.eval()
		logits, _ = self.forward(s)

		# Ensure logits has correct shape
		if logits.shape[1] != self.a_dim:
			logits = logits[:, :self.a_dim]

		prob = F.softmax(logits, dim=1).data
		m = self.distribution(prob)
		action = m.sample().numpy()[0]

		# Ensure action is within valid range as additional safety
		action = np.clip(action, 0, self.a_dim - 1)

		return action

	def loss_func(self, s, a, v_t):
		self.train()
		logits, values = self.forward(s)

		# Ensure proper dimensions
		values = values.squeeze()  # Shape: [batch_size]
		td = v_t - values
		c_loss = td.pow(2)

		probs = F.softmax(logits, dim=1)
		m = self.distribution(probs)

		# Ensure action tensor is 1D
		if a.dim() > 1:
			a = a.squeeze()

		exp_v = m.log_prob(a) * td.detach()
		a_loss = -exp_v
		total_loss = (c_loss + a_loss).mean()
		return total_loss


class Worker(mp.Process):
	def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
		super(Worker, self).__init__()
		self.name = 'w%02i' % name
		self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
		self.gnet, self.opt = gnet, opt
		self.lnet = Net(N_S, N_A)           # local network
		# self.env = gym.make('CartPole-v1')
		# self.env = gym.make(args.env_id)
		# self.env = gym.make('CartPole-v0').unwrapped
		# self.env = gym.make('BasicTwoQubit-v0', target = target)
		# self.env = gym.make('MiniGrid-SimpleCrossingS9N1-v0')
		# self.env = ImgObsFlatWrapper(gym.make('MiniGrid-SimpleCrossingS9N1-v0'))

		self.env = ImgObsFlatWrapper(gym.make(ENV_NAME))

	def run(self):
		total_step = 1
		while self.g_ep.value < MAX_EP:

			# s = self.env.reset() # Old GYM
			s, _ = self.env.reset() # New Gymnasium

			buffer_s, buffer_a, buffer_r = [], [], []
			ep_r = 0.
			epi_step = 1
			while True:
				# if self.name == 'w00':
					# self.env.render()

				a = self.lnet.choose_action(v_wrap(s[None, :])) # 2022 09 11: need to fix here
				
				# s_, r, done, _ = self.env.step(a) # Old GYM

				s_, r, terminated, truncated, info = self.env.step(a)
				done = terminated or truncated  # Combine both flags into a single `done` flag

				epi_step += 1

				# 2022 09 11: not good for QAS (originally for CartPole)
				# if done: r = -1

				ep_r += r
				buffer_a.append(a)
				buffer_s.append(s)
				buffer_r.append(r)

				if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
					# sync
					push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
					buffer_s, buffer_a, buffer_r = [], [], []

					if done or epi_step >= 500:  # done and print information
						record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
						break
				s = s_
				total_step += 1
		# return 
		self.res_queue.put(None)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Quantum DDQN with Parallel Environments for MountainCar')
	parser.add_argument('--env_id', type=str, default='CartPole-v1', help='Gym environment ID')
	parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
	parser.add_argument('--n_episodes', type=int, default=2000, help='Maximum number of episodes')
	parser.add_argument('--target_score', type=float, default=-110.0, help='Target score to solve environment')
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	# parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
	parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for neural networks')
	parser.add_argument('--eps_start', type=float, default=1.0, help='initial entropy')
	parser.add_argument('--eps_decay', type=float, default=0.995, help='entropy decay rate')
	parser.add_argument('--eps_end', type=float, default=0.01, help='entropy last')
	parser.add_argument('--std', type=float, default=2.0, help='model init std')

	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--lr_H', type=float, default=1e-1)

	parser.add_argument('--n-local', type=int, default=3)
	parser.add_argument('--vqc-depth', type=int, default=4)
	args = parser.parse_args()

	# Set random seeds
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	gnet = Net(N_S, N_A)        # global network
	gnet.share_memory()         # share the global parameters in multiprocessing
	opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
	global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

	# parallel training
	# workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(min(mp.cpu_count(), 80))]
	workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(16)]
	[w.start() for w in workers]
	
	# [w.join() for w in workers]

	res = []                    # record episode reward to plot

	# print("FINISHED join()")

	# while not res_queue.empty():
	# 	print("in the while loop")
	# 	r = res_queue.get()
	# 	print(r)

	# 	res.append(r)



	while True:
		r = res_queue.get()
		# if r is not None:
		# 	res.append(r)
		# else:
		# 	break

		if len(res) < MAX_EP and (r is not None):
			res.append(r)

		elif res_queue.empty():
			break
			
		else:
			pass

	[w.join() for w in workers]

	print(len(res))

	

	# Plot and save
	full_plotting(_fileTitle = "Nonlocal_A3C_{}".format(ENV_NAME), _trainingLength = len(res), _currentRewardList = res)
	path_model = join(dirname(abspath(__file__)), "trained_model_Nonlocal_A3C_{}.pth".format(ENV_NAME))
	torch.save(gnet.state_dict(), path_model)

	# Save the raw data of res
	path_episode_reward = join(dirname(abspath(__file__)), "episode_returns_Nonlocal_A3C_{}".format(ENV_NAME))
	with open(path_episode_reward, "wb") as fp:
		pickle.dump(res, fp)





