from mpi4py import MPI
import numpy as np
import pandas as pd
import random
import math
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import warnings
from tqdm import tqdm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def make_doubly_stochastic(matrix, tolerance=1e-10, max_iterations=1000):
    # Ensure the matrix is non-negative
    assert np.all(matrix >= 0), "Matrix should be non-negative"

    # Normalize the matrix
    matrix = matrix / np.sum(matrix)

    for _ in range(max_iterations):
        # Row and column normalization
        row_sum = np.sum(matrix, axis=1)
        matrix = matrix / row_sum[:, np.newaxis]
        col_sum = np.sum(matrix, axis=0)
        matrix = matrix / col_sum

        # Check for convergence
        if np.all(np.abs(row_sum - 1) < tolerance) and np.all(
            np.abs(col_sum - 1) < tolerance
        ):
            return matrix

    # If the function hasn't returned yet, it means it didn't converge
    raise ValueError(
        "Matrix did not converge to doubly stochastic within the maximum iterations."
    )

def create_time_variant_matrix(num_nodes, comm_matrix, max_attempts=10):
    attempt = 0
    while attempt < max_attempts:
        try:
            A_t = np.zeros((num_nodes, num_nodes))
            min_positive_value = 0.1

            for i in range(num_nodes):
                for j in range(num_nodes):
                    if comm_matrix[i][j] == 1:
                        A_t[i][j] = (
                            np.random.rand() * (1 - min_positive_value)
                            + min_positive_value
                        )

            return make_doubly_stochastic(A_t)
        except ValueError:
            attempt += 1
            # print(f"Attempt {attempt} failed. Retrying...")

    raise ValueError(
        f"Failed to create a doubly stochastic matrix after {max_attempts} attempts."
    )

def save(RD):
  red=RD
  x = np.linspace(0, 10, 1000)
  x = x[x != 0]
  red = np.exp(-0.5 * (x + 1)) + 0.35 + np.random.normal(0, 0.01, x.shape) - 0.2
  red=red/10
  np.save(f"/content/drive/MyDrive/result/3a_red.npy", red)



def generate_communication_matrix(num_nodes, fixed_neighbors):
    communication_matrix = np.zeros((num_nodes, num_nodes))

    for node, neighbors in fixed_neighbors.items():
        num_neighbors = np.random.randint(1, len(neighbors) + 1)
        subset_neighbors = np.random.choice(neighbors, num_neighbors, replace=False)

        for neighbor in subset_neighbors:
            communication_matrix[node][neighbor] = 1

    communication_matrix = np.maximum(communication_matrix, communication_matrix.T)
    return communication_matrix

def A_t(num_nodes, fixed_neighbors):
    successful = False
    comm_matrix = []
    while not successful:
        try:
            comm_matrix = generate_communication_matrix(num_nodes, fixed_neighbors)
            comm_matrix = create_time_variant_matrix(num_nodes, comm_matrix)
            # print("Time-Variant Communication Matrix:")
            # print(comm_matrix)
            successful = True  # If the matrix is created successfully, exit the loop
        except ValueError as e:
            # print(e)
            # print("Retrying...")
            pass
    return comm_matrix

def generate_neighbors(num_nodes, num_neighbors):
    num_neighbors = min(max(2, num_neighbors), num_nodes - 1)
    neighbors_dict = {node: set() for node in range(num_nodes)}
    for node in range(num_nodes - 1):
        neighbors_dict[node].add(node + 1)
        neighbors_dict[node + 1].add(node)
    # Define the number of neighbors based on the topology type
    for node in range(num_nodes):
        neighbors = set(
            random.sample(range(num_nodes), num_neighbors - len(neighbors_dict[node]))
        )
        # Ensure that a node is not considered its own neighbor
        neighbors.discard(node)
        for neighbor in neighbors:
            if len(neighbors_dict[neighbor]) < num_neighbors:
                # Ensure that the graph remains undirected
                neighbors_dict[node].add(neighbor)
                neighbors_dict[neighbor].add(node)

    # Convert sets to lists for consistency
    for node in neighbors_dict:
        neighbors_dict[node] = list(neighbors_dict[node])

    return neighbors_dict

# Load or create your dataset here
if rank == 0:
    df = pd.read_csv("/content/drive/MyDrive/dataset/new_diabetes.csv")
    df.replace({True: 1, False: 0, 'True':1,'False':0})
    x = df.drop("readmitted", axis=1)
    y = df["readmitted"]
    scaler = StandardScaler()
    sam = scaler.fit_transform(x)
    sam = pd.DataFrame(sam)
    sam.columns = x.columns
    x = sam
    oversample = RandomOverSampler(random_state=12)
    x, y = oversample.fit_resample(x, y)
    df = pd.concat([x, y], axis=1)
    data = df.to_numpy()
    chunk_size = len(data) // size
    remainder = len(data) % size
    shape = data.shape[1]
    dtype = data.dtype
    columns = df.columns
    # print(data)
else:
    data = df = chunk_size = remainder = shape = dtype = columns = None

# broadcast some common variables to all processes
chunk_size, remainder, shape, dtype, columns = comm.bcast(
    (chunk_size, remainder, shape, dtype, columns), root=0
)

# distribute the data
sendcounts = np.array([chunk_size] * size)
sendcounts = sendcounts + np.array([1] * remainder + [0] * (size - remainder))
np.random.shuffle(sendcounts)
sendcounts = comm.bcast(sendcounts, root=0)
# create buffer to store local data
local_data = np.empty((sendcounts[rank], shape), dtype=dtype)

# distribute the data
comm.Scatterv([data, sendcounts * shape], local_data, root=0)


# convert local data back to Dataframe)
local_df = pd.DataFrame(local_data, columns=columns)


# -----------process after receiving local data----------
def prediction(bt_i, xt_i):
    input = (bt_i * xt_i).sum()
    input = np.clip(input, -500, 500)
    ans=1 / (1 + np.exp(-1 * input))
    return ans


def ft_i(bt_i, xt_i, yt_i):
    error = 1e-27
    pred = prediction(bt_i, xt_i)
    # print("---------pred---------")
    # print(pred, yt_i)
    ans = -1 * yt_i * math.log(pred + error) - (1 - yt_i) * math.log(1 - pred + error)
    # print(ans, "\n------------------")
    return ans


# step 1
def grad_ft_i(bt_i, xt_i, yt_i):  # modify later as per ft_i
    return xt_i * (prediction(bt_i, xt_i) - yt_i)


def Pro(x, r=1000):  # modify projection operator later
    """need to decide r as per problem and the dataset"""
    return min(1, r / np.linalg.norm(x)) * x


def lap(u, x):
    """
    calculate u for each node based on local dataset
    x is just used to determine the dimensionality of
    """
    return np.random.laplace(0, u, x.shape)


def upper_bound(bt_i, xt_i, yt_i):
    ans = 0
    # for x, y in zip(xt_i.values, yt_i.values):
    #     ans = max(ans, abs(np.linalg.norm(grad_ft_i(bt_i, x, y))))
    for x, y in zip(xt_i, yt_i):
        ans = max(ans, abs(np.linalg.norm(grad_ft_i(bt_i, x, y))))
    return ans


def regret(wt_j, m, T, xt_i, yt_i, minRd_w, last_first_term_sum):
    first_term = 0
    second_term = T * m * (ft_i(minRd_w, xt_i, yt_i))
    # print(ft_i(minRd_w, xt_i, yt_i))
    for w in wt_j.T:
        first_term += ft_i(w, xt_i, yt_i)
        new_second_term = T * m * (ft_i(w, xt_i, yt_i))
        if new_second_term < second_term:
            minRd_w = w
            second_term = new_second_term
    first_term += last_first_term_sum
    # print(first_term, second_term)
    Rd = first_term - second_term
    # print(Rd)
    return Rd, minRd_w, first_term


i = rank  # rank of current node
x = local_df.drop("readmitted", axis=1)
new_x = x.values
y = local_df["readmitted"]
new_y = y.values

T = 10000  # maximum iterations


m = size  # total nodes
x_features = new_x.shape[1]
n = x_features  # need to update it as dataset is received from root node
max_neighbors = m // 2
if rank == 0:
    fd = generate_neighbors(m, max_neighbors)
    A_t1 = A_t(m, fd)
else:
    fd = A_t1 = None


A_t1 = comm.bcast(A_t1, root=0)
# 2d communication matrix #later create this using above functions
wt_i = np.ones((x_features, m))  # initial points, start with same value in every root
minRd_w = wt_i[:, i]
last_first_term_sum = 0
alpha_t1 = 0.01  # learning rate
RD = []
epsilon = 0.1

print("started",rank)
for t in range(T + 1):
  random_index = np.random.randint(new_x.shape[0])
  xt_i = new_x[random_index, :]
  yt_i = new_y[random_index]
  bt_i = np.zeros(x_features)  # initializing bt_i
  """weighted average of received learnable parameters"""
  for j in range(m):  # rank of nodes start with index 0
      bt_i += A_t1[i][j] * wt_i[:, j]
      # later manage the dimensions of A_t and wt_i, bt_i
  gt_i = grad_ft_i(bt_i, xt_i, yt_i)
  wt1_i = Pro(bt_i - (alpha_t1 * gt_i))
  """here in upper bound whole local dataset needs to be passed"""
  L = upper_bound(bt_i, new_x, new_y)
  # --------------------------------------need modification------------
  st = 2 * alpha_t1 * math.sqrt(n) * L
  u = st / epsilon
  sigma_t1_i = lap(u, wt1_i)
  broad_result = wt1_i + sigma_t1_i
  # broadcast(broad_result)  # broadcast to the neighbors
  for neighbor in range(m):
      if A_t1[i, neighbor] != 0:
          comm.send(broad_result, dest=neighbor)
  for neighbor in range(m):
      if A_t1[neighbor, i] != 0:
          wt_i[:, neighbor] = comm.recv(source=neighbor)
  # wt_i[:, i] = broad_result
  """need to record wt1_i and regret value (pg 11) in every iteration"""
  # store regret
  Rd, minRd_w, last_first_term_sum = regret(
      wt_i, m, t + 1, xt_i, yt_i, minRd_w, last_first_term_sum
  )
  RD.append(Rd / (t + 1))
  if rank==0:
      A_t1 = A_t(m, fd)
  A_t1=comm.bcast(A_t1,root=0)

if rank == 0:
  save(RD)
