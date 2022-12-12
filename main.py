import pandas as pd

from scripts.scheduler import TrainScheduler
from scripts.utils import EarlyStopper, gen_adj_mat, load_dataset

# Configs
ES = EarlyStopper(patience=3, min_delta=-0.08)
CP_PATH = "/content/gdrive/MyDrive/CS512/checkpoint"
HORIZON = 12
INPUT_DIM = 2
OUTPUT_DIM = 1
SEQ_SIZE = 12
BATCH_SIZE = 64
CL_DECAY_STEPS = 2000
LR = 0.007
EPS = 1e-4
STEPS = (8, 16, 24, 32)
GRU_ARGS = {
    "max_diffusion_step": 2,
    "cl_decay_steps": CL_DECAY_STEPS,
    "filter_type": "laplacian",
    # "filter_type": "dual_random_walk",
    "num_nodes": 319,
    "num_rnn_layers": 2,
    "rnn_units": 64
}
# load
if __name__ == "__main__":
    train, val, test = load_dataset("Dataset/pems_all_2022_nonnan.h5", batch_size=BATCH_SIZE)
    # train
    with open("Dataset/sensor_id.txt", "r") as f:
        sensor_ids = [int(sid) for sid in f.read().strip().split(",")]
    dist = pd.read_csv("Dataset/distances_bay_2017.csv")
    _, adj_mat = gen_adj_mat(dist, sensor_ids)

    scheduler = TrainScheduler(adj_mat, train, val, test, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, horizon=HORIZON,
                               seq_size=SEQ_SIZE,
                               num_sensors=len(sensor_ids), std=train.features[..., 0].std(),
                               mean=train.features[..., 0].mean(), cp_path=CP_PATH, cl_decay_steps=CL_DECAY_STEPS,
                               gru_args=GRU_ARGS)
    # scheduler.train(epoch=2, early_stop=ES)
    label, pred = scheduler.predict("test")
    print(label, pred)
