import pandas as pd

from scripts.scheduler import TrainScheduler
from scripts.utils import EarlyStopper, gen_adj_mat, load_dataset

# Configs
ES = EarlyStopper(patience=50, min_delta=0.5)
CP_PATH = "/content/gdrive/MyDrive/CS512/checkpoint"
HORIZON = 12
INPUT_DIM = 2
OUTPUT_DIM = 1
SEQ_SIZE = 12
GRU_ARGS = {
    "max_diffusion_step": 2,
    "cl_decay_steps": 1000,
    "filter_type": "laplacian",
    "num_nodes": 319,
    "num_rnn_layers": 1,
    "rnn_units": 2
}
BATCH_SIZE = 64
CL_DECAY_STEPS = 2000
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
    scheduler.train(epoch=2, early_stop=ES)
