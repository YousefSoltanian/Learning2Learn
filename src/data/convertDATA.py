import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def read_jld2_file(filepath):
    simulations = []
    with h5py.File(filepath, 'r') as f:
        # Grab the first dataset in file
        dataset_key = list(f.keys())[0]
        ref_list = f[dataset_key]

        for ref in ref_list:
            sim_dict = {}
            if isinstance(ref, h5py.Reference):
                group = f[ref]
                for key in group.keys():
                    try:
                        sim_dict[key] = group[key][()]
                    except Exception:
                        sim_dict[key] = group[key]  # Keep reference if not directly readable
                simulations.append(sim_dict)
    return simulations

def save_as_npz(sim_data, filename):
    np.savez_compressed(filename, data=sim_data, allow_pickle=True)
    print(f"Saved converted data to: {filename}")

def plot_trajectory(sim_dict):
    x_history = sim_dict["x_history"]
    # Ensure x_history is loaded as numpy array
    x_history = np.array(x_history)
    px = x_history[:, 0]
    py = x_history[:, 1]

    plt.figure()
    plt.plot(px, py, marker='o')
    plt.xlabel("x (px)")
    plt.ylabel("y (py)")
    plt.title("Lunar Lander Trajectory (First Simulation)")
    plt.grid(True)
    plt.show()

def main():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    jld2_path = os.path.join(data_dir, "LunarLander_Complete_Info_Peer.jld2")
    npz_path = os.path.join(data_dir, "LunarLander_Complete_Info_Peer.npz")

    print("Reading JLD2...")
    sim_data = read_jld2_file(jld2_path)

    print("Saving to NPZ...")
    save_as_npz(sim_data, npz_path)

    print("Plotting first trajectory...")
    plot_trajectory(sim_data[0])

if __name__ == "__main__":
    main()
