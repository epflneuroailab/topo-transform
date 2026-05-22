import config
import os
import h5py
import numpy as np
import pandas as pd

from .store import pickle_store
from .utils import compute_ceiling


class Assembly:
    """
    Neural assembly with efficient HDF5 random access.
    - Neural responses stored in h5py dataset (chunked by presentation for fast random access).
    - Stimulus metadata stored as pandas DataFrame in HDFStore.
    """

    def __init__(self, assembly_name, assembly=None):
        self.assembly_name = assembly_name
        self.h5_store = pickle_store.add_node("h5_assy")
        self.stim_store = pickle_store.add_node("h5_stimuli")  # store stimulus_set
        self.meta_store = pickle_store.add_node("meta")  # store metadata like ceiling

        if self.h5_store.exists(self.assembly_name) and self.stim_store.exists(self.assembly_name):
            # Reuse cached data
            self.h5_path = self.h5_store.filename(self.assembly_name)
            self.h5_file = h5py.File(self.h5_path, "r")
            self.stimuli_df = pd.read_hdf(self.stim_store.filename(self.assembly_name), key="stimuli")
            self.meta = self.meta_store.load(self.assembly_name)
        else:
            # First time: load xarray, standardize, build HDF5 and stimulus cache
            if assembly is None:
                from brainscore_vision import load_dataset
                assembly = load_dataset(self.assembly_name)

            stimuli = assembly.stimulus_set.copy()
            assembly, ceiling = self._standardize_assembly(assembly)

            # HDF5 neural data
            self.h5_path = self.h5_store.filename(self.assembly_name)
            with h5py.File(self.h5_path, "w") as f:
                f.create_dataset(
                    "assembly",
                    data=assembly.values,
                    chunks=(1, min(2, assembly.sizes["time_bin"]), assembly.sizes["neuroid"]),
                    compression="gzip",
                )
                f.create_dataset(
                    "stimulus_ids",
                    data=np.array(assembly.coords["stimulus_id"].values.astype("S")),
                )
            self.h5_file = h5py.File(self.h5_path, "r")

            # Stimulus metadata as pandas
            stimuli["stimulus_path"] = [stimuli.stimulus_paths[sid] for sid in stimuli["stimulus_id"]]
            stim_path = self.stim_store.filename(self.assembly_name)
            stimuli.to_hdf(stim_path, key="stimuli", mode="w")
            self.stimuli_df = pd.read_hdf(stim_path, key="stimuli")

            # Store metadata
            self.meta = {
                "num_presentations": self.h5_file["assembly"].shape[0],
                "num_time_bins": self.h5_file["assembly"].shape[1],
                "num_neuroids": self.h5_file["assembly"].shape[2],
                "time_bin_duration": assembly.coords["time_bin_end"].values[0] - assembly.coords["time_bin_start"].values[0],
                "ceiling": ceiling,
            }
            self.meta_store.store(self.meta, self.assembly_name)

    # ----------------------------
    # Standardization (only at build time)
    # ----------------------------
    def _standardize_assembly(self, assembly, dims=("presentation", "time_bin", "neuroid")):
        if "time_bin" not in assembly.dims:
            assembly = assembly.expand_dims("time_bin")
            assembly = assembly.assign_coords(time_bin=[-np.inf, np.inf])

        # NOTE: very coarse operation: remove all nan timepoints
        assembly = assembly.dropna("time_bin", how="any")

        # Normalize neuroids
        mean = assembly.mean(["time_bin", "presentation"])
        std = assembly.std(["time_bin", "presentation"]) + 1e-6
        assembly = (assembly - mean) / std

        if "repetition" in str(assembly.presentation):
            aver_assy_store = pickle_store.add_node("aver_assy")
            ceiling_store = pickle_store.add_node("ceiling")
            if aver_assy_store.exists(self.assembly_name) and ceiling_store.exists(self.assembly_name):
                return aver_assy_store.load(self.assembly_name).transpose(*dims), ceiling_store.load(self.assembly_name)

            from brainscore_vision.benchmark_helpers.neural_common import average_repetition
            ceiling = compute_ceiling(assembly)
            assembly = average_repetition(assembly)
            aver_assy_store.store(assembly, self.assembly_name)
            ceiling_store.store(ceiling, self.assembly_name)

        return assembly.transpose(*dims), ceiling

    # ----------------------------
    # Random-access API
    # ----------------------------
    @property
    def num_presentations(self):
        return self.h5_file["assembly"].shape[0]

    @property
    def num_time_bins(self):
        return self.h5_file["assembly"].shape[1]

    @property
    def num_neuroids(self):
        return self.h5_file["assembly"].shape[2]

    def get_data(self, idx, time_slice=None):
        """
        Random-access neural data by presentation and optional time slice.
        Returns (stimulus metadata dict, neural_response[time_slice, neuroid]).
        """
        if time_slice is None:
            target = self.h5_file["assembly"][idx]
            time_start, time_end = 0, self.num_time_bins
        else:
            target = self.h5_file["assembly"][idx, time_slice]
            time_start = max(0, time_slice.start) if time_slice.start is not None else 0
            time_end = min(self.num_time_bins, time_slice.stop) if time_slice.stop is not None else self.num_time_bins

        stimulus_id = self.h5_file["stimulus_ids"][idx].decode("utf-8")
        stimulus = self.stimuli_df.loc[self.stimuli_df["stimulus_id"] == stimulus_id].iloc[0].to_dict()
        time_bin_duration = self.meta["time_bin_duration"]
        stimulus["time_start"] = time_start * time_bin_duration
        stimulus["time_end"] = time_end * time_bin_duration
        stimulus["time_bin_duration"] = time_bin_duration
        return stimulus, target

    # ----------------------------
    # Environment configs
    # ----------------------------
    @classmethod
    def set_configs(
        cls,
        resultcaching_home=None,
        mmap_home=None,
        brainio_home=None,
        brainscore_home=None,
        torch_home=None,
        hf_home=None,
        **configs,
    ):
        configs = {
            "RESULTCACHING_HOME": resultcaching_home,
            "MMAP_HOME": mmap_home,
            "BRAINIO_HOME": brainio_home,
            "BRAINSCORE_HOME": brainscore_home,
            "TORCH_HOME": torch_home,
            "HF_HOME": hf_home,
            **{k: v for k, v in configs.items() if v is not None},
        }
        for key, value in configs.items():
            if value is not None:
                os.environ[key] = value


if __name__ == "__main__":
    cache_dir = "/mnt/scratch/ytang/migrate/cache"
    Assembly.set_configs(
        resultcaching_home=os.path.join(cache_dir, ".resultcaching"),
        mmap_home=os.path.join(cache_dir, ".mmap"),
        brainio_home=os.path.join(cache_dir, ".brainio2"),
        brainscore_home=os.path.join(cache_dir, ".brain-score"),
        torch_home=os.path.join(cache_dir, ".torch"),
        hf_home=os.path.join(cache_dir, ".hf"),
    )

    data = Assembly("McMahon2023-fMRI")
    print("Presentations:", data.num_presentations)
    stim, resp = data.get_presentation(0)
    print("First stimulus metadata:", stim)
    print("Response shape:", resp.shape)
