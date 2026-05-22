import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

from config import PLOTS_DIR
from .get_localizers import localizers
from .microstimulation import test_stimulation
from .common import MODEL_CKPT
from .plot_utils import ensure_dir, savefig


def logistic(x, a, b):
    return 1 / (1 + np.exp(-(a + b * x)))

def plot_response_curve(response, label_signal_levels, color='blue'):
    # group responses by signal level
    signal_levels = np.unique(label_signal_levels)
    signal_levels = np.sort(signal_levels)
    mean_responses = []
    for level in signal_levels:
        inds = np.where(label_signal_levels == level)[0]
        mean_responses.append(np.mean(response[inds]))
    mean_responses = np.array(mean_responses)

    # scatter and plot a logistic curve fit
    plt.scatter(signal_levels, mean_responses, alpha=0.3, label='Data', color=color)

    # Fit logistic curve (robust fallback if fit fails)
    signal_levels = np.asarray(signal_levels)
    mean_responses = np.asarray(mean_responses)
    finite_mask = np.isfinite(signal_levels) & np.isfinite(mean_responses)
    signal_levels = signal_levels[finite_mask]
    mean_responses = mean_responses[finite_mask]

    if signal_levels.size < 2 or np.all(signal_levels == signal_levels[0]):
        return float(np.mean(mean_responses)) if mean_responses.size > 0 else float("nan")

    try:
        popt, _ = scipy.optimize.curve_fit(
            logistic, signal_levels, mean_responses, p0=[0, 1], maxfev=100000
        )
        x_fit = np.linspace(signal_levels.min(), signal_levels.max(), 100)
        y_fit = logistic(x_fit, *popt)
        plt.plot(x_fit, y_fit, color=color, label='Logistic Fit', linestyle='--')
        return float(logistic(0.0, *popt))
    except Exception as exc:
        print(f"Logistic fit failed ({exc}); using mean response fallback.")
        return float(np.mean(mean_responses)) if mean_responses.size > 0 else float("nan")


def plot_stimulation_results(ckpt_name, dataset_name, save_dir=PLOTS_DIR, stimulation_mode="amplify", perturbation_params=None):
    """
    Generate plots to visualize the effects of microstimulation on model behavior.
    
    Parameters:
    -----------
    ckpt_name : str
        Name of the model checkpoint used for stimulation testing.
    dataset_name : str
        Name of the dataset used for testing.
    save_dir : str, optional
        Directory to save plots. If None, uses PLOTS_DIR from config
    stimulation_mode : str, optional
        "amplify" (default) or "ablate" to zero activations in a radius
    perturbation_params : dict, optional
        Parameters passed to the perturbation (e.g., ablation_radius_mm)
    """

    save_dir = ensure_dir(save_dir)
    results = test_stimulation(
        ckpt_name,
        dataset_name,
        stimulation_mode=stimulation_mode,
        perturbation_params=perturbation_params,
    )
    
    pre_stim = np.array(results['pre_stimulation'])  # (N_samples, )
    post_stim = np.array(results['post_stimulation'])  # list of (N_samples, ) arrays
    stimulation_locations = results['stimulation_locations']  # list of locations
    # selecitivities = results['selecitivities']  # list of selectivities
    label_signal_levels = results['label_signal_levels']
    sampled_indices = results.get('sampled_indices')
    selectivities = results.get('selectivities')
    n_locations = len(stimulation_locations)

    # Plot pre vs post stimulation predictions for each location
    plt.figure(figsize=(6, 4))

    mid_post = []
    for i in range(len(post_stim)):
        mid_p = plot_response_curve(post_stim[i], label_signal_levels, color='blue')
        mid_post.append(mid_p)

    mid_pre = plot_response_curve(pre_stim, label_signal_levels, color='green')
    suffix = f"_{stimulation_mode}" if stimulation_mode != "amplify" else ""
    path = save_dir / f"stimulation_response_curve{suffix}.svg"
    savefig(path)
    print(f"Saved stimulation response curve plot to {path}")

    mid_post = np.array(mid_post)
    mid_shifts = mid_post - mid_pre  # (n_locations, )

    if selectivities is None and sampled_indices is not None:
        if dataset_name == "afraz2006":
            t_vals_dicts, _, _ = localizers(ckpt_name, ['afraz'], ret_merged=True)
            t_vals = t_vals_dicts['face_vs_nonface'][0].flatten()
        else:
            t_vals_dicts, _, _ = localizers(ckpt_name, ret_merged=True)
            if 'face' not in t_vals_dicts:
                raise ValueError("No 'face' selectivity found in localizers for this dataset.")
            t_vals = t_vals_dicts['face'][0].flatten()
        selectivities = t_vals[sampled_indices]

    if selectivities is not None:
        plt.figure(figsize=(6, 4))
        plt.scatter(selectivities, mid_shifts, alpha=0.7)
        plt.xlabel('Selectivity (t-values for face)')
        plt.ylabel('Midpoint Shift (Post - Pre)')
        plt.title('Stimulation Effect vs Selectivity')
        path = save_dir / f"stimulation_selectivity_vs_shift{suffix}.svg"
        savefig(path)
        print(f"Saved selectivity vs shift plot to {path}")
    else:
        print("Skipping selectivity vs shift plot (no selectivities or sampled_indices provided).")

def main():
    plot_stimulation_results(
        ckpt_name=MODEL_CKPT,
        dataset_name="afraz2006",
        stimulation_mode="ablate",
        perturbation_params={"ablation_radius_mm": 3.0}
    )


if __name__ == "__main__":
    main()
