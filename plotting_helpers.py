from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib.patheffects as patheffects

def rolling_average(arr, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    return np.convolve(
        np.pad(arr, window_size // 2, mode='edge'),
        np.ones(window_size) / window_size,
        mode='valid'
    )

def plot_fpr_tpr_warning_time(fprs, tprs, warning_times, title='', save_name=''):
    """
    Plot TPR vs FPR with line segments colored by Warning Time,
    and a black 'outline' around the colored line.
    """
    # Prepare line segments
    points = np.column_stack([fprs, tprs]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 1) Create a thicker black LineCollection for the "outline"
    lc_outline = LineCollection(
        segments,
        linewidth=6,      # Outline thickness
        color='black'     # Solid black color
    )

    # 2) Create the color-mapped LineCollection (on top)
    lc = LineCollection(
        segments,
        cmap=plt.cm.plasma,
        norm=plt.Normalize(vmin=400, vmax=1500),
        linewidth=4        # Slightly thinner
    )
    lc.set_array(warning_times)  # Assign data for color mapping

    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    ax.add_collection(lc_outline)  # Add black outline first
    ax.add_collection(lc)         # Add color-mapped line second
    # set font size for everything
    plt.rcParams.update({'font.size': 16})
    # Invisible line to set axis limits
    ax.plot(fprs, tprs, alpha=0)

    # Colorbar
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Median Warning Time (ms)")

    # Axis labels and limits
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(fprs.min(), fprs.max())
    ax.set_ylim(tprs.min(), tprs.max())
    if save_name != "":
        plt.savefig(f'plots/{save_name}.pdf', bbox_inches='tight')
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

def plot_two_roc_curves(
    fprs1, tprs1, warning_times1,
    fprs2, tprs2, warning_times2,
    title='',
    save_name='',
    vmin=400,  # Adjust based on your data
    vmax=1500
):
    """
    Plot two separate ROC curves (TPR vs FPR), each with line segments 
    colored by their respective Warning Time. Both curves have black outlines.
    """

    # Prepare line segments for the first curve
    points1 = np.column_stack([fprs1, tprs1]).reshape(-1, 1, 2)
    segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)

    # Outline
    lc_outline1 = LineCollection(
        segments1,
        linewidth=6,
        color='black'
    )
    # Color-mapped collection
    lc1 = LineCollection(
        segments1,
        cmap=plt.cm.plasma,
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
        linewidth=4
    )
    lc1.set_array(warning_times1)

    # Prepare line segments for the second curve
    points2 = np.column_stack([fprs2, tprs2]).reshape(-1, 1, 2)
    segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)

    lc_outline2 = LineCollection(
        segments2,
        linewidth=6,
        color='black'
    )
    lc2 = LineCollection(
        segments2,
        cmap=plt.cm.plasma,
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
        linewidth=4
    )
    lc2.set_array(warning_times2)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams.update({'font.size': 16})

    # Add the black-outlined collections (first) and color collections (second)
    ax.add_collection(lc_outline1)
    ax.add_collection(lc1)
    ax.add_collection(lc_outline2)
    ax.add_collection(lc2)

    # Force the axes to include all points by plotting invisible lines 
    ax.plot(fprs1, tprs1, alpha=0)
    ax.plot(fprs2, tprs2, alpha=0)

    # Create colorbar from one of the line collections (same norm/cmap)
    cbar = plt.colorbar(lc2, ax=ax)
    cbar.set_label("Median Warning Time (ms)")

    # Axis labels
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    # Compute combined axis limits
    all_fprs = np.concatenate([fprs1, fprs2])
    all_tprs = np.concatenate([tprs1, tprs2])
    ax.set_xlim(all_fprs.min(), all_fprs.max())
    ax.set_ylim(all_tprs.min(), all_tprs.max())

    # Title, grid, and optional save
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)

    if save_name:
        plt.savefig(f'plots/{save_name}.pdf', bbox_inches='tight')

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

def plot_heatmap_profile(x, profile, heatmap, profile_name=""):
    """
    Plots Temperature vs Radial Position with line segments colored by Density.
    
    Args:
        x (array-like): Radial positions.
        temperature (array-like): Temperature values corresponding to `x`.
        density (array-like): Density values corresponding to `x`.
        title (str): Title of the plot.
    """
    # Define a bright red-to-blue colormap
    bright_red_blue = LinearSegmentedColormap.from_list("BrightRdBu", ["blue", "white", "red"], N=256)

    # Create segments for the line
    points = np.array([x, profile]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a LineCollection, with colors mapped to density
    lc = LineCollection(segments, cmap=bright_red_blue, norm=plt.Normalize(vmin=min(heatmap), vmax=max(heatmap)))
    lc = LineCollection(segments, cmap=bright_red_blue, norm=plt.Normalize(vmin=-0.01, vmax=0.01))

    lc.set_array(heatmap)  # Set density values to color the segments

    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 2))
    line = ax.add_collection(lc)
    ax.plot(x, profile, alpha=0)  # Add an invisible line for proper axis scaling

    # Add a colorbar for density
    colorbar = plt.colorbar(line, ax=ax)
    colorbar.set_label('Tearing impact')

    # Set axis limits and labels
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(profile.min(), profile.max())
    ax.set_xlabel("$\mathbf{\psi_N}$")
    ax.set_ylabel(profile_name)
    #ax.set_title(title)

    # Show the plot
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_2_profiles_heatmaps(
    x, profile_index, profile_names, short_profile_names, profiles, heatmaps, time_indices,
    title="", averaging_window_size=9, save_name=""
):
    """
    Plot 2 profiles with associated 'heatmaps' (color-coded impact).
    
    Parameters
    ----------
    x : 1D array
        The radial coordinate, shape = (33,) typically.
    profile_index : int
        Which profile to plot (e.g. 0 for 'Te', 1 for 'Ti', etc.).
    profile_names : list
        List of profile names (e.g., ['Te', 'Ti', 'ne', ...]).
    profiles : 2D array
        Shape = (n_times, 33 * n_profiles).
        E.g., row i is the data for all profiles at time i.
    heatmaps : 2D array
        Same shape as `profiles`, containing "impact" values for color coding.
    time_indices : tuple or list of length 2
        The two time steps to plot, e.g. [t1, t2].
    title : str, optional
        Plot title.
    averaging_window_size : int, optional
        Size of the rolling average window (must be odd).
    save_name : str, optional
        If not empty, save figure to 'plots/{save_name}.svg'.

    Returns
    -------
    None
    """

    def rolling_average(arr, window_size):
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd.")
        return np.convolve(
            np.pad(arr, window_size // 2, mode='edge'),
            np.ones(window_size) / window_size,
            mode='valid'
        )
    
    profile_name = profile_names[profile_index]
    
    # Extract 2 profiles (no third one), smooth out profiles
    profile1 = rolling_average(profiles[time_indices[0], 33*profile_index : 33*(profile_index+1)], 3)
    profile2 = rolling_average(profiles[time_indices[1], 33*profile_index : 33*(profile_index+1)], 3)

   # Compute rolling averages for corresponding heatmaps
    heatmap1 = rolling_average(
        heatmaps[time_indices[0], 33*profile_index : 33*(profile_index+1)],
        averaging_window_size
    )
    heatmap2 = rolling_average(
        heatmaps[time_indices[1], 33*profile_index : 33*(profile_index+1)],
        averaging_window_size
    )

    # Define a bright red-to-blue colormap
    bright_red_blue = LinearSegmentedColormap.from_list(
        "BrightRdBu", ["blue", "white", "red"], N=256
    )

    # Combine both heatmaps to find a global color range or use a fixed one
    all_heatmaps = np.concatenate([heatmap1, heatmap2])
    # For example, a fixed range:
    vmin, vmax = -0.0075, 0.0075

    fig, ax = plt.subplots(figsize=(6, 4))

    # Helper function to build a LineCollection for a given profile & heatmap
    def make_linecollection(profile, heatmap):
        # Create segments for the line
        points = np.column_stack([x, profile]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap=bright_red_blue,
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
            linewidth=3,
        )
        lc.set_array(heatmap)
        lc.set_path_effects([
            patheffects.Stroke(linewidth=5, foreground='black'),
            patheffects.Normal()
        ])
        return lc

    # Create 2 line collections
    lc1 = make_linecollection(profile1, heatmap1)
    lc2 = make_linecollection(profile2, heatmap2)

    # Add them to the same Axes for overplotting
    for lc in [lc1, lc2]:
        ax.add_collection(lc)

    # If you'd like a legend for each line name, one simple approach is to 
    # plot invisible lines with labels:
    ax.plot(
        x, profile1, alpha=0,
        label=(
            rf'{short_profile_names[profile_index]} at $t_1$ is '
            f'{"destabilizing" if np.sum(heatmap1) > 0 else "stabilizing"} '
            f'by {np.abs(np.sum(heatmap1)):.2f}'
        )
    )
    ax.plot(
        x, profile2, alpha=0,
        label=(
            rf'{short_profile_names[profile_index]} at $t_2$ is '
            f'{"destabilizing" if np.sum(heatmap2) > 0 else "stabilizing"} '
            f'by {np.abs(np.sum(heatmap2)):.2f}'
        )
    )

    # Create one colorbar for both lines
    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=bright_red_blue)
    sm.set_array([])  
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("TM impact", fontweight='bold')

    # Set axis limits
    ax.set_xlim(x.min(), x.max())

    # Combine both profiles to set a global y-range if you wish
    all_profiles = np.concatenate([profile1, profile2])
    # ax.set_ylim(0, all_profiles.max())  # or any range you prefer

    # Labels and title
    ax.set_xlabel(r"$\mathbf{\psi_N}$")
    ax.set_ylabel(f"{profile_name}", fontweight='bold')
    ax.set_title(title, fontweight='bold')

    plt.grid(True, linestyle="--", alpha=0.5)

    # -------------------------
    # Annotate each curve on the left side (instead of at the peak)
    # -------------------------
    profiles_list = [profile1, profile2]
    time_names = ['t1', 't2']

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # for i, prof in enumerate(profiles_list):
    #     # Place label near left edge
    #     x_pos = x_min + 0.05 * x_range  # 5% in from left
    #     # Shift label above the start of the profile
    #     y_pos = prof[0] + 0.05 * y_range - 0.3  

    #     ax.text(
    #         x_pos,
    #         y_pos,
    #         time_names[i],
    #         ha="left",
    #         va="bottom",
    #         fontsize=10,
    #         bbox=dict(boxstyle="round,pad=0.3", alpha=0.7, color="white")
    #     )

    ax.legend(loc='upper right', fontsize=10)

    # Optionally save the figure
    if save_name != "":
        plt.savefig(f'plots/{save_name}.pdf', bbox_inches='tight')

    plt.show()

def plot_3_profiles_heatmaps(
    x,
    profile_index,
    profile_names,
    short_profile_names,
    profiles,
    heatmaps,
    time_indices,
    title="",
    averaging_window_size=9,
    save_name=""
):
    """
    Plot 3 profiles with associated 'heatmaps' (color-coded impact).

    Parameters
    ----------
    x : 1D array
        The radial coordinate, shape = (33,) typically.
    profile_index : int
        Which profile to plot (e.g. 0 for 'Te', 1 for 'Ti', etc.).
    profile_names : list
        List of profile names (e.g., ['Te', 'Ti', 'ne', ...]).
    short_profile_names : list
        Short version of profile names (for legend).
    profiles : 2D array
        Shape = (n_times, 33 * n_profiles).
        E.g., row i is the data for all profiles at time i.
    heatmaps : 2D array
        Same shape as `profiles`, containing "impact" values for color coding.
    time_indices : tuple or list of length 3
        The three time steps to plot, e.g. [t1, t2, t3].
    title : str, optional
        Plot title.
    averaging_window_size : int, optional
        Size of the rolling average window (must be odd).
    save_name : str, optional
        If not empty, save figure to 'plots/{save_name}.pdf'.

    Returns
    -------
    None
    """

    def rolling_average(arr, window_size):
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd.")
        # Pad with edge values and then convolve
        return np.convolve(
            np.pad(arr, window_size // 2, mode='edge'),
            np.ones(window_size) / window_size,
            mode='valid'
        )

    profile_name = profile_names[profile_index]

    # Extract 3 profiles for the chosen profile_index
    p1 = rolling_average(profiles[time_indices[0], 33*profile_index : 33*(profile_index+1)], 5) # should be 3
    p2 = rolling_average(profiles[time_indices[1], 33*profile_index : 33*(profile_index+1)], 5)
    p3 = rolling_average(profiles[time_indices[2], 33*profile_index : 33*(profile_index+1)], 5)

    # Compute rolling averages for the corresponding heatmaps
    h1 = rolling_average(
        heatmaps[time_indices[0], 33*profile_index : 33*(profile_index+1)],
        averaging_window_size
    )
    h2 = rolling_average(
        heatmaps[time_indices[1], 33*profile_index : 33*(profile_index+1)],
        averaging_window_size
    )
    h3 = rolling_average(
        heatmaps[time_indices[2], 33*profile_index : 33*(profile_index+1)],
        averaging_window_size
    )

    # Define a bright red-to-blue colormap
    bright_red_blue = LinearSegmentedColormap.from_list(
        "BrightRdBu", ["blue", "white", "red"], N=256
    )

    # Combine all heatmaps to determine the color range (or fix them if you prefer)
    all_heatmaps = np.concatenate([h1, h2, h3])
    # Example: fixed range
    vmin, vmax = -0.0075, 0.0075

    fig, ax = plt.subplots(figsize=(6, 4))

    # Helper function to build a LineCollection
    def make_linecollection(profile, heatmap):
        # Create segments for the line
        points = np.column_stack([x, profile]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap=bright_red_blue,
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
            linewidth=3,
        )
        lc.set_array(heatmap)
        lc.set_path_effects([
            patheffects.Stroke(linewidth=5, foreground='black'),
            patheffects.Normal()
        ])
        return lc

    # Create 3 line collections
    lc1 = make_linecollection(p1, h1)
    lc2 = make_linecollection(p2, h2)
    lc3 = make_linecollection(p3, h3)

    # Add them to the same Axes for over-plotting
    for lc in [lc1, lc2, lc3]:
        ax.add_collection(lc)

    # We create invisible lines for each profile to insert legend entries
    # This also showcases the net "destabilizing or stabilizing" effect from the sum of the heatmap
    sums = [np.sum(h1), np.sum(h2), np.sum(h3)]
    p_list = [p1, p2, p3]
    t_labels = [f"t{idx+1}" for idx in range(3)]  # or any custom labels you prefer
    t_labels = [r'$t_1$', r'$t_2$', r'$t_3$']
    for i, (prof, s) in enumerate(zip(p_list, sums)):
        ax.plot(
            x, prof, alpha=0,
            label=(
                f'{short_profile_names[profile_index]} at {t_labels[i]} is '
                f'{"destabilizing" if s > 0 else "stabilizing"} '
                f'by {abs(s):.2f}'
            )
        )

    # Create one colorbar for all lines
    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=bright_red_blue)
    sm.set_array([])
    ticks = np.linspace(vmin, vmax, 7)
    cbar = plt.colorbar(sm, ax=ax, ticks=ticks)
    cbar.set_label("TM impact", fontweight='bold', fontsize=8)
    cbar.ax.tick_params(labelsize=8)  # Set colorbar tick label size

    # Set axis limits
    ax.set_xlim(x.min(), x.max())

    # Optionally set a global y-range
    all_profiles = np.concatenate([p1, p2, p3])
    # e.g., ax.set_ylim(0, all_profiles.max())

    # Labels and title
    ax.set_xlabel(r"$\mathbf{\psi_N}$")
    ax.set_ylabel(f"{profile_name}", fontweight='bold')
    ax.set_title(title, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.5)

    # Annotate each curve on the left side
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # for i, prof in enumerate(p_list):
    #     x_pos = x_min + 0.05 * x_range  # 5% in from left
    #     # Shift label slightly from the starting value
    #     y_pos = prof[0] + 0.05 * y_range - 0.3 * i  # offset each label a bit

    #     ax.text(
    #         x_pos,
    #         y_pos,
    #         t_labels[i],
    #         ha="left",
    #         va="bottom",
    #         fontsize=9,
    #         bbox=dict(boxstyle="round,pad=0.3", alpha=0.7, color="white")
    #     )

    ax.legend(loc='upper right', fontsize=9)

    # Optionally save the figure
    if save_name != "":
        plt.savefig(f'plots/{save_name}.pdf', bbox_inches='tight')

    plt.show()