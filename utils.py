import re
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate
import numpy as np
import warnings
import matplotlib.pyplot as plt
from IPython import display

warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)


def interpolate_cb_from_simulation_results(
    simulation_results,
    unit,
    time_new=None,
    axial_positions_new=None,
    roi_channels=None,              # indices, slice, or boolean mask over channels
    roi_axial=None,                 # indices, slice, or boolean mask over axial cells
    phase_fraction=1.0,             # scalar, (Ch,), or (Ch, X_new)
    A_roi_override=None,          # None (use unit areas), "ones" (default), or (Ch,) array
    flow_rate=None,                 # scalar or (Ch,) -> returns flux (mol/s) grid
    volume_correct=False            # if True (and no flow_rate), multiply by effective volume
):
    """
    Interpolate bulk concentrations to (time_new, axial_positions_new).

    Returns:
        out : ndarray, shape (T_new, C, Ch, X_new)
            - Concentration grid (input units, e.g. mM) by default.
            - If flow_rate is given (scalar or (Ch,)): flux grid (mol/s) via Q * c.
            - If volume_correct=True (and no flow_rate):
                amount-per-cell grid via c * A_use[ch] * dz * phase_fraction * ROI_mask.

    Notes:
      - volume_correct multiplies only the ROI (roi_channels × roi_axial); values
        outside the ROI are left as-is (i.e., not multiplied by volume).
        If you prefer zero outside ROI, set those masks and then multiply yourself.
      - A_roi_override controls only the volume multiplier used when volume_correct=True.
        It does not change the geometry of the unit or the interpolation.
      - When flow_rate is provided, volume_correct is ignored to avoid mixed units.
    """
    # ---------- helpers ----------
    def _get_ncol(u):
        if hasattr(u, "ncol"):
            return int(u.ncol)
        if hasattr(u, "discretization") and hasattr(u.discretization, "ncol"):
            return int(u.discretization.ncol)
        raise AttributeError("Unit missing 'ncol' (or 'discretization.ncol').")

    def _to_mask(idx_like, length):
        if idx_like is None:
            return np.ones(length, dtype=bool)
        if isinstance(idx_like, slice):
            m = np.zeros(length, dtype=bool)
            m[idx_like] = True
            return m
        arr = np.asarray(idx_like)
        if arr.dtype == bool:
            if arr.shape != (length,):
                raise ValueError("Boolean mask has wrong length.")
            return arr
        # assume indices
        m = np.zeros(length, dtype=bool)
        m[arr] = True
        return m

    def _areas_for_integration(A_ch, override):
        if override is None:
            return A_ch
        if isinstance(override, str) and override.lower() == "ones":
            return np.ones_like(A_ch, dtype=float)
        arr = np.asarray(override, dtype=float)
        if arr.shape != A_ch.shape:
            raise ValueError("A_roi_override must be 'ones', None, or shape (Ch,).")
        return arr

    # ---------- geometry & raw data ----------
    name = unit.name
    n_ch = int(unit.nchannel)
    n_comp = int(unit.component_system.n_comp)

    A_ch = np.asarray(unit.channel_cross_section_areas, dtype=float)  # (Ch,)
    if A_ch.shape != (n_ch,):
        raise ValueError("unit.channel_cross_section_areas must have shape (nchannel,)")

    c_raw = simulation_results.solution[name].bulk.solution           # (T, X_native, Ch, C)
    time = simulation_results.solution[name].bulk.time
    ncol_native = _get_ncol(unit)
    axial_positions_native = np.linspace(0.0, float(unit.length), ncol_native)

    if time_new is None:
        time_new = time
    if axial_positions_new is None:
        axial_positions_new = axial_positions_native

    time_new = np.asarray(time_new, dtype=float)
    axial_positions_new = np.asarray(axial_positions_new, dtype=float)
    T_new = len(time_new)
    X_new = len(axial_positions_new)

    # ---------- interpolate per channel (time, axial) -> comp ----------
    interpolators = []
    for ch in range(n_ch):
        interpolators.append(
            RegularGridInterpolator(
                (time, axial_positions_native),
                c_raw[:, :, ch, :],                 # (T, X, C)
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
        )

    tt, xx = np.meshgrid(time_new, axial_positions_new, indexing="ij")
    pts = np.column_stack([tt.ravel(), xx.ravel()])  # (T_new*X_new, 2)

    out = np.empty((T_new, n_comp, n_ch, X_new), dtype=float)
    for ch, interp in enumerate(interpolators):
        vals = interp(pts).reshape(T_new, X_new, n_comp)     # (T_new, X_new, C)
        out[:, :, ch, :] = np.transpose(vals, (0, 2, 1))     # -> (T_new, C, Ch, X_new)

    # ---------- optional: flux mode via flow_rate ----------
    if flow_rate is not None:
        # Ignore volume_correct in flux mode to avoid mixed units
        Qch = np.asarray(flow_rate, dtype=float)
        if Qch.ndim == 0:
            Qch = np.full(n_ch, Qch, dtype=float)
        if Qch.shape != (n_ch,):
            raise ValueError("flow_rate must be scalar or shape (nchannel,).")
        Qch = Qch.reshape(1, 1, n_ch, 1)  # broadcast to (1,1,Ch,1)
        return out * Qch                  # mol/s if c in mol/m^3 and Q in m^3/s

    # ---------- optional: volume correction (amount per cell) ----------
    if volume_correct:
        dz = float(unit.length) / float(X_new) if X_new > 0 else 0.0
        A_use = _areas_for_integration(A_ch, A_roi_override)          # (Ch,)

        # ROI mask over (Ch, X_new)
        ch_mask = _to_mask(roi_channels, n_ch)
        x_mask = _to_mask(roi_axial, X_new)
        ROI = np.outer(ch_mask, x_mask).astype(float)                 # (Ch, X_new)

        # phase fraction to (Ch, X_new)
        phi = np.asarray(phase_fraction, dtype=float)
        if phi.ndim == 0:
            phi = np.full((n_ch, X_new), phi, dtype=float)
        elif phi.ndim == 1:
            if phi.shape[0] != n_ch:
                raise ValueError("phase_fraction with ndim=1 must have length nchannel.")
            phi = phi[:, None] * np.ones((1, X_new), dtype=float)
        elif phi.shape != (n_ch, X_new):
            raise ValueError("phase_fraction must be scalar, (Ch,), or (Ch, X_new).")

        V_eff = (A_use[:, None] * dz) * phi * ROI                     # (Ch, X_new)

        # Multiply only inside ROI; outside stays as concentrations
        out = out * np.where(V_eff > 0, V_eff, 1.0)[None, None, :, :]

    return out


def process_roi_txt_to_csv(input_txt, output_csv_path):

    # Read the input text file
    try:
        with open(input_txt, 'r') as file:
            input_txt = file.read()
    except Exception as e:
        raise Exception(f"Error reading file: {e}")

    # Split text into lines
    lines = input_txt.splitlines()

    # More flexible ROI header pattern
    # - Match "# ROI" or "#  Roi"
    # - Capture ROI number
    # - Capture cumulated distance [mm]
    roi_header_pattern = r"^#\s*Roi\s*(\d+).*?cumulated distance \[mm\]:\s*([\d\.]+)"
    data_row_pattern = r"^\s*([\d\.]+)\s+([\d\.]+)"

    # Containers for data
    roi_data = {}
    times = set()

    current_roi = None
    current_distance = None

    # Parse the text
    for line in lines:
        roi_match = re.match(roi_header_pattern, line, flags=re.IGNORECASE)
        data_match = re.match(data_row_pattern, line)

        if roi_match:
            # Extract ROI number and distance
            current_roi = int(roi_match.group(1))
            current_distance = float(roi_match.group(2))
            roi_data[current_distance] = {}  # Initialize for this ROI
        elif data_match and current_roi is not None:
            # Extract time and count data
            time = float(data_match.group(1))
            count = float(data_match.group(2))
            roi_data[current_distance][time] = count
            times.add(time)

    # Create a DataFrame
    sorted_times = sorted(times)
    df = pd.DataFrame(index=sorted_times)

    for distance, data in roi_data.items():
        column_data = [data.get(time, None) for time in sorted_times]
        df[distance] = column_data

    # Save to CSV
    df.index.name = "time_min"
    df.to_csv(output_csv_path)

    return df


def plot_all_columns(data_array):
    """
    Function to plot all columns in a 2D array. Each column is stored
    in a dynamically created variable named roi1, roi2, ..., roiN.

    Parameters:
    data_array (ndarray): 2D array where each column is a separate dataset to plot.
    """
    # Dictionary to hold dynamically created variables
    roi_dict = {}

    # Loop through columns in the data array
    for i in range(data_array.shape[1]):
        # Dynamically name the variable
        var_name = f'roi{i+1}'
        roi_dict[var_name] = data_array[:, i]

        # Plot the column
        plt.plot(roi_dict[var_name], label=var_name)

    # Add legend and labels
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plots of All Columns')
    plt.show()

    return roi_dict  # Return dictionary of dynamically created variables


def make_parameter_transforms(param_bounds):
    """
    Build normalize/denormalize/analyze functions bound to a specific param_bounds dict.

    param_bounds : dict mapping parameter name -> (min_val, max_val). Parameters with
    min_val > 0 and a ratio max_val/min_val > 100 are log-scaled; others are linear.

    Returns (normalize_parameters, denormalize_parameters, analyze_parameters).
    """
    def normalize_parameters(x):
        normalized_x = []
        for i, param in enumerate(param_bounds):
            min_val, max_val = param_bounds[param]

            if min_val > 0 and (max_val / min_val) > 100:  # If the range is large, use log scaling
                norm_param = (np.log(x[i]) - np.log(min_val)) / (np.log(max_val) - np.log(min_val))
            else:  # For compact ranges, use min-max scaling
                norm_param = (x[i] - min_val) / (max_val - min_val)

            normalized_x.append(norm_param)

        return np.array(normalized_x)

    def denormalize_parameters(norm_x):
        x = []
        for i, param in enumerate(param_bounds):
            min_val, max_val = param_bounds[param]

            if min_val > 0 and (max_val / min_val) > 100:  # Log scaling for large ranges
                param_value = np.exp(norm_x[i] * (np.log(max_val) - np.log(min_val)) + np.log(min_val))
            else:  # Min-max scaling for compact ranges
                param_value = norm_x[i] * (max_val - min_val) + min_val

            x.append(param_value)

        return np.array(x)

    def analyze_parameters(norm_param_values, param_bounds=param_bounds):
        """
        Analyze parameters and print their names, denormalized values, and normalized values.

        Args:
            norm_param_values (list): Normalized parameter values (0-1 range).
            param_bounds (dict): Dictionary with parameter names as keys and boundary tuples as values.

        Returns:
            None: Prints the analysis.
        """
        print(f"{'Name':<30}{'Denormalized Value':<20}{'Normalized Value':<10}")
        print("-" * 65)

        # Denormalize parameters
        param_values = denormalize_parameters(norm_param_values)

        for i, (name, bounds) in enumerate(param_bounds.items()):
            value = param_values[i]
            norm_param = norm_param_values[i]

            print(f"{name:<30}{value:<20.8e}{norm_param:<10.2f}")

    return normalize_parameters, denormalize_parameters, analyze_parameters


def find_roots(x, y):
    """Finds the roots of a signal crossing zero using interpolation."""
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)


def compute_hmax_times(results, time_range):
    """
    Computes the half-max times for each ROI in the results using find_roots.

    Parameters:
    - results (dict): Dictionary containing ROI results with cubic splines and half-max values.
    - time_range (array): Array of time values to use for interpolation (e.g., np.linspace).

    Returns:
    - hmax_times (dict): Dictionary with ROI names as keys and the first half-max time as values.
    """
    hmax_times = {}

    for roi_name, data in results.items():
        if roi_name == "ratios":  # Skip the "ratios" key if present
            continue

        cubic_spline = data['cubic_spline']
        half_max = data['half_max']

        y_interp = cubic_spline(time_range) - half_max
        roots = find_roots(time_range, y_interp)

        hmax_times[roi_name] = roots[0] if len(roots) > 0 else np.nan

    return hmax_times


def process_and_plot_rois(rois, t, savefig_name=None):
    """
    Processes and plots multiple ROIs from a dictionary using cubic spline interpolation and calculates max ratios.

    Parameters:
    - rois (dict): Dictionary where keys are ROI names (e.g., 'roi1', 'roi2') and values are 1D arrays of data.
    - t (array): Time array corresponding to the ROIs.
    - savefig_name (str, optional): if given, saves the figure to this path (PDF).

    Returns:
    - results (dict): Dictionary where keys are ROI names and values are a dictionary with:
        - 'cubic_spline': The cubic spline object for the ROI.
        - 'max_value': Maximum value of the interpolated data.
        - 'half_max': Half-maximum value of the interpolated data.
        - 'roots': Roots of the cubic spline.
        - 'ratios': List of ratios between consecutive max values (calculated for all ROIs together).
    """
    results = {}
    maxima = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure()
    plt.style.use('default')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    for i, (roi_name, roi_data) in enumerate(rois.items()):
        # Interpolate using Cubic Spline
        cu = interpolate.CubicSpline(t.ravel(), roi_data.ravel())
        x_interp = np.linspace(0, max(t), 500)
        y_interp = cu(x_interp)

        # Compute max and half-max
        max_value = np.max(y_interp)
        half_max = max_value / 2
        maxima.append(max_value)

        # Plot interpolated curve
        color = colors[i % len(colors)]
        plt.plot(x_interp, y_interp, 'k', linewidth=1.2, label="Spline")

        # Highlight max and half-max
        plt.hlines(max_value, -9, max(t) + 10, linestyle=(0, (1, 1)), color=color, label="Maximum")
        plt.hlines(half_max, -9, max(t) + 10, linestyle='--', color=color, label="Half-Maximum")

        # Plot the original points
        plt.plot(t, roi_data, marker='.', markersize=8, linestyle='none', color=color, label="Experimental Data")

        # Save results
        results[roi_name] = {
            'cubic_spline': cu,
            'max_value': max_value,
            'half_max': half_max,
            'roots': cu.roots()
        }

    # Compute ratios of maxima
    ratios = [maxima[i] / maxima[i + 1] for i in range(len(maxima) - 1)]
    results['ratios'] = ratios

    plt.xlabel("Time [min]", fontsize=15)
    plt.ylabel("Activity [a.u.]", fontsize=15)

    # Remove duplicate labels while keeping the first occurrence
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = {}
    unique_handles = []

    for handle, label in zip(handles, labels):
        if label not in unique_labels:  # Keep first occurrence
            unique_labels[label] = handle
            unique_handles.append(handle)

    plt.legend(unique_handles, unique_labels.keys(), fontsize=12)

    if savefig_name:
        plt.savefig(savefig_name, format="pdf")
    plt.show()

    return results


def compute_jacobian(f, x, eps=None, relative_step=1e-4):
    """Numerical Jacobian using central differences with a relative (multiplicative) step.

    The step is proportional to each parameter's own magnitude rather than an additive
    constant, so it never crosses zero even for parameters with very small real-unit
    values (e.g. dispersion coefficients ~1e-9, where an additive step sized for O(1)
    values would push the perturbed value negative and trip CADET-Process's bounds
    validation). The 1e-4 default matches the step size empirically found stable (not
    dominated by the simulation's own numerical noise floor) during the R6 migration.
    """
    x = np.asarray(x, dtype=float)
    n_params = len(x)
    f_x = np.asarray(f(x)).ravel()
    n_outputs = len(f_x)
    J = np.zeros((n_outputs, n_params))

    if eps is None:
        eps = relative_step * np.abs(x)

    for i in range(n_params):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps[i]
        x_minus[i] -= eps[i]

        f_plus = np.asarray(f(x_plus)).ravel()
        f_minus = np.asarray(f(x_minus)).ravel()

        J[:, i] = (f_plus - f_minus) / (2 * eps[i])

    return J


def plot_live_fit(time, calc_data, experimental_data, obj_history):
    """Live-updating two-panel plot: current fit (left) and objective progress (right)."""
    display.clear_output(wait=True)
    fig, (ax_fit, ax_obj) = plt.subplots(1, 2, figsize=(12, 4))

    if calc_data.ndim == 1:
        ax_fit.plot(time / 60, calc_data, linewidth=1)
        ax_fit.set_prop_cycle(None)
        ax_fit.plot(time / 60, experimental_data, ".")
    else:
        for col in range(calc_data.shape[1]):
            ax_fit.plot(time / 60, calc_data[:, col], linewidth=1)
        ax_fit.set_prop_cycle(None)
        for col in range(experimental_data.shape[1]):
            ax_fit.plot(time / 60, experimental_data[:, col], ".")
    ax_fit.set_xlabel("Time (min)")
    ax_fit.set_title(f"NMSRE = {obj_history[-1]:.6f}" if obj_history else "")

    ax_obj.plot(obj_history, linewidth=1, color="steelblue")
    ax_obj.set_xlabel("Evaluation")
    ax_obj.set_ylabel("NMSRE")
    ax_obj.set_title("Objective progress")
    if len(obj_history) > 1:
        ax_obj.set_yscale("log")

    plt.tight_layout()
    display.display(fig)
    plt.close(fig)


def compute_parameter_uncertainties(residuals_fn, x, use_pinv=True):
    """Absolute parameter uncertainties from the residual Jacobian/covariance.

    Uses a numerical Jacobian of residuals_fn at x to estimate the covariance
    C = (J^T J)^{-1} * var(r) and returns sqrt(|diag(C)|), in the same units as x.
    Divide by abs(x) to get relative uncertainties (fraction).

    use_pinv=True  (default): Moore-Penrose pseudo-inverse, stable for near-singular J^T J.
    use_pinv=False           : regular inverse, exact when J^T J is well-conditioned.
    """
    J = compute_jacobian(residuals_fn, x)
    inv_fn = np.linalg.pinv if use_pinv else np.linalg.inv
    JTJ_inv = inv_fn(J.T @ J)
    sigma_squared = np.var(residuals_fn(x))
    C = JTJ_inv * sigma_squared
    return np.sqrt(np.abs(np.diag(C)))
