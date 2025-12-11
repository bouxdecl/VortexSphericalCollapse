#!/usr/bin/env python3
"""
Generate a vorticity movie from Athena++ .athdf snapshots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from athena_collapse_analysis.utils import collapse_param_decomposition
from athena_collapse_analysis.io.ath_io import get_hdf_files, open_hdf_files_with_collapse



def make_vorticity_movie(path_simu, outname=None,
                         vort_type="simulation",
                         crop=None,
                         vmin=0.0, vmax=None, cmap='RdBu_r',
                         nz_slice=0, fps=10, dpi=150):
    """
    Create a vorticity movie from a folder of Athena++ snapshots.

    Parameters
    ----------
    vort_type : str
        "simulation" → ω = dv_y/dx - dv_x/dy
        "physical"   → ω_phys using metric coefficients and rescaled coords
    crop : tuple or None
        (xmin, xmax, ymin, ymax) in rescaled coordinates. If None → full domain.
    """

    if outname is None:
        outname = f"vorticity_movie_{vort_type}.mp4"

    # -------------------------------
    # 1. Collect files
    # -------------------------------
    files = get_hdf_files(path_simu)
    Nframes = len(files)
    print(f"Found {Nframes} frames.")

    # -------------------------------
    # 2. Load geometry + first collapse snapshot
    # -------------------------------
    data0 = open_hdf_files_with_collapse(path_simu, [files[0]], read_every=1)
    x = data0["x1"]
    y = data0["x2"]
    Nz = len(data0["x3"])
    midz = nz_slice if nz_slice is not None else (Nz // 2)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # -------------------------------
    # 3. Prepare figure
    # -------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    cbar = None

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    # -------------------------------
    # 4. Init function
    # -------------------------------
    def init():
        # nothing to initialise — each frame clears and redraws
        return []

    # -------------------------------
    # 5. Update function
    # -------------------------------
    def update(i):
        file_i = files[i]
        print(f"[{i+1}/{Nframes}] Loading {file_i}")
        data = open_hdf_files_with_collapse(path_simu, [file_i])

        vx = data["v1"][0, :, :, midz]
        vy = data["v2"][0, :, :, midz]
        t = data["time"][0]

        dvy_dx = np.gradient(vy, dx, axis=0)
        dvx_dy = np.gradient(vx, dy, axis=1)

        # -------------------------------
        # Compute vorticity
        # -------------------------------
        if vort_type == "simulation":
            vort = dvy_dx - dvx_dy
            S, alpha = 1.0, 1.0
            Xplot, Yplot = x, y

        elif vort_type == "physical":
            R = data['Rglobal'][0]
            Lz = data['Lzglobal'][0]
            S, alpha = collapse_param_decomposition(np.array([R]), np.array([Lz]))
            S, alpha = S[0], alpha[0]

            gxx = S**2 * alpha**-4
            gyy = S**2 * alpha**2
            omega_tilda = gyy * dvy_dx - gxx * dvx_dy
            vort = omega_tilda * alpha

            # Rescaled coordinates
            Xplot = x * alpha**(-3/2)
            Yplot = y * alpha**(3/2)

        else:
            raise ValueError("vort_type must be 'simulation' or 'physical'.")

        # -------------------------------
        # Crop if requested
        # -------------------------------
        # -------------------------------
        # Crop if requested (find index ranges on 1D coords)
        # -------------------------------
        if crop is not None:
            try:
                xmin, xmax, ymin, ymax = crop
            except Exception:
                raise ValueError("crop must be a 4-tuple: (xmin, xmax, ymin, ymax)")

            ix = np.where((Xplot >= xmin) & (Xplot <= xmax))[0]
            iy = np.where((Yplot >= ymin) & (Yplot <= ymax))[0]
            if ix.size == 0 or iy.size == 0:
                raise ValueError(
                    f"crop region yields empty selection: x in [{xmin},{xmax}], y in [{ymin},{ymax}]"
                )
            Xplot = Xplot[ix]
            Yplot = Yplot[iy]
            vort_plot = vort[np.ix_(ix, iy)]
        else:
            vort_plot = vort

        # -------------------------------
        # Determine color range like single-plot function
        # -------------------------------
        vmin_plot = 0.0 if vmin is None else vmin
        vmax_plot = np.max(vort_plot) if vmax is None else vmax
        if vmax_plot <= vmin_plot:
            vmax_plot = np.max(np.abs(vort_plot))
            vmin_plot = -vmax_plot

        # -------------------------------
        # Clear axis and draw
        # -------------------------------
        nonlocal cbar
        ax.clear()
        pc = ax.pcolormesh(Xplot, Yplot, vort_plot.T,
                           cmap=cmap, vmin=vmin_plot, vmax=vmax_plot,
                           shading='auto')

        # manage colorbar (remove old one if present)
        if cbar is not None:
            try:
                cbar.remove()
            except Exception:
                pass
        cbar = fig.colorbar(pc, ax=ax)
        cbar.set_label('vorticity')

        if vort_type == "simulation":
            ax.set_title(f'vorticity (simulation) at t = {t:.3f}')
        else:
            ax.set_title(f'vorticity (physical) at t = {t:.3f}, '
                         f'S={S:.3f}, α={alpha:.3f}')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    # -------------------------------
    # 6. Create animation
    # -------------------------------
    ani = animation.FuncAnimation(fig, update,
                                  frames=Nframes,
                                  init_func=init,
                                  blit=False)

    print("Saving movie...")
    ani.save(outname, writer='ffmpeg', dpi=150)
    print(f"Movie saved: {outname}")




if __name__ == "__main__":
    from athena_collapse_analysis.config import RAW_DIR
    # Example simulation path
    path_simu = os.path.join(RAW_DIR, "typical_simu_20251311/")

    # --- Make the vorticity movie ---
    make_vorticity_movie(path_simu, outname="omegaTilda_movie.mp4",
                         vort_type="physical", crop=None)
