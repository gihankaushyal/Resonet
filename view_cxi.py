"""Quick CXI frame viewer — arrow keys to scroll, escape to quit.

Usage:
  python view_cxi.py <file.cxi>                    # raw unassembled view
  python view_cxi.py <file.cxi> <detector.geom>    # assembled view
"""
import sys
import re
import h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def parse_geom_panels(geom_path):
    """Extract panel geometry from a CrystFEL .geom file.

    Returns list of dicts with: name, corner_x, corner_y, fs_x, fs_y, ss_x, ss_y,
    min_fs, max_fs, min_ss, max_ss.
    """
    panels = {}
    with open(geom_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(';') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key, val = key.strip(), val.strip()
            m = re.match(r'^(\w+)/(corner_x|corner_y|min_fs|max_fs|min_ss|max_ss|fs|ss)$', key)
            if not m:
                continue
            name, field = m.group(1), m.group(2)
            panels.setdefault(name, {})['name'] = name
            if field in ('fs', 'ss'):
                # parse axis vector e.g. "-0.999991x -0.004221y" or bare "-y"
                def _coeff(m):
                    if m is None: return 0.0
                    s = m.group(1)
                    return 1.0 if s in ('', '+') else (-1.0 if s == '-' else float(s))
                mx = re.search(r'([+-]?(?:[0-9.]+(?:[eE][+-]?[0-9]+)?)?)x', val)
                my = re.search(r'([+-]?(?:[0-9.]+(?:[eE][+-]?[0-9]+)?)?)y', val)
                panels.setdefault(name, {})[field + '_x'] = _coeff(mx)
                panels.setdefault(name, {})[field + '_y'] = _coeff(my)
            else:
                panels.setdefault(name, {})[field] = float(val)

    required = {'corner_x', 'corner_y', 'fs_x', 'fs_y', 'ss_x', 'ss_y',
                'min_fs', 'max_fs', 'min_ss', 'max_ss'}
    return [p for p in panels.values() if required.issubset(p)]


def assemble(raw_frame, panels):
    """Assemble panels onto a canvas using full affine placement.

    For each pixel (i_fs, i_ss) in a panel:
        x =   corner_x + i_fs * fs_x + i_ss * ss_x
        y = -(corner_y + i_fs * fs_y + i_ss * ss_y)  # negated: CrystFEL +y=up, canvas row 0=top

    Handles any axis orientation including flipped/rotated panels.
    Returns (canvas, x_min, y_min) where x_min/y_min are the canvas origin offsets.
    """
    # First pass: determine canvas bounds
    all_xs, all_ys = [], []
    for p in panels:
        nf = int(p['max_fs'] - p['min_fs'] + 1)
        ns = int(p['max_ss'] - p['min_ss'] + 1)
        for i_fs, i_ss in [(0, 0), (nf - 1, 0), (0, ns - 1), (nf - 1, ns - 1)]:
            all_xs.append(p['corner_x'] + i_fs * p['fs_x'] + i_ss * p['ss_x'])
            all_ys.append(-(p['corner_y'] + i_fs * p['fs_y'] + i_ss * p['ss_y']))

    x_min = int(np.floor(min(all_xs)))
    y_min = int(np.floor(min(all_ys)))
    x_max = int(np.ceil(max(all_xs))) + 1
    y_max = int(np.ceil(max(all_ys))) + 1

    canvas = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)

    # Second pass: scatter pixels onto canvas
    for p in panels:
        nf = int(p['max_fs'] - p['min_fs'] + 1)
        ns = int(p['max_ss'] - p['min_ss'] + 1)
        ss0 = int(p['min_ss'])
        fs0 = int(p['min_fs'])
        patch = raw_frame[ss0:ss0 + ns, fs0:fs0 + nf]  # (ns, nf)

        i_fs, i_ss = np.meshgrid(np.arange(nf), np.arange(ns))  # both (ns, nf)
        col = np.round(p['corner_x'] + i_fs * p['fs_x'] + i_ss * p['ss_x']).astype(int) - x_min
        row = np.round(-(p['corner_y'] + i_fs * p['fs_y'] + i_ss * p['ss_y'])).astype(int) - y_min

        mask = (row >= 0) & (row < canvas.shape[0]) & (col >= 0) & (col < canvas.shape[1])
        canvas[row[mask], col[mask]] = patch[mask]

    return canvas, x_min, y_min


# --- CLI args ---
show_labels = '--labels' in sys.argv
_positional = [a for a in sys.argv[1:] if not a.startswith('--')]
if not _positional:
    print(__doc__)
    sys.exit(1)
cxi_path = _positional[0]
geom_path = _positional[1] if len(_positional) > 1 else None

h = h5py.File(cxi_path, "r")
imgs = h["/entry_1/data_1/data"]
n = imgs.shape[0]
print(f"{cxi_path}: {n} frames, raw shape per frame: {imgs.shape[1:]}")

panels = None
if geom_path:
    panels = parse_geom_panels(geom_path)
    print(f"Loaded {len(panels)} panels from {geom_path} — showing assembled view")
else:
    print("No .geom file — showing raw unassembled view (pass geom as 2nd arg for assembled)")

matplotlib.rcParams['keymap.back'].remove('left')
matplotlib.rcParams['keymap.forward'].remove('right')


def get_img(i):
    raw = imgs[i]
    return assemble(raw, panels)[0] if panels else raw


fig, ax = plt.subplots(figsize=(8, 8) if panels else (4, 12))
fig.i = 0

if panels:
    img0, canvas_x_min, canvas_y_min = assemble(imgs[0], panels)
else:
    img0 = imgs[0]
nonzero = img0[img0 > 0]
vmax0 = float(np.mean(nonzero) + 3.5 * np.std(nonzero)) if len(nonzero) else 1.0
im = ax.imshow(img0, vmin=0, vmax=vmax0, cmap="gray_r",
               aspect="equal" if panels else "auto", origin="upper")
title_sfx = "(assembled)" if panels else "(unassembled)"
ax.set_title(f"frame 1/{n}  {title_sfx}")
fig.colorbar(im, ax=ax, fraction=0.02)

if panels and show_labels:
    for p in panels:
        nf = int(p['max_fs'] - p['min_fs'] + 1)
        ns = int(p['max_ss'] - p['min_ss'] + 1)
        cx = (p['corner_x'] + (nf - 1) / 2 * p['fs_x'] + (ns - 1) / 2 * p['ss_x']) - canvas_x_min
        ry = (-(p['corner_y'] + (nf - 1) / 2 * p['fs_y'] + (ns - 1) / 2 * p['ss_y'])) - canvas_y_min
        ax.text(cx, ry, p.get('name', '?'), fontsize=4, color='red',
                ha='center', va='center', clip_on=True)


def press(event):
    if event.key == 'right':
        fig.i = min(fig.i + 1, n - 1)
    elif event.key == 'left':
        fig.i = max(fig.i - 1, 0)
    elif event.key == 'escape':
        plt.close()
        return
    else:
        return
    img = get_img(fig.i)
    im.set_data(img)
    im.set_extent([0, img.shape[1], img.shape[0], 0])
    nonzero = img[img > 0]
    im.set_clim(0, float(np.mean(nonzero) + 3.5 * np.std(nonzero)) if len(nonzero) else 1.0)
    ax.set_title(f"frame {fig.i + 1}/{n}  {title_sfx}")
    fig.canvas.draw_idle()


fig.canvas.mpl_connect('key_press_event', press)
plt.tight_layout()
plt.show()
