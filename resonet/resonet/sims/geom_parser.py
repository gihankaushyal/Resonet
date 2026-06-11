"""Parse CrystFEL .geom files and convert to dxtbx Detector objects."""
import re
from typing import Any
from dxtbx.model.detector import DetectorFactory


def _parse_axis(s: str) -> tuple:
    """Parse CrystFEL axis string e.g. '-0.999991x +0.004221y' to (x, y, z)."""
    x = y = z = 0.0
    for m in re.finditer(r'([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)\s*([xyz])', s):
        v, a = float(m.group(1)), m.group(2)
        if a == 'x':
            x = v
        elif a == 'y':
            y = v
        else:
            z = v
    return x, y, z


def _panel_sort_key(p: dict) -> tuple:
    """Numeric sort key so p10a0 sorts after p9a3, not before p1a0."""
    return tuple(int(n) for n in re.findall(r'\d+', p['name']))


def parse_geom(path: str) -> tuple:
    """Parse a CrystFEL .geom file.

    Returns:
        detector  : dxtbx Detector with one Panel per ASIC block
        panel_map : list of dicts with keys:
                    name, min_ss, max_ss, min_fs, max_fs,
                    panel_idx, n_fast, n_slow
        globals_  : dict with keys clen (m), res (px/m), photon_energy (eV)
    """
    globals_: dict[str, Any] = {}
    panels: dict[str, dict[str, Any]] = {}

    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.split(';')[0].strip()
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            if '/' in key:
                panel_name, _, field = key.partition('/')
                panel_name = panel_name.strip()
                field = field.strip()
                panels.setdefault(panel_name, {})['name'] = panel_name
                if field == 'fs':
                    panels[panel_name]['fs'] = _parse_axis(value)
                elif field == 'ss':
                    panels[panel_name]['ss'] = _parse_axis(value)
                elif field in ('corner_x', 'corner_y',
                               'min_fs', 'max_fs', 'min_ss', 'max_ss'):
                    try:
                        panels[panel_name][field] = float(value)
                    except ValueError:
                        pass
            else:
                if key in ('clen', 'photon_energy', 'res'):
                    try:
                        globals_[key] = float(value)
                    except ValueError:
                        pass  # dynamic field like /LCLS/photon_energy_eV

    required_panel_fields = {'fs', 'ss', 'corner_x', 'corner_y',
                             'min_fs', 'max_fs', 'min_ss', 'max_ss'}
    valid_panels = [
        p for p in panels.values()
        if required_panel_fields.issubset(p.keys())
    ]
    valid_panels.sort(key=_panel_sort_key)

    clen = globals_['clen']
    res = globals_['res']
    pixel_size_mm = 1000.0 / res

    panel_dicts = []
    panel_map = []

    for idx, p in enumerate(valid_panels):
        fast_axis = p['fs']
        slow_axis = p['ss']

        dot = sum(f * s for f, s in zip(fast_axis, slow_axis))
        if abs(dot) > 0.01:
            raise ValueError(
                f"Panel {p['name']}: fast/slow axes not orthogonal (dot={dot:.4f}). "
                "Use Plan B (CBF-based) fallback."
            )

        # CrystFEL origin: pixels from beam center; dxtbx origin: mm from lab origin.
        # CrystFEL z is +downstream; dxtbx z is +upstream (beam travels in -z).
        # CrystFEL +y is upward; dxtbx +y is downward — so corner_y must be negated.
        origin_mm = (
            p['corner_x'] * pixel_size_mm,
            -p['corner_y'] * pixel_size_mm,
            -clen * 1000.0,
        )
        n_fast = int(p['max_fs'] - p['min_fs'] + 1)
        n_slow = int(p['max_ss'] - p['min_ss'] + 1)

        panel_dicts.append({
            'name': p['name'],
            'type': '',
            'fast_axis': fast_axis,
            'slow_axis': slow_axis,
            'origin': origin_mm,
            'pixel_size': (pixel_size_mm, pixel_size_mm),
            'image_size': (n_fast, n_slow),
            'trusted_range': (0.0, 65536.0),
            'thickness': 0.0,
            'material': 'Si',
            'mu': 0.0,
            'gain': 1.0,
            'pedestal': 0.0,
            'identifier': '',
            'mask': [],
            'raw_image_offset': (0, 0),
            'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
        })
        panel_map.append({
            'name': p['name'],
            'min_fs': int(p['min_fs']),
            'max_fs': int(p['max_fs']),
            'min_ss': int(p['min_ss']),
            'max_ss': int(p['max_ss']),
            'panel_idx': idx,
            'n_fast': n_fast,
            'n_slow': n_slow,
        })

    detector = DetectorFactory.from_dict({'panels': panel_dicts})
    return detector, panel_map, globals_
