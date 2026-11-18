"""Approximate patch translations measured solely from actual PNG pixels.

The values are auxiliary evidence, not an action or acceptance decision. A
reviewing model must inspect the full frames to confirm the matched identity.
"""
import hashlib
from pathlib import Path
import re
import subprocess

from .rendering import decode_rgba, png_dimensions


_SIZE = 29
_RADIUS = _SIZE // 2
_SEARCH_X = 120
_SEARCH_Y = 40
_COARSE_STEP = 4
_SPARSE_STEP = 5
_REFINE_RADIUS = 5


def _integer(value):
    return type(value) is int and value >= 0


def validate_spec(value):
    """Normalize a bounded JSON specification, raising ValueError if invalid."""
    if not isinstance(value, dict) or set(value) != {'reference_frame', 'frame_indices', 'points'}:
        raise ValueError('pixel motion spec requires reference_frame, frame_indices and points')
    reference = value['reference_frame']
    indices = value['frame_indices']
    if not _integer(reference):
        raise ValueError('pixel motion reference_frame must be a nonnegative integer')
    if not isinstance(indices, list) or not indices or any(not _integer(i) for i in indices):
        raise ValueError('pixel motion frame_indices must be a nonempty list of nonnegative integers')
    indices = sorted(set(indices))
    if len(indices) > 4 or reference not in indices:
        raise ValueError('pixel motion requires at most four distinct frames including the reference')
    points = value['points']
    if not isinstance(points, list) or not 1 <= len(points) <= 4:
        raise ValueError('pixel motion requires one to four points')
    normalized = []
    ids = set()
    for point in points:
        if not isinstance(point, dict) or set(point) != {'id', 'point_xy'}:
            raise ValueError('pixel motion points require only id and point_xy')
        identity, xy = point['id'], point['point_xy']
        if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9]{1,32}', identity):
            raise ValueError('pixel motion point id must contain 1–32 ASCII letters or digits')
        if identity in ids:
            raise ValueError('pixel motion point ids must be unique')
        if not isinstance(xy, list) or len(xy) != 2 or any(not _integer(i) for i in xy):
            raise ValueError('pixel motion point_xy must contain two nonnegative integers')
        ids.add(identity)
        normalized.append({'id': identity, 'point_xy': list(xy)})
    return {'reference_frame': reference, 'frame_indices': indices,
            'points': sorted(normalized, key=lambda point: point['id'])}


def _axis_candidates(low, high, origin):
    # Include the exact reference and bounds even when clamping shifts the grid.
    return sorted(set(range(low, high + 1, _COARSE_STEP)) | {low, high, origin})


def _template(pixels, width, x, y, step):
    return [((dy * width + dx) * 4 + channel,
             pixels[((y + dy) * width + x + dx) * 4 + channel])
            for dy in range(0, _SIZE, step) for dx in range(0, _SIZE, step)
            for channel in range(3)]


def _match(pixels, width, height, x0, y0, sparse, dense):
    xmin, xmax = max(0, x0 - _SEARCH_X), min(width - _SIZE, x0 + _SEARCH_X)
    ymin, ymax = max(0, y0 - _SEARCH_Y), min(height - _SIZE, y0 + _SEARCH_Y)

    def score(x, y, template):
        offset = (y * width + x) * 4
        error = sum(abs(pixels[offset + delta] - value) for delta, value in template)
        # Explicit, stable tie-break; repeated/untextured regions remain ambiguous.
        return error, (x - x0) ** 2 + (y - y0) ** 2, y, x

    _, _, best_y, best_x = min(score(x, y, sparse)
        for y in _axis_candidates(ymin, ymax, y0)
        for x in _axis_candidates(xmin, xmax, x0))
    error, _, y, x = min(score(x, y, dense)
        for y in range(max(ymin, best_y - _REFINE_RADIUS), min(ymax, best_y + _REFINE_RADIUS) + 1)
        for x in range(max(xmin, best_x - _REFINE_RADIUS), min(xmax, best_x + _REFINE_RADIUS) + 1))
    return x, y, round(error / len(dense), 6)


def measure(spec, frames: dict[int, Path]):
    """Return deterministic image-derived evidence; never infer motion from code.

All selected frames must have the same dimensions and fit a 29×29 template.
Points can touch image boundaries: the complete template is clamped and its
point offset is preserved. PNG read/decode or specification errors raise
ValueError. Unselected frames are ignored, and no files are written.
"""
    spec = validate_spec(spec)
    if not isinstance(frames, dict) or any(not _integer(index) for index in frames):
        raise ValueError('pixel motion frames must map nonnegative integer indices to PNG paths')
    decoded, sources = {}, []
    size = None
    for index in spec['frame_indices']:
        if index not in frames:
            raise ValueError(f'pixel motion source frame {index} is missing')
        try:
            data = Path(frames[index]).read_bytes()
            current_size = png_dimensions(data)
            if min(current_size) < _SIZE:
                raise ValueError('pixel motion frames must be at least 29×29 pixels')
            if size is not None and current_size != size:
                raise ValueError('pixel motion source frame dimensions must match')
            size = current_size
            decoded[index] = decode_rgba(data, size)
        except (OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
            raise ValueError(f'pixel motion could not read/decode frame {index}: {error}') from error
        sources.append({'frame_index': index, 'sha256': hashlib.sha256(data).hexdigest(),
                        'size': list(size)})
    width, height = size
    for point in spec['points']:
        x, y = point['point_xy']
        if x >= width or y >= height:
            raise ValueError(f'pixel motion point {point["id"]} is outside the actual PNG bounds')
    reference = spec['reference_frame']
    rows = []
    for point in spec['points']:
        cx, cy = point['point_xy']
        x0, y0 = min(max(cx - _RADIUS, 0), width - _SIZE), min(max(cy - _RADIUS, 0), height - _SIZE)
        sparse = _template(decoded[reference], width, x0, y0, _SPARSE_STEP)
        dense = _template(decoded[reference], width, x0, y0, 1)
        measurements = []
        for index in spec['frame_indices']:
            x, y, error = (x0, y0, 0.0) if index == reference else _match(
                decoded[index], width, height, x0, y0, sparse, dense)
            measurements.append({'frame_index': index, 'point_xy': [cx + x - x0, cy + y - y0],
                                 'displacement_xy': [x - x0, y - y0],
                                 'mean_absolute_channel_error': error})
        rows.append({'id': point['id'], 'reference_point_xy': list(point['point_xy']),
                     'template_xywh': [x0, y0, _SIZE, _SIZE], 'measurements': measurements})
    return {
        'version': 'pixel_motion_v1',
        'method': 'approximate_rgb_template_matching_from_actual_png_pixels',
        'limitations': 'Approximate image patch matching can be ambiguous under repeated texture, '
            'occlusion, deformation or camera motion. The reviewing model must confirm feature '
            'identity from the full frames. These measurements do not establish action correctness '
            'or acceptance.',
        'spec': spec,
        'algorithm': {'template_size': [_SIZE, _SIZE], 'search_radius_xy': [_SEARCH_X, _SEARCH_Y],
                      'coarse_step': _COARSE_STEP, 'sparse_step': _SPARSE_STEP,
                      'refine_radius': _REFINE_RADIUS, 'channels': 'RGB',
                      'error': 'mean_absolute_channel_error_0_255',
                      'coarse_grid': 'clamped_start_step_plus_reference_and_bounds',
                      'tie_break': 'lowest_error_then_shortest_squared_displacement_then_y_then_x',
                      'boundary': 'clamp_complete_template_and_search; preserve_point_offset'},
        'source_frames': sources,
        'points': rows,
    }
