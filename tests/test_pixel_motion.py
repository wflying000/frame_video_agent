"""Actual PNG patch translations, bounded inputs and reproducible evidence."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from agent.pixel_motion import measure, validate_spec


def png(width, height, pixels):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    rows = b''.join(b'\0' + pixels[y * width * 4:(y + 1) * width * 4] for y in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


def frame(width, height, patches=()):
    pixels = bytearray(bytes((17, 29, 41, 255)) * width * height)
    for x0, y0 in patches:
        for y in range(29):
            for x in range(29):
                # Smooth distinctive block, so sparse coarse matches remain nearby.
                color = (70 + x * 5, 65 + y * 5, 110 + (x + y) * 2, 255)
                offset = ((y0 + y) * width + x0 + x) * 4
                pixels[offset:offset + 4] = bytes(color)
    return png(width, height, pixels)


class PixelMotionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        return path

    def spec(self, xy, indices=(0, 1)):
        return {'reference_frame': 0, 'frame_indices': list(indices),
                'points': [{'id': 'A', 'point_xy': list(xy)}]}

    def test_normalizes_without_mutating_and_rejects_invalid_specs(self):
        value = {'reference_frame': 2, 'frame_indices': [3, 2, 3, 0],
                 'points': [{'id': 'B', 'point_xy': [5, 6]}, {'id': 'A', 'point_xy': [0, 0]}]}
        before = copy.deepcopy(value)
        normalized = validate_spec(value)
        self.assertEqual(normalized['frame_indices'], [0, 2, 3])
        self.assertEqual([p['id'] for p in normalized['points']], ['A', 'B'])
        self.assertEqual(value, before)
        normalized['points'][0]['point_xy'][0] = 99
        self.assertEqual(value, before)
        valid = self.spec((0, 0))
        invalid = [None, {}, dict(valid, extra=True), dict(valid, reference_frame=True),
                   dict(valid, reference_frame=-1), dict(valid, reference_frame=2),
                   dict(valid, frame_indices=[]), dict(valid, frame_indices=[0, False]),
                   dict(valid, frame_indices=[0, 1, 2, 3, 4]), dict(valid, frame_indices=[0, 1.5]),
                   dict(valid, points=[]), dict(valid, points=[valid['points'][0]] * 5),
                   dict(valid, points=[valid['points'][0]] * 2)]
        for point in ({'id': 'not_valid', 'point_xy': [0, 0]}, {'id': 'é', 'point_xy': [0, 0]},
                      {'id': 'A' * 33, 'point_xy': [0, 0]}, {'id': 'A', 'point_xy': [True, 0]},
                      {'id': 'A', 'point_xy': [-1, 0]}, {'id': 'A', 'point_xy': [1.0, 0]},
                      {'id': 'A', 'point_xy': [0]}, {'id': 'A', 'point_xy': [0, 0], 'extra': 1}):
            invalid.append(dict(valid, points=[point]))
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_spec(candidate)

    def test_known_positive_and_negative_translation_uses_real_pixels(self):
        paths = {0: self.write('reference.png', frame(320, 180, [(135, 75)])),
                 1: self.write('right.png', frame(320, 180, [(175, 86)])),
                 2: self.write('left.png', frame(320, 180, [(100, 58)]))}
        result = measure(self.spec((149, 89), (2, 0, 1)), paths)
        rows = result['points'][0]['measurements']
        self.assertEqual([row['displacement_xy'] for row in rows], [[0, 0], [40, 11], [-35, -17]])
        self.assertEqual([row['mean_absolute_channel_error'] for row in rows], [0, 0, 0])
        self.assertEqual(result, measure(self.spec((149, 89), (0, 2, 1)), dict(reversed(list(paths.items())))))
        encoded = json.dumps(result, sort_keys=True)
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn('PASS', encoded)
        self.assertNotIn('FAIL', encoded)
        self.assertEqual(result['source_frames'][0]['sha256'], hashlib.sha256(paths[0].read_bytes()).hexdigest())
        self.assertEqual(result['source_frames'][0]['size'], [320, 180])
        self.assertIn('full frames', result['limitations'])

    def test_static_and_ambiguous_solid_frames_tie_to_zero_displacement(self):
        for name, data in [('textured', frame(95, 80, [(29, 25)])), ('solid', frame(95, 80))]:
            with self.subTest(name=name):
                paths = {0: self.write(name + '0.png', data), 1: self.write(name + '1.png', data)}
                row = measure(self.spec((43, 39)), paths)['points'][0]['measurements'][1]
                self.assertEqual(row['displacement_xy'], [0, 0])
                self.assertEqual(row['mean_absolute_channel_error'], 0)

    def test_clamps_full_templates_and_search_at_both_image_edges(self):
        for original, moved, point, delta in [((0, 0), (7, 5), (0, 0), [7, 5]),
                                               ((66, 51), (57, 44), (94, 79), [-9, -7])]:
            with self.subTest(point=point):
                paths = {0: self.write('edge0.png', frame(95, 80, [original])),
                         1: self.write('edge1.png', frame(95, 80, [moved]))}
                result = measure(self.spec(point), paths)['points'][0]
                self.assertEqual(result['template_xywh'], [*original, 29, 29])
                self.assertEqual(result['measurements'][1]['displacement_xy'], delta)
        path = self.write('minimum.png', frame(29, 29, [(0, 0)]))
        row = measure(self.spec((28, 28)), {0: path, 1: path})['points'][0]['measurements'][1]
        self.assertEqual(row['displacement_xy'], [0, 0])

    def test_actual_image_bounds_dimensions_missing_or_corrupt_frames_are_rejected(self):
        good = self.write('good.png', frame(95, 80))
        bad = self.write('bad.png', b'not a PNG')
        small = self.write('small.png', frame(28, 29))
        other = self.write('other.png', frame(96, 80))
        for spec, frames in [(self.spec((95, 0)), {0: good, 1: good}),
                             (self.spec((0, 80)), {0: good, 1: good}),
                             (self.spec((0, 0)), {0: good}),
                             (self.spec((0, 0)), {0: good, 1: bad}),
                             (self.spec((0, 0)), {0: small, 1: small}),
                             (self.spec((0, 0)), {0: good, 1: other}),
                             (self.spec((0, 0)), {0: good, True: good}),
                             (self.spec((0, 0)), {0: good, 1: self.root / 'missing.png'})]:
            with self.subTest(spec=spec, frames=frames), self.assertRaises(ValueError):
                measure(spec, frames)


if __name__ == '__main__':
    unittest.main()
