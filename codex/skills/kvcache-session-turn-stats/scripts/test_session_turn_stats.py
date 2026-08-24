"""验证图片筛选、去重及原有轮次统计。"""
import tempfile
import unittest
from datetime import time
from pathlib import Path

from session_turn_stats import analyze


class ImageStatsTest(unittest.TestCase):
    def run_log(self, images, early=False):
        lines = list(images)
        for session, base in [('text', 100), ('valid', 200), ('mixed', 300)]:
            for i in range(25):
                task = f'{base+i:08x}'
                hour = '05' if early and session == 'text' and i == 0 else '07'
                lines.append(f'2026-08-27 {hour}:00:{i:02d},000 INFO kv_manager.py:174 allocate kv for task {task} with token_num {1000+i*100}, capacity 5000, hit_length {i*100}\n')
                lines.append(f'INFO kvcache_manager.py:2835 task {task} session {session} release\n')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'log'
            path.write_text(''.join(lines))
            return analyze(path, time(6), 10, 50)

    def image(self, task, width=632, height=1400, tokens=495, identity='image-a'):
        return f'INFO worker.py:495 task {task:08x} image MMImage(mode=binary, metadata=ImageMetadata(num_tokens={tokens}, width={width}, height={height}, hash=abc, image_id={identity}, input_offset=123))\n'

    def test_mixed_session_excluded_and_duplicates_removed(self):
        valid = self.image(200)
        result = self.run_log([valid, valid, self.image(201), self.image(300), self.image(324, width=630)])
        self.assertEqual(result['eligible_sessions'], 2)
        self.assertEqual(result['eligible_requests'], 50)
        self.assertEqual(result['diagnostics']['excluded_image_size_or_unknown_dimensions'], 1)
        self.assertEqual(result['images_632x1400']['unique_images'], 1)
        self.assertEqual(result['images_632x1400']['tokens_per_image'], 495)
        self.assertEqual(len(result['turns_15_to_25']), 12)
        self.assertEqual(result['turns_15_to_25'][-1]['requests'], 22)
        self.assertEqual(result['turns_15_to_25'][-1]['average_new_tokens'], 100)
        self.assertEqual(result['turns_15_to_25'][-1]['average_rewrite_tokens'], 900)

    def test_rotated_and_unknown_dimensions_excluded(self):
        result = self.run_log([self.image(200, width=1400, height=632), self.image(300, width='None')])
        self.assertEqual(result['eligible_sessions'], 1)
        self.assertIsNone(result['images_632x1400']['tokens_per_image'])
        self.assertIsNone(result['images_632x1400']['average_tokens_per_image'])

    def test_distribution_missing_tokens_and_early_filter(self):
        result = self.run_log([self.image(200), self.image(201, tokens=500, identity='b'), self.image(202, tokens='None', identity='c')], early=True)
        self.assertEqual(result['diagnostics']['excluded_seen_before_start'], 1)
        stats = result['images_632x1400']
        self.assertIsNone(stats['tokens_per_image'])
        self.assertEqual(stats['average_tokens_per_image'], 497.5)
        self.assertEqual(stats['images_missing_token_count'], 1)
        self.assertEqual(stats['token_distribution'], [{'tokens': 495, 'images': 1}, {'tokens': 500, 'images': 1}])


if __name__ == '__main__':
    unittest.main()
