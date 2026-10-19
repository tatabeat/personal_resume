"""Small, device-free smoke tests for the research prototype."""

import unittest
from pathlib import Path

import numpy as np

from digivocal.frontend import tokens_from_numbered_pinyin
from digivocal.gesture_planner import plan_gestures
from digivocal.synthesizer import synthesize_timeline
from digivocal.voice_profile import VoiceProfile


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_pinyin_tokens(self):
        tokens = tokens_from_numbered_pinyin("ni3 hao3")
        self.assertEqual(len(tokens), 2)
        self.assertEqual([token.tone for token in tokens], [3, 3])

    def test_shipped_profile_loads(self):
        profile = VoiceProfile.load(ROOT / "profiles" / "digivocal_voice_profile.json")
        self.assertEqual(profile.sample_rate, 48_000)
        self.assertTrue(profile.is_digital_throat_profile)

    def test_short_synthesis_is_finite(self):
        profile = VoiceProfile.wave_analysis_clone()
        timeline = plan_gestures(tokens_from_numbered_pinyin("a1"), profile)
        audio = synthesize_timeline(timeline, profile, compute_backend="cpu")
        self.assertGreater(audio.size, 0)
        self.assertTrue(np.all(np.isfinite(audio)))


if __name__ == "__main__":
    unittest.main()
