"""Fictitious records and synthetic pixels only; no study artifacts are loaded."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from PIL import Image

from policy_confidence.evidence_gate import resolve, derive_all_gated
from policy_confidence.roi import render
from policy_confidence.vlm_interface import (
    parse_verdict, strict_verdict, build_messages, softmax,
    verify_candidate_interface, extract_step0_hidden,
)

ROOT = Path(__file__).resolve().parents[1]
TABLE = json.loads((ROOT / 'POLICY_RULE_TABLE.json').read_text())
INCLUDE, EXCLUDE = 'PB-EXHAUST-INCLUDE-v1', 'PB-EXHAUST-EXCLUDE-v1'


def record(flame, exhaust):
    facts = {key: {'state': value, 'sources': [] if value == 'UNREVIEWED' else
                  [{'type': 'SYNTHETIC_TEST_ONLY', 'note': 'No human observation'}]}
             for key, value in [('flame_connection', flame), ('exhaust_connection', exhaust)]}
    return {'image_id': 'synthetic-image', 'fact_record_id': 'synthetic-facts',
            'revision': 1, 'facts': facts, 'definition_issue': None,
            'image_sha256': '0' * 64, 'target_identifier': 'synthetic-target',
            'possibilities': {'status': 'UNREVIEWED', 'branches': []},
            'policy_links': [{'review_item_id': f'synthetic-policy-{i}', 'policy_id': p,
                              'policy_sha256': TABLE['policies'][p]['policy_sha256']}
                             for i, p in enumerate((INCLUDE, EXCLUDE))]}


class PolicyRuleTests(unittest.TestCase):
    def test_flame_priority_skips_exhaust(self):
        r = record('TRUE', 'UNREVIEWED')
        for p in (INCLUDE, EXCLUDE):
            result = resolve(TABLE, r, p)
            self.assertEqual(result['y_rule_derived_for_evaluation'], 'YES')
            self.assertEqual(result['needed_fact_ids'], [])
            self.assertEqual(result['applied_clauses'][0]['priority'], 1)
        self.assertEqual(r['facts']['exhaust_connection']['state'], 'UNREVIEWED')

    def test_difference_only_under_confirmed_conditions(self):
        for flame in ('TRUE', 'FALSE'):
            for exhaust in ('TRUE', 'FALSE'):
                r = record(flame, exhaust)
                labels = [resolve(TABLE, r, p)['y_rule_derived_for_evaluation'] for p in (INCLUDE, EXCLUDE)]
                self.assertEqual(labels[0] != labels[1], flame == 'FALSE' and exhaust == 'TRUE')

    def test_unknown_observation_is_not_false_or_unknown_label(self):
        r = record('FALSE', 'UNDETERMINABLE')
        self.assertIsNone(resolve(TABLE, r, INCLUDE)['y_rule_derived_for_evaluation'])
        self.assertEqual(resolve(TABLE, r, EXCLUDE)['y_rule_derived_for_evaluation'], 'NO')
        self.assertEqual(r['facts']['exhaust_connection']['state'], 'UNDETERMINABLE')

    def test_unreviewed_is_not_unknown_label(self):
        for p in (INCLUDE, EXCLUDE):
            result = resolve(TABLE, record('UNREVIEWED', 'UNREVIEWED'), p)
            self.assertIsNone(result['y_rule_derived_for_evaluation'])

    def test_raw_no_remains_excluded_when_possibility_support_unresolved(self):
        r = record('FALSE', 'UNDETERMINABLE')
        r['possibilities'] = {'status': 'SUPPORTED_SET_RECORDED', 'complete_set_confirmed': True,
            'sources': [{'type': 'SYNTHETIC_TEST_ONLY'}],
            'branches': [{'branch_id': 'NEITHER', 'rationale': 'Synthetic branch selection only.'}]}
        result = resolve(TABLE, r, INCLUDE)
        self.assertEqual(result['raw_rule_output'], 'NO')
        self.assertEqual(result['rule_resolution_status'], 'EVIDENCE_SUPPORT_UNRESOLVED')
        self.assertIsNone(result['y_rule_derived_for_evaluation'])

    def test_two_policies_share_facts_without_mutation(self):
        r = record('FALSE', 'TRUE')
        original = copy.deepcopy(r)
        rows = derive_all_gated(TABLE, {'images': {r['image_id']: r}})
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['facts_sha256'], rows[1]['facts_sha256'])
        self.assertEqual(rows[0]['shared_fact_record_id'], rows[1]['shared_fact_record_id'])
        self.assertEqual(r, original)

    def test_definition_issue_remains_null(self):
        r = record('TRUE', 'TRUE')
        r['definition_issue'] = 'Synthetic unresolved scope'
        self.assertIsNone(resolve(TABLE, r, INCLUDE)['y_rule_derived_for_evaluation'])

    def test_missing_observation_provenance_rejected(self):
        r = record('TRUE', 'FALSE')
        r['facts']['flame_connection']['sources'] = []
        with self.assertRaises(ValueError):
            resolve(TABLE, r, INCLUDE)


class InterfaceTests(unittest.TestCase):
    def test_unknown_parse_and_format_are_separate(self):
        self.assertEqual(parse_verdict('판단불가, synthetic'), 'UNKNOWN')
        self.assertEqual(strict_verdict('판단불가, synthetic'), 'UNKNOWN')
        self.assertEqual(parse_verdict('판'), 'UNKNOWN')
        self.assertIsNone(strict_verdict('판'))
        self.assertIsNone(parse_verdict('unparseable text'))

    def test_no_extra_evaluation_fields_in_messages(self):
        marker = object()
        messages = build_messages(marker, '<image> fixed policy question')
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['role'], 'user')
        self.assertIs(messages[0]['content'][0]['image'], marker)
        self.assertEqual(messages[0]['content'][1]['text'], 'fixed policy question')

    def test_candidate_softmax_is_stable(self):
        scores = np.array(softmax([1000., 999., 998.]))
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(float(scores.sum()), 1.)

    def test_chat_interface_prefix_and_first_token_contract(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return {'네': [10], '아니요': [20, 21], '판단불가': [30, 31], '아': [20], '판': [30]}[text]
            def decode(self, ids):
                return {10: '네', 20: '아', 30: '판'}[ids[0]]
        class Processor:
            tokenizer = Tokenizer()
            mismatch = False
            def apply_chat_template(self, messages, **kwargs):
                self.last_mode = messages[0]['content'][0]['image'].mode
                prefix = [1, 2]
                if messages[-1]['role'] == 'assistant':
                    prefix = ([9, 2] if self.mismatch else prefix) + self.tokenizer.encode(messages[-1]['content'])
                return {'input_ids': torch.tensor([prefix])}
        processor = Processor()
        with tempfile.TemporaryDirectory() as directory:
            Image.new('L', (8, 8), 0).save(Path(directory) / 'synthetic.png')
            row = {'image_project_relative': 'synthetic.png', 'prompt': 'synthetic prompt'}
            out = verify_candidate_interface(processor, row, Path(directory))
            self.assertEqual(out['candidate_ids'], {'YES': 10, 'NO': 20, 'UNKNOWN': 30})
            self.assertEqual(processor.last_mode, 'RGB')
            processor.mismatch = True
            with self.assertRaises(RuntimeError):
                verify_candidate_interface(processor, row, Path(directory))

    def test_hidden_is_final_nonpadding_prompt_position(self):
        hidden = torch.zeros((1, 3, 4096))
        hidden[0, 1, 0] = 2.
        class DummyProjection:
            def lm_head(self, values):
                return values[:, :3]
        inputs = {'input_ids': torch.tensor([[1, 2, 0]]), 'attention_mask': torch.tensor([[1, 1, 0]])}
        vector, meta = extract_step0_hidden(SimpleNamespace(hidden_states=((hidden,),)), inputs,
                                            DummyProjection(), torch.tensor([2., 0., 0.]))
        self.assertEqual(vector.shape, (4096,))
        self.assertEqual(meta['prompt_token_index'], 1)
        self.assertEqual(meta['lm_head_full_vocab_max_abs_diff_vs_generation_raw_logits'], 0.)


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font = os.environ.get('POLICY_TEST_FONT')
        if not cls.font or not Path(cls.font).is_file():
            raise unittest.SkipTest('Supply POLICY_TEST_FONT for the synthetic renderer checks')
        cls.config = json.loads((ROOT / 'ROI_RENDERING_CONFIG.json').read_text())

    def test_interior_unchanged_and_no_input_mutation(self):
        arr = np.arange(160 * 120 * 3, dtype=np.uint32).reshape(120, 160, 3).astype(np.uint8)
        original = Image.fromarray(arr)
        marked, changed = render(original, {'roi_bounds_inclusive_xyxy': [50, 30, 120, 90]}, self.config, self.font)
        self.assertTrue(np.array_equal(np.asarray(original), arr))
        self.assertTrue(np.array_equal(np.asarray(marked)[33:87, 53:117], arr[33:87, 53:117]))
        self.assertEqual(marked.size, original.size)
        self.assertGreater(changed, 0)

    def test_existing_top_right_edge_behavior(self):
        marked, _ = render(Image.new('RGB', (160, 120), 'white'),
                            {'roi_bounds_inclusive_xyxy': [50, 0, 159, 90]}, self.config, self.font)
        self.assertEqual(marked.getpixel((159, 40)), (0, 0, 255))
        self.assertEqual(marked.getpixel((80, 0)), (0, 0, 255))
