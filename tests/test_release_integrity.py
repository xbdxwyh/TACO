"""Small release-interface checks with constructed records and mocked inference."""
from argparse import Namespace
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from taco import cli
from taco.core.math12_actions_v1 import apply
from taco.core.math12_final_policy_v1 import PRIMARY
from taco.evaluation import score_directory
from taco.io_utils import digest, read_json, read_jsonl, sha, write_json, write_jsonl


def observation(answer):
    q = '解方程：3x+3=27'
    return dict(question=dict(text=q, latex=q),
                steps=[dict(node_id='s1', position=1, display=q, latex=q),
                       dict(node_id='s2', position=2, display=f'x={answer}', latex=f'x={answer}')],
                student_answer=str(answer))


def reference(final=False):
    return dict(source_gold=dict(repair_step=None, final_correct=final),
                steps=[dict(step_index=1, globally_valid=True),
                       dict(step_index=2, globally_valid=final)])


class ReleaseIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix='taco-integrity-')
        self.root = Path(self.temp.name)
        self.data = self.root/'data'
        self.output = self.root/'predictions'

    def tearDown(self):
        self.temp.cleanup()

    def make_fixture(self, seeds=(17,), arms=(PRIMARY,)):
        cases = [('error', 'logical_error'), ('unknown_error', 'execution_slip'),
                 ('clean', 'correct'), ('unknown_clean', 'self_corrected')]
        public = []
        labels = []
        admin = []
        for iid, variant in cases:
            p = observation(8 if iid == 'clean' else 9)
            if iid.startswith('unknown'):
                # If the wrapper mistakenly enters the frozen parser, this
                # unsupported public problem raises before status is handled.
                p['question'] = dict(text='Unsupported natural-language task',
                                     latex='Unsupported natural-language task')
            public.append(dict(item_id=iid, payload=p))
            labels.append(dict(item_id=iid, **reference(iid == 'clean')))
            admin.append(dict(item_id=iid, family_key='family1', family_split='test',
                              skill_id='linear_shift', variant=variant))
        write_jsonl(self.data/'public/test.jsonl', public)
        write_jsonl(self.data/'labels/test.jsonl', labels)
        write_jsonl(self.data/'admin/case_groups.jsonl', admin)
        records = []
        for seed in seeds:
            for row in public:
                iid = row['item_id']
                p = row['payload']
                if iid.startswith('unknown'):
                    base = dict(status='UNSUPPORTED', reason='OUTSIDE_SUPPORTED_SYNTAX',
                                final_payload=p, accepted=[], attempts=[], declared_complete=False,
                                tool_cost={}, rank_model_unique_inputs=0)
                else:
                    edits = [] if iid == 'clean' else [dict(target='s2', text='x=8'),
                                                      dict(target='answer', text='8')]
                    revised = p
                    for edit in edits:
                        revised = apply(revised, edit)
                    base = dict(status='SUPPORTED', final_payload=revised, accepted=edits,
                                attempts=[], declared_complete=True,
                                tool_cost={} if iid == 'clean' else dict(question_solves=1,
                                    step_audit_checks=1, candidate_checks=2),
                                rank_model_unique_inputs=0 if iid == 'clean' else 100)
                records.append(dict(item_id=iid, seed=seed, public_payload_sha256=digest(p),
                                    generation_cost={'construction_operations': 999},
                                    results=[dict(copy.deepcopy(base), arm=arm) for arm in arms]))
        run = dict(schema='taco-run-v1', split='test',
                   public_sha256=sha(self.data/'public/test.jsonl'),
                   item_ids=[r['item_id'] for r in public], seeds=list(seeds),
                   configurations=[dict(name=arm) for arm in arms])
        write_json(self.output/'RUN.json', run)
        write_jsonl(self.output/'predictions.jsonl', records)
        seal = dict(schema='taco-prediction-seal-v1', prediction_file='predictions.jsonl',
                    predictions_sha256=sha(self.output/'predictions.jsonl'),
                    public_sha256=run['public_sha256'], split='test', records=len(records),
                    run_sha256=sha(self.output/'RUN.json'), item_ids=run['item_ids'],
                    seeds=list(seeds), arms=list(arms), full_split=True)
        write_json(self.output/'PREDICTION_SEAL.json', seal)
        return records

    def replace_predictions_preserving_declared_coverage(self, records):
        write_jsonl(self.output/'predictions.jsonl', records)
        seal = read_json(self.output/'PREDICTION_SEAL.json')
        seal['predictions_sha256'] = sha(self.output/'predictions.jsonl')
        seal['records'] = len(records)
        write_json(self.output/'PREDICTION_SEAL.json', seal)

    def test_missing_whole_seed_is_rejected(self):
        records = self.make_fixture(seeds=(17, 29))
        self.replace_predictions_preserving_declared_coverage([r for r in records if r['seed'] == 17])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            score_directory(self.data, self.output)

    def test_missing_arm_is_rejected_even_when_record_counts_match(self):
        records = self.make_fixture(arms=(PRIMARY, 'S1_rule_uncertain'))
        for record in records:
            record['results'] = record['results'][:1]
        self.replace_predictions_preserving_declared_coverage(records)
        with self.assertRaisesRegex(ValueError, 'Conditions or seeds'):
            score_directory(self.data, self.output)

    def test_changed_run_metadata_is_rejected(self):
        self.make_fixture()
        run = read_json(self.output/'RUN.json')
        run['seeds'] = [29]
        write_json(self.output/'RUN.json', run)
        with self.assertRaisesRegex(ValueError, 'Run binding changed'):
            score_directory(self.data, self.output)

    def test_mismatched_per_record_public_hash_is_rejected(self):
        records = self.make_fixture()
        records[0]['public_payload_sha256'] = 'incorrect'
        self.replace_predictions_preserving_declared_coverage(records)
        with self.assertRaisesRegex(ValueError, 'Public payload'):
            score_directory(self.data, self.output)

    def test_unsupported_examples_stay_in_error_and_protection_denominators(self):
        self.make_fixture()
        summary = score_directory(self.data, self.output)
        panel = summary['panels'][f'{PRIMARY}/seed17']['overall']
        self.assertTrue(summary['complete_split'])
        self.assertEqual(panel['n'], 4)
        self.assertEqual(panel['error_n'], 2)
        self.assertEqual(panel['protection_n'], 2)
        self.assertEqual(panel['unsupported'], 2)
        self.assertEqual(panel['unknown'], 2)
        self.assertEqual(panel['repair'], .5)
        self.assertEqual(panel['delivery'], .5)
        self.assertEqual(panel['answer'], .5)
        self.assertEqual(panel['protection_failure'], .5)
        self.assertEqual(panel['self_corrected_failure'], 1.)

    def test_costs_include_context_solves_and_exclude_neural_and_construction(self):
        self.make_fixture()
        summary = score_directory(self.data, self.output)
        panel = summary['panels'][f'{PRIMARY}/seed17']['overall']
        self.assertEqual(panel['calls'], 1.)  # (one solve + one audit + two candidates) / four inputs
        self.assertEqual(panel['mean_tool_breakdown'],
                         dict(candidate_checks=.5, question_solves=.25, step_audit_checks=.25))
        self.assertEqual(panel['rank_inputs_cold'], 25.)
        rows = read_jsonl(self.output/'evaluation/scores.jsonl')
        repaired = next(r for r in rows if r['item_id'] == 'error')
        self.assertEqual(repaired['candidate_checks'], 2)
        self.assertEqual(repaired['tool_calls'], 4)

    def test_cached_payload_corruption_is_rejected_before_model_reload(self):
        checkpoints = self.root/'checkpoints'
        checkpoints.mkdir()
        (checkpoints/'diagnostic_seed17.pt').write_bytes(b'fixture-only; model is mocked')
        (checkpoints/'rank_seed17.pt').write_bytes(b'fixture-only; model is mocked')
        row = dict(item_id='example', payload=observation(8))
        write_jsonl(self.data/'public/train.jsonl', [row])
        result = dict(item_id='example', seed=17, public_payload_sha256=digest(row['payload']),
                      results=[dict(arm=PRIMARY, status='SUPPORTED')])
        args = Namespace(data=self.data, checkpoints=checkpoints, split='train', seeds=[17],
                         configs=['taco'], limit=1, device='cpu', output=self.output, resume=False)
        with patch('taco.pipeline.RepairPipeline') as model:
            model.return_value.predict.return_value = result
            self.assertEqual(cli.predict(args), 0)
            self.assertEqual(model.call_count, 1)
        cache = self.output/'cache/seed17'/f'{digest("example")}.json'
        saved = read_json(cache)
        saved['payload']['results'][0]['status'] = 'CORRUPTED'
        write_json(cache, saved)
        args.resume = True
        with patch('taco.pipeline.RepairPipeline') as model:
            with self.assertRaisesRegex(ValueError, 'Cached record integrity mismatch'):
                cli.predict(args)
            model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
