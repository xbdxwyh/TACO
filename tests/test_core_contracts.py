"""Small constructed regressions for the published repair contracts.

No benchmark sweep, learned-model inference, or training is performed here.
"""
import copy
import unittest

from taco.core.math12_actions_v1 import apply, learned_state
from taco.core.math12_actions_v2 import build
from taco.core.math12_final_policy_v1 import configurations, operate, PRIMARY
from taco.core.math12_model_v1 import record
from taco.core.math12_obligation_policy_v1 import Meter, closes_obligation
from taco.core.math12_score_v1 import final_score
from taco.core.math12_step_goal_policy_v1 import choose_audit, StepMeter
from taco.core.math12_syntax_v2 import parse_public


def payload(question, steps, answer):
    return dict(question=dict(text=question, latex=question),
                steps=[dict(node_id=f's{i}', position=i, display=s, latex=s)
                       for i, s in enumerate(steps, 1)], student_answer=answer)


def gold(valid, repair=None, final=False):
    return dict(steps=[dict(step_index=i, globally_valid=v)
                       for i, v in enumerate(valid, 1)],
                source_gold=dict(repair_step=repair, final_correct=final,
                                 error_persisted=not final))


def menus(target, texts):
    return {target: [dict(target=target, text=text, candidate_id=f'c{i}', operation='fixture')
                     for i, text in enumerate(texts)]}


def zero_rank(records, head):
    if head != 'rank':
        raise AssertionError('The repair controller must request the rank head')
    return [0.] * len(records)


def spec(name=PRIMARY):
    return next(s for s in configurations() if s['name'] == name)


class CoreContracts(unittest.TestCase):
    def test_observed_self_correction_is_preserved(self):
        q = '解方程：3x+3=27'
        original = payload(q, [q, '3x=30', '3x=24', 'x=8'], '8')
        probabilities = dict(s2=.1, s3=.9, s4=.9, answer=.9)
        data = build(original)
        result = operate(original, data['menus'], probabilities, zero_rank, spec())
        self.assertEqual(result['accepted'], [])
        self.assertIn('s2', result['initial_state']['retired'])
        self.assertTrue(result['declared_complete'])
        self.assertTrue(final_score(original, gold([True, False, True, True], 3, True), result)['complete'])

    def test_correct_answer_cannot_hide_an_unrepaired_step(self):
        q = '解方程：3x+3=27'
        original = payload(q, [q, '3x=30', 'x=10'], '10')
        edits = [dict(target='s3', text='x=8'), dict(target='answer', text='8')]
        current = original
        for edit in edits:
            current = apply(current, edit)
        result = dict(status='SUPPORTED', final_payload=current, accepted=edits,
                      declared_complete=True, tool_cost={}, rank_model_unique_inputs=0)
        score = final_score(original, gold([True, False, False]), result)
        self.assertFalse(score['complete'])
        self.assertTrue(score['false_accept'])
        self.assertIn('s2', score['failures'])

    def test_valid_progress_does_not_close_the_requested_goal(self):
        q = '甲组 15 人，平均 74 分；乙组 10 人，平均 86 分。求总平均分。'
        original = payload(q, [q, 'mean=80'], '394/5')
        probabilities = dict(s2=.1, answer=.9)
        partial = operate(original, menus('s2', ['S=1970']), probabilities, zero_rank, spec())
        self.assertTrue(partial['attempts'][0]['valid'])
        self.assertFalse(partial['accepted'][0]['closes_obligation'])
        self.assertEqual(partial['remaining_targets'], ['s2'])
        self.assertFalse(partial['declared_complete'])
        self.assertFalse(final_score(original, gold([True, False], final=True), partial)['complete'])
        complete = operate(original, menus('s2', ['S=1970', 'mean=394/5']),
                           probabilities, zero_rank, spec())
        self.assertEqual([a['closes_obligation'] for a in complete['accepted']], [False, True])
        self.assertTrue(final_score(original, gold([True, False], final=True), complete)['complete'])

    def test_function_inner_value_is_not_outer_completion(self):
        q = 'f(x)=5x+2，g(x)=3x-4，求 f(g(-4))'
        prepared = parse_public(payload(q, [q, 'f(g(-4))=-77'], '-77'))
        self.assertTrue(Meter(prepared['question']).check('s2', 'g(-4)=-16'))
        self.assertFalse(closes_obligation(prepared, 's2', 'g(-4)=-16'))
        self.assertTrue(closes_obligation(prepared, 's2', 'f(g(-4))=-78'))

    def test_audit_recovers_only_the_checked_missed_step(self):
        q = '化简：(x^2)^5'
        original = payload(q, [q, 'x^9'], 'x^9')
        candidates = menus('s2', ['x^10'])
        candidates.update(menus('answer', ['x^10']))
        probabilities = dict(s2=.61, answer=.49)
        audited = operate(original, candidates, probabilities, zero_rank, spec())
        unaudited = operate(original, candidates, probabilities, zero_rank,
                            spec('S1_learned_none'))
        self.assertEqual(audited['neural_initial_targets'], ['answer'])
        self.assertEqual(audited['reopened_targets'], ['s2'])
        self.assertIs(audited['audit_valid'], False)
        self.assertEqual(audited['tool_cost'], dict(step_audit_checks=1,
                                                  question_solves=1, candidate_checks=2))
        self.assertTrue(final_score(original, gold([True, False]), audited)['complete'])
        self.assertTrue(final_score(original, gold([True, False]), unaudited)['false_accept'])

    def test_unknown_audit_blocks_a_completion_declaration(self):
        q = '化简：x^3*x^2'
        original = payload(q, [q, 'x^(3/2)', 'x^5'], 'x^5')
        result = operate(original, {}, dict(s2=.51, s3=.9, answer=.9), zero_rank, spec())
        self.assertEqual(result['audited_target'], 's2')
        self.assertIsNone(result['audit_valid'])
        self.assertEqual(result['reopened_targets'], [])
        self.assertTrue(result['unknown_audit'])
        self.assertFalse(result['declared_complete'])

    def test_budget_reserves_one_slot_even_when_no_audit_is_possible(self):
        q = '解方程：3x+3=27'
        original = payload(q, [q, 'x=9'], '9')
        candidates = menus('s2', ['x=8'])
        candidates.update(menus('answer', ['8']))
        result = operate(original, candidates, dict(s2=.1, answer=.1), zero_rank, spec())
        self.assertIsNone(result['audited_target'])
        self.assertTrue(result['audit_unused_reservation'])
        self.assertEqual(result['candidate_budget'], 7)
        self.assertEqual(result['tool_cost'].get('step_audit_checks', 0), 0)

    def test_audit_and_candidate_checks_share_the_eight_call_budget(self):
        q = '解方程：3x+3=27'
        original = payload(q, [q, '3x=24', 'x=9'], '8')
        candidates = menus('s3', ['x=9']*7 + ['x=8'])
        probabilities = dict(s2=.8, s3=.1, answer=.9)
        result = operate(original, candidates, probabilities, zero_rank, spec())
        no_audit = operate(original, candidates, probabilities, zero_rank, spec('S1_learned_none'))
        self.assertEqual(result['online_judgments'], 8)
        self.assertEqual(result['tool_cost']['candidate_checks'], 7)
        self.assertFalse(result['declared_complete'])
        self.assertTrue(no_audit['declared_complete'])

    def test_retired_and_pending_steps_are_excluded_from_audit(self):
        q = '解方程：3x+3=27'
        prepared = parse_public(payload(q, [q, '3x=30', '3x=24', 'x=9'], '9'))
        probabilities = dict(s2=.1, s3=.6, s4=.1, answer=.1)
        state = learned_state(prepared, probabilities, True)
        self.assertIn('s2', state['retired'])
        self.assertIn('s4', state['pending'])
        self.assertEqual(choose_audit(prepared, probabilities, state, 'uncertain'), ('s3', ['s3']))

    def test_hidden_metadata_changes_no_public_decision(self):
        q = '解方程：3x+3=27'
        original = payload(q, [q, 'x=9'], '8')
        hidden = copy.deepcopy(original)
        hidden.update(gold='8', variant='correct', family='private', skill_id='private')
        clean_data, hidden_data = build(original), build(hidden)
        self.assertEqual(clean_data['menus'], hidden_data['menus'])
        probabilities = dict(s2=.6, answer=.9)
        a = operate(original, clean_data['menus'], probabilities, zero_rank, spec())
        b = operate(hidden, hidden_data['menus'], probabilities, zero_rank, spec())
        for field in ('initial_state', 'audited_target', 'accepted', 'attempts', 'tool_cost'):
            self.assertEqual(a[field], b[field])

    def test_step_encoding_is_independent_of_future_steps_and_answer(self):
        q = '解方程：3x+3=27'
        a = parse_public(payload(q, [q, '3x=30', 'x=10'], '10'))
        b = parse_public(payload(q, [q, '3x=30', 'x=8'], '8'))
        self.assertEqual(record(a, 's2', '3x=30'), record(b, 's2', '3x=30'))

    def test_original_given_and_repair_candidate_have_distinct_contracts(self):
        q = '解方程：3x+3=27'
        prepared = parse_public(payload(q, [q, 'x=8'], '8'))
        self.assertTrue(StepMeter(prepared['question']).check_original_step('s2', '3x+3=27'))
        self.assertFalse(Meter(prepared['question']).check('s2', '3x+3=27'))

    def test_rule_ranking_never_requests_a_neural_score(self):
        q = '解方程：3x+3=27'
        original = payload(q, [q, 'x=9'], '8')

        def forbidden(*args):
            raise AssertionError('Rule ranking requested neural inference')

        result = operate(original, menus('s2', ['x=8']), dict(s2=.6, answer=.9),
                         forbidden, spec('S1_rule_uncertain'))
        self.assertEqual(result['rank_model_unique_inputs'], 0)
        self.assertTrue(result['declared_complete'])


if __name__ == '__main__':
    unittest.main()
