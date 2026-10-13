"""Text-only extraction of the frozen MathRepairBench construction code.

The mathematical templates, generate_families, flatten_cases, and public
observation constructors below retain their original implementations. Image
rendering, simulated learner episodes, reporting, and unrelated export paths
are deliberately excluded. See docs/DATASET.md for provenance and scope.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Callable, Iterable

ORIGINAL_GENERATOR_SHA256 = 'df153a72daf5ef1f65619caf978413bb2850e6331e109dd094413a67b6253167'
ORIGINAL_EXPORT_CONTRACT_SHA256 = 'e2204219bed7f9b7f8c51e60afa1ab0835aa968ce047f5287a562257e35c9f3b'

SCHEMA_VERSION = "controlled-logic-state-v2"
VARIANTS = ("correct", "logical_error", "execution_slip", "self_corrected")



@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    name_zh: str
    reasoning_family: str
    misconception_id: str
    misconception_zh: str
    prerequisites: tuple[str, ...]


SKILLS = (
    SkillSpec(
        "linear_shift",
        "移项与加法逆运算",
        "equation_equivalence",
        "transfer_without_inverse",
        "移项时没有改变运算方向",
        ("integer_arithmetic",),
    ),
    SkillSpec(
        "negative_distribution",
        "负号作用域与括号展开",
        "symbol_scope",
        "negative_scope_drop",
        "负号只作用于括号内第一项",
        ("signed_numbers", "distributive_property"),
    ),
    SkillSpec(
        "fraction_denominator",
        "分式方程的同乘与通分",
        "equation_equivalence",
        "incomplete_denominator_scaling",
        "去分母时漏乘了一个分子项",
        ("fraction_equivalence", "linear_shift"),
    ),
    SkillSpec(
        "proportion_cross_product",
        "比例式的交叉相乘",
        "proportional_reasoning",
        "same_side_product",
        "把同侧量相乘而不是交叉相乘",
        ("fraction_equivalence", "multiplicative_inverse"),
    ),
    SkillSpec(
        "exponent_product",
        "同底数幂相乘",
        "rule_selection",
        "multiply_exponents_in_product",
        "同底数幂相乘时误把指数相乘",
        ("multiplication",),
    ),
    SkillSpec(
        "power_of_power",
        "幂的乘方",
        "rule_selection",
        "add_exponents_in_power",
        "幂的乘方时误把指数相加",
        ("exponent_product",),
    ),
    SkillSpec(
        "difference_of_squares",
        "平方差结构",
        "algebraic_structure",
        "square_difference_sign",
        "平方差展开时把负号写成正号",
        ("distributive_property", "signed_numbers"),
    ),
    SkillSpec(
        "inequality_negative_scale",
        "不等式负数缩放",
        "order_reasoning",
        "inequality_direction_not_flipped",
        "不等式两边除以负数时没有改变方向",
        ("signed_numbers", "multiplicative_inverse"),
    ),
    SkillSpec(
        "absolute_value_branching",
        "绝对值方程的双分支",
        "case_reasoning",
        "absolute_value_one_branch",
        "绝对值方程只保留了一个分支",
        ("linear_shift", "signed_numbers"),
    ),
    SkillSpec(
        "function_composition",
        "函数复合的顺序与代入",
        "operator_composition",
        "reverse_function_composition",
        "把 f(g(x)) 的复合顺序颠倒",
        ("linear_substitution", "distributive_property"),
    ),
    SkillSpec(
        "weighted_average",
        "加权平均与总体均值",
        "quantitative_modeling",
        "unweighted_group_means",
        "忽略组大小而直接平均两个组均值",
        ("multiplication", "fraction_equivalence"),
    ),
    SkillSpec(
        "linear_system_elimination",
        "二元一次方程组消元",
        "equation_system_reasoning",
        "elimination_side_mismatch",
        "消元时两边没有执行同一种组合",
        ("linear_shift", "signed_numbers"),
    ),
)
SKILL_BY_ID = {skill.skill_id: skill for skill in SKILLS}


@dataclass(frozen=True)
class TraceStep:
    step_index: int
    display: str
    latex: str
    relation_to_previous: str
    locally_valid: bool
    globally_valid: bool
    error_origin_step: int | None = None


@dataclass(frozen=True)
class VariantTrace:
    variant: str
    steps: tuple[TraceStep, ...]
    student_answer: str
    final_correct: bool
    contains_error: bool
    error_persisted: bool
    first_error_step: int | None
    repair_step: int | None
    error_mechanism: str
    error_level: str
    gold_diagnosis_zh: str
    state_operation: str
    evidence_strength: float


@dataclass(frozen=True)
class ProblemFamily:
    family_id: str
    skill_id: str
    question_text: str
    question_latex: str
    correct_answer: str
    parameters: dict[str, int]
    variants: tuple[VariantTrace, ...]
    solution_path_id: str = "canonical"
    repair_style: str = "rewrite_without_cue"


def _stable_seed(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _rng(seed: int, *parts: object) -> random.Random:
    return random.Random(_stable_seed(seed, *parts))


def _fmt_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _latex_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"


def _step(
    index: int,
    display: str,
    latex: str,
    relation: str,
    local: bool,
    global_: bool,
    origin: int | None = None,
) -> TraceStep:
    return TraceStep(index, display, latex, relation, local, global_, origin)


def _traces(
    *,
    given: tuple[str, str],
    correct_lines: list[tuple[str, str]],
    logical_lines: list[tuple[str, str]],
    slip_lines: list[tuple[str, str]],
    corrected_lines: list[tuple[str, str]],
    correct_answer: str,
    logical_answer: str,
    slip_answer: str,
    misconception_id: str,
    misconception_zh: str,
    logical_error_index: int = 0,
    corrected_error_index: int = 0,
    corrected_repair_index: int = 1,
) -> tuple[VariantTrace, ...]:
    logical_error_step = logical_error_index + 2
    corrected_error_step = corrected_error_index + 2
    corrected_repair_step = corrected_repair_index + 2

    def correct_steps() -> tuple[TraceStep, ...]:
        rows = [_step(1, given[0], given[1], "given", True, True)]
        for index, (display, latex) in enumerate(correct_lines, start=2):
            rows.append(_step(index, display, latex, "equivalent", True, True))
        return tuple(rows)

    def logical_steps() -> tuple[TraceStep, ...]:
        rows = [_step(1, given[0], given[1], "given", True, True)]
        for offset, (display, latex) in enumerate(logical_lines, start=2):
            if offset < logical_error_step:
                rows.append(_step(offset, display, latex, "equivalent", True, True))
            elif offset == logical_error_step:
                rows.append(
                    _step(
                        offset,
                        display,
                        latex,
                        "invalid_rule",
                        False,
                        False,
                        logical_error_step,
                    )
                )
            else:
                rows.append(
                    _step(
                        offset,
                        display,
                        latex,
                        "propagates_error",
                        True,
                        False,
                        logical_error_step,
                    )
                )
        return tuple(rows)

    def slip_steps() -> tuple[TraceStep, ...]:
        rows = [_step(1, given[0], given[1], "given", True, True)]
        for offset, (display, latex) in enumerate(slip_lines, start=2):
            is_last = offset == len(slip_lines) + 1
            rows.append(
                _step(
                    offset,
                    display,
                    latex,
                    "arithmetic_slip" if is_last else "equivalent",
                    not is_last,
                    not is_last,
                    offset if is_last else None,
                )
            )
        return tuple(rows)

    def repaired_steps() -> tuple[TraceStep, ...]:
        rows = [_step(1, given[0], given[1], "given", True, True)]
        for offset, (display, latex) in enumerate(corrected_lines, start=2):
            if offset < corrected_error_step:
                rows.append(_step(offset, display, latex, "equivalent", True, True))
            elif offset == corrected_error_step:
                rows.append(
                    _step(
                        offset,
                        display,
                        latex,
                        "invalid_rule",
                        False,
                        False,
                        corrected_error_step,
                    )
                )
            elif offset < corrected_repair_step:
                rows.append(
                    _step(
                        offset,
                        display,
                        latex,
                        "propagates_error",
                        True,
                        False,
                        corrected_error_step,
                    )
                )
            elif offset == corrected_repair_step:
                rows.append(
                    _step(
                        offset,
                        display,
                        latex,
                        "self_repair",
                        True,
                        True,
                        corrected_error_step,
                    )
                )
            else:
                rows.append(_step(offset, display, latex, "equivalent", True, True))
        return tuple(rows)

    return (
        VariantTrace(
            "correct",
            correct_steps(),
            correct_answer,
            True,
            False,
            False,
            None,
            None,
            "none",
            "none",
            "过程与结论均正确。",
            "increase_mastery",
            0.45,
        ),
        VariantTrace(
            "logical_error",
            logical_steps(),
            logical_answer,
            False,
            True,
            True,
            logical_error_step,
            None,
            misconception_id,
            "reasoning",
            misconception_zh,
            "activate_misconception",
            1.0,
        ),
        VariantTrace(
            "execution_slip",
            slip_steps(),
            slip_answer,
            False,
            True,
            True,
            len(slip_lines) + 1,
            None,
            "arithmetic_slip",
            "execution",
            "推理规则正确，但末步出现局部计算失误。",
            "weak_negative_update",
            0.25,
        ),
        VariantTrace(
            "self_corrected",
            repaired_steps(),
            correct_answer,
            True,
            True,
            False,
            corrected_error_step,
            corrected_repair_step,
            misconception_id,
            "reasoning_repaired",
            f"先出现“{misconception_zh}”，随后主动纠正并得到正确结论。",
            "record_self_correction",
            0.65,
        ),
    )


def _repair_trace(
    index: int,
    logical_lines: list[tuple[str, str]],
    correct_lines: list[tuple[str, str]],
    hint_zh: str,
    *,
    error_index: int = 0,
) -> tuple[list[tuple[str, str]], int, str]:
    """Create varied, semantically equivalent self-repair traces.

    The default form crosses out the invalid line and immediately rewrites the
    correct mathematics. Other forms use compact Chinese checking marks. None
    contains the old literal English ``Correction:`` shortcut.
    """

    style = ("rewrite_without_cue", "arrow_rewrite", "chinese_check", "restart")[
        (index - 1) % 4
    ]
    prefix = list(logical_lines[: error_index + 1])
    if style == "rewrite_without_cue":
        return [*prefix, *correct_lines], error_index + 1, style
    if style == "arrow_rewrite":
        note = ("=>", r"\Rightarrow")
    elif style == "chinese_check":
        note = (f"检查：{hint_zh}", rf"\text{{检查：{hint_zh}}}")
    else:
        note = ("重算", r"\text{重算}")
    return [*prefix, note, *correct_lines], error_index + 2, style


def _linear_shift(index: int, seed: int) -> ProblemFamily:
    rng = _rng(seed, "linear_shift", index)
    a = rng.randint(2, 9)
    k = rng.randint(1, 7)
    b = a * k
    x = rng.choice([n for n in range(-9, 13) if n not in (0, -2 * k)])
    c = a * x + b
    slip = x + rng.choice((-1, 1))
    given = (f"{a}x + {b} = {c}", rf"{a}x+{b}={c}")
    if index % 2:
        correct = [
            (f"{a}x = {c} - {b} = {c-b}", rf"{a}x={c}-{b}={c-b}"),
            (f"x = {c-b}/{a} = {x}", rf"x=\frac{{{c-b}}}{{{a}}}={x}"),
        ]
        wrong = Fraction(c + b, a)
        logical = [
            (f"{a}x = {c} + {b} = {c+b}", rf"{a}x={c}+{b}={c+b}"),
            (f"x = {_fmt_fraction(wrong)}", rf"x={_latex_fraction(wrong)}"),
        ]
        mechanism = "transfer_without_inverse"
        diagnosis = "移项时没有改变运算方向"
        path_id = "subtract_then_divide"
    else:
        b_over_a, c_over_a = Fraction(b, a), Fraction(c, a)
        correct = [
            (
                f"x + {_fmt_fraction(b_over_a)} = {_fmt_fraction(c_over_a)}",
                rf"x+{_latex_fraction(b_over_a)}={_latex_fraction(c_over_a)}",
            ),
            (
                f"x = {_fmt_fraction(c_over_a)} - {_fmt_fraction(b_over_a)} = {x}",
                rf"x={_latex_fraction(c_over_a)}-{_latex_fraction(b_over_a)}={x}",
            ),
        ]
        wrong = c_over_a - b
        logical = [
            (f"x + {b} = {_fmt_fraction(c_over_a)}", rf"x+{b}={_latex_fraction(c_over_a)}"),
            (f"x = {_fmt_fraction(wrong)}", rf"x={_latex_fraction(wrong)}"),
        ]
        mechanism = "partial_equation_scaling"
        diagnosis = "方程两边同除以系数时，常数项没有同步缩放"
        path_id = "normalize_then_subtract"
    slip_lines = [correct[0], (f"x = {c-b}/{a} = {slip}", rf"x=\frac{{{c-b}}}{{{a}}}={slip}")]
    corrected, repair_index, repair_style = _repair_trace(
        index, logical, correct, diagnosis, error_index=0
    )
    traces = _traces(
        given=given,
        correct_lines=correct,
        logical_lines=logical,
        slip_lines=slip_lines,
        corrected_lines=corrected,
        correct_answer=str(x),
        logical_answer=_fmt_fraction(wrong),
        slip_answer=str(slip),
        misconception_id=mechanism,
        misconception_zh=diagnosis,
        corrected_repair_index=repair_index,
    )
    return ProblemFamily(
        f"LEF-LS-{index:03d}",
        "linear_shift",
        f"解方程：{given[0]}",
        rf"\text{{Solve }}\;{given[1]}",
        str(x),
        {"a": a, "b": b, "c": c, "x": x},
        traces,
        path_id,
        repair_style,
    )


def _negative_distribution(index: int, seed: int) -> ProblemFamily:
    rng = _rng(seed, "negative_distribution", index)
    a = rng.randint(2, 9)
    b = rng.randint(1, 9)
    x = rng.choice([n for n in range(-8, 11) if n != 0])
    c = rng.randint(-8, 8)
    r = a * x + b + c
    slip = x + rng.choice((-1, 1))
    given = (f"{r} - ({a}x + {b}) = {c}", rf"{r}-({a}x+{b})={c}")
    if index % 2:
        correct = [
            (f"{r} - {a}x - {b} = {c}", rf"{r}-{a}x-{b}={c}"),
            (f"-{a}x = {c} - {r} + {b} = {-a*x}", rf"-{a}x={c}-{r}+{b}={-a*x}"),
            (f"x = {x}", rf"x={x}"),
        ]
        wrong_value = Fraction(r + b - c, a)
        logical = [
            (f"{r} - {a}x + {b} = {c}", rf"{r}-{a}x+{b}={c}"),
            (f"-{a}x = {c} - {r} - {b}", rf"-{a}x={c}-{r}-{b}"),
            (f"x = {_fmt_fraction(wrong_value)}", rf"x={_latex_fraction(wrong_value)}"),
        ]
        mechanism = "negative_scope_drop"
        diagnosis = "负号只作用于括号内第一项"
        path_id = "expand_parentheses"
    else:
        correct = [
            (f"{r} - {c} = {a}x + {b}", rf"{r}-{c}={a}x+{b}"),
            (f"{a}x = {r} - {c} - {b} = {a*x}", rf"{a}x={r}-{c}-{b}={a*x}"),
            (f"x = {x}", rf"x={x}"),
        ]
        wrong_value = Fraction(r - c + b, a)
        logical = [
            (f"{r} - {c} = {a}x - {b}", rf"{r}-{c}={a}x-{b}"),
            (f"{a}x = {r} - {c} + {b}", rf"{a}x={r}-{c}+{b}"),
            (f"x = {_fmt_fraction(wrong_value)}", rf"x={_latex_fraction(wrong_value)}"),
        ]
        mechanism = "group_transfer_sign_flip"
        diagnosis = "整体移走括号时又额外改变了括号内常数的符号"
        path_id = "move_group_then_solve"
    slip_lines = [*correct[:-1], (f"x = {slip}", rf"x={slip}")]
    corrected, repair_index, repair_style = _repair_trace(
        index, logical, correct, diagnosis, error_index=0
    )
    traces = _traces(
        given=given,
        correct_lines=correct,
        logical_lines=logical,
        slip_lines=slip_lines,
        corrected_lines=corrected,
        correct_answer=str(x),
        logical_answer=_fmt_fraction(wrong_value),
        slip_answer=str(slip),
        misconception_id=mechanism,
        misconception_zh=diagnosis,
        corrected_repair_index=repair_index,
    )
    return ProblemFamily(
        f"LEF-ND-{index:03d}",
        "negative_distribution",
        f"解方程：{given[0]}",
        rf"\text{{Solve }}\;{given[1]}",
        str(x),
        {"a": a, "b": b, "c": c, "r": r, "x": x},
        traces,
        path_id,
        repair_style,
    )


def _fraction_denominator(index: int, seed: int) -> ProblemFamily:
    rng = _rng(seed, "fraction_denominator", index)
    pairs = ((2, 3), (3, 4), (3, 5), (4, 5), (5, 6), (5, 7), (7, 8))
    a, c = rng.choice(pairs)
    b = rng.randint(1, 8)
    t = rng.choice([n for n in range(-5, 9) if n != 0])
    x = a * t
    r = c * t + b
    slip = x + rng.choice((-1, 1))
    given = (f"x/{a} + {b}/{c} = {r}/{c}", rf"\frac{{x}}{{{a}}}+\frac{{{b}}}{{{c}}}=\frac{{{r}}}{{{c}}}")
    if index % 2:
        correct = [
            (f"{c}x + {a*b} = {a*r}", rf"{c}x+{a*b}={a*r}"),
            (f"{c}x = {a*r-a*b}", rf"{c}x={a*r-a*b}"),
            (f"x = {x}", rf"x={x}"),
        ]
        wrong_value = Fraction(a * r - b, c)
        logical = [
            (f"{c}x + {b} = {a*r}", rf"{c}x+{b}={a*r}"),
            (f"{c}x = {a*r-b}", rf"{c}x={a*r-b}"),
            (f"x = {_fmt_fraction(wrong_value)}", rf"x={_latex_fraction(wrong_value)}"),
        ]
        mechanism = "incomplete_denominator_scaling"
        diagnosis = "去分母时漏乘了一个分子项"
        path_id = "clear_all_denominators"
    else:
        correct = [
            (f"x/{a} = ({r}-{b})/{c} = {t}", rf"\frac{{x}}{{{a}}}=\frac{{{r}-{b}}}{{{c}}}={t}"),
            (f"x = {a}*{t} = {x}", rf"x={a}\times {t}={x}"),
        ]
        wrong_value = a * (r - b)
        logical = [
            (f"x/{a} = {r}-{b} = {r-b}", rf"\frac{{x}}{{{a}}}={r}-{b}={r-b}"),
            (f"x = {a}*{r-b} = {wrong_value}", rf"x={a}\times {r-b}={wrong_value}"),
        ]
        mechanism = "drop_common_denominator_after_subtraction"
        diagnosis = "同分母分数相减后把公共分母直接丢掉"
        path_id = "isolate_fraction_first"
    slip_lines = [*correct[:-1], (f"x = {slip}", rf"x={slip}")]
    corrected, repair_index, repair_style = _repair_trace(
        index, logical, correct, diagnosis, error_index=0
    )
    traces = _traces(
        given=given,
        correct_lines=correct,
        logical_lines=logical,
        slip_lines=slip_lines,
        corrected_lines=corrected,
        correct_answer=str(x),
        logical_answer=_fmt_fraction(wrong_value),
        slip_answer=str(slip),
        misconception_id=mechanism,
        misconception_zh=diagnosis,
        corrected_repair_index=repair_index,
    )
    return ProblemFamily(
        f"LEF-FD-{index:03d}",
        "fraction_denominator",
        f"解方程：{given[0]}",
        rf"\text{{Solve }}\;{given[1]}",
        str(x),
        {"a": a, "b": b, "c": c, "r": r, "x": x},
        traces,
        path_id,
        repair_style,
    )


def _proportion(index: int, seed: int) -> ProblemFamily:
    rng = _rng(seed, "proportion", index)
    a = rng.randint(2, 9)
    k = rng.randint(2, 6)
    c = a * k
    x = rng.choice([value for value in range(2, 15) if value != a])
    b = k * x
    slip = x + rng.choice((-1, 1))
    given = (f"x/{a} = {b}/{c}", rf"\frac{{x}}{{{a}}}=\frac{{{b}}}{{{c}}}")
    if index % 2:
        correct = [
            (f"{c}x = {a}*{b}", rf"{c}x={a}\times {b}"),
            (f"x = {a*b}/{c} = {x}", rf"x=\frac{{{a*b}}}{{{c}}}={x}"),
        ]
        wrong = Fraction(b * c, a)
        logical = [
            (f"{a}x = {b}*{c}", rf"{a}x={b}\times {c}"),
            (f"x = {_fmt_fraction(wrong)}", rf"x={_latex_fraction(wrong)}"),
        ]
        mechanism = "same_side_product"
        diagnosis = "把同侧量相乘而不是交叉相乘"
        path_id = "cross_multiply"
    else:
        correct = [
            (f"{a}/x = {c}/{b}", rf"\frac{{{a}}}{{x}}=\frac{{{c}}}{{{b}}}"),
            (f"{c}x = {a}*{b}", rf"{c}x={a}\times {b}"),
            (f"x = {x}", rf"x={x}"),
        ]
        wrong = Fraction(a * c, b)
        logical = [
            (f"{a}/x = {b}/{c}", rf"\frac{{{a}}}{{x}}=\frac{{{b}}}{{{c}}}"),
            (f"{b}x = {a}*{c}", rf"{b}x={a}\times {c}"),
            (f"x = {_fmt_fraction(wrong)}", rf"x={_latex_fraction(wrong)}"),
        ]
        mechanism = "invert_one_ratio_only"
        diagnosis = "比例式变形时只倒置了一侧的比"
        path_id = "invert_both_then_cross"
    slip_lines = [correct[0], (f"x = {a*b}/{c} = {slip}", rf"x=\frac{{{a*b}}}{{{c}}}={slip}")]
    if index % 2 == 0:
        slip_lines = [*correct[:-1], (f"x = {slip}", rf"x={slip}")]
    corrected, repair_index, repair_style = _repair_trace(
        index, logical, correct, diagnosis, error_index=0
    )
    traces = _traces(
        given=given,
        correct_lines=correct,
        logical_lines=logical,
        slip_lines=slip_lines,
        corrected_lines=corrected,
        correct_answer=str(x),
        logical_answer=_fmt_fraction(wrong),
        slip_answer=str(slip),
        misconception_id=mechanism,
        misconception_zh=diagnosis,
        corrected_repair_index=repair_index,
    )
    return ProblemFamily(
        f"LEF-PC-{index:03d}",
        "proportion_cross_product",
        f"解比例：{given[0]}",
        rf"\text{{Solve }}\;{given[1]}",
        str(x),
        {"a": a, "b": b, "c": c, "x": x},
        traces,
        path_id,
        repair_style,
    )


def _exponent_parameters(index: int, seed: int) -> tuple[int, int]:
    rng = _rng(seed, "exponent_pair", index)
    while True:
        m, n = rng.randint(2, 12), rng.randint(2, 12)
        if m + n != m * n:
            return m, n


def _exponent_product(index: int, seed: int) -> ProblemFamily:
    m, n = _exponent_parameters(index, seed)
    correct_exp = m + n
    slip_exp = correct_exp + (1 if index % 2 else -1)
    given = (f"x^{m} * x^{n}", rf"x^{{{m}}}\times x^{{{n}}}")
    correct = [(f"= x^({m}+{n}) = x^{correct_exp}", rf"=x^{{{m}+{n}}}=x^{{{correct_exp}}}")]
    if index % 2:
        wrong_exp = m * n
        logical = [(f"= x^({m}*{n}) = x^{wrong_exp}", rf"=x^{{{m}\times {n}}}=x^{{{wrong_exp}}}")]
        mechanism = "multiply_exponents_in_product"
        diagnosis = "同底数幂相乘时误把指数相乘"
    else:
        wrong_exp = m - n
        logical = [(f"= x^({m}-{n}) = x^{wrong_exp}", rf"=x^{{{m}-{n}}}=x^{{{wrong_exp}}}")]
        mechanism = "subtract_exponents_in_product"
        diagnosis = "把同底数幂相乘误套成了商的指数相减规则"
    slip_lines = [(f"= x^({m}+{n}) = x^{slip_exp}", rf"=x^{{{m}+{n}}}=x^{{{slip_exp}}}")]
    corrected, repair_index, repair_style = _repair_trace(
        index, logical, correct, diagnosis, error_index=0
    )
    traces = _traces(
        given=given,
        correct_lines=correct,
        logical_lines=logical,
        slip_lines=slip_lines,
        corrected_lines=corrected,
        correct_answer=f"x^{correct_exp}",
        logical_answer=f"x^{wrong_exp}",
        slip_answer=f"x^{slip_exp}",
        misconception_id=mechanism,
        misconception_zh=diagnosis,
        corrected_repair_index=repair_index,
    )
    return ProblemFamily(
        f"LEF-EP-{index:03d}",
        "exponent_product",
        f"化简：{given[0]}",
        rf"\text{{Simplify }}\;{given[1]}",
        f"x^{correct_exp}",
        {"m": m, "n": n},
        traces,
        "exponent_rule",
        repair_style,
    )


def _power_of_power(index: int, seed: int) -> ProblemFamily:
    m, n = _exponent_parameters(index, seed)
    correct_exp = m * n
    slip_exp = correct_exp + (1 if index % 2 else -1)
    given = (f"(x^{m})^{n}", rf"(x^{{{m}}})^{{{n}}}")
    correct = [(f"= x^({m}*{n}) = x^{correct_exp}", rf"=x^{{{m}\times {n}}}=x^{{{correct_exp}}}")]
    if index % 2:
        wrong_exp = m + n
        logical = [(f"= x^({m}+{n}) = x^{wrong_exp}", rf"=x^{{{m}+{n}}}=x^{{{wrong_exp}}}")]
        mechanism = "add_exponents_in_power"
        diagnosis = "幂的乘方时误把指数相加"
    else:
        wrong_exp = m**n
        logical = [(f"= x^({m}^{n}) = x^{wrong_exp}", rf"=x^{{{m}^{{{n}}}}}=x^{{{wrong_exp}}}")]
        mechanism = "exponentiate_the_exponent"
        diagnosis = "把幂的乘方误理解为指数本身再做乘方"
    slip_lines = [(f"= x^({m}*{n}) = x^{slip_exp}", rf"=x^{{{m}\times {n}}}=x^{{{slip_exp}}}")]
    corrected, repair_index, repair_style = _repair_trace(
        index, logical, correct, diagnosis, error_index=0
    )
    traces = _traces(
        given=given,
        correct_lines=correct,
        logical_lines=logical,
        slip_lines=slip_lines,
        corrected_lines=corrected,
        correct_answer=f"x^{correct_exp}",
        logical_answer=f"x^{wrong_exp}",
        slip_answer=f"x^{slip_exp}",
        misconception_id=mechanism,
        misconception_zh=diagnosis,
        corrected_repair_index=repair_index,
    )
    return ProblemFamily(
        f"LEF-PP-{index:03d}",
        "power_of_power",
        f"化简：{given[0]}",
        rf"\text{{Simplify }}\;{given[1]}",
        f"x^{correct_exp}",
        {"m": m, "n": n},
        traces,
        "power_rule",
        repair_style,
    )


def _difference_of_squares(index: int, seed: int) -> ProblemFamily:
    rng = _rng(seed, "difference_of_squares", index)
    a = rng.randint(2, 80)
    while a == index:
        a = rng.randint(2, 80)
    square = a * a
    slip_square = square + (1 if index % 2 else -1)
    given = (f"(x + {a})(x - {a})", rf"(x+{a})(x-{a})")
    if index % 2:
        correct = [(f"= x^2 - {square}", rf"=x^2-{square}")]
        logical = [(f"= x^2 + {square}", rf"=x^2+{square}")]
        mechanism = "square_difference_sign"
        diagnosis = "平方差展开时把负号写成正号"
        path_id = "identity_direct"
    else:
        correct = [
            (f"= x^2 - {a}x + {a}x - {a}^2", rf"=x^2-{a}x+{a}x-{a}^2"),
            (f"= x^2 - {square}", rf"=x^2-{square}"),
        ]
        logical = [
            (f"= x^2 - {a}x + {a}x - {a}", rf"=x^2-{a}x+{a}x-{a}"),
            (f"= x^2 - {a}", rf"=x^2-{a}"),
        ]
        mechanism = "constant_product_not_squared"
        diagnosis = "逐项相乘时把常数项 (-a)×a 写成了 -a"
        path_id = "foil_then_combine"
    slip_lines = [*correct[:-1], (f"= x^2 - {slip_square}", rf"=x^2-{slip_square}")]
    corrected, repair_index, repair_style = _repair_trace(
        index, logical, correct, diagnosis, error_index=0
    )
    traces = _traces(
        given=given,
        correct_lines=correct,
        logical_lines=logical,
        slip_lines=slip_lines,
        corrected_lines=corrected,
        correct_answer=f"x^2 - {square}",
        logical_answer=f"x^2 + {square}",
        slip_answer=f"x^2 - {slip_square}",
        misconception_id=mechanism,
        misconception_zh=diagnosis,
        corrected_repair_index=repair_index,
    )
    return ProblemFamily(
        f"LEF-DS-{index:03d}",
        "difference_of_squares",
        f"展开并化简：{given[0]}",
        rf"\text{{Expand }}\;{given[1]}",
        f"x^2 - {square}",
        {"a": a},
        traces,
        path_id,
        repair_style,
    )


def _inequality(index: int, seed: int) -> ProblemFamily:
    rng = _rng(seed, "inequality_negative_scale", index)
    a = rng.randint(2, 12)
    k = rng.choice([n for n in range(-12, 13) if n != 0])
    b = a * k
    boundary = -k
    slip_boundary = boundary + (1 if index % 2 else -1)
    given = (f"-{a}x > {b}", rf"-{a}x>{b}")
    if index % 2:
        correct = [(f"x < {boundary}", rf"x<{boundary}")]
        logical = [(f"x > {boundary}", rf"x>{boundary}")]
        logical_answer = f"x > {boundary}"
        mechanism = "inequality_direction_not_flipped"
        diagnosis = "不等式两边除以负数时没有改变方向"
        error_index = 0
        path_id = "divide_by_negative_direct"
    else:
        correct = [
            (f"{a}x < {-b}", rf"{a}x<{-b}"),
            (f"x < {-b}/{a} = {boundary}", rf"x<\frac{{{-b}}}{{{a}}}={boundary}"),
        ]
        logical = [
            correct[0],
            (f"x < {b}/{a} = {k}", rf"x<\frac{{{b}}}{{{a}}}={k}"),
        ]
        logical_answer = f"x < {k}"
        mechanism = "negative_boundary_sign_drop"
        diagnosis = "方向已经翻转，但计算边界时又丢掉了负号"
        error_index = 1
        path_id = "negate_then_divide_positive"
    slip_lines = [*correct[:-1], (f"x < {slip_boundary}", rf"x<{slip_boundary}")]
    corrected, repair_index, repair_style = _repair_trace(
        index, logical, correct, diagnosis, error_index=error_index
    )
    traces = _traces(
        given=given,
        correct_lines=correct,
        logical_lines=logical,
        slip_lines=slip_lines,
        corrected_lines=corrected,
        correct_answer=f"x < {boundary}",
        logical_answer=logical_answer,
        slip_answer=f"x < {slip_boundary}",
        misconception_id=mechanism,
        misconception_zh=diagnosis,
        logical_error_index=error_index,
        corrected_error_index=error_index,
        corrected_repair_index=repair_index,
    )
    return ProblemFamily(
        f"LEF-IN-{index:03d}",
        "inequality_negative_scale",
        f"解不等式：{given[0]}",
        rf"\text{{Solve }}\;{given[1]}",
        f"x < {boundary}",
        {"a": a, "b": b, "boundary": boundary},
        traces,
        path_id,
        repair_style,
    )


def _absolute_value(index: int, seed: int) -> ProblemFamily:
    rng = _rng(seed, "absolute_value_branching", index)
    center = rng.choice([value for value in range(-9, 10) if value != 0])
    radius = rng.randint(2, 9)
    low, high = center - radius, center + radius
    given = (f"|x - ({center})| = {radius}", rf"|x-({center})|={radius}")
    correct = [
        (
            f"x - ({center}) = {radius} or x - ({center}) = {-radius}",
            rf"x-({center})={radius}\;\text{{或}}\;x-({center})={-radius}",
        ),
        (f"x = {high} or x = {low}", rf"x={high}\;\text{{or}}\;x={low}"),
    ]
    if index % 2:
        logical = [
            (f"x - ({center}) = {radius}", rf"x-({center})={radius}"),
            (f"x = {high}", rf"x={high}"),
        ]
        wrong_answer = f"x = {high}"
        mechanism = "absolute_value_one_branch"
        diagnosis = "绝对值方程只保留了正分支，漏掉另一个解"
        error_index = 0
    else:
        wrong_low, wrong_high = -center - radius, -center + radius
        logical = [
            correct[0],
            (
                f"x = {wrong_high} or x = {wrong_low}",
                rf"x={wrong_high}\;\text{{或}}\;x={wrong_low}",
            ),
        ]
        wrong_answer = f"x = {wrong_high} or x = {wrong_low}"
        mechanism = "absolute_value_center_sign"
        diagnosis = "展开双分支后，移项时把绝对值中心的符号处理反了"
        error_index = 1
    slip_low = low + (1 if index % 2 else -1)
    slip_lines = [correct[0], (f"x = {high} or x = {slip_low}", rf"x={high}\;\text{{or}}\;x={slip_low}")]
    corrected, repair_index, repair_style = _repair_trace(
        index, logical, correct, diagnosis, error_index=error_index
    )
    answer = f"x = {high} or x = {low}"
    traces = _traces(
        given=given,
        correct_lines=correct,
        logical_lines=logical,
        slip_lines=slip_lines,
        corrected_lines=corrected,
        correct_answer=answer,
        logical_answer=wrong_answer,
        slip_answer=f"x = {high} or x = {slip_low}",
        misconception_id=mechanism,
        misconception_zh=diagnosis,
        logical_error_index=error_index,
        corrected_error_index=error_index,
        corrected_repair_index=repair_index,
    )
    return ProblemFamily(
        f"LEF-AV-{index:03d}",
        "absolute_value_branching",
        f"解方程：{given[0]}",
        rf"\text{{Solve }}\;{given[1]}",
        answer,
        {"center": center, "radius": radius, "low": low, "high": high},
        traces,
        "two_case_expansion",
        repair_style,
    )


def _function_composition(index: int, seed: int) -> ProblemFamily:
    rng = _rng(seed, "function_composition", index)
    while True:
        a, c = rng.randint(2, 6), rng.randint(2, 6)
        b = rng.choice([value for value in range(-6, 7) if value != 0])
        d = rng.randint(-6, 6)
        t = rng.choice([value for value in range(-5, 6) if value != 0])
        inner = c * t + d
        correct_value = a * inner + b
        reverse_value = c * (a * t + b) + d
        if reverse_value != correct_value:
            break
    given = (
        f"f(x)={a}x+({b})，g(x)={c}x+({d})，求 f(g({t}))",
        rf"f(x)={a}x+({b}),\;g(x)={c}x+({d}),\;f(g({t}))",
    )
    correct = [
        (f"g({t}) = {c}*({t}) + ({d}) = {inner}", rf"g({t})={c}\times({t})+({d})={inner}"),
        (f"f(g({t})) = {a}*({inner}) + ({b}) = {correct_value}", rf"f(g({t}))={a}\times({inner})+({b})={correct_value}"),
    ]
    if index % 2:
        ft = a * t + b
        logical = [
            (f"f({t}) = {a}*({t}) + ({b}) = {ft}", rf"f({t})={a}\times({t})+({b})={ft}"),
            (f"g(f({t})) = {c}*({ft}) + ({d}) = {reverse_value}", rf"g(f({t}))={c}\times({ft})+({d})={reverse_value}"),
        ]
        wrong_answer = str(reverse_value)
        mechanism = "reverse_function_composition"
        diagnosis = "把 f(g(x)) 的复合顺序颠倒成了 g(f(x))"
        error_index = 0
        path_id = "evaluate_inner_then_outer"
    else:
        expanded = a * c * t + a * d
        correct = [
            (f"f(g({t})) = {a}*({c}*({t})+({d}))+({b})", rf"f(g({t}))={a}({c}({t})+({d}))+({b})"),
            (f"= {a*c*t} + ({a*d}) + ({b}) = {correct_value}", rf"={a*c*t}+({a*d})+({b})={correct_value}"),
        ]
        logical = [
            correct[0],
            (f"= {a*c*t} + ({a*d}) = {expanded}", rf"={a*c*t}+({a*d})={expanded}"),
        ]
        wrong_answer = str(expanded)
        mechanism = "drop_outer_function_bias"
        diagnosis = "代入内层函数后漏掉了外层函数的常数项"
        error_index = 1
        path_id = "symbolic_substitute_then_expand"
    slip_value = correct_value + (1 if index % 2 else -1)
    slip_lines = [*correct[:-1], (f"= {slip_value}", rf"={slip_value}")]
    corrected, repair_index, repair_style = _repair_trace(
        index, logical, correct, diagnosis, error_index=error_index
    )
    traces = _traces(
        given=given,
        correct_lines=correct,
        logical_lines=logical,
        slip_lines=slip_lines,
        corrected_lines=corrected,
        correct_answer=str(correct_value),
        logical_answer=wrong_answer,
        slip_answer=str(slip_value),
        misconception_id=mechanism,
        misconception_zh=diagnosis,
        logical_error_index=error_index,
        corrected_error_index=error_index,
        corrected_repair_index=repair_index,
    )
    return ProblemFamily(
        f"LEF-FC-{index:03d}",
        "function_composition",
        given[0],
        given[1],
        str(correct_value),
        {"a": a, "b": b, "c": c, "d": d, "t": t},
        traces,
        path_id,
        repair_style,
    )


def _weighted_average(index: int, seed: int) -> ProblemFamily:
    rng = _rng(seed, "weighted_average", index)
    n1 = rng.randint(3, 15)
    n2 = rng.choice([value for value in range(3, 16) if value != n1])
    mean1 = rng.randint(45, 92)
    mean2 = rng.choice([value for value in range(45, 93) if value != mean1])
    total = n1 * mean1 + n2 * mean2
    correct_value = Fraction(total, n1 + n2)
    given = (
        f"甲组 {n1} 人，平均 {mean1} 分；乙组 {n2} 人，平均 {mean2} 分。求总平均分。",
        rf"n_1={n1},\bar x_1={mean1};\;n_2={n2},\bar x_2={mean2}",
    )
    correct = [
        (f"S = {n1}*{mean1} + {n2}*{mean2} = {total}", rf"S={n1}\times {mean1}+{n2}\times {mean2}={total}"),
        (f"mean = {total}/({n1}+{n2}) = {_fmt_fraction(correct_value)}", rf"\bar x=\frac{{{total}}}{{{n1}+{n2}}}={_latex_fraction(correct_value)}"),
    ]
    if index % 2:
        wrong_value = Fraction(mean1 + mean2, 2)
        logical = [
            (f"mean = ({mean1}+{mean2})/2", rf"\bar x=\frac{{{mean1}+{mean2}}}{{2}}"),
            (f"= {_fmt_fraction(wrong_value)}", rf"={_latex_fraction(wrong_value)}"),
        ]
        mechanism = "unweighted_group_means"
        diagnosis = "忽略两组人数不同，直接平均了两个组均值"
        error_index = 0
        path_id = "recover_total_then_divide"
    else:
        wrong_value = Fraction(total, n1 + n2 - 1)
        logical = [
            correct[0],
            (f"mean = {total}/({n1}+{n2}-1) = {_fmt_fraction(wrong_value)}", rf"\bar x=\frac{{{total}}}{{{n1}+{n2}-1}}={_latex_fraction(wrong_value)}"),
        ]
        mechanism = "weighted_average_wrong_total_count"
        diagnosis = "总分计算正确，但把总体人数少算了 1"
        error_index = 1
        path_id = "recover_total_then_divide"
    slip_value = correct_value + 1
    slip_lines = [correct[0], (f"mean = {_fmt_fraction(slip_value)}", rf"\bar x={_latex_fraction(slip_value)}")]
    corrected, repair_index, repair_style = _repair_trace(
        index, logical, correct, diagnosis, error_index=error_index
    )
    traces = _traces(
        given=given,
        correct_lines=correct,
        logical_lines=logical,
        slip_lines=slip_lines,
        corrected_lines=corrected,
        correct_answer=_fmt_fraction(correct_value),
        logical_answer=_fmt_fraction(wrong_value),
        slip_answer=_fmt_fraction(slip_value),
        misconception_id=mechanism,
        misconception_zh=diagnosis,
        logical_error_index=error_index,
        corrected_error_index=error_index,
        corrected_repair_index=repair_index,
    )
    return ProblemFamily(
        f"LEF-WA-{index:03d}",
        "weighted_average",
        given[0],
        given[1],
        _fmt_fraction(correct_value),
        {"n1": n1, "n2": n2, "mean1": mean1, "mean2": mean2, "total": total},
        traces,
        path_id,
        repair_style,
    )


def _linear_system(index: int, seed: int) -> ProblemFamily:
    rng = _rng(seed, "linear_system_elimination", index)
    while True:
        x = rng.choice([value for value in range(-9, 10) if value != 0])
        y = rng.choice([value for value in range(-9, 10) if value not in (0, x)])
        s, d = x + y, x - y
        if s != 0:
            break
    given = (
        f"解方程组：x+y={s}，x-y={d}",
        rf"\begin{{cases}}x+y={s}\\x-y={d}\end{{cases}}",
    )
    correct = [
        (f"(1)+(2): 2x = {s}+({d}) = {2*x}", rf"2x={s}+({d})={2*x}"),
        (f"x = {x}", rf"x={x}"),
        (f"y = {s}-({x}) = {y}", rf"y={s}-({x})={y}"),
    ]
    if index % 2:
        wrong_x = Fraction(s - d, 2)
        wrong_y = Fraction(s, 1) - wrong_x
        logical = [
            (f"(1)+(2): 2x = {s}-({d}) = {s-d}", rf"2x={s}-({d})={s-d}"),
            (f"x = {_fmt_fraction(wrong_x)}", rf"x={_latex_fraction(wrong_x)}"),
            (f"y = {_fmt_fraction(wrong_y)}", rf"y={_latex_fraction(wrong_y)}"),
        ]
        mechanism = "elimination_side_mismatch"
        diagnosis = "左边把两式相加，右边却执行了相减"
        error_index = 0
    else:
        wrong_x = Fraction(s, 1) + Fraction(d, 2)
        wrong_y = Fraction(s, 1) - wrong_x
        logical = [
            correct[0],
            (f"x = {s} + ({d})/2 = {_fmt_fraction(wrong_x)}", rf"x={s}+\frac{{{d}}}{{2}}={_latex_fraction(wrong_x)}"),
            (f"y = {_fmt_fraction(wrong_y)}", rf"y={_latex_fraction(wrong_y)}"),
        ]
        mechanism = "divide_only_one_rhs_term"
        diagnosis = "由 2x=s+d 求 x 时只把右边第二项除以 2"
        error_index = 1
    slip_y = y + (1 if index % 2 else -1)
    slip_lines = [*correct[:-1], (f"y = {slip_y}", rf"y={slip_y}")]
    corrected, repair_index, repair_style = _repair_trace(
        index, logical, correct, diagnosis, error_index=error_index
    )
    answer = f"x={x}, y={y}"
    traces = _traces(
        given=given,
        correct_lines=correct,
        logical_lines=logical,
        slip_lines=slip_lines,
        corrected_lines=corrected,
        correct_answer=answer,
        logical_answer=f"x={_fmt_fraction(wrong_x)}, y={_fmt_fraction(wrong_y)}",
        slip_answer=f"x={x}, y={slip_y}",
        misconception_id=mechanism,
        misconception_zh=diagnosis,
        logical_error_index=error_index,
        corrected_error_index=error_index,
        corrected_repair_index=repair_index,
    )
    return ProblemFamily(
        f"LEF-SE-{index:03d}",
        "linear_system_elimination",
        given[0],
        given[1],
        answer,
        {"x": x, "y": y, "s": s, "d": d},
        traces,
        "add_equations_then_back_substitute",
        repair_style,
    )


FAMILY_BUILDERS: dict[str, Callable[[int, int], ProblemFamily]] = {
    "linear_shift": _linear_shift,
    "negative_distribution": _negative_distribution,
    "fraction_denominator": _fraction_denominator,
    "proportion_cross_product": _proportion,
    "exponent_product": _exponent_product,
    "power_of_power": _power_of_power,
    "difference_of_squares": _difference_of_squares,
    "inequality_negative_scale": _inequality,
    "absolute_value_branching": _absolute_value,
    "function_composition": _function_composition,
    "weighted_average": _weighted_average,
    "linear_system_elimination": _linear_system,
}


def generate_families(families_per_skill: int, seed: int) -> list[ProblemFamily]:
    families: list[ProblemFamily] = []
    for skill in SKILLS:
        seen_questions: set[str] = set()
        generated = 0
        candidate_index = 1
        while generated < families_per_skill:
            family = FAMILY_BUILDERS[skill.skill_id](candidate_index, seed)
            candidate_index += 1
            if family.question_text in seen_questions:
                continue
            seen_questions.add(family.question_text)
            generated += 1
            expected_id = family.family_id.rsplit("-", 1)[0] + f"-{generated:03d}"
            if family.family_id != expected_id:
                family = ProblemFamily(
                    expected_id,
                    family.skill_id,
                    family.question_text,
                    family.question_latex,
                    family.correct_answer,
                    family.parameters,
                    family.variants,
                    family.solution_path_id,
                    family.repair_style,
                )
            families.append(family)
    return families


def flatten_cases(families: Iterable[ProblemFamily], seed: int) -> list[dict[str, Any]]:
    families = list(families)
    mechanisms_by_skill: dict[str, set[str]] = {}
    for family in families:
        logical = next(trace for trace in family.variants if trace.variant == "logical_error")
        mechanisms_by_skill.setdefault(family.skill_id, set()).add(logical.error_mechanism)
    held_out_mechanism = {
        skill_id: sorted(mechanisms)[-1]
        for skill_id, mechanisms in mechanisms_by_skill.items()
    }
    cases: list[dict[str, Any]] = []
    counter = 1
    for family in families:
        skill = SKILL_BY_ID[family.skill_id]
        family_error_mechanism = next(
            trace.error_mechanism
            for trace in family.variants
            if trace.variant == "logical_error"
        )
        writer_style_id = f"writer_{_stable_seed(seed, family.family_id, 'writer') % 6 + 1}"
        split_bucket = _stable_seed(seed, family.family_id, "family_split") % 100
        family_split = "train" if split_bucket < 70 else "validation" if split_bucket < 85 else "test"
        for trace in family.variants:
            case_id = f"LEC{counter:05d}"
            counter += 1
            cases.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "case_id": case_id,
                    "family_id": family.family_id,
                    "counterfactual_group": family.family_id,
                    "skill_id": skill.skill_id,
                    "skill_name_zh": skill.name_zh,
                    "reasoning_family": skill.reasoning_family,
                    "prerequisite_skills": list(skill.prerequisites),
                    "question_text": family.question_text,
                    "question_latex": family.question_latex,
                    "parameters": family.parameters,
                    "family_error_mechanism": family_error_mechanism,
                    "solution_path_id": family.solution_path_id,
                    "repair_style": family.repair_style if trace.variant == "self_corrected" else "none",
                    "writer_style_id": writer_style_id,
                    "family_split": family_split,
                    "writer_ood_split": (
                        "test" if writer_style_id == "writer_6" else "validation" if writer_style_id == "writer_5" else "train"
                    ),
                    "mechanism_ood_split": (
                        "test"
                        if family_error_mechanism
                        == held_out_mechanism.get(skill.skill_id)
                        else "train"
                    ),
                    "variant": trace.variant,
                    "trace_steps": [asdict(step) for step in trace.steps],
                    "trace_display": [step.display for step in trace.steps],
                    "trace_latex": [step.latex for step in trace.steps],
                    "correct_answer": family.correct_answer,
                    "student_answer": trace.student_answer,
                    "final_correct": trace.final_correct,
                    "contains_error": trace.contains_error,
                    "error_persisted": trace.error_persisted,
                    "first_error_step": trace.first_error_step,
                    "error_position": (
                        "none"
                        if trace.first_error_step is None
                        else "early"
                        if trace.first_error_step == 2
                        else "late"
                    ),
                    "repair_step": trace.repair_step,
                    "error_mechanism": trace.error_mechanism,
                    "error_level": trace.error_level,
                    "gold_diagnosis_zh": trace.gold_diagnosis_zh,
                    "gold_state_operation": trace.state_operation,
                    "gold_state_key": skill.skill_id,
                    "evidence_strength": trace.evidence_strength,
                    "source_type": "deterministic_programmatic",
                    "uses_llm": False,
                    "uses_diffusion": False,
                    "generation_seed": seed,
                }
            )
    return cases


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def opaque(namespace, identifier):
    return digest(['original-data-prepare-v1', namespace, identifier])[:24]


def text(value):
    if not isinstance(value, str):
        raise ValueError('Expected a visible string, not a converted annotation')
    return value


def visible_cues(display, latex):
    """Surface markers only. An arrow does not establish a retraction target."""
    cues = []
    if display.strip() in ('=>', '→', '⇒') or latex.strip() == r'\Rightarrow':
        cues.append('arrow_marker')
    if '检查' in display:
        cues.append('check_word')
    if '重算' in display or '重新计算' in display:
        cues.append('restart_word')
    return cues


def surface_features(display, latex):
    """Non-exclusive lexical inventory; never called a successful math parse."""
    s = display + '\n' + latex
    patterns = {
        'equality': r'=', 'inequality': r'[<>≤≥]|\\(?:leq|geq)',
        'fraction_or_division': r'/|\\frac', 'power': r'\^',
        'absolute_bars': r'\|', 'branch_word': r'\bor\b|或',
        'function_call': r'[fg]\s*\(', 'system_environment': r'\\begin\{cases\}',
        'chinese_text': r'[\u4e00-\u9fff]', 'geometry_symbol': r'[⊥∥△∠]',
    }
    return sorted(k for k, p in patterns.items() if re.search(p, s))


def written_payload(question, question_latex, steps, answer):
    if not steps:
        raise ValueError('Empty student trace')
    nodes = []
    for index, (display, latex) in enumerate(steps, 1):
        display, latex = text(display), text(latex)
        nodes.append({'node_id': f's{index}', 'position': index, 'display': display,
                      'latex': latex, 'visible_cues': visible_cues(display, latex),
                      'surface_features': surface_features(display, latex)})
    return {
        'schema': 'written-observation-v1',
        'question': {'text': text(question), 'latex': text(question_latex)},
        'steps': nodes, 'student_answer': None if answer is None else text(answer),
        'sequence_edges': [{'from': f's{i}', 'to': f's{i+1}', 'type': 'written_after'}
                           for i in range(1, len(nodes))],
        'semantic_dependencies': None, 'retractions': None,
        'semantic_status': 'NOT_INFERRED',
    }


def controlled_payload(case):
    # Deliberately do not read step_index, relation_to_previous, parameters,
    # repair_style, or any correctness/diagnosis annotations.
    return written_payload(case['question_text'], case['question_latex'],
                           [(s['display'], s['latex']) for s in case['trace_steps']],
                           case['student_answer'])
