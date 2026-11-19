"""The sizing advisor: the pick is computed by replay, and its wording is held to the plan's numbers."""

from __future__ import annotations

from tcs import config
from tcs.advisor.sizing import LoggedRequest, Plan, advise, check, draft


def log(rate_per_s: float, rows: int, n: int = 120) -> list[LoggedRequest]:
    return [LoggedRequest(i / rate_per_s, rows, 20, 200, True, 3.0 + 4.0 * rows / 1e5, f"c{i % 6}") for i in range(n)]


def test_a_light_small_workload_gets_one_cheap_card():
    plan = advise(config.design("platform"), config.platform(), log(0.2, 5_000), 0.97, ["l4", "a100", "h100"], 3)
    assert (plan.kind, plan.count) == ("l4", 1) and plan.slo_attainment >= 0.97


def test_contexts_too_big_for_small_cards_move_up_a_kind():
    plan = advise(config.design("platform"), config.platform(), log(0.05, 900_000, 20), 0.97, ["l4", "a100", "h100"], 2)
    assert plan.kind in ("a100", "h100")


def test_the_draft_holds_only_numbers_from_the_plan():
    plan = Plan("l4", 2, 1.7, 0.991, 0.97, (("l4", 1, 0.85, 0.94), ("l4", 2, 1.7, 0.991), ("a100", 1, 4.6, 0.995)))
    assert check(draft(plan), plan) == []
    assert check("Two L4s at EUR 1.70 per hour give 99.9% in SLO.", plan) == ["99.9"]
