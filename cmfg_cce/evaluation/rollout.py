from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from cmfg_cce.envs.toy import (
    ManufacturerState,
    MarketStats,
    ToyEnvConfig,
    add_platform_job,
    advance_queues,
    advance_queues_with_external,
    base_cost,
    build_observation,
    expediting_cost,
    generate_order,
    generate_population,
    is_eligible,
    platform_available_capacity,
    remaining_capacity,
    update_exogenous_state,
    utilization,
)
from cmfg_cce.mechanisms.base import Bid, PriceMechanism
from cmfg_cce.mechanisms.delivery_critical import DeliveryCriticalMechanism
from cmfg_cce.mechanisms.delivery_first import DeliveryFirstMechanism
from cmfg_cce.mechanisms.price_critical import PriceCriticalMechanism
from cmfg_cce.mechanisms.price_first import PriceFirstMechanism
from cmfg_cce.policies.base import BiddingPolicy, PolicyContext


Profile = tuple[str, ...]


@dataclass(frozen=True)
class RolloutSeeds:
    type_seed: int
    order_seed: int
    tie_break_seed: int
    rollout_replication_seed: int
    outside_seed: int = 4000
    availability_seed: int = 5000


@dataclass
class RolloutEstimate:
    profile: Profile
    n_rollouts: int
    mean_returns: np.ndarray
    var_returns: np.ndarray
    ci_radius: np.ndarray
    mean_metrics: dict[str, float]
    var_metrics: dict[str, float]

    def to_jsonable(self) -> dict:
        return {
            "profile": list(self.profile),
            "n_rollouts": self.n_rollouts,
            "mean_returns": self.mean_returns.tolist(),
            "var_returns": self.var_returns.tolist(),
            "ci_radius": self.ci_radius.tolist(),
            "mean_metrics": self.mean_metrics,
            "var_metrics": self.var_metrics,
        }


def mechanism_from_id(mechanism_id: str, config: ToyEnvConfig) -> PriceMechanism:
    if mechanism_id == PriceFirstMechanism.mechanism_id:
        return PriceFirstMechanism()
    if mechanism_id == PriceCriticalMechanism.mechanism_id:
        return PriceCriticalMechanism(config.single_bidder_markup_cap)
    if mechanism_id == DeliveryFirstMechanism.mechanism_id:
        return DeliveryFirstMechanism()
    if mechanism_id == DeliveryCriticalMechanism.mechanism_id:
        return DeliveryCriticalMechanism(config.beta_p, config.beta_l, config.single_bidder_markup_cap)
    raise ValueError(f"Unsupported toy mechanism: {mechanism_id}")


def condition_order_on_provider_eligibility(order, population, rng: np.random.Generator):
    eligible_flags = [is_eligible(manufacturer, order) for manufacturer in population]
    if any(eligible_flags):
        return order, eligible_flags, False
    provider = population[int(rng.integers(0, len(population)))]
    capable_processes = [idx for idx, can_process in enumerate(provider.machine_mask) if can_process]
    if not capable_processes:
        return order, eligible_flags, False
    required_count = max(1, min(sum(order.required_mask), len(capable_processes)))
    required_indices = rng.choice(np.array(capable_processes, dtype=int), size=required_count, replace=False)
    required = np.zeros(len(provider.machine_mask), dtype=int)
    required[required_indices] = 1
    conditioned_order = replace(order, required_mask=tuple(int(x) for x in required))
    return conditioned_order, [is_eligible(manufacturer, conditioned_order) for manufacturer in population], True


def run_episode(
    profile: Profile,
    mechanism: PriceMechanism,
    policies: dict[str, BiddingPolicy],
    config: ToyEnvConfig,
    n_agents: int,
    seeds: RolloutSeeds,
    replication: int,
    markup_grid: tuple[float, ...],
    lead_time_grid: tuple[float, ...] = (0.55, 0.70, 0.85, 1.00),
) -> tuple[np.ndarray, dict[str, float]]:
    population = generate_population(config, n_agents, seeds.type_seed)
    states = [ManufacturerState() for _ in range(n_agents)]
    policy_context = PolicyContext(
        markup_grid=markup_grid,
        lead_time_grid=lead_time_grid,
        delivery_score_weight=config.beta_l,
    )
    returns = np.zeros(n_agents, dtype=float)
    undiscounted_profit = np.zeros(n_agents, dtype=float)
    win_counts = np.zeros(n_agents, dtype=float)
    utilization_by_agent = np.zeros(n_agents, dtype=float)
    markup_sum_by_agent = np.zeros(n_agents, dtype=float)
    markup_count_by_agent = np.zeros(n_agents, dtype=float)
    lead_sum_by_agent = np.zeros(n_agents, dtype=float)
    lead_count_by_agent = np.zeros(n_agents, dtype=float)
    market = MarketStats()
    participation_events = 0
    invitation_events = 0
    skip_events = 0
    eligibility_resample_events = 0
    eligibility_conditioned_events = 0
    total_bidders_possible = n_agents * config.horizon
    total_markup = 0.0
    total_lead_time = 0.0
    markup_count = 0
    lead_count = 0
    fulfilled = 0
    completed_total = 0
    late_total = 0
    overload_total = 0
    total_payment = 0.0
    utilization_sum = 0.0
    capacity_sum = 0.0
    external_pressure_sum = 0.0
    replication_offset = seeds.rollout_replication_seed * 100_000 + 10_000 * replication
    rng_order = np.random.default_rng(seeds.order_seed + replication_offset)
    rng_tie = np.random.default_rng(seeds.tie_break_seed + replication_offset)
    rng_outside = np.random.default_rng(seeds.outside_seed + replication_offset)
    rng_availability = np.random.default_rng(seeds.availability_seed + replication_offset)

    for t in range(config.horizon):
        completed, late, external_revenues = advance_queues_with_external(states)
        completed_total += completed
        late_total += late
        for i, external_revenue in enumerate(external_revenues):
            if external_revenue:
                returns[i] += (config.gamma**t) * external_revenue
                undiscounted_profit[i] += external_revenue
        for i, (manufacturer, state) in enumerate(zip(population, states, strict=True)):
            update_exogenous_state(manufacturer, state, config, rng_outside, rng_availability)
        order = generate_order(config, t, rng_order)
        eligible_flags = [is_eligible(manufacturer, order) for manufacturer in population]
        while not any(eligible_flags):
            order = generate_order(config, t, rng_order)
            eligible_flags = [is_eligible(manufacturer, order) for manufacturer in population]
            eligibility_resample_events += 1
            if eligibility_resample_events % 1000 == 0:
                order, eligible_flags, conditioned = condition_order_on_provider_eligibility(order, population, rng_order)
                eligibility_conditioned_events += int(conditioned)
        feasible_flags = [
            bool(
                state.available
                and eligible
                and platform_available_capacity(manufacturer, state, config) >= order.workload
            )
            for manufacturer, state, eligible in zip(population, states, eligible_flags, strict=True)
        ]
        invitation_events += sum(feasible_flags)
        invited_fraction = sum(feasible_flags) / max(1, n_agents)
        available_caps = [
            platform_available_capacity(manufacturer, state, config)
            for manufacturer, state in zip(population, states, strict=True)
        ]
        bids: list[Bid] = []
        for i, (manufacturer, state) in enumerate(zip(population, states, strict=True)):
            util = utilization(manufacturer, state)
            utilization_sum += util
            utilization_by_agent[i] += util
            capacity_sum += max(0.0, available_caps[i]) / manufacturer.capacity
            external_pressure_sum += state.outside_pressure
            obs = build_observation(
                manufacturer,
                state,
                order,
                config,
                market,
                invited_fraction,
                available_capacity=available_caps[i],
                eligible=eligible_flags[i],
            )
            action = policies[profile[i]].act(obs, feasible_flags[i], policy_context)
            base = base_cost(manufacturer, order.workload, util, config.rho)
            promised_duration = max(1, int(round(order.due_date * action.lead_time_multiplier)))
            ell_realized = promised_duration / max(1, order.due_date)
            cost = expediting_cost(base, manufacturer.alpha, ell_realized) if "delivery" in mechanism.mechanism_id else base
            if action.skip or not feasible_flags[i]:
                skip_events += 1
                bids.append(
                    Bid(
                        i,
                        price=0.0,
                        markup=0.0,
                        cost=cost,
                        lead_time_multiplier=1.0,
                        p_ref=1.0,
                        skip=True,
                    )
                )
                continue
            price = cost * (1.0 + action.markup)
            participation_events += 1
            total_markup += action.markup
            markup_sum_by_agent[i] += action.markup
            markup_count_by_agent[i] += 1
            markup_count += 1
            total_lead_time += ell_realized
            lead_sum_by_agent[i] += ell_realized
            lead_count_by_agent[i] += 1
            lead_count += 1
            bids.append(
                Bid(
                    i,
                    price=price,
                    markup=action.markup,
                    cost=cost,
                    lead_time_multiplier=ell_realized,
                    skip=False,
                )
            )

        reference_cost = float(np.mean([bid.cost for bid in bids]))
        max_valid_bid = max([bid.price for bid in bids if not bid.skip] or [0.0])
        reserve_price = max(config.reserve_price_multiplier * reference_cost, max_valid_bid)
        if "delivery" in mechanism.mechanism_id:
            scored_bids: list[Bid] = []
            for bid in bids:
                if bid.skip:
                    scored_bids.append(bid)
                    continue
                score = config.beta_p * bid.price / max(1e-9, reference_cost) + config.beta_l * bid.lead_time_multiplier
                scored_bids.append(
                    Bid(
                        bid.manufacturer_id,
                        price=bid.price,
                        markup=bid.markup,
                        cost=bid.cost,
                        lead_time_multiplier=bid.lead_time_multiplier,
                        score=score,
                        p_ref=reference_cost,
                        skip=False,
                    )
                )
            bids = scored_bids
        outcome = mechanism.allocate_and_pay(bids, reserve_price, rng_tie)
        if outcome.winner is not None:
            winner = outcome.winner
            winner_bid = next(bid for bid in bids if bid.manufacturer_id == winner)
            profit = outcome.payment - winner_bid.cost
            returns[winner] += (config.gamma**t) * profit
            undiscounted_profit[winner] += profit
            win_counts[winner] += 1
            states[winner].recent_profit = 0.8 * states[winner].recent_profit + 0.2 * profit
            committed_duration = max(1, int(round(order.due_date * winner_bid.lead_time_multiplier)))
            overloaded = add_platform_job(population[winner], states[winner], order, committed_duration)
            overload_total += int(overloaded)
            fulfilled += 1
            total_payment += outcome.payment
            market.recent_average_winning_bid = (
                0.8 * market.recent_average_winning_bid + 0.2 * outcome.payment
                if market.recent_average_winning_bid > 0
                else outcome.payment
            )

        participation_rate_step = len(outcome.valid_bids) / max(1, n_agents)
        for i, state in enumerate(states):
            participated = any((not bid.skip) and bid.manufacturer_id == i for bid in bids)
            state.recent_participation = 0.8 * state.recent_participation + 0.2 * float(participated)
            state.recent_win_rate = 0.8 * state.recent_win_rate + 0.2 * float(outcome.winner == i)
        market.recent_participation_rate = 0.8 * market.recent_participation_rate + 0.2 * participation_rate_step

    completed, late, external_revenues = advance_queues_with_external(states)
    completed_total += completed
    late_total += late
    for i, external_revenue in enumerate(external_revenues):
        if external_revenue:
            returns[i] += (config.gamma**config.horizon) * external_revenue
            undiscounted_profit[i] += external_revenue
    participation_rate = participation_events / max(1, total_bidders_possible)
    invitation_rate = invitation_events / max(1, total_bidders_possible)
    fulfillment_rate = fulfilled / max(1, config.horizon)
    late_completion_rate = late_total / max(1, completed_total)
    overload_rate = overload_total / max(1, fulfilled)
    platform_score = (
        10.0 * fulfilled
        - 0.01 * total_payment
        - 5.0 * late_total
        - 2.0 * overload_total
        + participation_rate
    )
    profit_mean = float(np.mean(undiscounted_profit))
    profit_gini = (
        float(np.sum(np.abs(undiscounted_profit[:, None] - undiscounted_profit[None, :])) / (2 * n_agents**2 * abs(profit_mean)))
        if abs(profit_mean) > 1e-9
        else 0.0
    )
    win_shares = win_counts / max(1.0, float(np.sum(win_counts)))
    winner_hhi = float(np.sum(win_shares**2))
    metrics = {
        "platform_operating_score": platform_score,
        "platform_total_payment": total_payment,
        "manufacturer_total_profit": float(np.sum(returns)),
        "manufacturer_profit_gini": profit_gini,
        "participation_rate": participation_rate,
        "invitation_rate": invitation_rate,
        "winner_concentration_hhi": winner_hhi,
        "fulfillment_rate": fulfillment_rate,
        "late_completion_rate": late_completion_rate,
        "overload_rate": overload_rate,
        "average_platform_utilization": utilization_sum / max(1, total_bidders_possible),
        "average_available_capacity": capacity_sum / max(1, total_bidders_possible),
        "average_markup": total_markup / max(1, markup_count),
        "average_lead_time_multiplier": total_lead_time / max(1, lead_count),
        "skip_rate": skip_events / max(1, total_bidders_possible),
        "eligibility_resample_events": float(eligibility_resample_events),
        "eligibility_conditioned_events": float(eligibility_conditioned_events),
        "no_provider_order_rate": 0.0,
        "external_pressure_mean": external_pressure_sum / max(1, total_bidders_possible),
        "completed_platform_jobs": float(completed_total),
    }
    for i in range(n_agents):
        metrics[f"win_rate_manufacturer_{i}"] = float(win_counts[i] / max(1, config.horizon))
        metrics[f"profit_manufacturer_{i}"] = float(undiscounted_profit[i])
        metrics[f"alpha_manufacturer_{i}"] = float(population[i].alpha)
        metrics[f"capacity_manufacturer_{i}"] = float(population[i].capacity)
        metrics[f"cost_manufacturer_{i}"] = float(population[i].c_base)
        metrics[f"utilization_manufacturer_{i}"] = float(utilization_by_agent[i] / max(1, config.horizon))
        metrics[f"markup_manufacturer_{i}"] = float(markup_sum_by_agent[i] / max(1.0, markup_count_by_agent[i]))
        metrics[f"lead_time_manufacturer_{i}"] = float(lead_sum_by_agent[i] / max(1.0, lead_count_by_agent[i]))
    alpha_values = np.array([manufacturer.alpha for manufacturer in population], dtype=float)
    median_alpha = float(np.median(alpha_values))
    for label, mask in {
        "low_alpha": alpha_values <= median_alpha,
        "high_alpha": alpha_values > median_alpha,
    }.items():
        if not np.any(mask):
            continue
        denom_wins = max(1.0, float(np.sum(win_counts)))
        metrics[f"{label}_win_share"] = float(np.sum(win_counts[mask]) / denom_wins)
        metrics[f"{label}_avg_markup"] = float(
            np.sum(markup_sum_by_agent[mask]) / max(1.0, float(np.sum(markup_count_by_agent[mask])))
        )
        metrics[f"{label}_avg_lead_time_multiplier"] = float(
            np.sum(lead_sum_by_agent[mask]) / max(1.0, float(np.sum(lead_count_by_agent[mask])))
        )
        metrics[f"{label}_profit"] = float(np.mean(undiscounted_profit[mask]))
        metrics[f"{label}_utilization"] = float(np.mean(utilization_by_agent[mask] / max(1, config.horizon)))
    return returns, metrics


def estimate_profile(
    profile: Profile,
    mechanism_id: str,
    policies: dict[str, BiddingPolicy],
    config: ToyEnvConfig,
    n_agents: int,
    seeds: RolloutSeeds,
    n_rollouts: int,
    markup_grid: tuple[float, ...],
    lead_time_grid: tuple[float, ...] = (0.55, 0.70, 0.85, 1.00),
) -> RolloutEstimate:
    mechanism = mechanism_from_id(mechanism_id, config)
    returns_list: list[np.ndarray] = []
    metrics_list: list[dict[str, float]] = []
    for replication in range(n_rollouts):
        returns, metrics = run_episode(
            profile,
            mechanism,
            policies,
            config,
            n_agents,
            seeds,
            replication,
            markup_grid,
            lead_time_grid,
        )
        returns_list.append(returns)
        metrics_list.append(metrics)
    returns_arr = np.vstack(returns_list)
    mean_returns = np.mean(returns_arr, axis=0)
    var_returns = np.var(returns_arr, axis=0, ddof=1) if n_rollouts > 1 else np.zeros(n_agents)
    ci_radius = 1.96 * np.sqrt(var_returns / max(1, n_rollouts))
    metric_keys = metrics_list[0].keys()
    mean_metrics = {key: float(np.mean([metrics[key] for metrics in metrics_list])) for key in metric_keys}
    var_metrics = {
        key: float(np.var([metrics[key] for metrics in metrics_list], ddof=1)) if n_rollouts > 1 else 0.0
        for key in metric_keys
    }
    return RolloutEstimate(profile, n_rollouts, mean_returns, var_returns, ci_radius, mean_metrics, var_metrics)


def estimates_to_records(estimates: Iterable[RolloutEstimate], mechanism_id: str, n_agents: int) -> list[dict]:
    records: list[dict] = []
    for estimate in estimates:
        record = {
            "mechanism": mechanism_id,
            "profile": "|".join(estimate.profile),
            "n_rollouts": estimate.n_rollouts,
        }
        for i in range(n_agents):
            record[f"return_agent_{i}"] = float(estimate.mean_returns[i])
            record[f"var_agent_{i}"] = float(estimate.var_returns[i])
            record[f"ci_agent_{i}"] = float(estimate.ci_radius[i])
        for key, value in estimate.mean_metrics.items():
            record[key] = value
        records.append(record)
    return records
