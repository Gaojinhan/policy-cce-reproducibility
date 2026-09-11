from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from cmfg_cce.envs.route_capacity import (
    MachineGroup,
    OrderFamily,
    RouteCapacityConfig,
    RouteManufacturer,
    advance_reservations,
    available_capacity,
    build_cnc_fleet,
    build_route_observation,
    generate_machining_order,
    group_capacity,
    initialize_route_state,
    is_aggregate_capacity_feasible,
    is_capability_feasible,
    is_route_capacity_feasible,
    outside_workload,
    reserve_order,
    route_base_cost,
    route_expediting_cost,
    update_operating_state,
)
from cmfg_cce.envs.toy import MarketStats
from cmfg_cce.evaluation.rollout import Profile, RolloutEstimate, RolloutSeeds
from cmfg_cce.mechanisms.base import Bid, PriceMechanism
from cmfg_cce.mechanisms.delivery_critical import DeliveryCriticalMechanism
from cmfg_cce.mechanisms.delivery_first import DeliveryFirstMechanism
from cmfg_cce.mechanisms.price_critical import PriceCriticalMechanism
from cmfg_cce.mechanisms.price_first import PriceFirstMechanism
from cmfg_cce.policies.base import BiddingPolicy, PolicyContext


def route_mechanism_from_id(mechanism_id: str, config: RouteCapacityConfig) -> PriceMechanism:
    if mechanism_id == PriceFirstMechanism.mechanism_id:
        return PriceFirstMechanism()
    if mechanism_id == PriceCriticalMechanism.mechanism_id:
        return PriceCriticalMechanism(config.single_bidder_markup_cap)
    if mechanism_id == DeliveryFirstMechanism.mechanism_id:
        return DeliveryFirstMechanism()
    if mechanism_id == DeliveryCriticalMechanism.mechanism_id:
        return DeliveryCriticalMechanism(config.beta_p, config.beta_l, config.single_bidder_markup_cap)
    raise ValueError(f"Unsupported route-capacity mechanism: {mechanism_id}")


def _replication_rngs(seeds: RolloutSeeds, replication: int) -> tuple[np.random.Generator, ...]:
    offset = int(seeds.rollout_replication_seed) * 100_000 + 10_000 * int(replication)
    return (
        np.random.default_rng(int(seeds.order_seed) + offset),
        np.random.default_rng(int(seeds.tie_break_seed) + offset),
        np.random.default_rng(int(seeds.outside_seed) + offset),
        np.random.default_rng(int(seeds.availability_seed) + offset),
    )


def _group_state_snapshot(
    fleet: tuple[RouteManufacturer, ...],
    states,
    config: RouteCapacityConfig,
) -> tuple[dict[MachineGroup, float], dict[MachineGroup, float], dict[MachineGroup, float]]:
    utilization: dict[MachineGroup, list[float]] = {group: [] for group in MachineGroup}
    slack: dict[MachineGroup, list[float]] = {group: [] for group in MachineGroup}
    outside: dict[MachineGroup, list[float]] = {group: [] for group in MachineGroup}
    for manufacturer, state in zip(fleet, states, strict=True):
        for group in manufacturer.capabilities:
            capacity = group_capacity(manufacturer, state, group, config)
            utilization[group].append(float(np.clip(state.platform_workload(group) / max(capacity, 1e-9), 0.0, 1.0)))
            slack[group].append(float(np.clip(available_capacity(manufacturer, state, group, config) / max(capacity, 1e-9), 0.0, 1.0)))
            outside[group].append(float(np.clip(outside_workload(manufacturer, state, group, config) / max(capacity, 1e-9), 0.0, 1.0)))
    return (
        {group: float(np.mean(values)) if values else 0.0 for group, values in utilization.items()},
        {group: float(np.mean(values)) if values else 0.0 for group, values in slack.items()},
        {group: float(np.mean(values)) if values else 0.0 for group, values in outside.items()},
    )


def _finite_metric_moments(values: list[float]) -> tuple[float, float]:
    """Return a mean and sample variance over defined rollout values only."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(finite))
    variance = float(np.var(finite, ddof=1)) if finite.size > 1 else 0.0
    return mean, variance


def run_route_episode(
    profile: Profile,
    mechanism: PriceMechanism,
    policies: dict[str, BiddingPolicy],
    config: RouteCapacityConfig,
    seeds: RolloutSeeds,
    replication: int,
    markup_grid: tuple[float, ...],
    lead_time_grid: tuple[float, ...] = (0.55, 0.70, 0.85, 1.00),
    fleet: tuple[RouteManufacturer, ...] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    fleet = fleet or build_cnc_fleet()
    n_agents = len(fleet)
    if len(profile) != n_agents:
        raise ValueError(f"Route profile has {len(profile)} policies; the CNC fleet has {n_agents} providers.")
    unknown = [policy_id for policy_id in profile if policy_id not in policies]
    if unknown:
        raise KeyError(f"Unknown policy IDs in profile: {unknown}")

    states = [initialize_route_state(manufacturer, config) for manufacturer in fleet]
    rng_order, rng_tie, rng_outside, rng_rate = _replication_rngs(seeds, replication)
    for _ in range(config.outside_burn_in):
        for manufacturer, state in zip(fleet, states, strict=True):
            update_operating_state(manufacturer, state, config, rng_rate, rng_outside)

    policy_context = PolicyContext(
        markup_grid=markup_grid,
        lead_time_grid=lead_time_grid,
        delivery_score_weight=config.beta_l,
    )
    returns = np.zeros(n_agents, dtype=float)
    undiscounted_profit = np.zeros(n_agents, dtype=float)
    win_counts = np.zeros(n_agents, dtype=float)
    utilization_by_agent = np.zeros(n_agents, dtype=float)
    markup_by_agent = np.zeros(n_agents, dtype=float)
    markup_count_by_agent = np.zeros(n_agents, dtype=float)
    lead_by_agent = np.zeros(n_agents, dtype=float)
    lead_count_by_agent = np.zeros(n_agents, dtype=float)
    market = MarketStats()

    total_opportunities = n_agents * config.horizon
    participation_events = 0
    invitation_events = 0
    capability_opportunities = 0
    scalar_capacity_opportunities = 0
    route_capacity_opportunities = 0
    aggregate_route_mismatches = 0
    mismatch_orders = 0
    assignment_count = 0
    completed_reservations = 0
    total_payment = 0.0
    total_markup = 0.0
    markup_count = 0
    total_lead = 0.0
    total_commitment_periods = 0.0
    lead_count = 0
    single_valid_bidder_orders = 0
    at_least_three_valid_bidder_orders = 0
    single_route_provider_orders = 0
    no_route_provider_orders = 0
    at_least_three_route_provider_orders = 0
    capability_provider_sum = 0
    scalar_capacity_provider_sum = 0
    route_provider_sum = 0
    total_order_workload = 0.0
    total_due_date = 0.0

    family_offered = {family: 0 for family in OrderFamily}
    family_assigned = {family: 0 for family in OrderFamily}
    family_at_least_three_route_providers = {family: 0 for family in OrderFamily}
    family_capability_opportunities = {family: 0 for family in OrderFamily}
    family_scalar_capacity_opportunities = {family: 0 for family in OrderFamily}
    family_route_capacity_opportunities = {family: 0 for family in OrderFamily}
    family_scalar_route_false_positives = {family: 0 for family in OrderFamily}
    policy_wins = {f"A{index}": 0 for index in range(1, 9)}
    group_utilization_sum = {group: 0.0 for group in MachineGroup}
    group_slack_sum = {group: 0.0 for group in MachineGroup}
    group_outside_sum = {group: 0.0 for group in MachineGroup}
    group_snapshot_count = {group: 0 for group in MachineGroup}
    group_shortfall_count = {group: 0 for group in MachineGroup}
    group_requirement_count = {group: 0 for group in MachineGroup}

    for t in range(config.horizon):
        for state in states:
            completed_reservations += advance_reservations(state)
        for manufacturer, state in zip(fleet, states, strict=True):
            update_operating_state(manufacturer, state, config, rng_rate, rng_outside)

        utilization_snapshot, slack_snapshot, outside_snapshot = _group_state_snapshot(fleet, states, config)
        for group in MachineGroup:
            if any(group in manufacturer.capabilities for manufacturer in fleet):
                group_utilization_sum[group] += utilization_snapshot[group]
                group_slack_sum[group] += slack_snapshot[group]
                group_outside_sum[group] += outside_snapshot[group]
                group_snapshot_count[group] += 1

        order = generate_machining_order(config, t, rng_order)
        family_offered[order.family] += 1
        total_order_workload += order.total_workload
        total_due_date += order.due_date
        capability_flags = [is_capability_feasible(manufacturer, order) for manufacturer in fleet]
        feasible_flags = [
            is_route_capacity_feasible(manufacturer, state, order, config)
            for manufacturer, state in zip(fleet, states, strict=True)
        ]
        aggregate_flags = [
            is_aggregate_capacity_feasible(manufacturer, state, order, config)
            for manufacturer, state in zip(fleet, states, strict=True)
        ]
        capability_count = int(sum(capability_flags))
        scalar_capacity_count = int(sum(aggregate_flags))
        route_provider_count = int(sum(feasible_flags))
        capability_provider_sum += capability_count
        scalar_capacity_provider_sum += scalar_capacity_count
        route_provider_sum += route_provider_count
        capability_opportunities += capability_count
        scalar_capacity_opportunities += scalar_capacity_count
        route_capacity_opportunities += route_provider_count
        family_capability_opportunities[order.family] += capability_count
        family_scalar_capacity_opportunities[order.family] += scalar_capacity_count
        family_route_capacity_opportunities[order.family] += route_provider_count
        invitation_events += capability_count
        if route_provider_count == 0:
            no_route_provider_orders += 1
        if route_provider_count == 1:
            single_route_provider_orders += 1
        if route_provider_count >= 3:
            at_least_three_route_provider_orders += 1
            family_at_least_three_route_providers[order.family] += 1
        order_has_mismatch = False
        for manufacturer, state, capability, aggregate, route_feasible in zip(
            fleet,
            states,
            capability_flags,
            aggregate_flags,
            feasible_flags,
            strict=True,
        ):
            if capability and aggregate and not route_feasible:
                aggregate_route_mismatches += 1
                family_scalar_route_false_positives[order.family] += 1
                order_has_mismatch = True
            if not capability:
                continue
            workloads = order.workload_by_group
            for group, workload in workloads.items():
                group_requirement_count[group] += 1
                if available_capacity(manufacturer, state, group, config) + 1e-10 < workload:
                    group_shortfall_count[group] += 1
        mismatch_orders += int(order_has_mismatch)

        invited_fraction = capability_count / max(1, n_agents)
        base_costs = [
            route_base_cost(manufacturer, state, order, config)
            for manufacturer, state in zip(fleet, states, strict=True)
        ]
        finite_costs = [cost for cost in base_costs if np.isfinite(cost)]
        if not finite_costs:
            raise RuntimeError(f"Order {order.id} has no capability-feasible CNC provider.")
        capability_reference_cost = float(np.mean(finite_costs))

        bids: list[Bid] = []
        for i, (manufacturer, state) in enumerate(zip(fleet, states, strict=True)):
            capacity_values = [
                group_capacity(manufacturer, state, group, config)
                for group in manufacturer.capabilities
            ]
            platform_values = [state.platform_workload(group) for group in manufacturer.capabilities]
            total_capacity = max(sum(capacity_values), 1e-9)
            manufacturer_utilization = float(np.clip(sum(platform_values) / total_capacity, 0.0, 1.0))
            utilization_by_agent[i] += manufacturer_utilization
            observation = build_route_observation(
                manufacturer,
                state,
                order,
                config,
                market,
                invited_fraction,
                capability_feasible=capability_flags[i],
                route_feasible=feasible_flags[i],
            )
            action = policies[profile[i]].act(observation, feasible_flags[i], policy_context)
            nominal_multiplier = action.lead_time_multiplier if "delivery" in mechanism.mechanism_id else 1.0
            committed_duration = max(1, int(round(order.due_date * nominal_multiplier)))
            realized_multiplier = committed_duration / max(1, order.due_date)
            base_cost = base_costs[i] if np.isfinite(base_costs[i]) else capability_reference_cost
            cost = (
                route_expediting_cost(base_cost, manufacturer.alpha, realized_multiplier)
                if "delivery" in mechanism.mechanism_id
                else base_cost
            )
            if action.skip or not feasible_flags[i]:
                bids.append(
                    Bid(
                        manufacturer_id=i,
                        price=0.0,
                        markup=0.0,
                        cost=cost,
                        lead_time_multiplier=1.0,
                        p_ref=capability_reference_cost,
                        skip=True,
                    )
                )
                continue

            price = cost * (1.0 + action.markup)
            participation_events += 1
            total_markup += action.markup
            markup_count += 1
            markup_by_agent[i] += action.markup
            markup_count_by_agent[i] += 1
            total_lead += realized_multiplier
            total_commitment_periods += committed_duration
            lead_count += 1
            lead_by_agent[i] += realized_multiplier
            lead_count_by_agent[i] += 1
            bids.append(
                Bid(
                    manufacturer_id=i,
                    price=price,
                    markup=action.markup,
                    cost=cost,
                    lead_time_multiplier=realized_multiplier,
                    skip=False,
                )
            )

        reference_cost = float(
            np.mean([bid.cost for bid, capability in zip(bids, capability_flags, strict=True) if capability])
        )
        max_valid_bid = max([bid.price for bid in bids if not bid.skip] or [0.0])
        reserve_price = max(config.reserve_price_multiplier * reference_cost, max_valid_bid)
        if "delivery" in mechanism.mechanism_id:
            bids = [
                bid
                if bid.skip
                else Bid(
                    manufacturer_id=bid.manufacturer_id,
                    price=bid.price,
                    markup=bid.markup,
                    cost=bid.cost,
                    lead_time_multiplier=bid.lead_time_multiplier,
                    score=(
                        config.beta_p * bid.price / max(reference_cost, 1e-9)
                        + config.beta_l * bid.lead_time_multiplier
                    ),
                    p_ref=reference_cost,
                    skip=False,
                )
                for bid in bids
            ]

        outcome = mechanism.allocate_and_pay(bids, reserve_price, rng_tie)
        if len(outcome.valid_bids) == 1:
            single_valid_bidder_orders += 1
        if len(outcome.valid_bids) >= 3:
            at_least_three_valid_bidder_orders += 1
        if outcome.winner is not None:
            winner = int(outcome.winner)
            winner_bid = next(bid for bid in bids if bid.manufacturer_id == winner)
            profit = float(outcome.payment - winner_bid.cost)
            returns[winner] += (config.gamma**t) * profit
            undiscounted_profit[winner] += profit
            win_counts[winner] += 1
            states[winner].recent_profit = 0.8 * states[winner].recent_profit + 0.2 * profit
            committed_duration = max(1, int(round(order.due_date * winner_bid.lead_time_multiplier)))
            if not reserve_order(fleet[winner], states[winner], order, committed_duration, config):
                raise RuntimeError("The allocated route became infeasible before its reservation was recorded.")
            assignment_count += 1
            family_assigned[order.family] += 1
            total_payment += float(outcome.payment)
            if profile[winner] in policy_wins:
                policy_wins[profile[winner]] += 1
            market.recent_average_winning_bid = (
                0.8 * market.recent_average_winning_bid + 0.2 * outcome.payment
                if market.recent_average_winning_bid > 0
                else float(outcome.payment)
            )

        participation_rate_step = len(outcome.valid_bids) / max(1, n_agents)
        for i, state in enumerate(states):
            participated = any((not bid.skip) and bid.manufacturer_id == i for bid in bids)
            state.recent_participation = 0.8 * state.recent_participation + 0.2 * float(participated)
            state.recent_win_rate = 0.8 * state.recent_win_rate + 0.2 * float(outcome.winner == i)
        market.recent_participation_rate = (
            0.8 * market.recent_participation_rate + 0.2 * participation_rate_step
        )

    participation_rate = participation_events / max(1, total_opportunities)
    invitation_rate = invitation_events / max(1, total_opportunities)
    conditional_bid_rate = (
        participation_events / invitation_events
        if invitation_events
        else float("nan")
    )
    assignment_rate = assignment_count / max(1, config.horizon)
    payment_per_assignment = (
        total_payment / assignment_count
        if assignment_count
        else float("nan")
    )
    if assignment_count:
        win_shares = win_counts / float(np.sum(win_counts))
        winner_hhi = float(np.sum(win_shares**2))
        normalized_hhi = float(
            (winner_hhi - 1.0 / n_agents) / (1.0 - 1.0 / n_agents)
        )
    else:
        winner_hhi = float("nan")
        normalized_hhi = float("nan")
    group_mean_utilization = {
        group: group_utilization_sum[group] / max(1, group_snapshot_count[group])
        for group in MachineGroup
    }
    group_mean_slack = {
        group: group_slack_sum[group] / max(1, group_snapshot_count[group])
        for group in MachineGroup
    }
    group_mean_outside = {
        group: group_outside_sum[group] / max(1, group_snapshot_count[group])
        for group in MachineGroup
    }
    bottleneck_group = max(MachineGroup, key=lambda group: group_mean_utilization[group])
    platform_score = 10.0 * assignment_count - 0.01 * total_payment + participation_rate

    metrics: dict[str, float] = {
        "platform_operating_score": float(platform_score),
        "platform_total_payment": float(total_payment),
        "manufacturer_discounted_profit_sum": float(np.sum(returns)),
        "payment_per_assignment": float(payment_per_assignment),
        "manufacturer_total_profit": float(np.sum(returns)),
        "manufacturer_undiscounted_profit": float(np.sum(undiscounted_profit)),
        "participation_rate": float(participation_rate),
        "invitation_rate": float(invitation_rate),
        "conditional_bid_rate": float(conditional_bid_rate),
        "assignment_rate": float(assignment_rate),
        "assignment_count": float(assignment_count),
        "winner_concentration_hhi": float(winner_hhi),
        "normalized_winner_hhi": float(normalized_hhi),
        "mean_platform_queue_utilization": float(np.mean(list(group_mean_utilization.values()))),
        "average_platform_utilization": float(np.mean(list(group_mean_utilization.values()))),
        "bottleneck_machine_utilization": float(group_mean_utilization[bottleneck_group]),
        "bottleneck_machine_slack": float(group_mean_slack[bottleneck_group]),
        "average_markup": float(total_markup / max(1, markup_count)),
        "submitted_lead_time_multiplier": float(total_lead / max(1, lead_count)),
        "submitted_commitment_periods": float(total_commitment_periods / max(1, lead_count)),
        "average_lead_time_multiplier": float(total_lead / max(1, lead_count)),
        "capability_feasible_providers_per_order": float(capability_provider_sum / config.horizon),
        "scalar_capacity_feasible_manufacturers_per_order": float(
            scalar_capacity_provider_sum / config.horizon
        ),
        "route_capacity_feasible_manufacturers_per_order": float(
            route_provider_sum / config.horizon
        ),
        "route_capacity_feasible_providers_per_order": float(route_provider_sum / config.horizon),
        "no_route_feasible_provider_rate": float(no_route_provider_orders / config.horizon),
        "single_route_feasible_provider_rate": float(single_route_provider_orders / config.horizon),
        "at_least_three_route_capacity_feasible_provider_rate": float(
            at_least_three_route_provider_orders / config.horizon
        ),
        "single_valid_bidder_rate": float(single_valid_bidder_orders / config.horizon),
        "at_least_three_valid_bidder_rate": float(
            at_least_three_valid_bidder_orders / config.horizon
        ),
        "aggregate_feasible_but_route_infeasible_rate": float(
            aggregate_route_mismatches / max(1, capability_opportunities)
        ),
        "conditional_route_false_positive_rate": float(
            aggregate_route_mismatches / max(1, scalar_capacity_opportunities)
        ),
        "orders_with_aggregate_route_mismatch_rate": float(mismatch_orders / config.horizon),
        # Raw counts are retained so the formal bootstrap recomputes ratios and
        # HHI within each resample instead of averaging precomputed ratios.
        "orders_offered_count": float(config.horizon),
        "invitation_count": float(invitation_events),
        "valid_bid_count": float(participation_events),
        "submitted_markup_sum": float(total_markup),
        "submitted_markup_count": float(markup_count),
        "submitted_lead_time_sum": float(total_lead),
        "submitted_lead_time_count": float(lead_count),
        "submitted_commitment_periods_sum": float(total_commitment_periods),
        "capability_feasible_manufacturer_opportunities": float(capability_opportunities),
        "scalar_capacity_feasible_manufacturer_opportunities": float(
            scalar_capacity_opportunities
        ),
        "route_capacity_feasible_manufacturer_opportunities": float(
            route_capacity_opportunities
        ),
        "scalar_route_false_positive_opportunities": float(aggregate_route_mismatches),
        "orders_with_scalar_route_mismatch_count": float(mismatch_orders),
        "mean_order_workload": float(total_order_workload / config.horizon),
        "mean_order_due_date": float(total_due_date / config.horizon),
        "completed_capacity_reservations": float(completed_reservations),
    }
    for group in MachineGroup:
        code = group.value
        metrics[f"machine_group_utilization_{code}"] = float(group_mean_utilization[group])
        metrics[f"machine_group_slack_{code}"] = float(group_mean_slack[group])
        metrics[f"machine_group_remaining_capacity_{code}"] = float(group_mean_slack[group])
        metrics[f"machine_group_remaining_capacity_sum_{code}"] = float(
            group_slack_sum[group]
        )
        metrics[f"machine_group_snapshot_count_{code}"] = float(
            group_snapshot_count[group]
        )
        metrics[f"outside_pressure_{code}"] = float(group_mean_outside[group])
        metrics[f"capacity_shortfall_rate_{code}"] = float(
            group_shortfall_count[group] / max(1, group_requirement_count[group])
        )
        metrics[f"bottleneck_is_{code}"] = float(group == bottleneck_group)
    for family in OrderFamily:
        offered = family_offered[family]
        metrics[f"orders_offered_{family.value}"] = float(offered)
        metrics[f"orders_assigned_{family.value}"] = float(family_assigned[family])
        metrics[f"capability_feasible_manufacturer_opportunities_{family.value}"] = float(
            family_capability_opportunities[family]
        )
        metrics[f"scalar_capacity_feasible_manufacturer_opportunities_{family.value}"] = float(
            family_scalar_capacity_opportunities[family]
        )
        metrics[f"route_capacity_feasible_manufacturer_opportunities_{family.value}"] = float(
            family_route_capacity_opportunities[family]
        )
        metrics[f"scalar_route_false_positive_opportunities_{family.value}"] = float(
            family_scalar_route_false_positives[family]
        )
        metrics[f"assignment_rate_{family.value}"] = float(
            family_assigned[family] / max(1, offered)
        )
        metrics[
            f"orders_with_at_least_three_route_capacity_feasible_providers_{family.value}"
        ] = float(family_at_least_three_route_providers[family])
        metrics[
            f"orders_with_at_least_three_route_capacity_feasible_manufacturers_{family.value}"
        ] = float(family_at_least_three_route_providers[family])
        metrics[f"at_least_three_route_capacity_feasible_provider_rate_{family.value}"] = float(
            family_at_least_three_route_providers[family] / offered
            if offered
            else float("nan")
        )
    for policy_id, wins in policy_wins.items():
        metrics[f"winner_policy_{policy_id}_count"] = float(wins)
        metrics[f"winner_policy_{policy_id}_share"] = float(wins / max(1, assignment_count))
    for i, manufacturer in enumerate(fleet):
        metrics[f"win_rate_manufacturer_{i}"] = float(win_counts[i] / config.horizon)
        metrics[f"wins_manufacturer_{i}_count"] = float(win_counts[i])
        metrics[f"profit_manufacturer_{i}"] = float(undiscounted_profit[i])
        metrics[f"alpha_manufacturer_{i}"] = float(manufacturer.alpha)
        metrics[f"cost_multiplier_manufacturer_{i}"] = float(manufacturer.cost_multiplier)
        metrics[f"utilization_manufacturer_{i}"] = float(utilization_by_agent[i] / config.horizon)
        metrics[f"markup_manufacturer_{i}"] = float(
            markup_by_agent[i] / max(1.0, markup_count_by_agent[i])
        )
        metrics[f"lead_time_manufacturer_{i}"] = float(
            lead_by_agent[i] / max(1.0, lead_count_by_agent[i])
        )
    return returns, metrics


def estimate_route_profile(
    profile: Profile,
    mechanism_id: str,
    policies: dict[str, BiddingPolicy],
    config: RouteCapacityConfig,
    seeds: RolloutSeeds,
    n_rollouts: int,
    markup_grid: tuple[float, ...],
    lead_time_grid: tuple[float, ...] = (0.55, 0.70, 0.85, 1.00),
    fleet: tuple[RouteManufacturer, ...] | None = None,
) -> RolloutEstimate:
    if n_rollouts <= 0:
        raise ValueError("n_rollouts must be positive.")
    fleet = fleet or build_cnc_fleet()
    mechanism = route_mechanism_from_id(mechanism_id, config)
    returns_list: list[np.ndarray] = []
    metrics_list: list[dict[str, float]] = []
    for replication in range(n_rollouts):
        returns, metrics = run_route_episode(
            profile=profile,
            mechanism=mechanism,
            policies=policies,
            config=config,
            seeds=seeds,
            replication=replication,
            markup_grid=markup_grid,
            lead_time_grid=lead_time_grid,
            fleet=fleet,
        )
        returns_list.append(returns)
        metrics_list.append(metrics)
    returns_array = np.vstack(returns_list)
    n_agents = len(fleet)
    variance = (
        np.var(returns_array, axis=0, ddof=1)
        if n_rollouts > 1
        else np.zeros(n_agents, dtype=float)
    )
    metric_keys = metrics_list[0]
    metric_moments = {
        key: _finite_metric_moments([metrics[key] for metrics in metrics_list])
        for key in metric_keys
    }
    mean_metrics = {key: moments[0] for key, moments in metric_moments.items()}
    var_metrics = {key: moments[1] for key, moments in metric_moments.items()}
    return RolloutEstimate(
        profile=profile,
        n_rollouts=n_rollouts,
        mean_returns=np.mean(returns_array, axis=0),
        var_returns=variance,
        ci_radius=1.96 * np.sqrt(variance / n_rollouts),
        mean_metrics=mean_metrics,
        var_metrics=var_metrics,
    )


@dataclass(frozen=True)
class RouteCncBackend:
    """Pluggable profile evaluator for the route-aware CNC case study."""

    mechanism_id: str
    policies: dict[str, BiddingPolicy]
    config: RouteCapacityConfig
    seeds: RolloutSeeds
    markup_grid: tuple[float, ...] = (0.03, 0.06, 0.12, 0.22)
    lead_time_grid: tuple[float, ...] = (0.55, 0.70, 0.85, 1.00)
    fleet: tuple[RouteManufacturer, ...] = ()
    stream_label: str = "main"

    def __post_init__(self) -> None:
        if not self.fleet:
            object.__setattr__(self, "fleet", build_cnc_fleet())
        if len(self.fleet) != 4:
            raise ValueError("The published CNC case uses exactly four manufacturers.")
        route_mechanism_from_id(self.mechanism_id, self.config)

    @property
    def n_agents(self) -> int:
        return len(self.fleet)

    @property
    def horizon(self) -> int:
        return int(self.config.horizon)

    @property
    def env_version(self) -> str:
        return self.config.env_version

    @property
    def cache_identity(self) -> dict[str, Any]:
        return {
            "env_version": self.config.env_version,
            "fleet_version": self.config.fleet_version,
            "order_generator_version": self.config.order_generator_version,
            "pressure_cell": self.config.pressure_cell,
            "mechanism": self.mechanism_id,
            "seed": int(self.seeds.type_seed),
            "order_seed": int(self.seeds.order_seed),
            "outside_seed": int(self.seeds.outside_seed),
            "rate_seed": int(self.seeds.availability_seed),
            "tie_break_seed": int(self.seeds.tie_break_seed),
            "rollout_replication_seed": int(self.seeds.rollout_replication_seed),
            "stream": self.stream_label,
        }

    def run_episode(self, profile: Profile, replication: int) -> tuple[np.ndarray, dict[str, float]]:
        return run_route_episode(
            profile=profile,
            mechanism=route_mechanism_from_id(self.mechanism_id, self.config),
            policies=self.policies,
            config=self.config,
            seeds=self.seeds,
            replication=replication,
            markup_grid=self.markup_grid,
            lead_time_grid=self.lead_time_grid,
            fleet=self.fleet,
        )

    def estimate(self, profile: Profile, n_rollouts: int) -> RolloutEstimate:
        return estimate_route_profile(
            profile=profile,
            mechanism_id=self.mechanism_id,
            policies=self.policies,
            config=self.config,
            seeds=self.seeds,
            n_rollouts=n_rollouts,
            markup_grid=self.markup_grid,
            lead_time_grid=self.lead_time_grid,
            fleet=self.fleet,
        )

    def holdout_backend(self, seed_offset: int = 10_000_000) -> RouteCncBackend:
        """Return an evaluator with independent streams for post-selection audits."""

        if seed_offset <= 0:
            raise ValueError("Holdout seed offset must be positive.")
        holdout_seeds = RolloutSeeds(
            type_seed=self.seeds.type_seed + seed_offset,
            order_seed=self.seeds.order_seed + seed_offset,
            tie_break_seed=self.seeds.tie_break_seed + seed_offset,
            rollout_replication_seed=self.seeds.rollout_replication_seed + seed_offset,
            outside_seed=self.seeds.outside_seed + seed_offset,
            availability_seed=self.seeds.availability_seed + seed_offset,
        )
        return replace(self, seeds=holdout_seeds, stream_label="holdout")


# Descriptive alias retained for callers that use the generic evaluator name.
RouteProfileEvaluator = RouteCncBackend
