from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Manufacturer:
    id: int
    capacity: float
    c_base: float
    c_setup: float
    alpha: float = 0.0
    p_out: float = 0.0
    machine_mask: tuple[int, ...] = (1, 1, 1, 1, 1)


@dataclass(frozen=True)
class Order:
    id: int
    workload: float
    due_date: int
    required_mask: tuple[int, ...] = (1, 0, 0, 0, 0)


@dataclass
class QueueJob:
    workload: float
    remaining_steps: int
    due_date: int
    age: int = 0
    revenue: float = 0.0


@dataclass
class ManufacturerState:
    queue: list[QueueJob] = field(default_factory=list)
    recent_profit: float = 0.0
    recent_participation: float = 0.0
    recent_win_rate: float = 0.0
    outside_pressure: float = 0.0
    available: bool = True
    external_jobs: list[QueueJob] = field(default_factory=list)

    @property
    def queue_workload(self) -> float:
        return float(sum(job.workload for job in self.queue))

    @property
    def queue_depth(self) -> int:
        return len(self.queue)


@dataclass(frozen=True)
class Observation:
    own_available_capacity_ratio: float
    own_platform_utilization: float
    own_queue_depth: int
    own_estimated_cost_for_current_order: float
    own_recent_profit: float
    own_recent_participation: float
    own_alpha_i: float
    own_outside_pressure: float
    own_machine_capability_mask_or_eligibility: float
    workload_normalized: float
    due_date_normalized: float
    due_tightness: float
    invitation_signal: float
    invited_fraction: float
    recent_average_winning_bid: float
    recent_participation_rate: float
    market_tightness: float
    own_recent_win_rate: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "own_available_capacity_ratio": self.own_available_capacity_ratio,
            "own_platform_utilization": self.own_platform_utilization,
            "own_queue_depth": float(self.own_queue_depth),
            "own_estimated_cost_for_current_order": self.own_estimated_cost_for_current_order,
            "own_recent_profit": self.own_recent_profit,
            "own_recent_participation": self.own_recent_participation,
            "own_recent_win_rate": self.own_recent_win_rate,
            "own_alpha_i": self.own_alpha_i,
            "own_outside_pressure": self.own_outside_pressure,
            "own_machine_capability_mask_or_eligibility": self.own_machine_capability_mask_or_eligibility,
            "workload_normalized": self.workload_normalized,
            "due_date_normalized": self.due_date_normalized,
            "due_tightness": self.due_tightness,
            "invitation_signal": self.invitation_signal,
            "invited_fraction": self.invited_fraction,
            "recent_average_winning_bid": self.recent_average_winning_bid,
            "recent_participation_rate": self.recent_participation_rate,
            "market_tightness": self.market_tightness,
        }


@dataclass(frozen=True)
class ToyEnvConfig:
    layer: str = "toy"
    horizon: int = 30
    gamma: float = 0.99
    rho: float = 2.0
    workload_range: tuple[int, int] = (5, 35)
    due_date_range: tuple[int, int] = (5, 25)
    capacity_range: tuple[float, float] = (70.0, 130.0)
    c_base_range: tuple[float, float] = (1.0, 4.0)
    c_setup_range: tuple[float, float] = (4.0, 12.0)
    p_out_range: tuple[float, float] = (0.05, 0.20)
    alpha_range: tuple[float, float] = (0.1, 1.5)
    outside_pressure_mode: str = "none"
    pressure_decay: float = 0.90
    shock_size_range: tuple[float, float] = (0.05, 0.25)
    max_pressure: float = 0.60
    external_job_workload: tuple[int, int] = (5, 30)
    external_job_duration: tuple[int, int] = (3, 15)
    external_revenue_multiplier: float = 1.2
    availability_shocks: bool = False
    beta_p: float = 0.6
    beta_l: float = 0.4
    reserve_price_multiplier: float = 1.75
    single_bidder_markup_cap: float = 0.20


@dataclass
class MarketStats:
    recent_average_winning_bid: float = 0.0
    recent_participation_rate: float = 0.0
    fulfilled_jobs: int = 0
    total_payment: float = 0.0


def base_cost(manufacturer: Manufacturer, workload: float, utilization: float, rho: float = 2.0) -> float:
    utilization = max(0.0, min(0.999, float(utilization)))
    return manufacturer.c_setup + manufacturer.c_base * (1.0 + utilization**rho) * workload


def expediting_cost(base: float, alpha: float, ell_realized: float) -> float:
    return (1.0 + alpha * (1.0 - ell_realized)) * base


def remaining_capacity(manufacturer: Manufacturer, state: ManufacturerState) -> float:
    return manufacturer.capacity - state.queue_workload


def utilization(manufacturer: Manufacturer, state: ManufacturerState) -> float:
    return 1.0 - remaining_capacity(manufacturer, state) / manufacturer.capacity


def generate_population(config: ToyEnvConfig, n: int, seed: int) -> list[Manufacturer]:
    rng = np.random.default_rng(seed)
    capacities = rng.uniform(config.capacity_range[0], config.capacity_range[1], size=n)
    c_bases = rng.uniform(config.c_base_range[0], config.c_base_range[1], size=n)
    c_setups = rng.uniform(config.c_setup_range[0], config.c_setup_range[1], size=n)
    alphas = rng.uniform(config.alpha_range[0], config.alpha_range[1], size=n)
    p_outs = rng.uniform(config.p_out_range[0], config.p_out_range[1], size=n)
    masks = []
    for _ in range(n):
        mask = np.zeros(5, dtype=int)
        count = int(rng.integers(2, 5))
        mask[rng.choice(np.arange(5), size=count, replace=False)] = 1
        masks.append(tuple(int(x) for x in mask))
    return [
        Manufacturer(
            id=i,
            capacity=float(capacities[i]),
            c_base=float(c_bases[i]),
            c_setup=float(c_setups[i]),
            alpha=float(alphas[i]),
            p_out=float(p_outs[i]),
            machine_mask=masks[i],
        )
        for i in range(n)
    ]


def generate_order(config: ToyEnvConfig, order_id: int, rng: np.random.Generator) -> Order:
    low_w, high_w = config.workload_range
    low_d, high_d = config.due_date_range
    required = np.zeros(5, dtype=int)
    length = 1 if config.layer == "toy" else int(rng.integers(1, 3))
    required[rng.choice(np.arange(5), size=length, replace=False)] = 1
    return Order(
        id=order_id,
        workload=float(rng.integers(low_w, high_w + 1)),
        due_date=int(rng.integers(low_d, high_d + 1)),
        required_mask=tuple(int(x) for x in required),
    )


def is_eligible(manufacturer: Manufacturer, order: Order) -> bool:
    return all(m >= r for m, r in zip(manufacturer.machine_mask, order.required_mask, strict=True))


def external_workload(state: ManufacturerState) -> float:
    return float(sum(job.workload for job in state.external_jobs))


def platform_available_capacity(manufacturer: Manufacturer, state: ManufacturerState, config: ToyEnvConfig) -> float:
    base_remaining = remaining_capacity(manufacturer, state)
    if config.layer == "core":
        return max(0.0, base_remaining - state.outside_pressure * manufacturer.capacity)
    if config.layer == "full":
        return max(0.0, base_remaining - external_workload(state))
    return base_remaining


def build_observation(
    manufacturer: Manufacturer,
    state: ManufacturerState,
    order: Order,
    config: ToyEnvConfig,
    market: MarketStats,
    invited_fraction: float,
    available_capacity: float | None = None,
    eligible: bool = True,
) -> Observation:
    rem = platform_available_capacity(manufacturer, state, config) if available_capacity is None else available_capacity
    util = utilization(manufacturer, state)
    cost = base_cost(manufacturer, order.workload, util, config.rho)
    feasible = bool(state.available and eligible and rem >= order.workload)
    workload_norm = order.workload / max(1.0, float(config.workload_range[1]))
    due_norm = order.due_date / max(1.0, float(config.due_date_range[1]))
    due_tightness = workload_norm / max(due_norm, 1e-9)
    return Observation(
        own_available_capacity_ratio=max(0.0, rem / manufacturer.capacity),
        own_platform_utilization=util,
        own_queue_depth=state.queue_depth,
        own_estimated_cost_for_current_order=cost,
        own_recent_profit=state.recent_profit,
        own_recent_participation=state.recent_participation,
        own_alpha_i=manufacturer.alpha,
        own_outside_pressure=state.outside_pressure,
        own_machine_capability_mask_or_eligibility=float(eligible),
        workload_normalized=workload_norm,
        due_date_normalized=due_norm,
        due_tightness=due_tightness,
        invitation_signal=1.0 if feasible else 0.0,
        invited_fraction=invited_fraction,
        recent_average_winning_bid=market.recent_average_winning_bid,
        recent_participation_rate=market.recent_participation_rate,
        market_tightness=1.0 - invited_fraction,
        own_recent_win_rate=state.recent_win_rate,
    )


def advance_queues(states: list[ManufacturerState]) -> tuple[int, int]:
    completed = 0
    late = 0
    for state in states:
        next_queue: list[QueueJob] = []
        for job in state.queue:
            job.age += 1
            job.remaining_steps -= 1
            if job.remaining_steps <= 0:
                completed += 1
                if job.age > job.due_date:
                    late += 1
            else:
                next_queue.append(job)
        state.queue = next_queue
        next_external: list[QueueJob] = []
        for job in state.external_jobs:
            job.age += 1
            job.remaining_steps -= 1
            if job.remaining_steps > 0:
                next_external.append(job)
        state.external_jobs = next_external
    return completed, late


def advance_queues_with_external(states: list[ManufacturerState]) -> tuple[int, int, np.ndarray]:
    completed = 0
    late = 0
    external_revenues = np.zeros(len(states), dtype=float)
    for idx, state in enumerate(states):
        next_queue: list[QueueJob] = []
        for job in state.queue:
            job.age += 1
            job.remaining_steps -= 1
            if job.remaining_steps <= 0:
                completed += 1
                if job.age > job.due_date:
                    late += 1
            else:
                next_queue.append(job)
        state.queue = next_queue
        next_external: list[QueueJob] = []
        for job in state.external_jobs:
            job.age += 1
            job.remaining_steps -= 1
            if job.remaining_steps <= 0:
                external_revenues[idx] += job.revenue
            else:
                next_external.append(job)
        state.external_jobs = next_external
    return completed, late, external_revenues


def update_exogenous_state(
    manufacturer: Manufacturer,
    state: ManufacturerState,
    config: ToyEnvConfig,
    rng_outside: np.random.Generator,
    rng_availability: np.random.Generator,
) -> None:
    state.available = True
    if config.availability_shocks:
        state.available = bool(rng_availability.random() > 0.04)
    if config.layer == "core":
        state.outside_pressure *= config.pressure_decay
        if rng_outside.random() < manufacturer.p_out:
            state.outside_pressure += float(rng_outside.uniform(config.shock_size_range[0], config.shock_size_range[1]))
        state.outside_pressure = min(config.max_pressure, state.outside_pressure)
    elif config.layer == "full":
        if rng_outside.random() < manufacturer.p_out:
            workload = float(rng_outside.integers(config.external_job_workload[0], config.external_job_workload[1] + 1))
            duration = int(rng_outside.integers(config.external_job_duration[0], config.external_job_duration[1] + 1))
            revenue = config.external_revenue_multiplier * base_cost(manufacturer, workload, 0.0, config.rho)
            state.external_jobs.append(
                QueueJob(workload=workload, remaining_steps=duration, due_date=duration, revenue=revenue)
            )
        state.outside_pressure = min(config.max_pressure, external_workload(state) / max(1.0, manufacturer.capacity))
    else:
        state.outside_pressure = 0.0


def service_duration(manufacturer: Manufacturer, order: Order) -> int:
    daily_capacity = max(1.0, manufacturer.capacity / 10.0)
    return max(1, int(ceil(order.workload / daily_capacity)))


def add_platform_job(
    manufacturer: Manufacturer,
    state: ManufacturerState,
    order: Order,
    committed_duration: int | None = None,
) -> bool:
    overload = remaining_capacity(manufacturer, state) < order.workload
    state.queue.append(
        QueueJob(
            workload=order.workload,
            remaining_steps=committed_duration or service_duration(manufacturer, order),
            due_date=committed_duration or order.due_date,
        )
    )
    return overload


def config_from_mapping(mapping: dict[str, Any]) -> ToyEnvConfig:
    pop = mapping.get("population", {})
    orders = mapping.get("orders", {})
    defaults = mapping.get("mechanism_defaults", {})
    matrix = (
        mapping.get("toy")
        or mapping.get("core")
        or mapping.get("scalability")
        or mapping.get("full_robustness")
        or mapping.get("ablations")
        or mapping.get("pair_repair")
        or mapping.get("oracle_baselines")
        or mapping.get("reviewer_audits")
        or mapping.get("reviewer_layer2")
        or {}
    )
    return ToyEnvConfig(
        horizon=int(matrix.get("horizon", 30)),
        layer=str(mapping.get("env_version", "layer0_toy_v1")).split("_")[0].replace("layer0", "toy"),
        gamma=float(defaults.get("gamma", 0.99)),
        rho=float(defaults.get("rho", 2.0)),
        workload_range=tuple(orders.get("workload_range", [5, 35])),
        due_date_range=tuple(orders.get("due_date_range", [5, 25])),
        capacity_range=tuple(pop.get("capacity_range", [70.0, 130.0])),
        c_base_range=tuple(pop.get("c_base_range", [1.0, 4.0])),
        c_setup_range=tuple(pop.get("c_setup_range", [4.0, 12.0])),
        p_out_range=tuple(pop.get("p_out_range", [0.05, 0.20])),
        alpha_range=tuple(pop.get("alpha_range", [0.1, 1.5])),
        outside_pressure_mode=mapping.get("outside_pressure", {}).get("mode", "none"),
        pressure_decay=float(mapping.get("outside_pressure", {}).get("decay", 0.90)),
        shock_size_range=tuple(mapping.get("outside_pressure", {}).get("shock_size_range", [0.05, 0.25])),
        max_pressure=float(mapping.get("outside_pressure", {}).get("max_pressure", 0.60)),
        external_job_workload=tuple(mapping.get("external_jobs", {}).get("workload", [5, 30])),
        external_job_duration=tuple(mapping.get("external_jobs", {}).get("duration", [3, 15])),
        external_revenue_multiplier=float(mapping.get("external_jobs", {}).get("revenue_multiplier", 1.2)),
        availability_shocks=bool(mapping.get("availability_shocks", False)),
        beta_p=float(defaults.get("beta_p", 0.6)),
        beta_l=float(defaults.get("beta_l", 0.4)),
        reserve_price_multiplier=float(defaults.get("reserve_price_multiplier", 1.75)),
        single_bidder_markup_cap=float(defaults.get("single_bidder_markup_cap", 0.20)),
    )
