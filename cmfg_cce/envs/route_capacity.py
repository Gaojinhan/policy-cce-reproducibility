from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import ceil
from typing import Mapping

import numpy as np

from cmfg_cce.envs.toy import MarketStats, Observation


class MachineGroup(str, Enum):
    """Machine groups used by the case-inspired CNC platform."""

    TURNING = "T"
    MILLING_3AXIS = "M3"
    MILLING_5AXIS = "M5"
    GRINDING = "G"
    EDM = "EDM"


class OrderFamily(str, Enum):
    PRECISION_SHAFT = "F1"
    PRISMATIC_HOUSING = "F2"
    COMPLEX_BRACKET = "F3"
    MOLD_INSERT = "F4"


@dataclass(frozen=True)
class EquipmentGroup:
    machine_group: MachineGroup
    count: int
    nominal_effective_rate: float
    running_cost: float
    setup_cost: float
    capable: bool = True

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("Equipment count must be positive.")
        if self.nominal_effective_rate <= 0:
            raise ValueError("Nominal effective rate must be positive.")
        if self.running_cost < 0 or self.setup_cost < 0:
            raise ValueError("Equipment costs cannot be negative.")


@dataclass(frozen=True)
class RouteOperation:
    machine_group: MachineGroup
    setup_workload: float
    batch_processing_workload: float

    def __post_init__(self) -> None:
        if self.setup_workload < 0 or self.batch_processing_workload < 0:
            raise ValueError("Operation workloads cannot be negative.")
        if self.total_workload <= 0:
            raise ValueError("An operation must have positive total workload.")

    @property
    def total_workload(self) -> float:
        return float(self.setup_workload + self.batch_processing_workload)


@dataclass(frozen=True)
class MachiningOrder:
    id: int
    family: OrderFamily
    family_name: str
    batch_size: int
    route: tuple[RouteOperation, ...]
    due_date: int
    reference_route_duration: float

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if not self.route:
            raise ValueError("A machining order must contain at least one route operation.")
        if self.due_date <= 0 or self.reference_route_duration <= 0:
            raise ValueError("Due date and reference route duration must be positive.")

    @property
    def workload_by_group(self) -> dict[MachineGroup, float]:
        workloads: dict[MachineGroup, float] = {}
        for operation in self.route:
            workloads[operation.machine_group] = (
                workloads.get(operation.machine_group, 0.0) + operation.total_workload
            )
        return workloads

    @property
    def total_workload(self) -> float:
        return float(sum(operation.total_workload for operation in self.route))

    @property
    def required_groups(self) -> tuple[MachineGroup, ...]:
        return tuple(dict.fromkeys(operation.machine_group for operation in self.route))


@dataclass(frozen=True)
class RouteManufacturer:
    id: int
    name: str
    equipment: tuple[EquipmentGroup, ...]
    cost_multiplier: float
    alpha: float

    def __post_init__(self) -> None:
        if not self.equipment:
            raise ValueError("A manufacturer must own at least one equipment group.")
        groups = [item.machine_group for item in self.equipment]
        if len(groups) != len(set(groups)):
            raise ValueError("Each machine group must appear at most once per manufacturer.")
        if self.cost_multiplier <= 0:
            raise ValueError("Cost multiplier must be positive.")
        if self.alpha < 0:
            raise ValueError("Expediting parameter alpha cannot be negative.")

    @property
    def capabilities(self) -> frozenset[MachineGroup]:
        return frozenset(item.machine_group for item in self.equipment if item.capable)

    def equipment_group(self, group: MachineGroup) -> EquipmentGroup | None:
        return next((item for item in self.equipment if item.machine_group == group and item.capable), None)


@dataclass
class ReservedJob:
    order_id: int
    family: OrderFamily
    workload_by_group: dict[MachineGroup, float]
    remaining_commitment_periods: int

    def __post_init__(self) -> None:
        if self.remaining_commitment_periods <= 0:
            raise ValueError("A reservation must last at least one period.")
        if not self.workload_by_group or any(value <= 0 for value in self.workload_by_group.values()):
            raise ValueError("Reserved workloads must be positive.")


@dataclass
class RouteManufacturerState:
    """Private operating state; none of these fields is shared with rivals."""

    current_effective_rates: dict[MachineGroup, float]
    reservations: list[ReservedJob] = field(default_factory=list)
    outside_pressure: dict[MachineGroup, float] = field(default_factory=dict)
    recent_profit: float = 0.0
    recent_participation: float = 0.0
    recent_win_rate: float = 0.0
    available: bool = True

    @property
    def queue_depth(self) -> int:
        return len(self.reservations)

    def platform_workload(self, group: MachineGroup) -> float:
        return float(sum(job.workload_by_group.get(group, 0.0) for job in self.reservations))


@dataclass(frozen=True)
class OrderFamilySpec:
    family: OrderFamily
    name: str
    route: tuple[tuple[MachineGroup, float, float], ...]
    batch_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.route or not self.batch_sizes:
            raise ValueError("Order-family routes and batch-size supports cannot be empty.")
        if any(batch <= 0 for batch in self.batch_sizes):
            raise ValueError("Batch sizes must be positive.")
        if any(setup < 0 or unit <= 0 for _, setup, unit in self.route):
            raise ValueError("Route templates require nonnegative setup and positive unit workloads.")

    def total_workload(self, batch_size: int) -> float:
        return float(sum(setup + unit * batch_size for _, setup, unit in self.route))

    @property
    def expected_total_workload(self) -> float:
        return float(np.mean([self.total_workload(batch) for batch in self.batch_sizes]))


# The batch supports and workload functions below are frozen case inputs.  The
# factorial design equalizes expected total workload at the cell level rather
# than altering these manufacturing-order definitions.
ORDER_FAMILY_SPECS: dict[OrderFamily, OrderFamilySpec] = {
    OrderFamily.PRECISION_SHAFT: OrderFamilySpec(
        family=OrderFamily.PRECISION_SHAFT,
        name="precision shaft",
        route=(
            (MachineGroup.TURNING, 0.5, 0.25),
            (MachineGroup.GRINDING, 0.3, 0.08),
        ),
        batch_sizes=(20, 30, 40),
    ),
    OrderFamily.PRISMATIC_HOUSING: OrderFamilySpec(
        family=OrderFamily.PRISMATIC_HOUSING,
        name="prismatic housing",
        route=(
            (MachineGroup.MILLING_3AXIS, 0.8, 0.50),
            (MachineGroup.GRINDING, 0.3, 0.06),
        ),
        batch_sizes=(10, 15, 20),
    ),
    OrderFamily.COMPLEX_BRACKET: OrderFamilySpec(
        family=OrderFamily.COMPLEX_BRACKET,
        name="impeller/complex bracket",
        route=(
            (MachineGroup.MILLING_5AXIS, 1.2, 1.00),
            (MachineGroup.GRINDING, 0.3, 0.10),
        ),
        batch_sizes=(4, 7, 10),
    ),
    OrderFamily.MOLD_INSERT: OrderFamilySpec(
        family=OrderFamily.MOLD_INSERT,
        name="mold insert",
        route=(
            (MachineGroup.MILLING_3AXIS, 0.6, 0.25),
            (MachineGroup.EDM, 1.0, 0.75),
            (MachineGroup.GRINDING, 0.3, 0.08),
        ),
        batch_sizes=(6, 10, 14),
    ),
}


BALANCED_ROUTE_MIX: dict[OrderFamily, float] = {
    OrderFamily.PRECISION_SHAFT: 0.30,
    OrderFamily.PRISMATIC_HOUSING: 0.30,
    OrderFamily.COMPLEX_BRACKET: 0.20,
    OrderFamily.MOLD_INSERT: 0.20,
}

BOTTLENECK_HEAVY_ROUTE_MIX: dict[OrderFamily, float] = {
    OrderFamily.PRECISION_SHAFT: 0.15,
    OrderFamily.PRISMATIC_HOUSING: 0.15,
    OrderFamily.COMPLEX_BRACKET: 0.35,
    OrderFamily.MOLD_INSERT: 0.35,
}


@dataclass(frozen=True)
class RouteCapacityConfig:
    horizon: int = 60
    gamma: float = 0.99
    rho: float = 2.0
    capacity_window_hours: float = 16.0
    reference_machine_hours_per_period: float = 3.0
    due_slack_range: tuple[float, float] = (1.4, 2.2)
    load_level: str = "nominal"
    route_mix: str = "balanced"
    outside_regime: str = "normal"
    outside_burn_in: int = 20
    outside_decay: float = 0.85
    outside_volatility: float = 0.015
    max_outside_pressure: float = 0.60
    rate_decay: float = 0.85
    rate_volatility: float = 0.025
    rate_multiplier_bounds: tuple[float, float] = (0.85, 1.15)
    effective_rate_scale: float = 1.0
    cost_scale: float = 1.0
    beta_p: float = 0.6
    beta_l: float = 0.4
    reserve_price_multiplier: float = 1.75
    single_bidder_markup_cap: float = 0.20
    env_version: str = "route_capacity_v2"
    fleet_version: str = "cnc_fleet_v2"
    order_generator_version: str = "cnc_orders_v1"

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.outside_burn_in < 0:
            raise ValueError("Horizon must be positive and burn-in cannot be negative.")
        if self.load_level not in {"nominal", "high"}:
            raise ValueError("load_level must be 'nominal' or 'high'.")
        if self.route_mix not in {"balanced", "bottleneck_heavy"}:
            raise ValueError("route_mix must be 'balanced' or 'bottleneck_heavy'.")
        if self.outside_regime not in {"normal", "critical_machine"}:
            raise ValueError("outside_regime must be 'normal' or 'critical_machine'.")
        if self.capacity_window_hours <= 0 or self.reference_machine_hours_per_period <= 0:
            raise ValueError("Capacity and reference-rate constants must be positive.")
        if not 0 <= self.outside_decay < 1 or not 0 <= self.rate_decay < 1:
            raise ValueError("State-decay parameters must lie in [0, 1).")
        if self.outside_volatility < 0 or self.rate_volatility < 0:
            raise ValueError("State volatility cannot be negative.")
        if self.effective_rate_scale <= 0 or self.cost_scale <= 0:
            raise ValueError("Sensitivity scales must be positive.")

    @property
    def load_multiplier(self) -> float:
        return self.offered_load_multiplier * self.route_mix_normalization

    @property
    def offered_load_multiplier(self) -> float:
        return 1.0 if self.load_level == "nominal" else 1.25

    @property
    def raw_mix_expected_standard_hours(self) -> float:
        return float(
            sum(
                probability * ORDER_FAMILY_SPECS[family].expected_total_workload
                for family, probability in self.family_probabilities.items()
            )
        )

    @property
    def route_mix_normalization(self) -> float:
        balanced_expected = sum(
            probability * ORDER_FAMILY_SPECS[family].expected_total_workload
            for family, probability in BALANCED_ROUTE_MIX.items()
        )
        return float(balanced_expected / self.raw_mix_expected_standard_hours)

    @property
    def family_probabilities(self) -> Mapping[OrderFamily, float]:
        return BALANCED_ROUTE_MIX if self.route_mix == "balanced" else BOTTLENECK_HEAVY_ROUTE_MIX

    @property
    def pressure_cell(self) -> str:
        return f"{self.load_level}__{self.route_mix}__{self.outside_regime}"

    @property
    def maximum_standard_workload(self) -> float:
        maximum = max(
            spec.total_workload(max(spec.batch_sizes))
            for spec in ORDER_FAMILY_SPECS.values()
        )
        return float(1.25 * maximum)

    @property
    def maximum_due_date(self) -> int:
        max_reference_duration = self.maximum_standard_workload / self.reference_machine_hours_per_period
        return max(1, int(ceil(self.due_slack_range[1] * max_reference_duration)))

    def outside_target(self, group: MachineGroup) -> float:
        if self.outside_regime == "critical_machine" and group in {
            MachineGroup.MILLING_5AXIS,
            MachineGroup.GRINDING,
            MachineGroup.EDM,
        }:
            return 0.30
        return 0.10


def build_factorial_configs(base: RouteCapacityConfig | None = None) -> tuple[RouteCapacityConfig, ...]:
    base = base or RouteCapacityConfig()
    return tuple(
        replace(base, load_level=load, route_mix=mix, outside_regime=outside)
        for load in ("nominal", "high")
        for mix in ("balanced", "bottleneck_heavy")
        for outside in ("normal", "critical_machine")
    )


_GROUP_COSTS: dict[MachineGroup, tuple[float, float]] = {
    MachineGroup.TURNING: (2.4, 6.0),
    MachineGroup.MILLING_3AXIS: (3.0, 8.0),
    MachineGroup.MILLING_5AXIS: (4.5, 12.0),
    MachineGroup.GRINDING: (3.6, 7.0),
    MachineGroup.EDM: (4.2, 10.0),
}


def _equipment(group: MachineGroup, count: int, rate: float) -> EquipmentGroup:
    running, setup = _GROUP_COSTS[group]
    return EquipmentGroup(group, count, rate, running, setup)


def build_cnc_fleet() -> tuple[RouteManufacturer, ...]:
    """Return the fixed four-provider fleet used by every reporting seed."""

    return (
        RouteManufacturer(
            0,
            "P1",
            (
                _equipment(MachineGroup.TURNING, 2, 1.15),
                _equipment(MachineGroup.MILLING_3AXIS, 2, 0.90),
                _equipment(MachineGroup.GRINDING, 1, 1.05),
            ),
            cost_multiplier=0.95,
            alpha=0.4,
        ),
        RouteManufacturer(
            1,
            "P2",
            (
                _equipment(MachineGroup.TURNING, 2, 0.95),
                _equipment(MachineGroup.MILLING_3AXIS, 2, 1.05),
                _equipment(MachineGroup.MILLING_5AXIS, 3, 0.90),
                _equipment(MachineGroup.GRINDING, 1, 0.95),
                _equipment(MachineGroup.EDM, 3, 1.00),
            ),
            cost_multiplier=1.00,
            alpha=0.8,
        ),
        RouteManufacturer(
            2,
            "P3",
            (
                _equipment(MachineGroup.TURNING, 2, 0.85),
                _equipment(MachineGroup.MILLING_3AXIS, 2, 0.90),
                _equipment(MachineGroup.MILLING_5AXIS, 3, 1.10),
                _equipment(MachineGroup.GRINDING, 1, 0.90),
                _equipment(MachineGroup.EDM, 3, 1.10),
            ),
            cost_multiplier=1.08,
            alpha=1.2,
        ),
        RouteManufacturer(
            3,
            "P4",
            (
                _equipment(MachineGroup.TURNING, 2, 1.00),
                _equipment(MachineGroup.MILLING_3AXIS, 2, 0.95),
                _equipment(MachineGroup.MILLING_5AXIS, 3, 0.95),
                _equipment(MachineGroup.GRINDING, 2, 1.10),
                _equipment(MachineGroup.EDM, 3, 0.90),
            ),
            cost_multiplier=0.92,
            alpha=0.6,
        ),
    )


def initialize_route_state(
    manufacturer: RouteManufacturer,
    config: RouteCapacityConfig,
) -> RouteManufacturerState:
    rates = {
        item.machine_group: item.nominal_effective_rate
        for item in manufacturer.equipment
        if item.capable
    }
    pressures = {group: config.outside_target(group) for group in manufacturer.capabilities}
    return RouteManufacturerState(current_effective_rates=rates, outside_pressure=pressures)


def generate_machining_order(
    config: RouteCapacityConfig,
    order_id: int,
    rng: np.random.Generator,
) -> MachiningOrder:
    families = tuple(config.family_probabilities)
    probabilities = np.asarray([config.family_probabilities[family] for family in families], dtype=float)
    probabilities /= probabilities.sum()
    family = families[int(rng.choice(len(families), p=probabilities))]
    spec = ORDER_FAMILY_SPECS[family]
    batch_size = int(spec.batch_sizes[int(rng.integers(0, len(spec.batch_sizes)))])
    operations = tuple(
        RouteOperation(
            machine_group=group,
            setup_workload=config.load_multiplier * setup,
            batch_processing_workload=config.load_multiplier * unit * batch_size,
        )
        for group, setup, unit in spec.route
    )
    total_workload = sum(operation.total_workload for operation in operations)
    reference_duration = total_workload / config.reference_machine_hours_per_period
    due_slack = float(rng.uniform(*config.due_slack_range))
    return MachiningOrder(
        id=order_id,
        family=family,
        family_name=spec.name,
        batch_size=batch_size,
        route=operations,
        due_date=max(1, int(ceil(due_slack * reference_duration))),
        reference_route_duration=reference_duration,
    )


def expected_standard_machine_hours(config: RouteCapacityConfig) -> float:
    """Expected workload per offered order under the selected family mix."""

    return float(
        config.load_multiplier * config.raw_mix_expected_standard_hours
    )


def is_capability_feasible(manufacturer: RouteManufacturer, order: MachiningOrder) -> bool:
    return all(group in manufacturer.capabilities for group in order.required_groups)


def group_capacity(
    manufacturer: RouteManufacturer,
    state: RouteManufacturerState,
    group: MachineGroup,
    config: RouteCapacityConfig,
) -> float:
    equipment = manufacturer.equipment_group(group)
    if equipment is None:
        return 0.0
    current_rate = state.current_effective_rates.get(group, equipment.nominal_effective_rate)
    return float(
        config.capacity_window_hours
        * equipment.count
        * current_rate
        * config.effective_rate_scale
    )


def outside_workload(
    manufacturer: RouteManufacturer,
    state: RouteManufacturerState,
    group: MachineGroup,
    config: RouteCapacityConfig,
) -> float:
    return float(state.outside_pressure.get(group, 0.0) * group_capacity(manufacturer, state, group, config))


def available_capacity(
    manufacturer: RouteManufacturer,
    state: RouteManufacturerState,
    group: MachineGroup,
    config: RouteCapacityConfig,
) -> float:
    return max(
        0.0,
        group_capacity(manufacturer, state, group, config)
        - state.platform_workload(group)
        - outside_workload(manufacturer, state, group, config),
    )


def is_route_capacity_feasible(
    manufacturer: RouteManufacturer,
    state: RouteManufacturerState,
    order: MachiningOrder,
    config: RouteCapacityConfig,
) -> bool:
    if not state.available or not is_capability_feasible(manufacturer, order):
        return False
    return all(
        available_capacity(manufacturer, state, group, config) + 1e-10 >= workload
        for group, workload in order.workload_by_group.items()
    )


def is_aggregate_capacity_feasible(
    manufacturer: RouteManufacturer,
    state: RouteManufacturerState,
    order: MachiningOrder,
    config: RouteCapacityConfig,
) -> bool:
    """Scalar diagnostic that ignores which group supplies each machine-hour."""

    if not state.available or not is_capability_feasible(manufacturer, order):
        return False
    aggregate_available = sum(
        available_capacity(manufacturer, state, group, config)
        for group in order.required_groups
    )
    return aggregate_available + 1e-10 >= order.total_workload


def reserve_order(
    manufacturer: RouteManufacturer,
    state: RouteManufacturerState,
    order: MachiningOrder,
    commitment_periods: int,
    config: RouteCapacityConfig,
) -> bool:
    if commitment_periods <= 0:
        raise ValueError("commitment_periods must be positive.")
    if not is_route_capacity_feasible(manufacturer, state, order, config):
        return False
    state.reservations.append(
        ReservedJob(
            order_id=order.id,
            family=order.family,
            workload_by_group=dict(order.workload_by_group),
            remaining_commitment_periods=commitment_periods,
        )
    )
    return True


def advance_reservations(state: RouteManufacturerState) -> int:
    remaining: list[ReservedJob] = []
    completed = 0
    for job in state.reservations:
        job.remaining_commitment_periods -= 1
        if job.remaining_commitment_periods <= 0:
            completed += 1
        else:
            remaining.append(job)
    state.reservations = remaining
    return completed


def update_operating_state(
    manufacturer: RouteManufacturer,
    state: RouteManufacturerState,
    config: RouteCapacityConfig,
    rng_rate: np.random.Generator,
    rng_outside: np.random.Generator,
) -> None:
    """Advance private rates and group-specific outside pressure by one period."""

    lower, upper = config.rate_multiplier_bounds
    for equipment in manufacturer.equipment:
        group = equipment.machine_group
        previous_rate = state.current_effective_rates.get(group, equipment.nominal_effective_rate)
        previous_multiplier = previous_rate / equipment.nominal_effective_rate
        multiplier = (
            config.rate_decay * previous_multiplier
            + (1.0 - config.rate_decay)
            + float(rng_rate.normal(0.0, config.rate_volatility))
        )
        state.current_effective_rates[group] = equipment.nominal_effective_rate * float(
            np.clip(multiplier, lower, upper)
        )

        target = config.outside_target(group)
        pressure = (
            config.outside_decay * state.outside_pressure.get(group, target)
            + (1.0 - config.outside_decay) * target
            + float(rng_outside.normal(0.0, config.outside_volatility))
        )
        state.outside_pressure[group] = float(np.clip(pressure, 0.0, config.max_outside_pressure))


def route_platform_utilization(
    manufacturer: RouteManufacturer,
    state: RouteManufacturerState,
    order: MachiningOrder,
    config: RouteCapacityConfig,
) -> float:
    values = [
        state.platform_workload(group) / max(group_capacity(manufacturer, state, group, config), 1e-9)
        for group in order.required_groups
        if group in manufacturer.capabilities
    ]
    return float(np.clip(max(values, default=0.0), 0.0, 1.0))


def route_capacity_slack(
    manufacturer: RouteManufacturer,
    state: RouteManufacturerState,
    order: MachiningOrder,
    config: RouteCapacityConfig,
) -> float:
    if not is_capability_feasible(manufacturer, order):
        return 0.0
    residual_ratios = [
        (
            available_capacity(manufacturer, state, group, config) - workload
        )
        / max(group_capacity(manufacturer, state, group, config), 1e-9)
        for group, workload in order.workload_by_group.items()
    ]
    return float(np.clip(min(residual_ratios, default=0.0), 0.0, 1.0))


def route_outside_pressure(
    manufacturer: RouteManufacturer,
    state: RouteManufacturerState,
    order: MachiningOrder,
    config: RouteCapacityConfig,
) -> float:
    total = max(order.total_workload, 1e-9)
    weighted_pressure = 0.0
    for group, workload in order.workload_by_group.items():
        capacity = group_capacity(manufacturer, state, group, config)
        ratio = outside_workload(manufacturer, state, group, config) / max(capacity, 1e-9)
        weighted_pressure += (workload / total) * ratio
    return float(np.clip(weighted_pressure, 0.0, config.max_outside_pressure))


def route_base_cost(
    manufacturer: RouteManufacturer,
    state: RouteManufacturerState,
    order: MachiningOrder,
    config: RouteCapacityConfig,
) -> float:
    if not is_capability_feasible(manufacturer, order):
        return float("inf")
    cost = 0.0
    for operation in order.route:
        equipment = manufacturer.equipment_group(operation.machine_group)
        if equipment is None:
            return float("inf")
        capacity = group_capacity(manufacturer, state, operation.machine_group, config)
        utilization = float(
            np.clip(state.platform_workload(operation.machine_group) / max(capacity, 1e-9), 0.0, 0.999)
        )
        cost += equipment.setup_cost + equipment.running_cost * (1.0 + utilization**config.rho) * operation.total_workload
    return float(config.cost_scale * manufacturer.cost_multiplier * cost)


def route_expediting_cost(base_cost: float, alpha: float, realized_multiplier: float) -> float:
    return float((1.0 + alpha * (1.0 - realized_multiplier)) * base_cost)


def build_route_observation(
    manufacturer: RouteManufacturer,
    state: RouteManufacturerState,
    order: MachiningOrder,
    config: RouteCapacityConfig,
    market: MarketStats,
    invited_fraction: float,
    *,
    capability_feasible: bool | None = None,
    route_feasible: bool | None = None,
) -> Observation:
    """Map a private route state to the unchanged A1--A8 observation schema.

    The platform invitation is based on equipment capability.  Current
    route-capacity feasibility remains private and is enforced separately by
    the action mask passed to the policy.
    """

    capability = is_capability_feasible(manufacturer, order) if capability_feasible is None else capability_feasible
    # ``route_feasible`` remains in the adapter signature for callers that
    # construct both private feasibility flags together.  It must not alter
    # the public invitation signal.
    workload_normalized = float(np.clip(order.total_workload / config.maximum_standard_workload, 0.0, 1.0))
    due_normalized = float(np.clip(order.due_date / max(1, config.maximum_due_date), 0.0, 1.0))
    due_tightness = float(np.clip(order.reference_route_duration / max(1, order.due_date), 0.0, 1.0))
    estimated_cost = route_base_cost(manufacturer, state, order, config)
    if not np.isfinite(estimated_cost):
        estimated_cost = 0.0
    return Observation(
        own_available_capacity_ratio=route_capacity_slack(manufacturer, state, order, config),
        own_platform_utilization=route_platform_utilization(manufacturer, state, order, config),
        own_queue_depth=state.queue_depth,
        own_estimated_cost_for_current_order=estimated_cost,
        own_recent_profit=state.recent_profit,
        own_recent_participation=state.recent_participation,
        own_alpha_i=manufacturer.alpha,
        own_outside_pressure=route_outside_pressure(manufacturer, state, order, config),
        own_machine_capability_mask_or_eligibility=float(capability),
        workload_normalized=workload_normalized,
        due_date_normalized=due_normalized,
        due_tightness=due_tightness,
        invitation_signal=float(capability),
        invited_fraction=float(invited_fraction),
        recent_average_winning_bid=market.recent_average_winning_bid,
        recent_participation_rate=market.recent_participation_rate,
        market_tightness=1.0 - float(invited_fraction),
        own_recent_win_rate=state.recent_win_rate,
    )
