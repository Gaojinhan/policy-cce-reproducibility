from __future__ import annotations

from dataclasses import dataclass

from cmfg_cce.envs.toy import Observation
from cmfg_cce.policies.base import BiddingPolicy, PolicyAction, PolicyContext


M_VERY_LOW = 0.03
M_LOW = 0.06
M_MED = 0.12
M_HIGH = 0.22

L_VERY_FAST = 0.70
L_FAST = 0.80
L_REG = 1.00


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class ArchetypeFeatures:
    capacity_slack: float
    queue_utilization: float
    expediting_rate: float
    outside_pressure: float
    due_tightness: float
    delivery_weight: float
    recent_win_rate: float
    recent_profit_rate: float


def features_from_observation(obs: Observation, context: PolicyContext) -> ArchetypeFeatures:
    estimated_cost = max(1.0, float(obs.own_estimated_cost_for_current_order))
    return ArchetypeFeatures(
        capacity_slack=clip(obs.own_available_capacity_ratio, 0.0, 1.0),
        queue_utilization=clip(obs.own_platform_utilization, 0.0, 1.0),
        expediting_rate=clip((obs.own_alpha_i - 0.1) / 1.4, 0.0, 1.0),
        outside_pressure=clip(obs.own_outside_pressure / 0.60, 0.0, 1.0),
        due_tightness=clip(obs.due_tightness, 0.0, 1.0),
        delivery_weight=clip(context.delivery_score_weight, 0.0, 1.0),
        recent_win_rate=clip(obs.own_recent_win_rate, 0.0, 1.0),
        recent_profit_rate=clip(obs.own_recent_profit / estimated_cost, -1.0, 1.0),
    )


@dataclass(frozen=True)
class ArchetypePolicy(BiddingPolicy):
    name: str
    mechanism_family: str = "price"
    state_coefficient_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.state_coefficient_scale) <= 2.0:
            raise ValueError(
                "state_coefficient_scale must lie in (0, 2] for the frozen "
                f"policy sensitivity design; got {self.state_coefficient_scale!r}."
            )

    def _scale_state_terms(self, value: float, intercept: float) -> float:
        return float(intercept) + float(self.state_coefficient_scale) * (
            float(value) - float(intercept)
        )

    def act(self, obs: Observation, feasible: bool, context: PolicyContext) -> PolicyAction:
        if not feasible:
            return PolicyAction(skip=True)
        f = features_from_observation(obs, context)
        if self._participation_score(f) < 0.5:
            return PolicyAction(skip=True)
        markup = self._markup(f)
        lead_time_multiplier = self._lead_time_multiplier(f) if self.mechanism_family == "price_delivery" else 1.0
        return PolicyAction(skip=False, markup=markup, lead_time_multiplier=lead_time_multiplier)

    def _participation_score(self, f: ArchetypeFeatures) -> float:
        s, u, e, o, h, a = (
            f.capacity_slack,
            f.queue_utilization,
            f.expediting_rate,
            f.outside_pressure,
            f.due_tightness,
            f.delivery_weight,
        )
        if self.name == "A1":
            raw = 0.75 + 0.35 * s - 0.30 * u - 0.25 * o - 0.10 * h
            return self._scale_state_terms(raw, 0.75)
        if self.name == "A2":
            raw = 0.70 + 0.15 * s - 0.20 * u - 0.15 * o - 0.08 * h
            return self._scale_state_terms(raw, 0.70)
        if self.name == "A3":
            raw = 0.55 + 0.35 * s - 0.40 * u - 0.35 * o - 0.20 * h
            return self._scale_state_terms(raw, 0.55)
        if self.name == "A4":
            fast = s * (1.0 - u) * (1.0 - e) * (1.0 - o)
            raw = 0.55 + 0.25 * s - 0.25 * u - 0.20 * o + 0.20 * h * fast - 0.15 * h * (1.0 - fast)
            return self._scale_state_terms(raw, 0.55)
        if self.name == "A5":
            service = s * (1.0 - u) * (1.0 - e) * (1.0 - o)
            raw = 0.55 + 0.25 * s - 0.25 * u - 0.20 * o + 0.25 * a * service - 0.10 * a * (1.0 - service)
            return self._scale_state_terms(raw, 0.55)
        if self.name == "A6":
            raw = 0.65 + 0.20 * s - 0.20 * u - 0.45 * o - 0.12 * h
            return self._scale_state_terms(raw, 0.65)
        if self.name == "A7":
            rel = s * (1.0 - u) * (1.0 - o)
            raw = 0.60 + 0.30 * rel - 0.25 * h * (1.0 - rel) - 0.10 * e
            return self._scale_state_terms(raw, 0.60)
        if self.name == "A8":
            delta_w, delta_r = self._feedback_deltas(f)
            raw = 0.60 + 0.20 * s - 0.25 * u - 0.20 * o + 0.15 * max(delta_w, 0.0) * s - 0.15 * max(-delta_r, 0.0) * u
            return self._scale_state_terms(raw, 0.60)
        raise ValueError(f"Unsupported policy archetype: {self.name}")

    def _markup(self, f: ArchetypeFeatures) -> float:
        s, u, e, o, h, a = (
            f.capacity_slack,
            f.queue_utilization,
            f.expediting_rate,
            f.outside_pressure,
            f.due_tightness,
            f.delivery_weight,
        )
        price_delivery = self.mechanism_family == "price_delivery"
        if self.name == "A1":
            raw = M_LOW + 0.05 * u + 0.03 * h + 0.04 * o
            return clip(self._scale_state_terms(raw, M_LOW), 0.03, 0.20)
        if self.name == "A2":
            raw = M_MED + 0.04 * u + 0.03 * h + 0.03 * o
            return clip(self._scale_state_terms(raw, M_MED), 0.08, 0.24)
        if self.name == "A3":
            raw = M_HIGH + 0.08 * u + 0.06 * h + 0.08 * o - 0.04 * s
            return clip(self._scale_state_terms(raw, M_HIGH), 0.15, 0.40)
        if self.name == "A4":
            if price_delivery:
                raw = M_MED + 0.06 * h + 0.08 * e * h + 0.04 * u
                return clip(self._scale_state_terms(raw, M_MED), 0.10, 0.34)
            raw = M_MED + 0.08 * h + 0.06 * e * h + 0.04 * u + 0.03 * o
            return clip(self._scale_state_terms(raw, M_MED), 0.10, 0.32)
        if self.name == "A5":
            if price_delivery:
                raw = M_HIGH + 0.08 * a + 0.06 * e * a + 0.04 * h + 0.04 * u
                return clip(self._scale_state_terms(raw, M_HIGH), 0.16, 0.40)
            raw = M_HIGH + 0.04 * h + 0.04 * u + 0.04 * e
            return clip(self._scale_state_terms(raw, M_HIGH), 0.16, 0.34)
        if self.name == "A6":
            if price_delivery:
                raw = M_MED + 0.10 * o + 0.05 * u + 0.04 * h + 0.03 * e
                return clip(self._scale_state_terms(raw, M_MED), 0.10, 0.38)
            raw = M_MED + 0.10 * o + 0.05 * u + 0.04 * h
            return clip(self._scale_state_terms(raw, M_MED), 0.10, 0.36)
        if self.name == "A7":
            rel = s * (1.0 - u) * (1.0 - o)
            if price_delivery:
                raw = M_MED + 0.05 * u + 0.05 * h + 0.04 * o
                return clip(self._scale_state_terms(raw, M_MED), 0.10, 0.34)
            raw = M_MED + 0.07 * u + 0.06 * h * (1.0 - rel) + 0.04 * o
            return clip(self._scale_state_terms(raw, M_MED), 0.10, 0.32)
        if self.name == "A8":
            delta_w, delta_r = self._feedback_deltas(f)
            if price_delivery:
                raw = M_MED - 0.04 * max(delta_w, 0.0) * s + 0.06 * max(delta_r, 0.0) + 0.04 * u + 0.03 * e + 0.03 * a
                return clip(self._scale_state_terms(raw, M_MED), 0.05, 0.36)
            raw = M_MED - 0.05 * max(delta_w, 0.0) * s + 0.06 * max(delta_r, 0.0) + 0.05 * u + 0.03 * o
            return clip(self._scale_state_terms(raw, M_MED), 0.05, 0.34)
        raise ValueError(f"Unsupported policy archetype: {self.name}")

    def _lead_time_multiplier(self, f: ArchetypeFeatures) -> float:
        s, u, e, o, h, a = (
            f.capacity_slack,
            f.queue_utilization,
            f.expediting_rate,
            f.outside_pressure,
            f.due_tightness,
            f.delivery_weight,
        )
        if self.name == "A1":
            raw = L_REG - 0.10 * s - 0.05 * (1.0 - u) + 0.08 * u + 0.08 * e + 0.05 * o
            return clip(self._scale_state_terms(raw, L_REG), 0.80, 1.10)
        if self.name == "A2":
            raw = L_REG - 0.05 * h * s * (1.0 - e) + 0.08 * u + 0.05 * o
            return clip(self._scale_state_terms(raw, L_REG), 0.90, 1.20)
        if self.name == "A3":
            raw = L_REG + 0.15 * u + 0.12 * o + 0.08 * e - 0.05 * s
            return clip(self._scale_state_terms(raw, L_REG), 0.95, 1.35)
        if self.name == "A4":
            fast = s * (1.0 - u) * (1.0 - e) * (1.0 - o)
            raw = L_REG - 0.20 * h * fast + 0.10 * u + 0.08 * e + 0.05 * o
            return clip(self._scale_state_terms(raw, L_REG), 0.75, 1.20)
        if self.name == "A5":
            service = s * (1.0 - u) * (1.0 - e) * (1.0 - o)
            raw = L_REG - 0.25 * a * service - 0.08 * h * service + 0.12 * u + 0.10 * e + 0.06 * o
            return clip(self._scale_state_terms(raw, L_REG), 0.70, 1.25)
        if self.name == "A6":
            raw = L_REG + 0.15 * o + 0.08 * u + 0.05 * e - 0.04 * s
            return clip(self._scale_state_terms(raw, L_REG), 0.95, 1.35)
        if self.name == "A7":
            rel = s * (1.0 - u) * (1.0 - o)
            raw = L_REG + 0.20 * u + 0.10 * o + 0.08 * e - 0.08 * s - 0.05 * h * rel
            return clip(self._scale_state_terms(raw, L_REG), 0.85, 1.40)
        if self.name == "A8":
            delta_w, _ = self._feedback_deltas(f)
            adapt = s * (1.0 - u) * (1.0 - e) * (1.0 - o)
            raw = L_REG - 0.08 * max(delta_w, 0.0) * adapt - 0.08 * a * adapt + 0.10 * u + 0.08 * e + 0.05 * o
            return clip(self._scale_state_terms(raw, L_REG), 0.80, 1.25)
        raise ValueError(f"Unsupported policy archetype: {self.name}")

    @staticmethod
    def _feedback_deltas(f: ArchetypeFeatures) -> tuple[float, float]:
        target_win_rate = 0.35
        target_profit_rate = 0.12
        return target_win_rate - f.recent_win_rate, target_profit_rate - f.recent_profit_rate
