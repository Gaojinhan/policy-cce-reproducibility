from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from itertools import product
import json
from typing import Any

from cmfg_cce.baselines.mwu_tuning import MwuTuningConfig, revision_full_v1_mwu_grid
from cmfg_cce.evaluation.rollout import RolloutSeeds


MECHANISMS = (
    "M1_price_first",
    "M2_price_critical",
    "M3_delivery_first",
    "M4_delivery_critical",
)
FULL_SOLVER_BUNDLE = (
    "FullTensor-CCE-LP",
    "ExhaustiveCG-CCE",
    "REPAIR-SAD-CCE",
    "MWU-PolicyTrace",
)
SPARSE_SOLVER_BUNDLE = (
    "REPAIR-SAD-CCE",
    "MWU-PolicyTrace",
)
SELECTION_SENSITIVITY_SELECTORS = (
    "platform_operating_score",
    "total_manufacturer_return",
    "max_entropy_posthoc",
)
TRANSPLANT_DIRECTIONS = (
    ("M1_price_first", "M2_price_critical"),
    ("M2_price_critical", "M1_price_first"),
    ("M3_delivery_first", "M4_delivery_critical"),
    ("M4_delivery_critical", "M3_delivery_first"),
)
LIBRARY_VARIANT_POLICY_COUNTS = {
    "base_a1_a6": 6,
    "expanded_a1_a8": 8,
    "leave_out_a1": 5,
    "leave_out_a2": 5,
    "leave_out_a3": 5,
    "leave_out_a4": 5,
    "leave_out_a5": 5,
    "leave_out_a6": 5,
    "coefficient_scale_0p8": 6,
    "coefficient_scale_1p2": 6,
}
SOLVER_FORMAL_AUDIT_VARIANTS = (
    # FullTensor and ExhaustiveCG return the same q on the frozen grid.  They
    # therefore share one replication audit rather than spending twice on an
    # identical support-deviation closure.
    "full_tensor_and_exhaustive_cg_shared_q",
    "mwu_policy_trace",
    "dss_cce",
)
MIXED_FORMAL_AUDIT_VARIANTS = SOLVER_FORMAL_AUDIT_VARIANTS
SCALABILITY_EXACT_AUDIT_VARIANTS = SOLVER_FORMAL_AUDIT_VARIANTS
SCALABILITY_SPARSE_AUDIT_VARIANTS = ("mwu_policy_trace", "dss_cce")
AUDIT_VARIANT_SOLVERS = {
    "full_tensor_and_exhaustive_cg_shared_q": (
        "FullTensor-CCE-LP",
        "ExhaustiveCG-CCE",
    ),
    "mwu_policy_trace": ("MWU-PolicyTrace",),
    "dss_cce": ("REPAIR-SAD-CCE",),
}
DYNAMIC_SEED_FIELDS = (
    "order_seed",
    "outside_seed",
    "availability_seed",
    "tie_break_seed",
    "rollout_replication_seed",
)
PIPELINE_STAGES = (
    "training",
    "formal_audit",
    "outcome_evaluation",
    "mwu_tuning_audit",
    "transplant_audit",
    "runtime_measurement",
)
RESOURCE_MODES = ("bulk", "exclusive_runtime")
TRANSPLANT_ARMS = (
    "same_mechanism_control",
    "cross_mechanism",
    "target_recomputed",
)


@dataclass(frozen=True)
class FormalSeedNamespaces:
    """Disjoint deterministic namespaces; calibration streams are never formal evidence."""

    training: int = 0
    mwu_tuning_audit: int = 15_000_000
    formal_audit: int = 20_000_000
    outcome_evaluation: int = 30_000_000
    transplant_audit: int = 40_000_000

    def offsets(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}

    def __post_init__(self) -> None:
        offsets = self.offsets()
        if len(set(offsets.values())) != len(offsets):
            raise ValueError("Every seed namespace must have a unique offset.")
        if any(value < 0 for value in offsets.values()):
            raise ValueError("Seed namespace offsets must be nonnegative.")

    def apply(
        self,
        base: RolloutSeeds,
        namespace: str,
        *,
        preserve_population: bool = True,
    ) -> RolloutSeeds:
        if namespace not in self.offsets():
            raise KeyError(f"Unknown seed namespace: {namespace}")
        offset = self.offsets()[namespace]
        return RolloutSeeds(
            type_seed=int(base.type_seed) if preserve_population else int(base.type_seed) + offset,
            order_seed=int(base.order_seed) + offset,
            outside_seed=int(base.outside_seed) + offset,
            availability_seed=int(base.availability_seed) + offset,
            tie_break_seed=int(base.tie_break_seed) + offset,
            rollout_replication_seed=int(base.rollout_replication_seed) + offset,
        )

    def validate_disjoint(self, base: RolloutSeeds) -> None:
        streams: dict[str, set[int]] = {}
        for namespace in self.offsets():
            derived = self.apply(base, namespace)
            streams[namespace] = {
                int(getattr(derived, field_name)) for field_name in DYNAMIC_SEED_FIELDS
            }
        for first, second in product(streams, repeat=2):
            if first >= second:
                continue
            if streams[first].intersection(streams[second]):
                raise ValueError(f"Seed namespaces overlap: {first} and {second}.")


def mwu_tuning_grid() -> tuple[MwuTuningConfig, ...]:
    """Compatibility export of the single frozen grid owned by the MWU module."""

    return revision_full_v1_mwu_grid()


@dataclass(frozen=True)
class RobustnessSetting:
    setting_id: str
    cost_dispersion_scale: float
    alpha_scale: float
    effective_rate_scale: float
    policy_coefficient_scale: float


def fractional_robustness_settings() -> tuple[RobustnessSetting, ...]:
    """Resolution-IV half fraction with D=ABC, plus the normalized base setting."""

    low_high = {
        "cost": (0.5, 1.5),
        "alpha": (0.5, 1.5),
        "rate": (0.85, 1.15),
        "coefficient": (0.8, 1.2),
    }
    rows = [
        RobustnessSetting("base", 1.0, 1.0, 1.0, 1.0),
    ]
    for index, (a, b, c) in enumerate(product((-1, 1), repeat=3), start=1):
        d = a * b * c
        rows.append(
            RobustnessSetting(
                setting_id=f"fraction_{index:02d}",
                cost_dispersion_scale=low_high["cost"][a > 0],
                alpha_scale=low_high["alpha"][b > 0],
                effective_rate_scale=low_high["rate"][c > 0],
                policy_coefficient_scale=low_high["coefficient"][d > 0],
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class RevisionJobKey:
    phase: str
    family: str
    n_agents: int
    policies_per_agent: int
    mechanism: str
    seed: int
    variant: str = "base"
    source_mechanism: str | None = None
    target_mechanism: str | None = None
    solvers: tuple[str, ...] = ()
    selectors: tuple[str, ...] = ()
    stages: tuple[str, ...] = ()
    tuning_config_ids: tuple[str, ...] = ()
    evaluation_arms: tuple[str, ...] = ()
    resource_mode: str = "bulk"

    def __post_init__(self) -> None:
        if self.n_agents <= 0 or self.policies_per_agent <= 0 or self.seed < 0:
            raise ValueError("Job dimensions and seeds must be positive/nonnegative.")
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
        if (self.source_mechanism is None) != (self.target_mechanism is None):
            raise ValueError("Transplant jobs require both source and target mechanisms.")
        if self.target_mechanism is not None:
            if self.mechanism != self.target_mechanism:
                raise ValueError("A transplant job's mechanism must equal its target mechanism.")
            if self.source_mechanism == self.target_mechanism:
                raise ValueError("A transplant direction must change the mechanism.")
        if len(set(self.solvers)) != len(self.solvers):
            raise ValueError("A job cannot repeat a solver.")
        if len(set(self.selectors)) != len(self.selectors):
            raise ValueError("A job cannot repeat a CCE selector.")
        unknown_selectors = set(self.selectors).difference(SELECTION_SENSITIVITY_SELECTORS)
        if unknown_selectors:
            raise ValueError(f"Unknown CCE selectors: {sorted(unknown_selectors)}")
        if len(set(self.stages)) != len(self.stages):
            raise ValueError("A physical pipeline cannot repeat a stage.")
        unknown_stages = set(self.stages).difference(PIPELINE_STAGES)
        if unknown_stages:
            raise ValueError(f"Unknown pipeline stages: {sorted(unknown_stages)}")
        if len(set(self.tuning_config_ids)) != len(self.tuning_config_ids):
            raise ValueError("A tuning pipeline cannot repeat a MWU configuration.")
        if self.evaluation_arms and self.evaluation_arms != TRANSPLANT_ARMS:
            raise ValueError("Policy-transplant jobs must use the frozen three-arm design.")
        if self.resource_mode not in RESOURCE_MODES:
            raise ValueError(f"Unknown resource mode: {self.resource_mode}")
        if self.resource_mode == "exclusive_runtime" and not (
            "runtime_measurement" in self.stages or self.family == "mwu_tuning_pipeline"
        ):
            raise ValueError("GCP-exclusive jobs must perform a frozen runtime measurement.")

    @property
    def job_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return (
            f"{self.phase}__{self.family}__n{self.n_agents}j{self.policies_per_agent}"
            f"__{self.mechanism}__s{self.seed}__{self.variant}__{digest}"
        )


@dataclass(frozen=True)
class RevisionFullV1Matrix:
    campaign: str = "revision-full-v1"
    train_rollouts: int = 200
    audit_rollouts: int = 2000
    outcome_rollouts: int = 500
    bootstrap_samples: int = 5000
    alpha: float = 0.05
    solver_sizes: tuple[tuple[int, int], ...] = (
        (4, 6),
        (4, 8),
        (5, 6),
        (5, 8),
        (6, 6),
        (6, 8),
    )
    solver_seeds: tuple[int, ...] = (0, 1, 2)
    mechanisms: tuple[str, ...] = MECHANISMS
    mixed_variants: tuple[str, ...] = (
        "tight_capacity",
        "tight_due_dates",
        "high_core_pressure",
        "high_alpha_heterogeneity",
        "narrow_capacity",
    )
    mixed_seeds: tuple[int, ...] = (10, 11, 12, 13, 14)
    cnc_seeds: tuple[int, ...] = tuple(range(10))
    cnc_conditions: tuple[str, ...] = tuple(
        f"{load}__{mix}__{outside}"
        for load, mix, outside in product(
            ("nominal", "high"),
            ("balanced", "m5_edm_intensive"),
            ("normal", "high_outside_m5_g_edm"),
        )
    )
    representative_cnc_conditions: tuple[str, ...] = (
        "nominal__balanced__normal",
        "high__m5_edm_intensive__high_outside_m5_g_edm",
    )
    library_variants: tuple[str, ...] = tuple(LIBRARY_VARIANT_POLICY_COUNTS)
    sensitivity_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    robustness_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    robustness_settings: tuple[RobustnessSetting, ...] = field(
        default_factory=fractional_robustness_settings
    )
    scalability_exact_sizes: tuple[tuple[int, int], ...] = ((7, 6),)
    scalability_sparse_sizes: tuple[tuple[int, int], ...] = ((8, 8), (10, 8), (12, 8))
    scalability_exact_mechanisms: tuple[str, ...] = MECHANISMS
    scalability_sparse_mechanisms: tuple[str, ...] = (
        "M1_price_first",
        "M4_delivery_critical",
    )
    scalability_seeds: tuple[int, ...] = (0, 1, 2)
    mwu_tuning_sizes: tuple[tuple[int, int], ...] = ((4, 6), (5, 8), (6, 8))
    mwu_tuning_seeds: tuple[int, ...] = (100, 101)
    mwu_tuning_configs: tuple[MwuTuningConfig, ...] = field(default_factory=mwu_tuning_grid)
    mwu_tuning_audit_rollouts: int = 500
    transplant_directions: tuple[tuple[str, str], ...] = TRANSPLANT_DIRECTIONS
    selection_sensitivity_selectors: tuple[str, ...] = SELECTION_SENSITIVITY_SELECTORS
    seed_namespaces: FormalSeedNamespaces = field(default_factory=FormalSeedNamespaces)

    def __post_init__(self) -> None:
        if self.train_rollouts != 200 or self.audit_rollouts != 2000:
            raise ValueError("revision-full-v1 freezes R_train=200 and R_audit=2000.")
        if self.outcome_rollouts < 2 or self.bootstrap_samples < 100:
            raise ValueError("Outcome and bootstrap sample sizes are too small.")
        if not 0.0 < self.alpha < 0.5:
            raise ValueError("alpha must lie in (0, 0.5).")
        if set(self.mechanisms) != set(MECHANISMS):
            raise ValueError("The formal matrix must retain all four mechanisms.")
        if len(set(self.cnc_conditions)) != 8:
            raise ValueError("The CNC matrix must have eight unique operating conditions.")
        if self.mwu_tuning_audit_rollouts != 500:
            raise ValueError("revision-full-v1 freezes MWU tuning audits at R=500.")
        if self.library_variants != tuple(LIBRARY_VARIANT_POLICY_COUNTS):
            raise ValueError("The ten policy-library variants are frozen for revision-full-v1.")
        if len(self.robustness_settings) != 9 or self.robustness_settings[0].setting_id != "base":
            raise ValueError("Parameter robustness requires the base plus eight fraction points.")
        if len({item.setting_id for item in self.robustness_settings}) != 9:
            raise ValueError("Parameter-robustness setting IDs must be unique.")
        if len(self.mwu_tuning_configs) != 16 or len(
            {item.config_id for item in self.mwu_tuning_configs}
        ) != 16:
            raise ValueError("MWU tuning requires the frozen sixteen-point global grid.")
        if self.scalability_exact_mechanisms != self.mechanisms:
            raise ValueError("The exact N=7,J=6 extension must retain all four mechanisms.")
        if set(self.scalability_sparse_mechanisms) != {
            "M1_price_first",
            "M4_delivery_critical",
        }:
            raise ValueError("The sparse extension is frozen to M1 and M4.")
        if self.selection_sensitivity_selectors != SELECTION_SENSITIVITY_SELECTORS:
            raise ValueError("The three CCE-selection sensitivity selectors are frozen.")
        if self.transplant_directions != TRANSPLANT_DIRECTIONS:
            raise ValueError("The four policy-transplant directions are frozen.")

    @property
    def matrix_hash(self) -> str:
        payload = self.canonical_payload()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def canonical_payload(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def solver_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "training",
                "solver_benchmark",
                n,
                j,
                mechanism,
                seed,
                solvers=FULL_SOLVER_BUNDLE,
                stages=("training", "formal_audit"),
            )
            for (n, j), mechanism, seed in product(
                self.solver_sizes, self.mechanisms, self.solver_seeds
            )
        )

    def solver_formal_audit_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "audit",
                "solver_benchmark_formal",
                n,
                j,
                mechanism,
                seed,
                audit_variant,
                solvers=AUDIT_VARIANT_SOLVERS[audit_variant],
                stages=("formal_audit",),
            )
            for (n, j), mechanism, seed, audit_variant in product(
                self.solver_sizes,
                self.mechanisms,
                self.solver_seeds,
                SOLVER_FORMAL_AUDIT_VARIANTS,
            )
        )

    def solver_runtime_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "runtime",
                "solver_benchmark_runtime",
                n,
                j,
                mechanism,
                seed,
                "empty_cache_wall_time",
                solvers=FULL_SOLVER_BUNDLE,
                stages=("runtime_measurement",),
                resource_mode="exclusive_runtime",
            )
            for (n, j), mechanism, seed in product(
                self.solver_sizes, self.mechanisms, self.solver_seeds
            )
        )

    def mixed_challenge_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "training",
                "mixed_challenge",
                4,
                6,
                mechanism,
                seed,
                variant,
                solvers=FULL_SOLVER_BUNDLE,
                stages=("training", "formal_audit"),
            )
            for variant, mechanism, seed in product(
                self.mixed_variants, self.mechanisms, self.mixed_seeds
            )
        )

    def mixed_challenge_formal_audit_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "audit",
                "mixed_challenge_formal",
                4,
                6,
                mechanism,
                seed,
                f"{variant}__{audit_variant}",
                solvers=AUDIT_VARIANT_SOLVERS[audit_variant],
                stages=("formal_audit",),
            )
            for variant, mechanism, seed, audit_variant in product(
                self.mixed_variants,
                self.mechanisms,
                self.mixed_seeds,
                MIXED_FORMAL_AUDIT_VARIANTS,
            )
        )

    def cnc_main_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "training",
                "cnc_main",
                4,
                6,
                mechanism,
                seed,
                condition,
                solvers=("REPAIR-SAD-CCE",),
                stages=("training", "formal_audit", "outcome_evaluation"),
            )
            for condition, mechanism, seed in product(
                self.cnc_conditions, self.mechanisms, self.cnc_seeds
            )
        )

    def cnc_formal_audit_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "audit",
                "cnc_formal",
                4,
                6,
                mechanism,
                seed,
                condition,
                solvers=("REPAIR-SAD-CCE",),
                stages=("formal_audit",),
            )
            for condition, mechanism, seed in product(
                self.cnc_conditions, self.mechanisms, self.cnc_seeds
            )
        )

    def cnc_outcome_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "evaluation",
                "cnc_outcomes",
                4,
                6,
                mechanism,
                seed,
                condition,
                solvers=("REPAIR-SAD-CCE",),
                stages=("outcome_evaluation",),
            )
            for condition, mechanism, seed in product(
                self.cnc_conditions, self.mechanisms, self.cnc_seeds
            )
        )

    def transplant_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "audit",
                "policy_transplant",
                4,
                6,
                target,
                seed,
                condition,
                source_mechanism=source,
                target_mechanism=target,
                solvers=("REPAIR-SAD-CCE",),
                stages=("transplant_audit",),
                evaluation_arms=TRANSPLANT_ARMS,
            )
            for condition, (source, target), seed in product(
                self.cnc_conditions, self.transplant_directions, self.cnc_seeds
            )
        )

    @staticmethod
    def _library_size(variant: str) -> int:
        try:
            return LIBRARY_VARIANT_POLICY_COUNTS[variant]
        except KeyError as exc:
            raise ValueError(f"Unknown policy-library variant: {variant}") from exc

    def policy_library_sensitivity_jobs(self, phase: str = "training") -> tuple[RevisionJobKey, ...]:
        if phase not in {"training", "audit", "evaluation"}:
            raise ValueError("Policy-library jobs support training, audit, or evaluation phases.")
        family = {
            "training": "policy_library_sensitivity",
            "audit": "policy_library_sensitivity_formal",
            "evaluation": "policy_library_sensitivity_outcomes",
        }[phase]
        stages = {
            "training": ("training", "formal_audit", "outcome_evaluation"),
            "audit": ("formal_audit",),
            "evaluation": ("outcome_evaluation",),
        }[phase]
        return tuple(
            RevisionJobKey(
                phase,
                family,
                4,
                self._library_size(library_variant),
                mechanism,
                seed,
                f"{condition}__{library_variant}",
                solvers=("REPAIR-SAD-CCE",),
                stages=stages,
            )
            for condition, library_variant, mechanism, seed in product(
                self.representative_cnc_conditions,
                self.library_variants,
                self.mechanisms,
                self.sensitivity_seeds,
            )
        )

    def parameter_robustness_jobs(self, phase: str = "training") -> tuple[RevisionJobKey, ...]:
        if phase not in {"training", "audit", "evaluation"}:
            raise ValueError("Parameter-robustness jobs support training, audit, or evaluation phases.")
        family = {
            "training": "parameter_robustness",
            "audit": "parameter_robustness_formal",
            "evaluation": "parameter_robustness_outcomes",
        }[phase]
        stages = {
            "training": ("training", "formal_audit", "outcome_evaluation"),
            "audit": ("formal_audit",),
            "evaluation": ("outcome_evaluation",),
        }[phase]
        return tuple(
            RevisionJobKey(
                phase,
                family,
                4,
                6,
                mechanism,
                seed,
                f"{condition}__{setting.setting_id}",
                solvers=("REPAIR-SAD-CCE",),
                stages=stages,
            )
            for condition, setting, mechanism, seed in product(
                self.representative_cnc_conditions,
                self.robustness_settings,
                self.mechanisms,
                self.robustness_seeds,
            )
        )

    def exact_scalability_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "training",
                "scalability_exact",
                n,
                j,
                mechanism,
                seed,
                "exact_full_tensor",
                solvers=FULL_SOLVER_BUNDLE,
                stages=("training", "formal_audit"),
            )
            for (n, j), mechanism, seed in product(
                self.scalability_exact_sizes,
                self.scalability_exact_mechanisms,
                self.scalability_seeds,
            )
        )

    def sparse_scalability_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "training",
                "scalability_sparse",
                n,
                j,
                mechanism,
                seed,
                "sparse",
                solvers=SPARSE_SOLVER_BUNDLE,
                stages=("training", "formal_audit"),
            )
            for (n, j), mechanism, seed in product(
                self.scalability_sparse_sizes,
                self.scalability_sparse_mechanisms,
                self.scalability_seeds,
            )
        )

    def scalability_jobs(self) -> tuple[RevisionJobKey, ...]:
        return self.exact_scalability_jobs() + self.sparse_scalability_jobs()

    def exact_scalability_formal_audit_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "audit",
                "scalability_exact_formal",
                n,
                j,
                mechanism,
                seed,
                audit_variant,
                solvers=AUDIT_VARIANT_SOLVERS[audit_variant],
                stages=("formal_audit",),
            )
            for (n, j), mechanism, seed, audit_variant in product(
                self.scalability_exact_sizes,
                self.scalability_exact_mechanisms,
                self.scalability_seeds,
                SCALABILITY_EXACT_AUDIT_VARIANTS,
            )
        )

    def exact_scalability_runtime_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "runtime",
                "scalability_exact_runtime",
                n,
                j,
                mechanism,
                seed,
                "empty_cache_wall_time",
                solvers=FULL_SOLVER_BUNDLE,
                stages=("runtime_measurement",),
                resource_mode="exclusive_runtime",
            )
            for (n, j), mechanism, seed in product(
                self.scalability_exact_sizes,
                self.scalability_exact_mechanisms,
                self.scalability_seeds,
            )
        )

    def sparse_scalability_formal_audit_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "audit",
                "scalability_sparse_formal",
                n,
                j,
                mechanism,
                seed,
                audit_variant,
                solvers=AUDIT_VARIANT_SOLVERS[audit_variant],
                stages=("formal_audit",),
            )
            for (n, j), mechanism, seed, audit_variant in product(
                self.scalability_sparse_sizes,
                self.scalability_sparse_mechanisms,
                self.scalability_seeds,
                SCALABILITY_SPARSE_AUDIT_VARIANTS,
            )
        )

    def sparse_scalability_runtime_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "runtime",
                "scalability_sparse_runtime",
                n,
                j,
                mechanism,
                seed,
                "empty_cache_wall_time",
                solvers=SPARSE_SOLVER_BUNDLE,
                stages=("runtime_measurement",),
                resource_mode="exclusive_runtime",
            )
            for (n, j), mechanism, seed in product(
                self.scalability_sparse_sizes,
                self.scalability_sparse_mechanisms,
                self.scalability_seeds,
            )
        )

    def scalability_formal_audit_jobs(self) -> tuple[RevisionJobKey, ...]:
        return (
            self.exact_scalability_formal_audit_jobs()
            + self.sparse_scalability_formal_audit_jobs()
        )

    def mwu_tuning_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "training",
                "mwu_global_tuning",
                n,
                j,
                mechanism,
                seed,
                tuning.config_id,
                solvers=("MWU-PolicyTrace",),
                stages=("training", "mwu_tuning_audit"),
                tuning_config_ids=(tuning.config_id,),
            )
            for (n, j), mechanism, seed, tuning in product(
                self.mwu_tuning_sizes,
                self.mechanisms,
                self.mwu_tuning_seeds,
                self.mwu_tuning_configs,
            )
        )

    def mwu_tuning_dss_reference_jobs(self) -> tuple[RevisionJobKey, ...]:
        """One empty-cache DSS runtime reference for each tuning game.

        All sixteen MWU configurations in a game use this single measured
        budget.  The reference is not multiplied by the tuning-grid size.
        """

        return tuple(
            RevisionJobKey(
                "training",
                "mwu_tuning_dss_reference",
                n,
                j,
                mechanism,
                seed,
                "empty_cache_matched_runtime",
                solvers=("REPAIR-SAD-CCE",),
                stages=("training",),
            )
            for (n, j), mechanism, seed in product(
                self.mwu_tuning_sizes,
                self.mechanisms,
                self.mwu_tuning_seeds,
            )
        )

    def mwu_tuning_pipeline_jobs(self) -> tuple[RevisionJobKey, ...]:
        """The 24 physical tuning pipelines used by the campaign.

        A pipeline measures DSS once from an empty payoff cache, runs all
        sixteen prespecified MWU configurations under that measured budget,
        and audits every returned distribution with the shared R=500
        calibration stream.  The 384 configuration evaluations remain
        available through :meth:`mwu_tuning_jobs` as conceptual units.
        """

        config_ids = tuple(config.config_id for config in self.mwu_tuning_configs)
        return tuple(
            RevisionJobKey(
                "training",
                "mwu_tuning_pipeline",
                n,
                j,
                mechanism,
                seed,
                "dss_plus_16_mwu_configs",
                solvers=("REPAIR-SAD-CCE", "MWU-PolicyTrace"),
                stages=("runtime_measurement", "training", "mwu_tuning_audit"),
                tuning_config_ids=config_ids,
                resource_mode="exclusive_runtime",
            )
            for (n, j), mechanism, seed in product(
                self.mwu_tuning_sizes,
                self.mechanisms,
                self.mwu_tuning_seeds,
            )
        )

    def selection_sensitivity_jobs(self) -> tuple[RevisionJobKey, ...]:
        """Forty shared-table pipelines, each returning three selected CCEs."""

        return tuple(
            RevisionJobKey(
                "training",
                "selection_sensitivity",
                4,
                6,
                mechanism,
                seed,
                condition,
                solvers=("FullTensor-CCE-LP",),
                selectors=self.selection_sensitivity_selectors,
                stages=("training", "formal_audit", "outcome_evaluation"),
            )
            for condition, mechanism, seed in product(
                self.representative_cnc_conditions,
                self.mechanisms,
                self.sensitivity_seeds,
            )
        )

    def selection_sensitivity_formal_audit_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "audit",
                "selection_sensitivity_formal",
                4,
                6,
                mechanism,
                seed,
                f"{condition}__{selector}",
                solvers=("FullTensor-CCE-LP",),
                selectors=(selector,),
                stages=("formal_audit",),
            )
            for condition, mechanism, seed, selector in product(
                self.representative_cnc_conditions,
                self.mechanisms,
                self.sensitivity_seeds,
                self.selection_sensitivity_selectors,
            )
        )

    def selection_sensitivity_outcome_jobs(self) -> tuple[RevisionJobKey, ...]:
        return tuple(
            RevisionJobKey(
                "evaluation",
                "selection_sensitivity_outcomes",
                4,
                6,
                mechanism,
                seed,
                f"{condition}__{selector}",
                solvers=("FullTensor-CCE-LP",),
                selectors=(selector,),
                stages=("outcome_evaluation",),
            )
            for condition, mechanism, seed, selector in product(
                self.representative_cnc_conditions,
                self.mechanisms,
                self.sensitivity_seeds,
                self.selection_sensitivity_selectors,
            )
        )

    def all_compute_jobs(self) -> tuple[RevisionJobKey, ...]:
        """Every conceptual stage/configuration unit in the frozen design.

        This deliberately expands selector audits, outcome evaluations and
        MWU configurations.  It is a design-accounting view, not the list
        sent to workers; :meth:`physical_jobs` groups shared simulation work.
        """

        return (
            self.mwu_tuning_jobs()
            + self.mwu_tuning_dss_reference_jobs()
            + self.solver_jobs()
            + self.solver_formal_audit_jobs()
            + self.solver_runtime_jobs()
            + self.mixed_challenge_jobs()
            + self.mixed_challenge_formal_audit_jobs()
            + self.cnc_main_jobs()
            + self.cnc_formal_audit_jobs()
            + self.cnc_outcome_jobs()
            + self.selection_sensitivity_jobs()
            + self.selection_sensitivity_formal_audit_jobs()
            + self.selection_sensitivity_outcome_jobs()
            + self.transplant_jobs()
            + self.policy_library_sensitivity_jobs("training")
            + self.policy_library_sensitivity_jobs("audit")
            + self.policy_library_sensitivity_jobs("evaluation")
            + self.parameter_robustness_jobs("training")
            + self.parameter_robustness_jobs("audit")
            + self.parameter_robustness_jobs("evaluation")
            + self.scalability_jobs()
            + self.scalability_formal_audit_jobs()
            + self.exact_scalability_runtime_jobs()
            + self.sparse_scalability_runtime_jobs()
        )

    def physical_jobs(self) -> tuple[RevisionJobKey, ...]:
        """Worker-level pipelines after grouping shared tables and stages."""

        jobs = (
            self.mwu_tuning_pipeline_jobs()
            + self.solver_jobs()
            + self.mixed_challenge_jobs()
            + self.cnc_main_jobs()
            + self.selection_sensitivity_jobs()
            + self.transplant_jobs()
            + self.policy_library_sensitivity_jobs("training")
            + self.parameter_robustness_jobs("training")
            + self.exact_scalability_jobs()
            + self.sparse_scalability_jobs()
            + self.solver_runtime_jobs()
            + self.exact_scalability_runtime_jobs()
            + self.sparse_scalability_runtime_jobs()
        )
        if len({job.job_id for job in jobs}) != len(jobs):
            raise RuntimeError("The physical revision campaign contains duplicate job IDs.")
        return jobs

    def expected_design_counts(self) -> dict[str, int]:
        """Frozen experiment-combination counts stated in the revision plan."""

        return {
            "mwu_calibration_games": 24,
            "mwu_configuration_evaluations": 384,
            "mwu_dss_runtime_references": 24,
            "solver_benchmark_games": 72,
            "exact_scalability_games": 12,
            "sparse_scalability_games": 18,
            "mixed_challenge_games": 100,
            "cnc_main_games": 320,
            "selection_sensitivity_base_games": 40,
            "selection_sensitivity_selector_outputs": 120,
            "policy_transplant_directions": 320,
            "policy_library_combinations": 400,
            "parameter_robustness_combinations": 360,
            "gcp_exclusive_runtime_measurements": 102,
        }

    def physical_family_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for job in self.physical_jobs():
            counts[job.family] = counts.get(job.family, 0) + 1
        return dict(sorted(counts.items()))

    def expected_family_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for job in self.all_compute_jobs():
            counts[job.family] = counts.get(job.family, 0) + 1
        return dict(sorted(counts.items()))


def default_revision_matrix() -> RevisionFullV1Matrix:
    return RevisionFullV1Matrix()
