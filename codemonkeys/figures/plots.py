from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import numpy as np
import pydra
from pathlib import Path
from collections import defaultdict
from typing import Callable
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import tqdm
import unidiff
import plotly.graph_objects as go

from codemonkeys.stages.context.relevance import _should_include_file
from codemonkeys.utils.concurrency import parallelize
from codemonkeys.utils.utils import pass_at_k
from codemonkeys.trajectory_data.correctness import (
    model_selection_machine_correct,
    selection_machine_correct,
    ensemble_machine_correct,
)
from codemonkeys.figures.constants import (
    LEADERBOARD_NAMES,
    NUM_INSTANCES,
    COLORS,
    TITLE_SIZE,
    LABEL_SIZE,
    TICK_SIZE,
    METHOD_TO_DISPLAY_NAME,
    LEGEND_LINEWIDTH,
)
from codemonkeys.utils.count_tokens import count_tokens
from codemonkeys.utils.leaderboard_solutions import get_leaderboard_solutions
from codemonkeys.utils.shared_config import CodeMonkeysConfig
from codemonkeys.utils.utils import mean
from codemonkeys.utils.majority_voting import (
    majority_voting_score,
    expected_majority_voting_score_per_attempt,
)
from codemonkeys.trajectory_data.store import TrajectoryStore
from codemonkeys.trajectory_data.structs import (
    GeneratedTestExecutionIntermediateEdits,
)


class PlotConfig(CodeMonkeysConfig):
    save_path: Path
    figure_name: str

    def __init__(self):
        super().__init__()
        self.save_path = pydra.REQUIRED  # type: ignore
        self.figure_name = pydra.REQUIRED  # type: ignore
        self.deepseek_trajectory_store_dir: str | None = None
        self.no_majority_voting_filtering_trajectory_store_dir: str | None = None
        self.num_workers: int = 64

        self.num_samples: int = 10
        self.num_attempts: int = 8
        self.num_trials: int = 100

    def finalize(self):
        super().finalize()

        self.save_path = Path(self.save_path)  # type: ignore
        self.trajectory_store_dir = Path(self.trajectory_store_dir)

        if self.deepseek_trajectory_store_dir is not None:
            self.deepseek_trajectory_store_dir = Path(self.deepseek_trajectory_store_dir)  # type: ignore

        if self.no_majority_voting_filtering_trajectory_store_dir is not None:
            self.no_majority_voting_filtering_trajectory_store_dir = Path(self.no_majority_voting_filtering_trajectory_store_dir)  # type: ignore

        assert isinstance(self.figure_name, str)


def get_leaderboard_scores():
    solutions = get_leaderboard_solutions(LEADERBOARD_NAMES)
    name_to_num_correct = defaultdict(int)
    for instance_results in solutions.values():
        for result in instance_results:
            if result.is_correct:
                name_to_num_correct[result.name] += 1
    return {k: v / NUM_INSTANCES for k, v in name_to_num_correct.items()}


def aggregate_metric_over_problems(
    config: CodeMonkeysConfig,
    metric_fn: Callable[[TrajectoryStore], float],
):
    problems = config.get_problems()
    metrics = [metric_fn(config.get_trajectory_store(problem)) for problem in problems]

    return mean(metrics)


def get_pass_at_1(config: PlotConfig):
    score = 0.0
    for problem in config.get_problems():
        trajectory_store = config.get_trajectory_store(problem)

        patch_to_correct = {
            patch: result.resolved
            for (
                patch,
                result,
            ) in trajectory_store.load_sample_evaluation().patch_to_results.items()
        }

        num_correct_samples = 0
        for sample_index in range(config.num_samples):
            editing_sm_data = (
                trajectory_store.editing_state_machine_data_store.load_state_data(
                    sample_index
                )
            )
            if not editing_sm_data.patches:
                continue

            patch = editing_sm_data.patches[-1]
            if not patch or len(patch.split("\n")) == 1:
                continue

            if patch_to_correct[patch]:
                num_correct_samples += 1
        score += num_correct_samples / config.num_samples

    return score / len(config.get_problems())


def get_coverage_for_problem(trajectory_store: TrajectoryStore):
    test_execution = trajectory_store.load_sample_evaluation()
    if len(test_execution.patch_to_results) == 0:
        return False
    return any([output.resolved for output in test_execution.patch_to_results.values()])


def get_recall_for_problem(trajectory_store: TrajectoryStore):
    context = trajectory_store.load_context(max_tokens=128_000)
    retrieved_context_file_paths = [file.path for file in context]
    gold_context_file_paths = [
        file.path for file in unidiff.PatchSet(trajectory_store.problem.gold_patch)
    ]
    if len(set(gold_context_file_paths) - set(retrieved_context_file_paths)) == 0:
        return 1
    else:
        return 0


def get_model_selection_score_no_majority_voting_filtering(config: PlotConfig):
    no_majority_voting_filtering_trajectory_stores = get_trajectory_stores(
        config, traj_store_dir="no_majority_voting_filtering"
    )
    model_selection_scores_no_majority_voting_filtering = []
    for traj_store in no_majority_voting_filtering_trajectory_stores:
        model_selection_scores_no_majority_voting_filtering.append(
            model_selection_machine_correct(traj_store)
        )

    return mean(model_selection_scores_no_majority_voting_filtering)


get_ensemble_score = partial(
    aggregate_metric_over_problems, metric_fn=ensemble_machine_correct
)
get_codemonkeys_score = partial(
    aggregate_metric_over_problems, metric_fn=selection_machine_correct
)
get_model_selection_score = partial(
    aggregate_metric_over_problems, metric_fn=model_selection_machine_correct
)
get_majority_voting_score = partial(
    aggregate_metric_over_problems, metric_fn=majority_voting_score
)
get_coverage = partial(
    aggregate_metric_over_problems, metric_fn=get_coverage_for_problem
)
get_recall = partial(
    aggregate_metric_over_problems, metric_fn=get_recall_for_problem
)


def get_trajectory_stores(config: PlotConfig, traj_store_dir: bool | str = "main"):
    problems = config.get_problems()
    if traj_store_dir == "deepseek":
        data_dir = config.deepseek_trajectory_store_dir
        assert isinstance(data_dir, Path)
    elif traj_store_dir == "no_majority_voting_filtering":
        data_dir = config.no_majority_voting_filtering_trajectory_store_dir
        assert isinstance(data_dir, Path)
    else:
        data_dir = config.trajectory_store_dir

    return [TrajectoryStore(problem=problem, data_dir=data_dir) for problem in problems]


def get_test_execution_results_with_intermediate_edits(
    config: PlotConfig,
) -> list[GeneratedTestExecutionIntermediateEdits]:
    test_executions = [
        trajectory_store.load_generated_test_execution_intermediate_edits()
        for trajectory_store in get_trajectory_stores(config)
    ]
    return test_executions


def final_scores(config: PlotConfig):
    leaderboard_scores = get_leaderboard_scores()
    codemonkeys_score = get_codemonkeys_score(config)
    ensemble_score = get_ensemble_score(config)
    scores = {
        **leaderboard_scores,
        "codemonkeys": codemonkeys_score,
        "barrel_of_monkeys": ensemble_score,
    }

    sorted_items = sorted(scores.items(), key=lambda x: x[1])

    methods = [METHOD_TO_DISPLAY_NAME[item[0]] for item in sorted_items]
    values = [item[1] * 100 for item in sorted_items]  # Convert to percentages

    colors = [
        COLORS[0] if method in ["CodeMonkeys", "Barrel of Monkeys"] else COLORS[-1]
        for method in methods
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, values, color=colors)

    plt.ylabel("% Resolved", fontsize=LABEL_SIZE, weight="bold")

    plt.ylim(40, 70)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.xticks(fontsize=LABEL_SIZE)
    plt.yticks(fontsize=LABEL_SIZE)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.savefig(config.save_path / f"{config.figure_name}.pdf")
    plt.savefig(config.save_path / f"{config.figure_name}.png")


def context_recall(config: PlotConfig):
    problems = config.get_problems()
    trajectory_stores = [config.get_trajectory_store(problem) for problem in problems]
    
    # Get the number of tokens that are in the files
    # that are edited by the gold patch.
    num_tokens_in_gold_context_files = []
    for problem in problems:
        gold_context_file_paths = [
            file.path for file in unidiff.PatchSet(problem.gold_patch) if not file.is_added_file and file.path.endswith(".py")
        ]
        tokens_in_curr_gold_context_file = 0
        for gold_context_file_path in gold_context_file_paths:
            file_content = problem.get_file(gold_context_file_path).content
            tokens_in_curr_gold_context_file += count_tokens(file_content)
        num_tokens_in_gold_context_files.append(tokens_in_curr_gold_context_file)

    # Calculate our recall and the oracle recall.
    max_tokens_list = [1000 * i for i in range(200)]
    max_tokens_to_context_recall = {}
    max_tokens_to_gold_context_recall = {}
    for num_tokens in max_tokens_list:
        curr_context_recall = []
        for trajectory_store, problem in zip(trajectory_stores, problems):
            context = trajectory_store.load_context(max_tokens=num_tokens)
            retrieved_context_file_paths = [file.path for file in context]
            gold_context_file_paths = [
                file.path for file in unidiff.PatchSet(problem.gold_patch)
            ]
            if (
                len(set(gold_context_file_paths) - set(retrieved_context_file_paths))
                == 0
            ):
                curr_context_recall.append(1)
            else:
                curr_context_recall.append(0)
        max_tokens_to_context_recall[num_tokens] = mean(curr_context_recall)
        max_tokens_to_gold_context_recall[num_tokens] = mean([
            instance_num_tokens_in_gold_context_files <= num_tokens for instance_num_tokens_in_gold_context_files in num_tokens_in_gold_context_files
        ])

    # Get tokens in context and total tokens for each problem.
    @dataclass
    class TokenCountInfo:
        retrieved: int 
        total: int

    def get_token_count_info_for_problem(problem):
        trajectory_store = config.get_trajectory_store(problem)
        context = trajectory_store.load_context(max_tokens=128_000)

        total_tokens = 0
        for file_path in problem.all_file_paths():
            if _should_include_file(file_path):
                file = problem.get_file(file_path)
                total_tokens += count_tokens(file.content)

        return TokenCountInfo(
            retrieved=sum([count_tokens(file.content) for file in context]),
            total=total_tokens
        )


    problems = config.get_problems()
    token_counts = parallelize(
        get_token_count_info_for_problem, 
        problems,
        num_workers=config.num_workers
    )

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot - Context Recall vs Num Tokens
    ax1.plot(
        [x / 1000 for x in max_tokens_to_context_recall.keys()],
        [100 * v for v in max_tokens_to_context_recall.values()],
        label="Our Context Retrieval"
    )
    ax1.plot(
        [x / 1000 for x in max_tokens_to_gold_context_recall.keys()],
        [100 * v for v in max_tokens_to_gold_context_recall.values()],
        label="Oracle Context Retrieval"
    )
    ax1.legend(fontsize=TICK_SIZE, frameon=False)
    ax1.set_title("Recall vs. Max Context Length", fontsize=TITLE_SIZE, weight="bold")
    ax1.set_xlabel(
        "Max Context Length (thousands of tokens)", fontsize=LABEL_SIZE, weight="bold"
    )
    ax1.set_ylabel("Recall (% of Instances)", fontsize=LABEL_SIZE, weight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(labelsize=TICK_SIZE)
    ax1.set_ylim(0, 100)

    # Right plot - distribution of compression factors
    compression_factors = [token_count.total / token_count.retrieved for token_count in token_counts]
    ax2.bar(range(len(compression_factors)), sorted(compression_factors), width=1.0)
    ax2.set_yscale("log")
    ax2.set_title(
        "Distribution of Compression Factors",
        fontsize=TITLE_SIZE,
        weight="bold",
    )
    ax2.set_xlabel(
        "Problem Index\n(Sorted by Compression Factor)", fontsize=LABEL_SIZE, weight="bold"
    )
    # ax2.set_xscale("log")
    ax2.set_ylabel("Codebase Length / Context Length", fontsize=LABEL_SIZE, weight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=TICK_SIZE)

    plt.tight_layout()
    plt.savefig(config.save_path / f"{config.figure_name}.pdf")
    plt.savefig(config.save_path / f"{config.figure_name}.png")


@dataclass
class MajorityVotingSweepResults:
    # indexed by [num_samples - 1][attempt_idx]
    expected_cost_by_samples_by_attempt: list[list[float]] = field(default_factory=list)
    expected_score_by_samples_by_attempt: list[list[float]] = field(
        default_factory=list
    )
    expected_coverage_by_samples_by_attempt: list[list[float]] = field(
        default_factory=list
    )

    def score_cost_and_coverage_for_num_samples(
        self, num_samples
    ) -> tuple[list[float], list[float], list[float]]:
        """Short form for (
            self.expected_score_by_samples_by_attempt[num_samples - 1],
            self.expected_cost_by_samples_by_attempt[num_samples - 1],
            self.expected_coverage_by_samples_by_attempt[num_samples - 1],
        )"""
        return (
            self.expected_score_by_samples_by_attempt[num_samples - 1],
            self.expected_cost_by_samples_by_attempt[num_samples - 1],
            self.expected_coverage_by_samples_by_attempt[num_samples - 1],
        )


def _do_majority_voting_scores_sweep_across_samples_and_attempts(
    config: PlotConfig,
    trajectory_stores: list[TrajectoryStore],
) -> MajorityVotingSweepResults:
    def _expected_attempt_score_cost_and_coverage(
        num_samples: int,
        attempt_idx: int,
    ):
        expected_cost = 0.0
        expected_coverage = 0.0
        expected_score = 0.0

        for trajectory_store in trajectory_stores:
            # costs and patch info if we limited the number of attempts to attempt_idx + 1
            # load_generated_test_execution_intermediate_edits stores the patches/tests/costs
            # of limiting the number of attempts of the state machine.
            attempt_limited_data = trajectory_store.load_generated_test_execution_intermediate_edits().generated_test_executions[
                attempt_idx
            ]

            expected_cost += mean(attempt_limited_data.costs) * num_samples

            # Compute expected coverage
            per_sample_test_results = (
                trajectory_store.load_sample_evaluation_of_intermediate_edits().patch_to_results
            )

            num_correct_samples = sum(
                [
                    per_sample_test_results[patch.patch].resolved
                    for patch in attempt_limited_data.patch_data
                ]
            )

            expected_coverage += pass_at_k(
                n=config.num_samples,
                k=num_samples,
                c=num_correct_samples,
            )

            # Compute expected scores by running many trials
            scores = []

            for _ in range(config.num_trials):
                sample_indexes = random.sample(range(config.num_samples), num_samples)

                score_this_trial = expected_majority_voting_score_per_attempt(
                    trajectory_store,
                    attempt_idx,
                    set(sample_indexes),
                )
                scores.append(score_this_trial)
            expected_score += mean(scores)

        return (
            expected_score / len(trajectory_stores),
            expected_cost,
            expected_coverage / len(trajectory_stores),
        )

    # Load all the trajectory data from disk in parallel
    def warm_caches(trajectory_store: TrajectoryStore):
        trajectory_store.load_generated_test_execution_intermediate_edits()
        trajectory_store.load_sample_evaluation_of_intermediate_edits()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(warm_caches, t) for t in trajectory_stores]
        [future.result() for future in tqdm.tqdm(futures, desc="Loading data")]

    # Actually sweep over results
    results = MajorityVotingSweepResults()

    for num_samples in tqdm.tqdm(range(1, config.num_samples + 1)):
        scores, costs, coverage = [], [], []
        for attempt_idx in range(config.num_attempts):
            expected_score, expected_cost, expected_coverage = (
                _expected_attempt_score_cost_and_coverage(num_samples, attempt_idx)
            )

            costs.append(expected_cost)
            scores.append(expected_score)
            coverage.append(expected_coverage)

        results.expected_cost_by_samples_by_attempt.append(costs)
        results.expected_score_by_samples_by_attempt.append(scores)
        results.expected_coverage_by_samples_by_attempt.append(coverage)

    return results


def compute_scaling(config: PlotConfig):
    sweep_results = _do_majority_voting_scores_sweep_across_samples_and_attempts(
        config, get_trajectory_stores(config)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, config.num_samples))
    lines = []
    labels = []

    for i, num_samples in enumerate(range(1, config.num_samples + 1)):
        _, costs, coverages = sweep_results.score_cost_and_coverage_for_num_samples(
            num_samples
        )
        line = ax1.plot(
            costs, coverages, color=colors[i], label=f"{num_samples} samples"
        )[0]
        ax1.scatter(costs, coverages, color=colors[i])

        scores, costs, _ = sweep_results.score_cost_and_coverage_for_num_samples(
            num_samples
        )
        ax2.plot(costs, scores, color=colors[i])
        ax2.scatter(costs, scores, color=colors[i])

        lines.append(line)
        labels.append(f"num_samples = {num_samples}")

    # Create legend above the plots
    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,
        frameon=False,
        fontsize=TICK_SIZE,
    )

    ax1.set_xlabel("Cost (USD)", fontsize=LABEL_SIZE, weight="bold")
    ax1.set_ylabel("Coverage", fontsize=LABEL_SIZE, weight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(labelsize=TICK_SIZE)
    ax1.set_xscale("log")
    ax1.set_ylim(0.2, 0.8)
    # ax1.set_title("Coverage", fontsize=TITLE_SIZE, weight='bold')

    ax2.set_xlabel("Cost (USD)", fontsize=LABEL_SIZE, weight="bold")
    ax2.set_ylabel("Majority Voting Score", fontsize=LABEL_SIZE, weight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=TICK_SIZE)
    ax2.set_xscale("log")
    ax2.set_ylim(0.2, 0.8)
    # ax2.set_title("Majority Voting Score", fontsize=TITLE_SIZE, weight='bold')

    plt.suptitle(
        "Scaling Serial and Parallel Inference Compute",
        fontsize=TITLE_SIZE,
        weight="bold",
    )
    plt.tight_layout()

    # Add padding to prevent cutoff
    plt.subplots_adjust(top=0.85)

    plt.savefig(config.save_path / f"{config.figure_name}.pdf", bbox_inches="tight")
    plt.savefig(config.save_path / f"{config.figure_name}.png", bbox_inches="tight")


def selection(config: PlotConfig):
    # Get data for left axis (selection methods)
    selection_score = get_codemonkeys_score(config)
    model_selection_score = get_model_selection_score(config)
    majority_voting_score = get_majority_voting_score(config)
    pass_at_1 = get_pass_at_1(config)
    coverage = get_coverage(config)

    methods = [
        "Random\nSelection",
        "Majority\nVoting\nwith Tests",
        "Model\nSelection\nAfter Top-3\nFiltering",
        "Selection\nState Machine\nAfter Top-3\nFiltering",
    ]
    scores = [
        pass_at_1,
        majority_voting_score,
        model_selection_score,
        selection_score,
    ]
    if config.no_majority_voting_filtering_trajectory_store_dir is not None:
        model_selection_score_no_majority_voting_filtering = (
            get_model_selection_score_no_majority_voting_filtering(config)
        )
        methods.insert(2, "Model\nSelection")
        scores.insert(2, model_selection_score_no_majority_voting_filtering)

    # Create figure with one axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Selection methods plot
    bars = ax.bar(
        range(len(methods)),
        [s * 100 for s in scores],
        width=0.6,
        color=[COLORS[i] for i in range(len(methods))]
    )

    ax.axhline(
        y=coverage * 100,
        color="black",
        linestyle="--",
        label=f"Oracle Selection = {coverage * 100:.1f}%",
    )
    ax.set_ylabel("Score", fontsize=LABEL_SIZE+3, weight="bold")
    ax.set_ylim(40, 71)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=TITLE_SIZE+3  # Increased font size for bar labels
        )

    # Legend
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(
        lines,
        labels,
        fontsize=TITLE_SIZE+3,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.95),
    )

    # Formatting
    ax.set_title("Comparing Selection Methods", fontsize=TITLE_SIZE+3, weight="bold")
    ax.set_xticks([i for i in range(len(methods))])
    ax.set_xticklabels(
        methods, fontsize=TICK_SIZE * 1.2, ha="center", weight="bold"
    )  # Made ticks 20% larger
    ax.tick_params(
        axis="both", which="major", labelsize=TICK_SIZE + 5
    )  # Increased tick size for both axes

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    assert isinstance(config.save_path, Path)
    plt.savefig(config.save_path / f"{config.figure_name}.pdf", bbox_inches="tight")
    plt.savefig(config.save_path / f"{config.figure_name}.png", bbox_inches="tight")


def deepseek_maj_voting_score_scaling(config: PlotConfig):

    deepseek_sweep_results = (
        _do_majority_voting_scores_sweep_across_samples_and_attempts(
            config,
            get_trajectory_stores(config, traj_store_dir="deepseek"),
        )
    )

    claude_sweep_results = _do_majority_voting_scores_sweep_across_samples_and_attempts(
        config,
        get_trajectory_stores(
            config,
        ),
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(
        "Scaling Serial and Parallel Inference Compute\n(100 Sample Subset)",
        fontsize=TITLE_SIZE,
        weight='bold'
    )

    sample_legend_elements = [
        Line2D([0], [0], color=COLORS[i], label=f"num_samples = {i}", linewidth=LEGEND_LINEWIDTH)
        for i in range(1, config.num_samples + 1)
    ]
    fig.legend(handles=sample_legend_elements, loc='upper center', 
            bbox_to_anchor=(0.5, 1.15), ncol=5, frameon=False, fontsize=LABEL_SIZE-2)

    # Create legend for model types
    model_legend_elements = [
        Line2D([0], [0], linestyle='-', color='gray', label='DeepSeek v3', linewidth=LEGEND_LINEWIDTH),
        Line2D([0], [0], linestyle='--', color='gray', label='Claude Sonnet 3.5', linewidth=LEGEND_LINEWIDTH)
    ]
    ax1.legend(handles=model_legend_elements, frameon=False, fontsize=LABEL_SIZE)

    for num_samples in range(1, config.num_samples + 1):
        # Plot deepseek on both axes
        scores_deepseek, costs_deepseek, coverages_deepseek = (
            deepseek_sweep_results.score_cost_and_coverage_for_num_samples(num_samples)
        )
        for ax in [ax1, ax2, ax3]:
            if ax == ax1 and num_samples not in [1, 5, 10]:
                continue

            if ax == ax3:
                results_deepseek = coverages_deepseek
            else:
                results_deepseek = scores_deepseek
            ax.scatter(costs_deepseek, results_deepseek, color=COLORS[num_samples])
            ax.plot(
                costs_deepseek,
                results_deepseek,
                color=COLORS[num_samples],
                linestyle='-'
            )
            ax.set_xscale('log')

        # Only plot claude results on the first axes, and for a fixed number of samples
        if num_samples not in [1, 5, 10]:
            continue

        scores_claude, costs_claude, _ = (
            claude_sweep_results.score_cost_and_coverage_for_num_samples(num_samples)
        )
        ax1.scatter(costs_claude, scores_claude, color=COLORS[num_samples])
        ax1.plot(
            costs_claude,
            scores_claude,
            color=COLORS[num_samples],
            linestyle='--'
        )

    ax1.set_xlabel("Cost (USD)", fontsize=LABEL_SIZE, weight='bold')
    ax1.set_ylabel("Majority Voting Score", fontsize=LABEL_SIZE, weight='bold')
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(labelsize=TICK_SIZE)

    ax2.set_xlabel("Cost (USD)", fontsize=LABEL_SIZE, weight='bold')
    ax2.set_ylabel("Majority Voting Score", fontsize=LABEL_SIZE, weight='bold')
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=TICK_SIZE)
    ax1.set_ylim(0.2,0.6)
    ax2.set_ylim(0.2,0.6)

    ax3.set_xlabel("Cost (USD)", fontsize=LABEL_SIZE, weight='bold')
    ax3.set_ylabel("Coverage", fontsize=LABEL_SIZE, weight='bold')
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.tick_params(labelsize=TICK_SIZE)
    ax3.set_ylim(0.2,0.6)

    plt.tight_layout()  # Add tight_layout to adjust spacing
    plt.savefig(config.save_path / f"{config.figure_name}.pdf", bbox_inches="tight")
    plt.savefig(config.save_path / f"{config.figure_name}.png", bbox_inches="tight")


def problem_resolution_flow(config: PlotConfig):
    recall = round(get_recall(config)*100, 2)
    coverage = round(get_coverage(config)*100, 2)
    score = round(get_codemonkeys_score(config)*100, 2)

    metrics = [
        recall, 100-recall,
        coverage, recall-coverage,
        score, coverage-score
    ]
        # Define the nodes and their labels
    labels = [
        '<b>SWE-bench<br>Verified<br>(100%)</b>',
        f'<b>Required Files<br>Retrieved<br>({metrics[0]:.2f}% = Recall)</b>',
        f'<b>Missing<br>Required Files<br>({metrics[1]:.2f}%)</b>',
        f'<b>Correct Edit<br>Generated<br>({metrics[2]:.2f}% = Coverage)</b>', 
        f'<b>Correct Edit<br>Not Generated<br>({metrics[3]:.2f}%)</b>',
        "",
        ""
    ]

    # Define the flow connections between nodes
    source = [
        0, 0,           # Problems -> (Required Files, Missing Files)
        1, 1,           # Required Files -> (Correct Edit Gen, No Correct Edit)
        3, 3            # Correct Edit Gen -> (Selected, Not Selected)
    ]

    target = [
        1, 2,           # Problems -> (Required Files, Missing Files)
        3, 4,           # Required Files -> (Correct Edit Gen, No Correct Edit)
        5, 6            # Correct Edit Gen -> (Selected, Not Selected)
    ]

    # Create the Sankey diagram with node styling
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=[
                COLORS[0], COLORS[2],
                COLORS[3], COLORS[2],
                COLORS[3], COLORS[2],
                COLORS[3]
            ],
            # Node x positions (0 to 1)
            x=[0.05,                    # Problems
            0.325, 0.325,           # Required Files, Missing Files
            0.6, 0.6,               # Correct Edit Gen, No Correct Edit
            0.9, 0.9],              # Selected, Not Selected
            # Node y positions (0 to 1) 
            y=[0.5,                    # Problems
            0.4, 0.95,              # Required Files at bottom, Missing Files at top
            0.3, 0.8,               # Correct Edit at bottom, No Correct Edit at top
            0.3, 0.7],              # Selected at bottom, Not Selected middle
        ),
        link=dict(
            source=source,
            target=target,
            value=metrics,
            color='rgba(200, 200, 200, 0.5)'  # Semi-transparent gray links
        )
    )])

    # Add stage labels at the top
    STAGE_HEIGHT = 1.15

    fig.add_annotation(
        x=0.125, y=STAGE_HEIGHT,
        text="<b>1. Context</b>",
        showarrow=False,
        font=dict(size=20)
    )

    fig.add_annotation(
        x=0.475, y=STAGE_HEIGHT,
        text="<b>2. Generation</b>",
        showarrow=False,
        font=dict(size=20)
    )

    fig.add_annotation(
        x=0.825, y=STAGE_HEIGHT,
        text="<b>3. Selection</b>",
        showarrow=False,
        font=dict(size=20)
    )

    # Add final stage labels on the right
    LAST_LABEL_X = 1.05
    fig.add_annotation(
        x=LAST_LABEL_X+0.035, y=0.8,  # Align with bottom flow
        text="<b>Correct Edit<br>Selected<br>(57.4% = Score)</b>",
        showarrow=False,
        font=dict(size=18),
        align='left'
    )

    fig.add_annotation(
        x=LAST_LABEL_X, y=0.15,  # Align with top flow
        text="<b>Correct Edit<br>Not Selected<br>(11.6%)</b>",
        showarrow=False,
        font=dict(size=18),
        align='left'
    )

    # Update layout and save plot
    fig.update_layout(
        font_size=18,
        height=375,
        width=1100,
        margin=dict(l=10, r=100, t=75, b=20)
    )

    fig.write_image(config.save_path / f"{config.figure_name}.png")
    fig.write_image(config.save_path / f"{config.figure_name}.pdf")



@pydra.main(PlotConfig)
def main(config: PlotConfig):
    config.save_path.mkdir(parents=True, exist_ok=True)

    globals()[config.figure_name](config)


if __name__ == "__main__":
    main()  # type: ignore
