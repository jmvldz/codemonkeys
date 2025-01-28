from dataclasses import dataclass
from codemonkeys.utils.shared_config import CodeMonkeysConfig

from codemonkeys.trajectory_data.structs import StateData
from codemonkeys.figures.cost import sonnet_usage_to_cost
from codemonkeys.figures.count_relevance_tokens import load_relevance_token_count
from codemonkeys.figures.estimate_ranking_cost import load_ranking_token_estimates
import pydra

from codemonkeys.utils.leaderboard_solutions import get_leaderboard_solutions
from collections import defaultdict
from codemonkeys.figures.constants import METHOD_TO_DISPLAY_NAME, NUM_INSTANCES
from codemonkeys.figures.plots import (
    get_codemonkeys_score,
    get_coverage,
    get_ensemble_score,
    get_coverage_for_problem,
)


class TableConfig(CodeMonkeysConfig):
    def __init__(self):
        super().__init__()
        self.num_samples: int = 10


@dataclass
class EnsembleData:
    name: str
    size: int
    coverage: float


@dataclass
class CostInfo:
    label: str
    qwen_cost: float = 0.0
    claude_cache_read: float = 0.0
    claude_cache_write: float = 0.0
    claude_input_tokens: float = 0.0
    claude_output: float = 0.0

    def update_cost(self, token_usage: dict):
        cost_dict = sonnet_usage_to_cost(token_usage)
        self.claude_cache_read += cost_dict["cache_read"]
        self.claude_cache_write += cost_dict["cache_write"]
        self.claude_output += cost_dict["output"]
        self.claude_input_tokens += cost_dict["input"]

    def update_from_state_machine_data(self, state_machine_data: StateData):
        for token_usage in state_machine_data.prompt_caching_usages:
            self.update_cost(token_usage)

    def row_cost(self) -> float:
        return (
            self.qwen_cost
            + self.claude_cache_read
            + self.claude_cache_write
            + self.claude_input_tokens
            + self.claude_output
        )

    def format_as_table(self, total_cost: float) -> str:
        def round_2(num):
            return round(num, 2)

        claude_input_tokens_rounded = round_2(self.claude_input_tokens)
        claude_output_rounded = round_2(self.claude_output)
        claude_cache_read_rounded = round_2(self.claude_cache_read)
        claude_cache_write_rounded = round_2(self.claude_cache_write)
        qwen_cost_rounded = round_2(self.qwen_cost)

        # round manually and don't use .row_cost(), because we want to sum with the rounded numbers.
        row_cost_rounded = round_2(
            claude_input_tokens_rounded
            + claude_output_rounded
            + claude_cache_read_rounded
            + claude_cache_write_rounded
            + qwen_cost_rounded
        )

        relative_cost = round((row_cost_rounded / total_cost) * 100, 1)

        return f"{self.label}  & {claude_input_tokens_rounded:.2f}  & {claude_output_rounded:.2f} & {claude_cache_read_rounded:.2f} & {claude_cache_write_rounded:.2f} & {qwen_cost_rounded:.2f}  & {row_cost_rounded:.2f} ({relative_cost:.1f}\\%) & \\\\"


# See appendix C for computation
TOKENS_PER_SECOND = 9_051
COST_PER_HOUR = 8.24


def table_1_cost_table(config: TableConfig):
    relevance = CostInfo(label="Relevance")
    ranking = CostInfo(label="Ranking")
    testing = CostInfo(label="Gen. tests")
    editing = CostInfo(label="Gen. edits")
    selection = CostInfo(label="Selection")

    problem_to_tokens = load_relevance_token_count(config)
    total_tokens_for_relevance = sum(problem_to_tokens.values())

    relevance_cost = (
        total_tokens_for_relevance / (TOKENS_PER_SECOND * 3600)
    ) * COST_PER_HOUR
    relevance.qwen_cost = relevance_cost

    ranking_token_estimate_by_problem = load_ranking_token_estimates(config)

    for problem in config.get_problems():
        trajectory_store = config.get_trajectory_store(problem)
        ranking_token_estimate = ranking_token_estimate_by_problem[problem.instance_id]

        ranking.update_cost(
            {
                "cache_creation_input_tokens": ranking_token_estimate.cache_write_tokens,
                "cache_read_input_tokens": ranking_token_estimate.cache_read_tokens,
                "output_tokens": ranking_token_estimate.output_tokens,
                # We always use prompt caching for the ranking, so the input token cost will be almost zero.
                # Sometimes there are 1-2 input tokens, but that can be ignored.
                "input_tokens": 0,
            }
        )

        for sample_index in range(config.num_samples):
            testing_state_data = (
                trajectory_store.testing_state_machine_data_store.load_state_data(
                    sample_index
                )
            )
            testing.update_from_state_machine_data(testing_state_data)

            editing_state_data = (
                trajectory_store.editing_state_machine_data_store.load_state_data(
                    sample_index
                )
            )
            editing.update_from_state_machine_data(editing_state_data)

        selection_state_data = (
            trajectory_store.selection_state_machine_data_store.load_state_data(0)
        )
        selection.update_from_state_machine_data(selection_state_data)

    print(
        """
\\begin{tabular}{lccccccr}
\\toprule
 &
 \\multicolumn{4}{c}{\\textbf{Claude Sonnet-3.5 API Costs}} &
 \\multicolumn{1}{c}{\\textbf{Local Costs}} &
 \\multicolumn{1}{c}{\\textbf{Total Cost}} 
 \\\\
\\cmidrule(lr){6-6} \\cmidrule(lr){2-5} \\cmidrule(lr){7-7}
\\textbf{Stage}  &
\\textbf{Input} &
\\textbf{Output} &
\\multicolumn{2}{c}{\\textbf{Input Cache}} &
\\textbf{Qwen-2.5} &
\\textbf{USD (\\%)} \\\\
\\cmidrule(lr){4-5}
&
&
&
\\textbf{Read} &
\\textbf{Write} &
&  \\\\
\\toprule"""
    )

    stages = [relevance, ranking, testing, editing, selection]
    total = CostInfo("\\textbf{Total}")

    for cost in stages:
        total.claude_cache_read += cost.claude_cache_read
        total.claude_cache_write += cost.claude_cache_write
        total.claude_input_tokens += cost.claude_input_tokens
        total.claude_output += cost.claude_output
        total.qwen_cost += cost.qwen_cost

    total_cost = total.row_cost()

    print(f"{relevance.format_as_table(total_cost)}")
    print(f"{ranking.format_as_table(total_cost)}")
    print("\\midrule")
    print(f"{testing.format_as_table(total_cost)}")
    print(f"{editing.format_as_table(total_cost)}")
    print("\\midrule")
    print(f"{selection.format_as_table(total_cost)}")

    print("\\midrule")
    print(total.format_as_table(total_cost))

    print(
        """\\bottomrule
\\end{tabular}"""
    )


LEADERBOARD_SUBMISSIONS = [
    "blackbox",
    "interact",
    "gru",
    "amazon",
    "devlo",
    "codestory",
    "open_hands",
]


def table_2_final_scores(config: TableConfig):
    solutions = get_leaderboard_solutions(LEADERBOARD_SUBMISSIONS)

    method_to_num_correct = defaultdict(int)
    for instance_results in solutions.values():
        for result in instance_results:
            if result.is_correct:
                method_to_num_correct[METHOD_TO_DISPLAY_NAME[result.name]] += 1

    def score_to_num_correct(score) -> int:
        num_correct_float = score * NUM_INSTANCES
        num_correct = int(num_correct_float)
        assert num_correct == num_correct_float
        return num_correct

    # O3's 71.7% of 500 is not an integer.
    method_to_num_correct[" o3"] = 0.717 * 500  # type: ignore
    method_to_num_correct["\\textbf{CodeMonkeys (Oracle Selection)}"] = (
        score_to_num_correct(get_coverage(config))
    )

    instances_with_coverage_ensemble = {
        problem.instance_id
        for problem in config.get_problems()
        if get_coverage_for_problem(config.get_trajectory_store(problem))
        or any(solution.is_correct for solution in solutions[problem.instance_id])
    }
    method_to_num_correct["\\textbf{Barrel of Monkeys (Oracle Selection)}"] = (
        score_to_num_correct(
            len(instances_with_coverage_ensemble) / len(config.get_problems())
        )
    )

    method_to_num_correct["\\textbf{CodeMonkeys}"] = score_to_num_correct(
        get_codemonkeys_score(config)
    )
    method_to_num_correct["\\textbf{Barrel of Monkeys}"] = score_to_num_correct(
        get_ensemble_score(config)
    )

    sorted_methods = sorted(
        [(score, method) for (method, score) in method_to_num_correct.items()],
        reverse=True,
    )

    latex_table = """\\begin{tabular}{@{}lr@{}}
    \\toprule
    \\textbf{Method} & \\textbf{\\% Resolved} \\\\
    \\midrule"""

    for num_correct, method in sorted_methods:
        score = num_correct / NUM_INSTANCES * 100
        latex_table += f"{method} & {score:.1f} \\\\\n"

    latex_table += """\\bottomrule\\end{tabular}"""

    print(latex_table)


@pydra.main(TableConfig)
def main(config: TableConfig):
    print("***Table 1***")
    table_1_cost_table(config)

    print("***Table 2***")
    table_2_final_scores(config)


if __name__ == "__main__":
    main()
