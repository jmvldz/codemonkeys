"""Estimates the cost of using claude to do context ranking.

When we ran context ranking, we did not store the token usage data from Claude.
We use the claude tokenization api to reconstruct it here.
"""

import tqdm

from codemonkeys.utils.shared_config import (
    CodeMonkeysConfig,
    CodeMonkeysConfigWithLLM,
)
from codemonkeys.trajectory_data.structs import (
    ProblemContext,
)
from codemonkeys.trajectory_data.store import _load_from_file
import pydra
from codemonkeys.stages.context.ranking import get_relevant_files, user_prompt
from codemonkeys.swe_bench.swe_bench_verified import SWEBenchProblem
from pathlib import Path
from pydantic import BaseModel


class CostEstimate(BaseModel):
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int


def count_tokens(client, messages) -> int:
    return client.messages.count_tokens(
        model="claude-3-5-sonnet-20241022", messages=messages
    ).input_tokens


def get_estimate_for_problem(
    problem: SWEBenchProblem, config: CodeMonkeysConfigWithLLM
) -> CostEstimate:
    trajectory_store = config.get_trajectory_store(problem)

    problem_context = _load_from_file(
        ProblemContext,
        trajectory_store._ranking_data_file_path(),
    )

    relevant_files = get_relevant_files(trajectory_store, problem)
    prompt = user_prompt(problem.problem_statement, relevant_files)

    prompt_tokens = count_tokens(
        config.client,
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    total_tokens = [
        count_tokens(
            config.client,
            [
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": output.message,
                },
            ],
        )
        for output in problem_context.model_rankings
    ]
    assert len(total_tokens) == 3

    return CostEstimate(
        cache_read_tokens=prompt_tokens * 2,
        cache_write_tokens=prompt_tokens,
        output_tokens=sum(total_tokens) - 3 * prompt_tokens,
    )


class CostEstimateByProblem(BaseModel):
    problem_to_cost_estimate: dict[str, CostEstimate]


def cost_estimate_path(config: CodeMonkeysConfigWithLLM) -> Path:
    return Path(config.trajectory_store_dir / "ranking_cost_estimate.json")


def get_cost_estimate(config: CodeMonkeysConfigWithLLM):
    problem_to_cost_estimate = {}

    for problem in tqdm.tqdm(config.get_problems(), desc="Estimating cost"):
        problem_to_cost_estimate[problem.instance_id] = get_estimate_for_problem(
            problem,
            config,
        )

    with open(Path(cost_estimate_path(config)), "w") as f:
        f.write(
            CostEstimateByProblem(
                problem_to_cost_estimate=problem_to_cost_estimate
            ).model_dump_json(indent=2)
        )


def load_ranking_token_estimates(config: CodeMonkeysConfig) -> dict[str, CostEstimate]:
    with open(cost_estimate_path(config), "r") as f:
        raw_data = CostEstimateByProblem.model_validate_json(f.read())

    return raw_data.problem_to_cost_estimate


@pydra.main(CodeMonkeysConfigWithLLM)
def main(config: CodeMonkeysConfigWithLLM):
    get_cost_estimate(config)


if __name__ == "__main__":
    main()
