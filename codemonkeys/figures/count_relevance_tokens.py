"""Count the number of tokens (input and output) used for relevance."""

import json
from pathlib import Path
import tqdm
from concurrent.futures import ProcessPoolExecutor

from codemonkeys.utils.shared_config import CodeMonkeysConfig
from codemonkeys.trajectory_data.store import TrajectoryStore

import pydra

from codemonkeys.swe_bench.swe_bench_verified import SWEBenchProblem

from functools import cache
from transformers import AutoTokenizer
from codemonkeys.prompts.relevance_prompts import get_user_prompt


@cache
def load_tokenizer():
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def count_tokens(trajectory_store: TrajectoryStore) -> int:
    problem = trajectory_store.problem

    relevance_decisions = trajectory_store.load_relevance_decisions()

    all_messages = []
    for file_path, decision in relevance_decisions.items():
        user_prompt = get_user_prompt(
            problem,
            file_path,
            problem.get_file(file_path).content,
        )
        model_response = decision.message
        all_messages.append(
            [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": model_response},
            ]
        )
    assert len(all_messages) > 0

    all_tokens = load_tokenizer().apply_chat_template(
        all_messages,  # type: ignore
        tokenize=True,
        add_generation_prompt=True,
    )

    token_count = 0
    assert isinstance(all_tokens, list)
    for tokens in all_tokens:
        assert isinstance(tokens, list)
        token_count += len(tokens)

    return token_count


def token_count_path(config: CodeMonkeysConfig) -> Path:
    return Path(config.trajectory_store_dir / "relevance_token_estimate.json")


@pydra.main(CodeMonkeysConfig)
def main(config: CodeMonkeysConfig):

    with ProcessPoolExecutor(max_workers=2) as executor:
        # Load the conversations
        instance_id_to_tokens_futures = {
            problem.instance_id: executor.submit(
                count_tokens, config.get_trajectory_store(problem)
            )
            for problem in tqdm.tqdm(
                config.get_problems(), desc="Loading conversations"
            )
        }

        # Submit the futures for doing the tokenization as they load

        # Wait for the tokenization to finish
        instance_id_to_tokens = {
            instance_id: future.result()
            for (instance_id, future) in tqdm.tqdm(
                instance_id_to_tokens_futures.items(), desc="Waiting for futures"
            )
        }

    with open(token_count_path(config), "w") as f:  # type: ignore
        json.dump(instance_id_to_tokens, f)


def load_relevance_token_count(config: CodeMonkeysConfig) -> dict[str, int]:
    with open(token_count_path(config), "r") as f:
        problem_to_tokens = json.load(f)

    assert isinstance(problem_to_tokens, dict)

    for k, v in problem_to_tokens.items():
        assert isinstance(k, str)
        assert isinstance(v, int)

    return problem_to_tokens


if __name__ == "__main__":
    main()
