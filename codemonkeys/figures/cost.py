def deepseek_usage_to_cost(usage_dict: dict):
    completion_cost_per_token = 0.28 / 1e6
    cache_miss_cost_per_token = 0.14 / 1e6
    cache_hit_cost_per_token = 0.014 / 1e6

    cost = {
        "output": usage_dict["completion_tokens"] * completion_cost_per_token,
        "cache_hit": usage_dict["prompt_cache_hit_tokens"] * cache_hit_cost_per_token,
        "cache_miss": usage_dict["prompt_cache_miss_tokens"]
        * cache_miss_cost_per_token,
    }
    return cost


def sonnet_usage_to_cost(usage_dict: dict):
    completion_cost_per_token = 15 / 1e6
    cache_write_cost_per_token = 3.75 / 1e6
    cache_read_cost_per_token = 0.3 / 1e6
    input_cost_per_token = 3 / 1e6

    cost = {
        "output": usage_dict["output_tokens"] * completion_cost_per_token,
        "cache_write": usage_dict["cache_creation_input_tokens"]
        * cache_write_cost_per_token,
        "cache_read": usage_dict["cache_read_input_tokens"] * cache_read_cost_per_token,
        "input": usage_dict["input_tokens"] * input_cost_per_token,
    }
    return cost


def usage_to_cost_dicts(model: str, usage_dict: dict):
    if model == "deepseek":
        return deepseek_usage_to_cost(usage_dict)
    elif model == "sonnet":
        return sonnet_usage_to_cost(usage_dict)
    elif model == "do-not-compute-costs":
        return {
            "output": 0.0,
            "cache_write": 0.0,
            "cache_read": 0.0,
            "input": 0.0,
        }
    else:
        raise ValueError(f"Unknown model: {model}")


def usage_to_cost(model: str, usage_dict: dict):
    return sum(usage_to_cost_dicts(model, usage_dict).values())
