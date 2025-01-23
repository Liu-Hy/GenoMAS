import asyncio
import os
import sys
import traceback

import pandas as pd

from agents import StatisticianAgent, PIAgent, GEOAgent, CodeReviewerAgent, DomainExpertAgent
from core.context import ActionUnit
from environment import Environment
from prompts import *
from utils.config import GLOBAL_MAX_TIME
from utils.config import setup_arg_parser
from utils.llm import get_llm_client
from utils.logger import Logger
from utils.path_config import StatisticianPathConfig


async def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    client = get_llm_client(args)

    pairs_df = pd.read_csv("./metadata/trait_condition_pairs.csv")
    pair_rows = list(pairs_df.iterrows())
    all_pairs = []
    seen_traits = set()
    for i, row in pair_rows:
        trait, condition = row['Trait'], row['Condition']
        if trait not in seen_traits:
            seen_traits.add(trait)
            all_pairs.append((trait, None))
        all_pairs.append((trait, condition))

    gene_info_path = './metadata/trait_related_genes.csv'
    in_data_root = './output/preprocess/b4'
    out_version_dir = './output/regress/b4_test'

    os.makedirs(out_version_dir, exist_ok=True)
    log_file = os.path.join(out_version_dir, 'log.txt')
    logger = Logger(log_file=log_file, max_msg_length=5000)

    tool_file = "./tools/statistics.py"
    with open(tool_file, 'r') as file:
        tools_code_full = file.read()
    extracted_code = tools_code_full
    tools = {"full": STATISTICIAN_TOOLS.format(tools_code=tools_code_full),
             "domain_focus": STATISTICIAN_TOOLS.format(tools_code=extracted_code)}
    action_units = [
        ActionUnit("Unconditional One-step Regression",
                   UNCONDITIONAL_ONE_STEP_PROMPT, UNCONDITIONAL_ONE_STEP_CODE),
        ActionUnit("Conditional One-step Regression",
                   CONDITIONAL_ONE_STEP_PROMPT, CONDITIONAL_ONE_STEP_CODE),
        ActionUnit("Two-step Regression", TWO_STEP_PROMPT, TWO_STEP_CODE),
        ActionUnit("TASK COMPLETED", TASK_COMPLETED_PROMPT)
    ]

    statistician = StatisticianAgent(client=client,
                                     logger=logger,
                                     role_prompt=STATISTICIAN_ROLE_PROMPT,
                                     guidelines=STATISTICIAN_GUIDELINES,
                                     tools=tools,
                                     setups='',
                                     action_units=action_units,
                                     args=args
                                     )

    agents = [
        PIAgent(client=client, logger=logger, args=args),
        statistician,
        CodeReviewerAgent(client=client, logger=logger, args=args),
        DomainExpertAgent(client=client, logger=logger, args=args)
    ]

    env = Environment(logger=logger, agents=agents, args=args)

    for index, pair in enumerate(all_pairs):
        try:
            trait, condition = pair
            question = f"\nThe question to solve is: What are the genetic factors related to the trait '{trait}' when considering the influence of the " \
                       f"condition '{condition}'?"
            print(trait, condition)
            path_config = StatisticianPathConfig(trait=trait, condition=condition,
                                                 in_data_root=in_data_root, gene_info_file=gene_info_path,
                                                 output_root=out_version_dir)
            statistician.update_path_config(path_config)
            # Clear states before processing new cohort
            env.clear_states()

            # Run the environment
            code = await asyncio.wait_for(env.run(), timeout=GLOBAL_MAX_TIME)

            print(code)

        except asyncio.TimeoutError:
            logger.error(f"Timeout error occurred while processing trait {trait}, condition {condition}")
        except Exception as e:
            logger.error(
                f"Error occurred while processing trait {trait}, condition {condition}\n {type(e).__name__}: {str(e)}\n{traceback.format_exc()}")
        finally:
            # Reset stdout and stderr to default
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == "__main__":
    asyncio.run(main())
