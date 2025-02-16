import asyncio
import json
import os

import pandas as pd

from agents import PIAgent, GEOAgent, TCGAAgent, StatisticianAgent, CodeReviewerAgent, DomainExpertAgent
from core.context import ActionUnit
from environment import Environment
from prompts import *
from utils.utils import normalize_trait
from utils.config import setup_arg_parser
from utils.llm import get_llm_client
from utils.logger import Logger
from utils.resource_monitor import ResourceMonitor
from utils.utils import extract_function_code, get_question_pairs, check_slow_inference


async def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    model = args.model
    scaler = 1.0
    if check_slow_inference(model):
        scaler = 6.0 if ('deepseek' in model.lower() and '671b' in model.lower()) else 3.0
    elif ('deepseek' in model.lower() and 'v3' in model.lower()):
        scaler = 3.0
    args.max_time = args.max_time * scaler
    task_info_file = './metadata/task_info.json'
    all_pairs = get_question_pairs(task_info_file)
    # Set the path containing input data on different devices. Change to yours.
    in_data_root = '/media/techt/DATA' if os.path.exists('/media/techt/DATA') else '../DATA'
    tcga_root = os.path.join(in_data_root, 'TCGA')
    output_root = './output/'
    version = args.version
    log_file = os.path.join(output_root, f"log_{version}.txt")
    
    # Initialize resource monitor with 60-second interval
    ResourceMonitor(log_interval=60).start()
    
    logger = Logger(log_file=log_file, max_msg_length=10000)
    client = get_llm_client(args, logger)

    prep_tool_file = "./tools/preprocess.py"
    with open(prep_tool_file, 'r') as file:
        prep_tools_code_full = file.read()
    geo_selected_code = extract_function_code(prep_tool_file,
                                              ["validate_and_save_cohort_info", "geo_select_clinical_features",
                                               "preview_df"])
    geo_tools = {"full": PREPROCESS_TOOLS.format(tools_code=prep_tools_code_full),
                 "domain_focus": PREPROCESS_TOOLS.format(tools_code=geo_selected_code)}
    # Create agents once
    geo_action_units = [
        ActionUnit("Initial Data Loading", GEO_DATA_LOADING_PROMPT),
        ActionUnit("Dataset Analysis and Clinical Feature Extraction", GEO_FEATURE_ANALYSIS_EXTRACTION_PROMPT),
        ActionUnit("Gene Data Extraction", GEO_GENE_DATA_EXTRACTION_PROMPT),
        ActionUnit("Gene Identifier Review", GEO_GENE_IDENTIFIER_REVIEW_PROMPT),
        ActionUnit("Gene Annotation", GEO_GENE_ANNOTATION_PROMPT),
        ActionUnit("Gene Identifier Mapping", GEO_GENE_IDENTIFIER_MAPPING_PROMPT),
        ActionUnit("Data Normalization and Linking", GEO_DATA_NORMALIZATION_LINKING_PROMPT),
        ActionUnit("TASK COMPLETED", TASK_COMPLETED_PROMPT)
    ]

    geo_agent = GEOAgent(
        client=client,
        logger=logger,
        role_prompt=GEO_ROLE_PROMPT,
        guidelines=GEO_GUIDELINES,
        tools=geo_tools,
        setups='',
        action_units=geo_action_units,
        args=args
    )

    tcga_selected_code = extract_function_code(prep_tool_file, ["tcga_get_relevant_filepaths",
                                                                "tcga_convert_trait",
                                                                "tcga_convert_age",
                                                                "tcga_convert_gender",
                                                                "tcga_select_clinical_features",
                                                                "preview_df"])
    tcga_tools = {"full": PREPROCESS_TOOLS.format(tools_code=prep_tools_code_full),
                  "domain_focus": PREPROCESS_TOOLS.format(tools_code=tcga_selected_code)}
    tcga_action_units = [
        ActionUnit("Initial Data Loading", TCGA_DATA_LOADING_PROMPT.format(list_subdirs=os.listdir(tcga_root))),
        ActionUnit("Find Candidate Demographic Features", TCGA_FIND_CANDIDATE_DEMOGRAPHIC_PROMPT),
        ActionUnit("Select Demographic Features", TCGA_SELECT_DEMOGRAPHIC_PROMPT),
        ActionUnit("Feature Engineering and Validation", TCGA_FEATURE_ENGINEERING_PROMPT),
        ActionUnit("TASK COMPLETED", TASK_COMPLETED_PROMPT)
    ]

    tcga_agent = TCGAAgent(
        client=client,
        logger=logger,
        role_prompt=TCGA_ROLE_PROMPT,
        guidelines=TCGA_GUIDELINES,
        tools=tcga_tools,
        setups='',
        action_units=tcga_action_units,
        args=args
    )

    stat_tool_file = "./tools/statistics.py"
    with open(stat_tool_file, 'r') as file:
        stat_tools_code_full = file.read()
    stat_selected_code = stat_tools_code_full
    stat_tools = {"full": STATISTICIAN_TOOLS.format(tools_code=stat_tools_code_full),
                  "domain_focus": STATISTICIAN_TOOLS.format(tools_code=stat_selected_code)}

    stat_action_units = [
        ActionUnit("Unconditional One-step Regression",
                   UNCONDITIONAL_ONE_STEP_PROMPT),
        ActionUnit("Conditional One-step Regression",
                   CONDITIONAL_ONE_STEP_PROMPT),
        ActionUnit("Two-step Regression", TWO_STEP_PROMPT),
        ActionUnit("TASK COMPLETED", TASK_COMPLETED_PROMPT)
    ]

    statistician = StatisticianAgent(client=client,
                                     logger=logger,
                                     role_prompt=STATISTICIAN_ROLE_PROMPT,
                                     guidelines=STATISTICIAN_GUIDELINES,
                                     tools=stat_tools,
                                     setups='',
                                     action_units=stat_action_units,
                                     args=args
                                     )

    agents = [
        PIAgent(client=client, logger=logger, args=args),
        geo_agent,
        tcga_agent,
        statistician,
        CodeReviewerAgent(client=client, logger=logger, args=args),
        DomainExpertAgent(client=client, logger=logger, args=args)
    ]

    env = Environment(logger=logger, agents=agents, args=args)

    await env.run(all_pairs, in_data_root, output_root, version, task_info_file)


if __name__ == "__main__":
    asyncio.run(main())
