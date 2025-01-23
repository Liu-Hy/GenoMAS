import asyncio
import io
import json
import os
import re
import shutil
import sys
import time
import traceback

from core.context import ActionUnit
from environment import Environment
from prompts import *
from agents import PIAgent, GEOAgent, CodeReviewerAgent, DomainExpertAgent
from tools.statistics import normalize_trait
from utils.llm import LLMClient, get_llm_client
from utils.logger import Logger
from utils.config import GLOBAL_MAX_TIME
from utils.path_config import GEOPathConfig, TCGAPathConfig
from utils.utils import extract_function_code, load_last_cohort_info, delete_corrupted_files, save_last_cohort_info
from utils.config import setup_arg_parser

async def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    client = get_llm_client(args)

    # Read tools code
    tool_file = "./tools/preprocess.py"
    with open(tool_file, 'r') as file:
        tools_code_full = file.read()
    extracted_code = extract_function_code(tool_file, ["validate_and_save_cohort_info", "geo_select_clinical_features", "preview_df"])
    tools = {"full": PREPROCESS_TOOLS.format(tools_code=tools_code_full),
             "domain_focus": PREPROCESS_TOOLS.format(tools_code=extracted_code)}
    # Load traits with special ordering
    with open("./metadata/all_traits.json", "r") as f:
        all_traits = json.load(f)
    all_traits = [normalize_trait(t) for t in all_traits]
    special_traits = ["Breast_Cancer", "Epilepsy", "Atherosclerosis"]
    all_traits = [t for t in all_traits if t not in special_traits]
    all_traits = special_traits + all_traits

    input_dir = '/media/techt/DATA/GEO' if os.path.exists('/media/techt/DATA/GEO') else '../DATA/GEO'
    output_root = './output/preprocess/'
    version = args.version
    out_version_dir = os.path.join(output_root, version)

    if not args.resume:
        confirm_delete = input(
            f"Do you want to delete all the output data in '{out_version_dir}'? [yes/no]: ")
        if confirm_delete.lower() == 'yes':
            shutil.rmtree(out_version_dir)

    os.makedirs(out_version_dir, exist_ok=True)
    log_file = os.path.join(out_version_dir, 'log.txt')
    logger = Logger(log_file=log_file, max_msg_length=5000)

    # Create agents once
    action_units = [
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
        tools=tools,
        setups='',
        action_units=action_units,
        args=args
    )

    agents = [
        PIAgent(client=client, logger=logger, args=args),
        geo_agent,
        CodeReviewerAgent(client=client, logger=logger, args=args),
        DomainExpertAgent(client=client, logger=logger, args=args)
    ]

    env = Environment(logger=logger, agents=agents, args=args)

    last_cohort_info = load_last_cohort_info(out_version_dir)
    for index, trait in enumerate(all_traits):
        in_trait_dir = os.path.join(input_dir, trait)
        if not os.path.isdir(in_trait_dir):
            logger.error(f"Trait directory not found: {in_trait_dir}")
            continue
        out_trait_dir = os.path.join(out_version_dir, trait)
        os.makedirs(out_trait_dir, exist_ok=True)
        out_gene_dir = os.path.join(out_trait_dir, 'gene_data')
        out_clinical_dir = os.path.join(out_trait_dir, 'clinical_data')
        out_log_dir = os.path.join(out_trait_dir, 'log')
        out_code_dir = os.path.join(out_trait_dir, 'code')
        for this_dir in [out_gene_dir, out_clinical_dir, out_log_dir, out_code_dir]:
            os.makedirs(this_dir, exist_ok=True)

        json_path = os.path.join(out_trait_dir, "cohort_info.json")

        cohorts = os.listdir(in_trait_dir)
        for cohort in cohorts:
            in_cohort_dir = os.path.join(in_trait_dir, cohort)
            if not os.path.isdir(in_cohort_dir):
                logger.error(f"Cohort directory not found: {in_cohort_dir}")
                continue
            out_data_file = os.path.join(out_trait_dir, f"{cohort}.csv")
            out_gene_data_file = os.path.join(out_gene_dir, f"{cohort}.csv")
            out_clinical_data_file = os.path.join(out_clinical_dir, f"{cohort}.csv")


            # Handle resume logic
            if args.resume:
                if last_cohort_info:
                    # Skip until we find the last processed cohort
                    if last_cohort_info['trait'] == trait and last_cohort_info['cohort'] == cohort:
                        logger.info(f"Found last processed: trait {trait}, cohort {cohort}")
                        last_cohort_info = None
                        continue
                    logger.info(f"Skipping previously processed: trait {trait}, cohort {cohort}")
                    continue
                elif os.path.exists(out_gene_data_file) or os.path.exists(out_clinical_data_file):
                    logger.info(f"Cleaning up partial files before resuming: trait {trait}, cohort {cohort}")
                    delete_corrupted_files(out_trait_dir, cohort)

            try:
                # Create path configuration for this cohort
                path_config = GEOPathConfig(
                    trait=trait,
                    cohort=cohort,
                    in_trait_dir=in_trait_dir,
                    in_cohort_dir=in_cohort_dir,
                    out_data_file=out_data_file,
                    out_gene_data_file=out_gene_data_file,
                    out_clinical_data_file=out_clinical_data_file,
                    json_path=json_path,
                )

                geo_agent.update_path_config(path_config)

                # Clear states before processing new cohort
                env.clear_states()
                
                # Run the environment
                code = await asyncio.wait_for(env.run(), timeout=GLOBAL_MAX_TIME)

                # Save the final code to a file
                code_file = os.path.join(out_code_dir, f"{cohort}.py")
                with open(code_file, "w") as cf:
                    cf.write(code)

                # Save the current state
                save_last_cohort_info(out_version_dir, {'trait': trait, 'cohort': cohort})
                logger.save()

            except asyncio.TimeoutError:
                logger.error(f"Timeout error occurred while processing trait {trait}, cohort {cohort}")
            except Exception as e:
                logger.error(f"Error occurred while processing trait {trait}, cohort {cohort}\n {type(e).__name__}: {str(e)}\n{traceback.format_exc()}")
            finally:
                # Reset stdout and stderr to default
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__


if __name__ == "__main__":
    asyncio.run(main())
