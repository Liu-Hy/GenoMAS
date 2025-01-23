import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

from agents import BaseAgent
from core.messaging import Role, Message, MessageType
from utils.config import GLOBAL_MAX_TIME
from utils.logger import Logger
from utils.path_config import GEOPathConfig, TCGAPathConfig, StatisticianPathConfig
from utils.utils import load_completed_tasks, add_completed_task, delete_corrupted_files


class Environment:
    def __init__(self, logger: Logger, agents: Optional[List[BaseAgent]] = None,
                 args: Optional[argparse.Namespace] = None):
        self.logger = logger
        self.agents: Dict[Role, BaseAgent] = {}
        if agents:
            for agent in agents:
                self.add_agent(agent)

        self.topology = {
            Role.PI: {Role.PI, Role.GEO_AGENT, Role.TCGA_AGENT, Role.STATISTICIAN_AGENT, Role.CODE_REVIEWER,
                      Role.DOMAIN_EXPERT},
            Role.GEO_AGENT: {Role.PI, Role.GEO_AGENT, Role.CODE_REVIEWER, Role.DOMAIN_EXPERT},
            Role.TCGA_AGENT: {Role.PI, Role.TCGA_AGENT, Role.CODE_REVIEWER, Role.DOMAIN_EXPERT},
            Role.STATISTICIAN_AGENT: {Role.PI, Role.STATISTICIAN_AGENT, Role.CODE_REVIEWER},
            Role.CODE_REVIEWER: {Role.PI, Role.GEO_AGENT, Role.TCGA_AGENT, Role.STATISTICIAN_AGENT},
            Role.DOMAIN_EXPERT: {Role.PI, Role.GEO_AGENT, Role.TCGA_AGENT}
        }

        self.message_queue: List[Message] = []
        self.start_time: Optional[float] = None

    def add_agent(self, agent: BaseAgent):
        self.agents[agent.role] = agent

    async def run_task(self, role: Role) -> Optional[str]:
        self.agents[role].clear_states()
        initial_message = await self.agents[Role.PI].assign_task(role)
        if initial_message:
            self.message_queue.append(initial_message)
        while self.message_queue:
            # Check global timeout
            if time.time() - self.start_time > GLOBAL_MAX_TIME:
                self.logger.error(f"TimeoutError! {GLOBAL_MAX_TIME}s exceeded, early stopped this cohort.")
                break

            current_message = self.message_queue.pop(0)

            valid_receivers = current_message.target_roles.intersection(
                self.topology[current_message.role]
            )

            # Let valid receivers observe and act
            for role in valid_receivers:
                if role in self.agents:
                    new_message = await self.agents[role].act(current_message)
                    if new_message:
                        if not MessageType.is_request(new_message):
                            msg_preview = '\n'.join(new_message.content.splitlines()[-50:])
                            self.logger.debug(f"\n【{new_message.role.value}】 ({new_message.type.value})\n{msg_preview}")
                        if new_message.type == MessageType.TIMEOUT:
                            self.logger.error(
                                f"TimeoutError! {new_message.role.value} reported timeout, early stopped this cohort.")
                            self.message_queue.clear()
                            break
                        self.message_queue.append(new_message)
        return self.agents[role].task_context.concatenate_snippets()

    async def run(self, questions: List[Tuple[str, str]], in_data_root: str, output_root: str, version: str,
                  gene_info_file: str) -> Optional[str]:
        """Run the multi-agent system"""
        GEO_root = os.path.join(in_data_root, 'GEO')
        TCGA_root = os.path.join(in_data_root, 'TCGA')
        out_prep_version_dir = os.path.join(output_root, 'preprocess', version)
        out_stat_version_dir = os.path.join(output_root, 'regress', version)

        self.start_time = time.time()
        for (trt, condition) in questions:
            if (trt, condition) in load_completed_tasks(out_stat_version_dir):
                self.logger.info(f"Statistical analysis of {trt}-{condition} already completed, skipped.")
                continue
            traits = [trt]
            if condition and condition not in ['Age', 'Gender']:
                traits.append(condition)

            for trait in traits:
                geo_trait_dir = os.path.join(GEO_root, trait)
                if not os.path.isdir(geo_trait_dir):
                    self.logger.error(f"GEO input trait directory not found: {geo_trait_dir}, skipped.")
                    continue

                out_prep_trait_dir = os.path.join(out_prep_version_dir, trait)
                out_gene_dir = os.path.join(out_prep_trait_dir, 'gene_data')
                out_clinical_dir = os.path.join(out_prep_trait_dir, 'clinical_data')
                out_code_dir = os.path.join(out_prep_trait_dir, 'code')
                for this_dir in [out_gene_dir, out_clinical_dir, out_code_dir]:
                    os.makedirs(this_dir, exist_ok=True)
                json_path = os.path.join(out_prep_trait_dir, "cohort_info.json")

                cohorts = os.listdir(geo_trait_dir) + ['TCGA']
                for cohort in cohorts:
                    if cohort != 'TCGA':
                        geo_cohort_dir = os.path.join(geo_trait_dir, cohort)
                        if not os.path.isdir(geo_cohort_dir):
                            self.logger.info(f"'{geo_cohort_dir}' is not a GEO cohort directory, skipped.")
                            continue
                    if (trait, cohort) in load_completed_tasks(out_prep_version_dir):
                        self.logger.info(f"Preprocessing of {trait}-{cohort} already completed, skipped.")
                        continue
                    delete_corrupted_files(out_prep_trait_dir, cohort)
                    out_data_file = os.path.join(out_prep_trait_dir, f"{cohort}.csv")
                    out_gene_data_file = os.path.join(out_gene_dir, f"{cohort}.csv")
                    out_clinical_data_file = os.path.join(out_clinical_dir, f"{cohort}.csv")
                    out_code_file = os.path.join(out_code_dir, f"{cohort}.py")
                    if cohort != 'TCGA':
                        path_config = GEOPathConfig(
                            trait=trait,
                            cohort=cohort,
                            in_trait_dir=geo_trait_dir,
                            in_cohort_dir=geo_cohort_dir,
                            out_data_file=out_data_file,
                            out_gene_data_file=out_gene_data_file,
                            out_clinical_data_file=out_clinical_data_file,
                            json_path=json_path,
                        )
                        self.agents[Role.GEO_AGENT].update_path_config(path_config)
                        code = await self.run_task(Role.GEO_AGENT)
                        with open(out_code_file, "w") as cf:
                            cf.write(code)

                    else:
                        path_config = TCGAPathConfig(
                            trait=trait,
                            tcga_root_dir=TCGA_root,
                            out_data_file=out_data_file,
                            out_gene_data_file=out_gene_data_file,
                            out_clinical_data_file=out_clinical_data_file,
                            json_path=json_path,
                        )
                        self.agents[Role.TCGA_AGENT].set_path_config(path_config)
                        code = self.run_task(Role.TCGA_AGENT)
                        with open(out_code_file, "w") as cf:
                            cf.write(code)

                    add_completed_task((trait, cohort), out_prep_version_dir)
            path_config = StatisticianPathConfig(trait=trait, condition=condition,
                                                 in_data_root=out_prep_version_dir, gene_info_file=gene_info_file,
                                                 output_root=out_stat_version_dir)
            self.agents[Role.STATISTICIAN_AGENT].update_path_config(path_config)
            self.run_task(Role.STATISTICIAN_AGENT)
            add_completed_task((trt, condition), out_stat_version_dir)

        return None

    def clear_states(self):
        """Clear all states of the environment"""
        for agent in self.agents.values():
            agent.clear_states()
        self.message_queue.clear()
        self.start_time = None
