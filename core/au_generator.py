"""
Action Unit generator that creates instruction prompts from guidelines.
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from utils.llm import LLMClient
from utils.logger import Logger
from core.prompt_loader import load_prompts
from core.action_unit_set import ActionUnitSet
from core.context import ActionUnit


async def generate_for_agent(
    agent_id: str,
    guidelines: str,
    client: LLMClient,
    logger: Optional[Logger] = None
) -> Optional[ActionUnitSet]:
    """
    Generate action unit set for a specific agent based on guidelines.
    
    Args:
        agent_id: Agent identifier ('GEO', 'TCGA', or 'STAT')
        guidelines: The agent's high-level guidelines
        client: LLM client for generation
        logger: Optional logger
        
    Returns:
        ActionUnitSet with generated action units, or None if generation fails
    """
    # Prepare the generation prompt
    system_prompt = """You are an expert in designing workflow decomposition for autonomous programming agents.
You will decompose agent guidelines into Action Units for a biomedical data analysis system.
Output strict JSON only, no additional text."""
    
    user_prompt = f"""CONTEXT: Action Units in Guided Planning Framework

The programming agents execute complex workflows using a guided planning framework that balances structure and flexibility.

Agents operate under textual task guidelines describing dependency structures, conditional logic, and termination criteria 
implicitly as a directed acyclic graph (DAG), enabling dynamic behavior adjustment based on evolving task context.

Within this DAG, agents decompose workflows into Action Units: semantically coherent operations corresponding to discrete 
subtasks. For example, the GEO agent's workflow comprises Action Units for data loading, clinical feature extraction, 
gene identifier mapping, data normalization and linking, etc. 
Each unit represents a self-contained sequence of operations executable atomically, without needing intermediate observations.

YOUR TASK:

Analyze the following guidelines for the {agent_id} agent and decompose them into appropriate Action Units.

GUIDELINES:
{guidelines}

For each Action Unit you create, provide:
1. A clear, descriptive name (e.g., "Initial Data Loading", "Clinical Feature Extraction")
2. Detailed step-by-step instructions that the agent can follow to execute this unit
3. Metadata indicating:
   - requires_domain_knowledge: true if this unit requires biomedical domain expertise
   - no_history: true if this unit should execute without full task history context

REQUIREMENTS:
- Decompose the workflow into a list of Action Units
- Each unit should be executable atomically, without needing intermediate observations
- Instructions should be concrete and actionable. Reference specific variables and file paths when appropriate
- Number the steps within each instruction for clarity
- Do not include a unit for task completion or termination - it's added automatically to all agents

OUTPUT FORMAT:

Generate a JSON array of action units:

[
  {{
    "name": "Action Unit Name",
    "instruction": "Detailed step-by-step instructions...",
    "metadata": {{
      "requires_domain_knowledge": true/false,
      "no_history": true/false
    }}
  }},
  ...
]

Generate the JSON array now:"""
    
    # Call the LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        # Use the client's configured timeout_per_message to enforce timeout at application level
        # This matches the timeout configuration in utils/llm.py lines 240-299
        timeout = client.timeout_per_message * 2
        
        if logger:
            logger.info(f"Generating action units for {agent_id} agent (timeout: {timeout}s)...")
        
        response = await asyncio.wait_for(
            client.generate_completion(messages),
            timeout=timeout
        )
        content = response.get("content", "")
        
        # Parse JSON response
        # First try to extract JSON if wrapped in markdown
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        
        au_data_list = json.loads(content)
        
        # Validate structure
        if not isinstance(au_data_list, list):
            raise ValueError("Expected JSON array of action units")
        
        # Convert to ActionUnit objects
        action_units = [ActionUnit.from_dict(au_data) for au_data in au_data_list]
        
        # Create ActionUnitSet
        au_set = ActionUnitSet(
            agent_id=agent_id.lower(),
            action_units=action_units,
            version="1.0",
            is_generated=True,
            generated_at=datetime.now().isoformat()
        )
        
        if logger:
            logger.info(f"Generated {len(action_units)} action units for {agent_id} agent")
        
        return au_set
        
    except asyncio.TimeoutError:
        if logger:
            logger.error(f"Timeout generating action units for {agent_id} (exceeded {timeout}s)")
        return None
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        if logger:
            logger.error(f"Failed to parse LLM response for {agent_id}: {e}")
        return None
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error generating action units for {agent_id}: {e}")
        return None


async def generate_and_save_all(
    planning_client: LLMClient,
    logger: Optional[Logger] = None,
) -> Dict[str, str]:
    """
    Generate and save all action unit sets for all agents.
    
    Args:
        planning_client: LLM client to use for generation
        logger: Optional logger
        
    Returns:
        Dictionary mapping agent_id to filepath of generated JSON file
    """
    # Load base prompts for guidelines
    base_prompts = load_prompts(use_generated=False)
    
    generated_files = {}
    output_dir = 'prompts/action_units/generated'
    
    # GEO Agent
    geo_au_set = await generate_for_agent(
        'GEO', base_prompts.GEO_GUIDELINES, planning_client, logger
    )
    
    if geo_au_set:
        filepath = os.path.join(output_dir, 'geo_action_units.json')
        geo_au_set.to_file(filepath)
        generated_files['GEO'] = filepath
        if logger:
            logger.info(f"Generated GEO action units: {filepath}")
    
    # TCGA Agent
    tcga_au_set = await generate_for_agent(
        'TCGA', base_prompts.TCGA_GUIDELINES, planning_client, logger
    )
    
    if tcga_au_set:
        filepath = os.path.join(output_dir, 'tcga_action_units.json')
        tcga_au_set.to_file(filepath)
        generated_files['TCGA'] = filepath
        if logger:
            logger.info(f"Generated TCGA action units: {filepath}")
    
    # Statistician Agent
    stat_au_set = await generate_for_agent(
        'STAT', base_prompts.STATISTICIAN_GUIDELINES, planning_client, logger
    )
    
    if stat_au_set:
        filepath = os.path.join(output_dir, 'statistician_action_units.json')
        stat_au_set.to_file(filepath)
        generated_files['STAT'] = filepath
        if logger:
            logger.info(f"Generated Statistician action units: {filepath}")
    
    return generated_files
