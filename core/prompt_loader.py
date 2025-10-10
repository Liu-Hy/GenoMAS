"""
Dynamic prompt loader that supports both base and generated prompts.
"""
import importlib
import importlib.util
from typing import List, Dict, Any, Optional

from core.context import ActionUnit
from core.action_unit_set import ActionUnitSet


class PromptContainer:
    """Container for all prompts used in the system (excluding action units)."""
    
    def __init__(self):
        # Shared prompts
        self.PREPROCESS_TOOLS = None
        self.STATISTICIAN_TOOLS = None
        self.MULTI_STEP_SETUPS = None
        self.TASK_COMPLETED_PROMPT = None
        self.CODE_INDUCER = None
        
        # GEO prompts (role and guidelines only)
        self.GEO_ROLE_PROMPT = None
        self.GEO_GUIDELINES = None
        
        # TCGA prompts (role and guidelines only)
        self.TCGA_ROLE_PROMPT = None
        self.TCGA_GUIDELINES = None
        
        # Statistician prompts (role and guidelines only)
        self.STATISTICIAN_ROLE_PROMPT = None
        self.STATISTICIAN_GUIDELINES = None


def load_prompts(use_generated: bool = False) -> PromptContainer:
    """
    Load prompts from the prompts package.
    Note: Action units are now loaded separately via load_action_units().
    
    Args:
        use_generated: Not used anymore for this function (kept for compatibility)
        
    Returns:
        PromptContainer with all loaded prompts (excluding action units)
    """
    container = PromptContainer()
    
    # Load base prompts
    shared = importlib.import_module('prompts.shared')
    geo = importlib.import_module('prompts.GEO')
    tcga = importlib.import_module('prompts.TCGA')
    statistics = importlib.import_module('prompts.statistics')
    
    # Load shared prompts (never generated)
    container.PREPROCESS_TOOLS = shared.PREPROCESS_TOOLS
    container.STATISTICIAN_TOOLS = shared.STATISTICIAN_TOOLS
    container.MULTI_STEP_SETUPS = shared.MULTI_STEP_SETUPS
    container.TASK_COMPLETED_PROMPT = shared.TASK_COMPLETED_PROMPT
    container.CODE_INDUCER = shared.CODE_INDUCER
    
    # Load GEO prompts (role and guidelines only)
    container.GEO_ROLE_PROMPT = geo.GEO_ROLE_PROMPT
    container.GEO_GUIDELINES = geo.GEO_GUIDELINES
    
    # Load TCGA prompts (role and guidelines only)
    container.TCGA_ROLE_PROMPT = tcga.TCGA_ROLE_PROMPT
    container.TCGA_GUIDELINES = tcga.TCGA_GUIDELINES
    
    # Load Statistician prompts (role and guidelines only)
    container.STATISTICIAN_ROLE_PROMPT = statistics.STATISTICIAN_ROLE_PROMPT
    container.STATISTICIAN_GUIDELINES = statistics.STATISTICIAN_GUIDELINES
    
    return container


def load_action_units(agent_id: str, use_generated: bool = False, 
                     context: Optional[Dict[str, Any]] = None) -> List[ActionUnit]:
    """
    Load action units for a specific agent.
    
    Args:
        agent_id: Agent identifier ('geo', 'tcga', or 'statistician')
        use_generated: If True, try to load generated version first
        context: Optional runtime context for augmentation (e.g., {'subdirs': [...]})
        
    Returns:
        List of ActionUnit objects with TASK_COMPLETED appended
    """
    au_set = ActionUnitSet.from_agent_id(agent_id, use_generated=use_generated)
    action_units = au_set.get_action_units()
    
    # Apply runtime augmentation for TCGA subdirectories
    if context and agent_id.lower() == 'tcga' and 'subdirs' in context:
        subdirs_prompt = f"\nAvailable subdirectories under `tcga_root_dir`:\n{context['subdirs']}\n"
        # Find the first AU (should be "Initial Data Loading") and append
        if action_units:
            action_units[0].instruction = action_units[0].instruction + subdirs_prompt
    
    # Append TASK_COMPLETED using shared prompt for consistency
    prompts = load_prompts()
    action_units.append(
        ActionUnit("TASK COMPLETED", prompts.TASK_COMPLETED_PROMPT)
    )
    
    return action_units
