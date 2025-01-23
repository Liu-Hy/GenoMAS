"""Path configuration utilities for GEOAgent."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class PathConfig(ABC):
    """Abstract base class for path configurations used in data analysis."""

    @abstractmethod
    def get_setup_code(self) -> str:
        """Generate Python code for setting up paths in the CodeExecutor namespace."""
        pass

    @abstractmethod
    def get_setup_prompt(self) -> str:
        """Generate the path setup section for the prompt."""
        pass


@dataclass
class GEOPathConfig(PathConfig):
    """Container for all path configurations used in preprocessing GEO data."""
    # Context identifiers
    trait: str
    cohort: str

    # Input paths
    in_trait_dir: str
    in_cohort_dir: str

    # Output paths
    out_data_file: str
    out_gene_data_file: str
    out_clinical_data_file: str
    json_path: str

    def get_setup_code(self) -> str:
        """Generate Python code for setting up paths in the CodeExecutor namespace."""
        return f"""# Path Configuration
from tools.preprocess import *

# Processing context
trait = "{self.trait}"
cohort = "{self.cohort}"

# Input paths
in_trait_dir = "{self.in_trait_dir}"
in_cohort_dir = "{self.in_cohort_dir}"

# Output paths
out_data_file = "{self.out_data_file}"
out_gene_data_file = "{self.out_gene_data_file}"
out_clinical_data_file = "{self.out_clinical_data_file}"
json_path = "{self.json_path}"
"""

    def get_setup_prompt(self) -> str:
        """Generate the path setup section for the prompt."""
        return f"""
1. Path Configuration
The following variables have been pre-configured in your execution environment,
to maintain consistent file organization:

Context Variables:
- trait: "{self.trait}"
  The current trait being processed.
  Use this instead of hardcoding the trait name in your code.

- cohort: "{self.cohort}"
  The current cohort being processed.
  Use this instead of hardcoding the cohort name.

Input Paths:
- in_trait_dir: "{self.in_trait_dir}"
  The directory containing raw data of all cohorts for the current trait.

- in_cohort_dir: "{self.in_cohort_dir}"
  The directory containing raw data for the current cohort.

Output Paths:
- out_data_file: "{self.out_data_file}"
  Where to save the processed linked data.

- out_gene_data_file: "{self.out_gene_data_file}"
  Where to save the processed gene expression data.

- out_clinical_data_file: "{self.out_clinical_data_file}"
  Where to save the processed clinical data.

- json_path: "{self.json_path}"
  Where to save cohort metadata about data usability and quality.

2. Pre-executed Setup Code
The following code has been automatically executed to prepare your environment. 
All functions from tools.preprocess have been imported and are ready to use.
You can use these variables and functions directly in your code without importing or defining them.

```python
{self.get_setup_code()}```
"""


@dataclass
class TCGAPathConfig(PathConfig):
    """Container for all path configurations used in preprocessing TCGA data."""
    # Context identifiers
    trait: str

    # Input paths
    tcga_root_dir: str

    # Output paths
    out_data_file: str
    out_gene_data_file: str
    out_clinical_data_file: str
    json_path: str

    def get_setup_code(self) -> str:
        """Generate Python code for setting up paths in the CodeExecutor namespace."""
        return f"""# Path Configuration
from tools.preprocess import *

# Processing context
trait = "{self.trait}"

# Input paths
tcga_root_dir = "{self.tcga_root_dir}"

# Output paths
out_data_file = "{self.out_data_file}"
out_gene_data_file = "{self.out_gene_data_file}"
out_clinical_data_file = "{self.out_clinical_data_file}"
json_path = "{self.json_path}"
"""

    def get_setup_prompt(self) -> str:
        """Generate the path setup section for the prompt."""
        return f"""
1. Path Configuration
The following variables have been pre-configured in your execution environment,
to maintain consistent file organization:

Context Variables:
- trait: "{self.trait}"
  The current trait being processed.
  Use this instead of hardcoding the trait name in your code.

Input Paths:
- tcga_root_dir: "{self.tcga_root_dir}"
  The root directory of the TCGA Xena dataset.

Output Paths:
- out_data_file: "{self.out_data_file}"
  Where to save the processed linked data.

- out_gene_data_file: "{self.out_gene_data_file}"
  Where to save the processed gene expression data.

- out_clinical_data_file: "{self.out_clinical_data_file}"
  Where to save the processed clinical data.

- json_path: "{self.json_path}"
  Where to save cohort metadata about data usability and quality.

2. Pre-executed Setup Code
The following code has been automatically executed to prepare your environment. 
All functions from tools.preprocess have been imported and are ready to use.
You can use these variables and functions directly in your code without importing or defining them.

```python
{self.get_setup_code()}```
"""



# TO DO: for statistician
"""
Setups:
1. All input data are stored in the directory: '{data_root}'.
2. The output should be saved to the directory '{output_root}', under a subdirectory named after the trait.
3. External knowledge about genes related to each trait is available in a file '{gene_info_path}'.
"""



@dataclass
class StatisticianPathConfig(PathConfig):
    """Container for all path configurations used in Statistical analysis."""
    # Context identifiers
    trait: str
    condition: str

    # Input paths
    in_data_root: str
    gene_info_file: str

    # Output paths
    output_root: str

    def get_setup_code(self) -> str:
        """Generate Python code for setting up paths in the CodeExecutor namespace."""
        return f"""# Path Configuration
from tools.statistics import *
from sklearn.linear_model import LogisticRegression, LinearRegression

# Processing context
trait = "{self.trait}"
condition = "{self.condition}"

# Input paths
in_data_root = "{self.in_data_root}"
gene_info_file = "{self.gene_info_file}"

# Output paths
output_root = "{self.output_root}"
"""

    def get_setup_prompt(self) -> str:
        """Generate the path setup section for the prompt."""
        return f"""
1. Path Configuration
The following variables have been pre-configured in your execution environment,
to maintain consistent file organization:

Context Variables:
- trait: "{self.trait}"
  The trait in the current question being addressed.
  Use this instead of hardcoding the trait name in your code.

- condition: "{self.condition}"
  The condition in the current question being addressed.
  Use this instead of hardcoding the condition name.

Input Paths:
- in_data_root: "{self.in_data_root}"
  The directory containing all the preprocessed data for statistical analysis.

- gene_info_file: "{self.gene_info_file}"
  The file containing external knowledge about genes related to each trait.

Output Paths:
- output_root: "{self.output_root}"
  Where to save all the analysis results.

2. Pre-executed Setup Code
The following code has been automatically executed to prepare your environment. 
All functions from tools.statistics have been imported and are ready to use.
You can use these variables and functions directly in your code without importing or defining them.

```python
{self.get_setup_code()}```
"""