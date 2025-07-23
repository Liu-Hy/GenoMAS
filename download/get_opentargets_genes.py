import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Optional

import mygene
import pandas as pd
import requests
from requests.exceptions import RequestException

# Suppress mygene debug messages
logging.getLogger('mygene').setLevel(logging.WARNING)


class OpenTargetsGeneMapper:
    """
    Download association dataset from Open Targets with the following command:
    rsync -rpltvz --delete rsync.ebi.ac.uk::pub/databases/opentargets/platform/24.09/output/etl/json/associationByOverallDirect .
    See https://platform.opentargets.org/downloads for more information
    """

    def __init__(self, associations_dir='~/associationByOverallDirect'):
        """Initialize the mapper with the directory containing association files"""
        self.associations_dir = associations_dir.replace('~', os.path.expanduser('~'))
        self.disease_id_cache = {}
        self.mg = mygene.MyGeneInfo()

        # Load gene synonyms
        with open("../metadata/gene_synonym.json", 'r') as f:
            self.gene_synonyms = json.load(f)

    def get_disease_id(self, trait: str, max_retries: int = 3, initial_delay: float = 1.0) -> Optional[str]:
        """
        Get disease ID from Open Targets API using trait name
        
        Args:
            trait: The trait name to search for
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds (will be exponentially increased)
            
        Returns:
            Disease ID if found, None otherwise
        """
        if trait in self.disease_id_cache:
            return self.disease_id_cache[trait]

        url = "https://api.platform.opentargets.org/api/v4/graphql"
        query = """
        query SearchQuery($queryString: String!) {
          search(queryString: $queryString, entityNames: ["disease"], page: {index: 0, size: 1}) {
            hits {
              id
              name
              entity
            }
          }
        }
        """

        variables = {"queryString": trait}

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json={'query': query, 'variables': variables})
                response.raise_for_status()  # Raise an exception for bad status codes

                data = response.json()
                hits = data.get('data', {}).get('search', {}).get('hits', [])

                if hits:
                    disease_id = hits[0]['id']
                    self.disease_id_cache[trait] = disease_id
                    return disease_id

                # If no hits found, no need to retry
                return None

            except (RequestException, json.JSONDecodeError) as e:
                delay = initial_delay * (2 ** attempt)  # Exponential backoff

                if attempt < max_retries - 1:
                    logging.warning(
                        f"Failed to get disease ID for trait '{trait}' (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay:.1f} seconds... Error: {str(e)}")
                    time.sleep(delay)
                else:
                    logging.error(
                        f"Failed to get disease ID for trait '{trait}' after {max_retries} attempts. Error: {str(e)}")

        return None

    def batch_get_gene_symbols(self, ensembl_ids: List[str], batch_size: int = 1000) -> Dict[str, str]:
        """Get gene symbols from mygene.info using batch query"""
        symbol_mapping = {}

        # Process in batches
        for i in range(0, len(ensembl_ids), batch_size):
            batch = ensembl_ids[i:i + batch_size]
            results = self.mg.querymany(batch, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)

            # Update the symbol_mapping dictionary with results from this batch
            for result in results:
                if 'symbol' in result:
                    symbol_mapping[result['query']] = result['symbol']

        return symbol_mapping

    def normalize_gene_list(self, genes: List[str]) -> List[str]:
        """Normalize gene symbols using synonym dictionary and remove duplicates while preserving order"""
        seen = set()
        normalized = []
        for gene in genes:
            # Get standard symbol if exists in synonym dictionary
            std_symbol = self.gene_synonyms.get(gene.upper())
            if (std_symbol is not None) and (std_symbol not in seen):
                seen.add(std_symbol)
                normalized.append(std_symbol)
        return normalized

    def load_associations(self) -> pd.DataFrame:
        """Load all association files into a pandas DataFrame"""
        all_data = []
        for file in Path(self.associations_dir).glob('part-*.json'):
            with open(file, 'r') as f:
                for line in f:
                    all_data.append(json.loads(line))
        return pd.DataFrame(all_data)

    def process_traits(self, input_file: str, score_threshold: float = 0.1):
        """
        Process traits from input file and save results back to the same file
        
        Args:
            input_file: Path to input JSON file
            score_threshold: Minimum score threshold for including gene associations
        """
        # Load input
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Extract traits list
        if isinstance(data, list):
            traits = data
        else:
            traits = list(data.keys())

        # Adjust the name of traits that would otherwise fail to match
        trait_mapper = {"Huntingtons Disease": "Huntington's Disease",
                        "Ocular Melanomas": "Eye Melanoma",
                        "Parkinsons Disease": "Parkinson's Disease",
                        "Sjögrens Syndrome": "Sjogren's Syndrome",
                        "Vitamin D Levels": "Vitamin D",
                        "lower grade glioma and glioblastoma": "low grade glioma"}

        # Load associations
        df = self.load_associations()

        # Get disease IDs for all traits
        trait_to_id = {}
        for trait in traits:
            trait_phrase = ' '.join(trait.split('_'))
            if trait_phrase in trait_mapper:
                trait_phrase = trait_mapper[trait_phrase]
            disease_id = self.get_disease_id(trait_phrase)
            trait_to_id[trait] = disease_id
            print(f"Trait: {trait} - Disease ID: {disease_id}")

        # print(f"Trait to disease ID mapping: {trait_to_id}")
        matched_count = len([d for d in trait_to_id.values() if d is not None])
        failed_matches = [k for k, v in trait_to_id.items() if v is None]
        print(f"Matched {matched_count} traits to disease IDs")
        print(f"Failed matches: {failed_matches}")

        # Process each trait
        result = {}
        for trait, disease_id in trait_to_id.items():
            if disease_id is None:
                result[trait] = []
                print(f"Processed trait: {trait} - Failed to get disease ID")
            else:
                # Filter associations for this disease
                trait_associations = df[df['diseaseId'] == disease_id]

                # Apply score threshold and sort
                trait_associations = trait_associations[trait_associations['score'] >= score_threshold]
                trait_associations = trait_associations.sort_values('score', ascending=False)

                # Batch get gene symbols
                ensembl_ids = trait_associations['targetId'].tolist()
                symbol_mapping = self.batch_get_gene_symbols(ensembl_ids)

                # Create ordered list of symbols
                gene_symbols = [symbol_mapping.get(eid) for eid in ensembl_ids]
                gene_symbols = [s for s in gene_symbols if s]  # Remove None values

                # Normalize gene symbols
                gene_symbols = self.normalize_gene_list(gene_symbols)

                result[trait] = gene_symbols
                print(f"Processed trait: {trait}, found {len(gene_symbols)} genes")
                print(f"Result: {gene_symbols}")

        # Save results back to file
        with open(input_file, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    try:
        mapper = OpenTargetsGeneMapper()
        mapper.process_traits("../metadata/task_info.json", score_threshold=0.2)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
