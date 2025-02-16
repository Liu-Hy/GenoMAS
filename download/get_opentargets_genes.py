import os
import json
import requests
from typing import List, Dict
from pathlib import Path
import pandas as pd

class OpenTargetsGeneMapper:
    def __init__(self, associations_dir='~/associationByOverallDirect'):
        """Initialize the mapper with the directory containing association files"""
        self.associations_dir = associations_dir.replace('~', os.path.expanduser('~'))
        self.disease_id_cache = {}
        self.gene_symbol_cache = {}
        
    def get_disease_id(self, trait: str) -> str:
        """Get disease ID from Open Targets API using trait name"""
        if trait in self.disease_id_cache:
            return self.disease_id_cache[trait]
        
        print(f"Querying disease ID for trait: {trait}")
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
        response = requests.post(url, json={'query': query, 'variables': variables})
        print(f"Disease ID API response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            hits = data.get('data', {}).get('search', {}).get('hits', [])
            if hits:
                disease_id = hits[0]['id']
                self.disease_id_cache[trait] = disease_id
                print(f"Found disease ID: {disease_id}")
                return disease_id
        return None
    
    def get_gene_symbol(self, ensembl_id: str) -> str:
        """Get gene symbol from Open Targets API using Ensembl ID"""
        if ensembl_id in self.gene_symbol_cache:
            return self.gene_symbol_cache[ensembl_id]
        
        url = "https://api.platform.opentargets.org/api/v4/graphql"
        query = """
        query GeneInfo($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            approvedSymbol
          }
        }
        """
        response = requests.post(url, json={'query': query, 'variables': {'ensemblId': ensembl_id}})
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data', {}).get('target', {}).get('approvedSymbol'):
                symbol = data['data']['target']['approvedSymbol']
                self.gene_symbol_cache[ensembl_id] = symbol
                return symbol
        return None

    def load_associations(self) -> pd.DataFrame:
        """Load all association files into a pandas DataFrame"""
        all_data = []
        print(f"Loading associations from: {self.associations_dir}")
        file_count = 0
        for file in Path(self.associations_dir).glob('part-*.json'):
            file_count += 1
            print(f"Reading file: {file.name}")
            with open(file, 'r') as f:
                for line in f:
                    all_data.append(json.loads(line))
        
        print(f"Found {file_count} files, loaded {len(all_data)} associations")
        df = pd.DataFrame(all_data)
        print(f"DataFrame shape: {df.shape}")
        return df

    def get_trait_associated_genes(self, traits: List[str], score_threshold: float = 0.1) -> Dict[str, List[str]]:
        """
        Get associated genes for a list of traits
        
        Args:
            traits: List of trait names in natural language
            score_threshold: Minimum score threshold for including gene associations
            
        Returns:
            Dictionary mapping traits to lists of gene symbols, ordered by association score
        """
        # Load all associations
        df = self.load_associations()
        
        # Get disease IDs for all traits
        trait_to_id = {}
        for trait in traits:
            disease_id = self.get_disease_id(trait)
            if disease_id:
                trait_to_id[trait] = disease_id
        
        print(f"Trait to disease ID mapping: {trait_to_id}")
        
        result = {}
        for trait, disease_id in trait_to_id.items():
            print(f"\nProcessing trait: {trait} (disease ID: {disease_id})")
            
            # Filter associations for this disease
            trait_associations = df[df['diseaseId'] == disease_id]
            print(f"Found {len(trait_associations)} associations for this disease")
            
            # Apply score threshold
            trait_associations = trait_associations[trait_associations['score'] >= score_threshold]
            print(f"After applying score threshold ({score_threshold}): {len(trait_associations)} associations")
            
            # Sort by score in descending order
            trait_associations = trait_associations.sort_values('score', ascending=False)
            
            # Get gene symbols for Ensembl IDs
            gene_symbols = []
            for ensembl_id in trait_associations['targetId']:
                symbol = self.get_gene_symbol(ensembl_id)
                if symbol:
                    gene_symbols.append(symbol)
                    print(f"Mapped {ensembl_id} to {symbol}")
            
            result[trait] = gene_symbols
            
        return result

# Example usage with error handling:
if __name__ == "__main__":
    try:
        mapper = OpenTargetsGeneMapper()
        traits = ["Type 2 diabetes", "Alzheimer's disease"]
        print(f"Processing traits: {traits}")
        
        trait_genes = mapper.get_trait_associated_genes(traits, score_threshold=0.5)
        
        for trait, genes in trait_genes.items():
            print(f"\n{trait}:")
            print(f"Number of associated genes: {len(genes)}")
            if genes:
                print(f"Top 10 genes: {genes[:10]}")
            else:
                print("No genes found")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")