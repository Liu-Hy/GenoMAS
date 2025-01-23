import requests
import time
from xml.etree import ElementTree


def search_genes_for_trait(trait, organism="human"):
    # E-utilities base URL
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    # First, search for gene IDs related to the trait
    search_url = f"{base_url}esearch.fcgi"
    search_params = {
        'db': 'gene',
        'term': f'"{trait}"[All Fields] AND "{organism}"[Organism]',
        'retmax': 100  # Adjust this number based on how many results you want
    }

    # Add sleep to comply with NCBI's rate limits
    time.sleep(0.34)  # Maximum 3 requests per second

    try:
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()

        # Parse XML response
        tree = ElementTree.fromstring(response.content)
        id_list = [id_elem.text for id_elem in tree.findall('.//Id')]

        if not id_list:
            return []

        # Get detailed information for each gene
        summary_url = f"{base_url}esummary.fcgi"
        summary_params = {
            'db': 'gene',
            'id': ','.join(id_list)
        }

        time.sleep(0.34)
        response = requests.get(summary_url, params=summary_params)
        response.raise_for_status()

        # Parse gene information
        tree = ElementTree.fromstring(response.content)
        genes = []

        for doc in tree.findall('.//DocumentSummary'):
            name = doc.find('Name').text if doc.find('Name') is not None else 'N/A'
            description = doc.find('Description').text if doc.find('Description') is not None else 'N/A'
            genes.append({
                'name': name,
                'description': description
            })

        return genes

    except requests.exceptions.RequestException as e:
        print(f"Error searching for trait {trait}: {e}")
        return []


# Example usage
traits = ["Breast_Cancer", "height", "eye color", "blood pressure"]

for trait in traits:
    print(f"\nSearching genes related to: {trait}")
    genes = search_genes_for_trait(trait)

    for i, gene in enumerate(genes, 1):
        print(f"{i}. {gene['name']}: {gene['description']}")