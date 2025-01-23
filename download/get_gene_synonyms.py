import gzip
import json

def build_gene_info_dict(gene_info_path, tax_id='9606'):
    """
    Return a dict of:
        gene_id -> {
            "official_symbol": str,
            "synonyms": set of str
        }
    """
    gene_dict = {}
    with gzip.open(gene_info_path, 'rt') as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split('\t')
            # typical columns: tax_id, GeneID, Symbol, LocusTag, Synonyms, ...
            if len(cols) < 6:
                continue
            tid, gene_id, official_symbol, locus_tag, synonyms_str = cols[:5]
            if tid != tax_id:
                continue

            synonyms = set(synonyms_str.split('|')) if synonyms_str else set()

            gene_dict[gene_id] = {
                "official_symbol": official_symbol,
                "synonyms": synonyms
            }
    return gene_dict


def build_synonym_dict(gene_info_path):
    """
    From gene_id -> {offical_symbol, synonyms},
    build a dictionary from uppercase alias -> official symbol,
    avoiding collisions that overwrite official symbols of different genes.
    """
    alias2official = {}
    gene_dict = build_gene_info_dict(gene_info_path)
    # Sort keys so we always process in a consistent order (optional).
    # But for big data, might skip sorting to save time.
    for gid, info in gene_dict.items():
        official_symbol = info["official_symbol"]
        synonyms = info["synonyms"]

        # Combine synonyms and the official symbol itself
        # so that official symbols also map to themselves
        all_aliases = synonyms.union({official_symbol})

        for alias in all_aliases:
            alias_upper = alias.upper()

            # If the alias is already taken as an official symbol for a different gene,
            # check if there's a conflict:
            if alias_upper in alias2official and alias2official[alias_upper] != official_symbol\
                    and alias2official[alias_upper] == alias_upper:
                # We have discovered the same alias points to two distinct official symbols.
                # If the symbol was previously identified as an official symbol, skip overwriting.
                continue
            else:
                alias2official[alias_upper] = official_symbol
    return alias2official

if __name__ == "__main__":
    synonym_dict = build_synonym_dict("/home/techt/Downloads/gene_info.gz")
    with open("../metadata/gene_synonym.json", "w") as f:
        json.dump(synonym_dict, f)