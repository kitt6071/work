import json
import re
from collections import Counter, defaultdict
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import sys # Added to redirect stdout

def parse_iucn_from_object(object_str: str) -> tuple[str, str | None, str | None]:
    """
    Parses the threat description, IUCN code, and IUCN name from the object string.
    Example: "agricultural expansion [IUCN: 2.1 Annual & perennial non-timber crops]"
    Returns: (description, code, name)
    """
    if not isinstance(object_str, str):
        return str(object_str), None, None
    
    match = re.match(r"^(.*?)\s*\[IUCN:\s*([\d\.]+)\s*(.*?)\]$", object_str, re.DOTALL)
    if match:
        description = match.group(1).strip()
        code = match.group(2).strip()
        name = match.group(3).strip()
        return description, code, name if name else code # Return code as name if name is missing
    return object_str.strip(), None, None

def load_and_parse_data(filepath: str) -> pd.DataFrame:
    """Loads enriched_triplets.json and parses it into a Pandas DataFrame."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return pd.DataFrame()

    if 'triplets' not in data or not isinstance(data['triplets'], list):
        print("Error: JSON file must contain a 'triplets' list.")
        return pd.DataFrame()

    parsed_triplets = []
    for triplet in data['triplets']:
        subject = triplet.get('subject')
        predicate = triplet.get('predicate')
        obj_raw = triplet.get('object')
        doi = triplet.get('doi')
        taxonomy = triplet.get('taxonomy', {})

        obj_desc, iucn_code, iucn_name = parse_iucn_from_object(obj_raw)

        parsed_triplets.append({
            'subject': subject,
            'predicate': predicate,
            'object_raw': obj_raw,
            'object_desc': obj_desc,
            'iucn_code': iucn_code,
            'iucn_name': iucn_name if iucn_name and iucn_name != iucn_code else (iucn_code if iucn_code else "Unknown"), # Ensure name is descriptive
            'doi': doi,
            'tax_class': taxonomy.get('class'),
            'tax_order': taxonomy.get('order'),
            'tax_family': taxonomy.get('family'),
            'tax_genus': taxonomy.get('genus'),
            'tax_species': taxonomy.get('scientific_name') or taxonomy.get('species')
        })
    
    df = pd.DataFrame(parsed_triplets)
    # Filter out rows where IUCN code might be None if parsing failed for some
    df = df.dropna(subset=['iucn_code'])
    return df

def analyze_iucn_unique_species(df: pd.DataFrame):
    """Analyzes IUCN categories affecting the most unique species."""
    if df.empty:
        print("No data to analyze for IUCN unique species.")
        return

    print("\n--- IUCN Category vs. Unique Species Affected ---")
    iucn_species_counts = df.groupby('iucn_name')['subject'].nunique().sort_values(ascending=False)
    print("Number of unique species affected by each IUCN category:")
    print(iucn_species_counts)

    if not iucn_species_counts.empty:
        most_impactful_iucn = iucn_species_counts.index[0]
        num_species = iucn_species_counts.iloc[0]
        print(f"\nIUCN category affecting the most unique species: '{most_impactful_iucn}' (affecting {num_species} unique species).")

    # Plotting
    plt.figure(figsize=(12, 8))
    iucn_species_counts.head(20).plot(kind='bar')
    plt.title('Top 20 IUCN Categories by Number of Unique Species Affected')
    plt.xlabel('IUCN Category Name')
    plt.ylabel('Number of Unique Species')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("iucn_unique_species.png")
    print("\nSaved plot to iucn_unique_species.png")
    plt.close()


def analyze_iucn_occurrences_per_species(df: pd.DataFrame):
    """Analyzes which IUCN category affects a singular species the most."""
    if df.empty:
        print("No data to analyze for IUCN occurrences per species.")
        return
        
    print("\n--- IUCN Category Occurrences per Species ---")
    # Count occurrences of each IUCN category for each species
    species_iucn_counts = df.groupby(['subject', 'iucn_name']).size().reset_index(name='threat_count')
    
    # Find the IUCN category with the maximum threat_count for each species
    idx_max_threats = species_iucn_counts.groupby('subject')['threat_count'].idxmax()
    max_threats_per_species = species_iucn_counts.loc[idx_max_threats].sort_values(by='threat_count', ascending=False)

    print("Top species most affected by a single IUCN category (most occurrences):")
    print(max_threats_per_species.head(10))

    if not max_threats_per_species.empty:
        top_species = max_threats_per_species.iloc[0]['subject']
        top_iucn = max_threats_per_species.iloc[0]['iucn_name']
        top_count = max_threats_per_species.iloc[0]['threat_count']
        print(f"\nSpecies most frequently impacted by a single IUCN category: '{top_species}' by '{top_iucn}' ({top_count} occurrences).")

def analyze_iucn_threat_diversity_vs_species_impact(df: pd.DataFrame):
    """
    Analyzes IUCN categories based on the diversity of their threat descriptions
    versus the number of unique species they impact.
    """
    if df.empty:
        print("No data for threat diversity vs. species impact analysis.")
        return

    print("\n--- IUCN Threat Diversity vs. Species Impact ---")
    iucn_analysis = df.groupby('iucn_name').agg(
        unique_threat_descs=pd.NamedAgg(column='object_desc', aggfunc='nunique'),
        unique_species_affected=pd.NamedAgg(column='subject', aggfunc='nunique')
    ).reset_index()

    # High threat diversity, low unique species (species really getting impacted by varied specific threats in this category)
    print("\nIUCN Categories with High Threat Description Diversity and Low Unique Species Impact:")
    high_threat_low_species = iucn_analysis[
        (iucn_analysis['unique_threat_descs'] > iucn_analysis['unique_threat_descs'].median()) &
        (iucn_analysis['unique_species_affected'] < iucn_analysis['unique_species_affected'].median())
    ].sort_values(by=['unique_threat_descs', 'unique_species_affected'], ascending=[False, True])
    print(high_threat_low_species.head(10))

    # Low threat diversity, high unique species (a few specific threats in this category impact many species)
    print("\nIUCN Categories with Low Threat Description Diversity and High Unique Species Impact:")
    low_threat_high_species = iucn_analysis[
        (iucn_analysis['unique_threat_descs'] < iucn_analysis['unique_threat_descs'].median()) &
        (iucn_analysis['unique_species_affected'] > iucn_analysis['unique_species_affected'].median())
    ].sort_values(by=['unique_species_affected', 'unique_threat_descs'], ascending=[False, True])
    print(low_threat_high_species.head(10))

    # Plotting this relationship
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=iucn_analysis, x='unique_threat_descs', y='unique_species_affected', hue='iucn_name', legend=False, size='unique_species_affected', sizes=(50,500))
    plt.title('IUCN Category: Threat Description Diversity vs. Unique Species Affected')
    plt.xlabel('Number of Unique Threat Descriptions')
    plt.ylabel('Number of Unique Species Affected')
    # Annotate some points (optional, can get crowded)
    # for i, row in iucn_analysis.sample(min(5, len(iucn_analysis))).iterrows(): # Sample a few points
    #     plt.text(row['unique_threat_descs'] + 0.1, row['unique_species_affected'] + 0.1, row['iucn_name'], fontsize=8)
    plt.tight_layout()
    plt.savefig("iucn_diversity_vs_impact.png")
    print("\nSaved plot to iucn_diversity_vs_impact.png")
    plt.close()


def novel_analyses(df: pd.DataFrame):
    """Performs and prints results for several novel analyses."""
    if df.empty:
        print("No data for novel analyses.")
        return

    print("\n--- Novel Analyses ---")

    # 1. Most Common Predicates (Impact Mechanisms)
    print("\n1. Most Common Impact Mechanisms (Predicates):")
    predicate_counts = df['predicate'].value_counts()
    print(predicate_counts.head(10))

    # 2. Species with Most Diverse Threats (IUCN Categories)
    print("\n2. Species Facing Most Diverse IUCN Threat Categories:")
    species_threat_diversity = df.groupby('subject')['iucn_code'].nunique().sort_values(ascending=False)
    print(species_threat_diversity.head(10))

    # 3. Threat Descriptions Impacting Most Species
    print("\n3. Specific Threat Descriptions Impacting Most Unique Species:")
    threat_desc_impact = df.groupby('object_desc')['subject'].nunique().sort_values(ascending=False)
    print(threat_desc_impact.head(10))

    # 4. IUCN Category Co-occurrence on Species
    print("\n4. IUCN Category Co-occurrence (Top Pairs):")
    # This is a bit more complex; for simplicity, we'll show species with multiple IUCN categories
    # A full co-occurrence matrix would be more involved.
    species_iucn_list = df.groupby('subject')['iucn_name'].apply(lambda x: list(set(x))).reset_index()
    species_iucn_list['num_iucn_categories'] = species_iucn_list['iucn_name'].apply(len)
    print("Species with multiple co-occurring IUCN threat categories:")
    print(species_iucn_list[species_iucn_list['num_iucn_categories'] > 1].sort_values('num_iucn_categories', ascending=False).head(5))
    
    # Simple pairwise co-occurrence count
    iucn_pairs = defaultdict(int)
    for _, group in df.groupby('subject'):
        unique_iucn_in_group = sorted(list(set(group['iucn_name'])))
        for i in range(len(unique_iucn_in_group)):
            for j in range(i + 1, len(unique_iucn_in_group)):
                pair = tuple(sorted((unique_iucn_in_group[i], unique_iucn_in_group[j])))
                iucn_pairs[pair] += 1
    sorted_iucn_pairs = sorted(iucn_pairs.items(), key=lambda item: item[1], reverse=True)
    print("Top co-occurring IUCN category pairs across species:")
    for pair, count in sorted_iucn_pairs[:5]:
        print(f"  {pair}: {count} species")


    # 5. Taxonomic Pattern of Threats (Focus on Order)
    print("\n5. Taxonomic Pattern: Threats per Bird Order (Top IUCN Categories):")
    if 'tax_order' in df.columns and not df['tax_order'].isnull().all():
        order_threats = df.groupby(['tax_order', 'iucn_name']).size().reset_index(name='count')
        top_order_threats = order_threats.sort_values(by=['tax_order', 'count'], ascending=[True, False])
        # Show top threat for a few orders
        for order, group in top_order_threats.groupby('tax_order'):
            if order and len(group) > 0 : # Check if order is not None or empty
                 print(f"  Order '{order}': Top threat is '{group.iloc[0]['iucn_name']}' ({group.iloc[0]['count']} times)")
    else:
        print("  Taxonomic order data not available or insufficient for this analysis.")

    # 6. Predicate-IUCN Category Affinity
    print("\n6. Predicate-IUCN Category Affinity (Top Combinations):")
    predicate_iucn_affinity = df.groupby(['predicate', 'iucn_name']).size().reset_index(name='count').sort_values('count', ascending=False)
    print(predicate_iucn_affinity.head(10))

    # 7. DOI Insights
    print("\n7. DOI Insights:")
    doi_insights = df.groupby('doi').agg(
        total_triplets=pd.NamedAgg(column='subject', aggfunc='size'),
        unique_species=pd.NamedAgg(column='subject', aggfunc='nunique'),
        unique_iucn_categories=pd.NamedAgg(column='iucn_code', aggfunc='nunique')
    ).sort_values('total_triplets', ascending=False)
    print("DOIs contributing most triplets:")
    print(doi_insights.head(5))

    # 8. Distribution of Predicate Lengths per IUCN Category
    print("\n8. Predicate Detail (Length) per IUCN Category (Median):")
    df['predicate_length'] = df['predicate'].astype(str).apply(len)
    predicate_length_per_iucn = df.groupby('iucn_name')['predicate_length'].median().sort_values(ascending=False)
    print(predicate_length_per_iucn.head(10))
    
    # 9. Bipartite Graph Analysis (Conceptual - showing data for it)
    print("\n9. Network Perspective (Data for Species-IUCN Category Graph):")
    print("  - Number of unique species (nodes):", df['subject'].nunique())
    print("  - Number of unique IUCN categories (nodes):", df['iucn_name'].nunique())
    print("  - Number of connections (edges):", len(df[['subject', 'iucn_name']].drop_duplicates()))
    # Degree of IUCN categories (already calculated in analyze_iucn_unique_species)
    iucn_degrees = df.groupby('iucn_name')['subject'].nunique().sort_values(ascending=False)
    print("  Top 5 IUCN Categories by number of species connected:")
    print(iucn_degrees.head())
    # Degree of species (already calculated in species_threat_diversity)
    species_degrees = df.groupby('subject')['iucn_name'].nunique().sort_values(ascending=False)
    print("  Top 5 Species by number of IUCN Categories connected:")
    print(species_degrees.head())


    # 10. Threat Hotspots within IUCN Categories
    print("\n10. Threat Hotspots (Most common threat descriptions within top IUCN categories):")
    top_iucn_categories = df['iucn_name'].value_counts().nlargest(3).index.tolist()
    for iucn_cat in top_iucn_categories:
        print(f"  Hotspots for IUCN Category '{iucn_cat}':")
        hotspots = df[df['iucn_name'] == iucn_cat]['object_desc'].value_counts().nlargest(3)
        for desc, count in hotspots.items():
            print(f"    - '{desc}': {count} times")

    # 11. Predicate Diversity per IUCN Category
    print("\n11. Predicate Diversity per IUCN Category:")
    predicate_diversity_iucn = df.groupby('iucn_name')['predicate'].nunique().sort_values(ascending=False)
    print(predicate_diversity_iucn.head(10))

    # 12. Average Number of Threats per Species
    print("\n12. Average Number of Threats (IUCN categories) per Species:")
    avg_threats_per_species = df.groupby('subject')['iucn_code'].nunique().mean()
    print(f"  Overall average: {avg_threats_per_species:.2f} unique IUCN categories per species.")
    if 'tax_order' in df.columns and not df['tax_order'].isnull().all():
        avg_threats_per_order = df.groupby(['tax_order', 'subject'])['iucn_code'].nunique().reset_index().groupby('tax_order')['iucn_code'].mean()
        print("  Average unique IUCN categories per species by Order:")
        print(avg_threats_per_order.sort_values(ascending=False).head())


def main():
    parser = argparse.ArgumentParser(description="Analyze enriched triplet data from a JSON file.")
    parser.add_argument("json_file", help="Path to the enriched_triplets.json file.")
    parser.add_argument("--output_file", help="Optional path to save the console output.", default="analysis_results.txt") # Added output_file argument
    args = parser.parse_args()

    # Redirect stdout to a file
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(args.output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f  # Change the standard output to the file we created.

        df = load_and_parse_data(args.json_file)

        if df.empty:
            print("Exiting due to data loading issues.")
            sys.stdout = original_stdout # Restore stdout before exiting
            return

        print(f"Loaded {len(df)} triplets for analysis.")
        print("Columns:", df.columns.tolist())
        print("Sample data:")
        print(df.head())

        # Run core analyses
        analyze_iucn_unique_species(df)
        analyze_iucn_occurrences_per_species(df)
        analyze_iucn_threat_diversity_vs_species_impact(df)

        # Run novel analyses
        novel_analyses(df)
        
        print("\n--- Analysis Complete ---")
        print("Consider the generated PNG files for visualizations.")
        print("The 'shape of things' (distributions, network properties) indeed matters and is reflected in these analyses.")
        print("For example, skewed distributions in threats per species or IUCN category prevalence highlight common vs. rare issues.")
        print("Network perspectives (like species connected to many IUCN types, or IUCN types connected to many species) identify critical nodes/categories.")
        
        sys.stdout = original_stdout # Restore stdout
    
    print(f"Analysis output saved to {args.output_file}")
    print(f"Plots saved to iucn_unique_species.png and iucn_diversity_vs_impact.png")


if __name__ == "__main__":
    main() 