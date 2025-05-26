import json
import re
import pandas as pd
import argparse
import networkx as nx
from networkx.algorithms import community as nx_comm
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Attempt to import optional libraries
try:
    import community as community_louvain  # For Louvain community detection
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("Warning: python-louvain library not found. Louvain community detection will not be available.")
    print("Install with: pip install python-louvain")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers or sklearn not found. Semantic similarity analysis will not be available.")
    print("Install with: pip install sentence-transformers scikit-learn")


def parse_iucn_from_object(object_str: str) -> tuple[str, str | None, str | None]:
    if not isinstance(object_str, str):
        return str(object_str), None, None
    match = re.match(r"^(.*?)\s*\[IUCN:\s*([\d\.]+)\s*(.*?)\]$", object_str, re.DOTALL)
    if match:
        description = match.group(1).strip()
        code = match.group(2).strip()
        name = match.group(3).strip()
        return description, code, name if name else code
    return object_str.strip(), None, None

def load_and_parse_data(filepath: str) -> pd.DataFrame:
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
        obj_raw = triplet.get('object')
        taxonomy = triplet.get('taxonomy', {})
        obj_desc, iucn_code, iucn_name = parse_iucn_from_object(obj_raw)
        publication_year = triplet.get('publication_year') # Attempt to get year

        parsed_triplets.append({
            'subject': subject,
            'object_raw': obj_raw,
            'object_desc': obj_desc,
            'iucn_code': iucn_code,
            'iucn_name': iucn_name if iucn_name and iucn_name != iucn_code else (iucn_code if iucn_code else "Unknown"),
            'tax_order': taxonomy.get('order'),
            'publication_year': publication_year
        })
    df = pd.DataFrame(parsed_triplets)
    df = df.dropna(subset=['subject', 'iucn_name'])
    return df

def build_species_threat_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph() # Using Graph for Louvain, which works better on undirected
    species_nodes = []
    threat_nodes = []
    edges = []

    for _, row in df.iterrows():
        species = f"SPECIES_{row['subject']}" # Prefix to distinguish node types
        threat = f"THREAT_{row['iucn_name']}"
        species_nodes.append(species)
        threat_nodes.append(threat)
        edges.append((species, threat))
    
    G.add_nodes_from(list(set(species_nodes)), bipartite=0, type='species')
    G.add_nodes_from(list(set(threat_nodes)), bipartite=1, type='threat')
    G.add_edges_from(list(set(edges)))
    return G

def analyze_community_structure(G: nx.Graph):
    if not LOUVAIN_AVAILABLE:
        print("Louvain community detection skipped as library is not available.")
        return
    if G.number_of_nodes() == 0:
        print("Graph is empty, skipping community detection.")
        return

    print("\n--- Community Detection (Louvain) ---")
    try:
        partition = community_louvain.best_partition(G)
        print(f"Found {len(set(partition.values()))} communities.")

        # Summarize top N communities
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)

        for i, (comm_id, nodes) in enumerate(communities.items()):
            if i >= 5 and len(communities) > 10: # Limit printing for very many communities
                print(f"... and {len(communities) - 5} more communities.")
                break
            species_in_comm = [n.replace("SPECIES_", "") for n in nodes if n.startswith("SPECIES_")]
            threats_in_comm = [n.replace("THREAT_", "") for n in nodes if n.startswith("THREAT_")]
            print(f"\nCommunity {comm_id} (Size: {len(nodes)} nodes):")
            print(f"  Top 5 Species: {species_in_comm[:5]}")
            print(f"  Top 5 Threats: {threats_in_comm[:5]}")
            if len(species_in_comm) > 5:
                print(f"    ... and {len(species_in_comm) - 5} more species.")
            if len(threats_in_comm) > 5:
                 print(f"    ... and {len(threats_in_comm) - 5} more threats.")
        # You could save partition to a file or return it for further use.
    except Exception as e:
        print(f"Error during Louvain community detection: {e}")

def analyze_centrality_measures(G: nx.Graph):
    if G.number_of_nodes() == 0:
        print("Graph is empty, skipping centrality analysis.")
        return

    print("\n--- Centrality Measures ---")
    
    # Degree Centrality
    degree_centrality = nx.degree_centrality(G)
    sorted_degree = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)
    print("\nTop 10 Nodes by Degree Centrality:")
    for node, centrality in sorted_degree[:10]:
        print(f"  {node}: {centrality:.4f}")

    # Betweenness Centrality
    # Note: Betweenness can be computationally expensive for large graphs
    print("\nCalculating Betweenness Centrality (may take time for large graphs)...")
    try:
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes())) # Use k for approximation if large
        sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)
        print("Top 10 Nodes by Betweenness Centrality:")
        for node, centrality in sorted_betweenness[:10]:
            print(f"  {node}: {centrality:.4f}")
    except Exception as e:
        print(f"Error calculating Betweenness Centrality: {e}")

    # Closeness Centrality
    print("\nCalculating Closeness Centrality...")
    try:
        closeness_centrality = nx.closeness_centrality(G)
        sorted_closeness = sorted(closeness_centrality.items(), key=lambda item: item[1], reverse=True)
        print("Top 10 Nodes by Closeness Centrality:")
        for node, centrality in sorted_closeness[:10]:
            print(f"  {node}: {centrality:.4f}")
    except Exception as e:
        print(f"Error calculating Closeness Centrality: {e}")

    # Eigenvector Centrality
    print("\nCalculating Eigenvector Centrality...")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-03) # Added max_iter and tol for convergence
        sorted_eigenvector = sorted(eigenvector_centrality.items(), key=lambda item: item[1], reverse=True)
        print("Top 10 Nodes by Eigenvector Centrality:")
        for node, centrality in sorted_eigenvector[:10]:
            print(f"  {node}: {centrality:.4f}")
    except Exception as e: # Catching generic Exception, can be refined to specific NetworkX errors
        print(f"Error calculating Eigenvector Centrality: {e}. It might not have converged.")

def analyze_semantic_similarity_threats(df: pd.DataFrame, top_n=5):
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Semantic similarity analysis skipped as sentence-transformers library is not available.")
        return
    if df.empty or 'object_desc' not in df.columns:
        print("No object descriptions found for semantic similarity analysis.")
        return

    print("\n--- Semantic Similarity of Threat Descriptions (Illustrative) ---")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2') # A lightweight model
        unique_threat_descs = df['object_desc'].dropna().unique().tolist()
        
        if len(unique_threat_descs) < 2:
            print("Not enough unique threat descriptions to compare.")
            return
            
        print(f"Generating embeddings for {len(unique_threat_descs)} unique threat descriptions...")
        embeddings = model.encode(unique_threat_descs, show_progress_bar=True)
        
        print(f"Calculating cosine similarity matrix ({len(unique_threat_descs)}x{len(unique_threat_descs)})...")
        similarity_matrix = cosine_similarity(embeddings)
        
        print(f"\nTop {top_n} most similar pairs of distinct threat descriptions:")
        
        # Find top N distinct pairs
        processed_pairs = set()
        similar_pairs = []

        for i in range(len(unique_threat_descs)):
            for j in range(i + 1, len(unique_threat_descs)):
                # Create a canonical representation of the pair to avoid duplicates like (A,B) and (B,A)
                pair_key = tuple(sorted((unique_threat_descs[i], unique_threat_descs[j])))
                if unique_threat_descs[i] != unique_threat_descs[j] and pair_key not in processed_pairs:
                    similar_pairs.append(((unique_threat_descs[i], unique_threat_descs[j]), similarity_matrix[i,j]))
                    processed_pairs.add(pair_key)
        
        # Sort by similarity score in descending order
        similar_pairs.sort(key=lambda x: x[1], reverse=True)

        for k, ((desc1, desc2), score) in enumerate(similar_pairs):
            if k >= top_n:
                break
            # Get original IUCN codes/names for context
            iucn1 = df[df['object_desc'] == desc1][['iucn_code', 'iucn_name']].iloc[0]
            iucn2 = df[df['object_desc'] == desc2][['iucn_code', 'iucn_name']].iloc[0]
            print(f"  Pair {k+1} (Similarity: {score:.4f}):")
            print(f"    Threat 1: '{desc1}' (IUCN: {iucn1['iucn_code']} - {iucn1['iucn_name']})")
            print(f"    Threat 2: '{desc2}' (IUCN: {iucn2['iucn_code']} - {iucn2['iucn_name']})")
            print("-" * 20)
            
    except Exception as e:
        print(f"Error during semantic similarity analysis: {e}")

def analyze_temporal_trends(df: pd.DataFrame):
    print("\n--- Temporal Analysis (Placeholder) ---")
    if 'publication_year' not in df.columns or df['publication_year'].isnull().all():
        print("  'publication_year' column not found or all values are null in the input data.")
        print("  To perform temporal analysis, ensure your enriched_triplets.json includes a 'publication_year' field for each triplet.")
        print("  This field could be extracted from DOIs using external APIs like CrossRef if not directly available.")
        return

    # Ensure year is numeric and drop NaNs for plotting
    df_temporal = df.copy()
    df_temporal['publication_year'] = pd.to_numeric(df_temporal['publication_year'], errors='coerce')
    df_temporal = df_temporal.dropna(subset=['publication_year'])
    df_temporal['publication_year'] = df_temporal['publication_year'].astype(int)

    if df_temporal.empty:
        print("  No valid publication year data remaining after cleaning.")
        return

    print("  (Example: Plotting total threat mentions over time if 'publication_year' was available)")
    
    # Overall threat reporting
    threats_over_time = df_temporal.groupby('publication_year').size()
    
    if threats_over_time.empty or len(threats_over_time) < 2: # Need at least 2 points to plot a line
        print("  Not enough data points to plot overall threat mentions over time.")
    else:
        plt.figure(figsize=(12, 6))
        threats_over_time.plot(kind='line', marker='o')
        plt.title('Total Threat Mentions Over Time')
        plt.xlabel('Publication Year')
        plt.ylabel('Number of Threat Mentions (Triplets)')
        plt.tight_layout()
        plt.savefig("adv_threats_over_time_total.png")
        print("\n  Saved example plot to adv_threats_over_time_total.png")
        plt.close()

    # Example: Reporting of a specific IUCN category over time
    top_iucn_category = df_temporal['iucn_name'].mode()
    if not top_iucn_category.empty:
        specific_iucn_name = top_iucn_category[0]
        specific_iucn_over_time = df_temporal[df_temporal['iucn_name'] == specific_iucn_name].groupby('publication_year').size()
        if not specific_iucn_over_time.empty and len(specific_iucn_over_time) >= 2:
            plt.figure(figsize=(12, 6))
            specific_iucn_over_time.plot(kind='line', marker='o', color='orange')
            plt.title(f'Mentions of IUCN: "{specific_iucn_name}" Over Time')
            plt.xlabel('Publication Year')
            plt.ylabel('Number of Mentions')
            plt.tight_layout()
            plt.savefig(f"adv_iucn_{specific_iucn_name.replace(' ', '_').replace('/', '_')}_over_time.png")
            print(f"\n  Saved example plot for {specific_iucn_name} to adv_iucn_{specific_iucn_name.replace(' ', '_').replace('/', '_')}_over_time.png")
            plt.close()
        else:
            print(f"  Not enough data points to plot for IUCN category '{specific_iucn_name}'.")
    else:
        print("  Could not determine a top IUCN category for temporal trend example.")

def main():
    parser = argparse.ArgumentParser(description="Advanced analysis of enriched triplet data.")
    parser.add_argument("json_file", help="Path to the enriched_triplets.json file.")
    parser.add_argument("--output_file", help="Optional path to save the console output.", default="advanced_analysis_results.txt")
    args = parser.parse_args()

    original_stdout = sys.stdout
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        sys.stdout = f_out # Redirect stdout

        df = load_and_parse_data(args.json_file)
        if df.empty:
            print("Exiting due to data loading issues.")
            sys.stdout = original_stdout
            return

        print(f"Loaded {len(df)} triplets for advanced analysis.")
        print(df.head())

        # Build graph for network analyses
        G = build_species_threat_graph(df)
        print(f"\nBuilt graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        # Run advanced analyses
        analyze_community_structure(G)
        analyze_centrality_measures(G)
        analyze_semantic_similarity_threats(df, top_n=10) # Show top 10 similar pairs
        analyze_temporal_trends(df)

        print("\n--- Advanced Analysis Complete ---")
        sys.stdout = original_stdout # Restore stdout
    
    print(f"Advanced analysis output saved to {args.output_file}")
    print("Any generated plots have also been saved as PNG files in the current directory.")

if __name__ == "__main__":
    import sys # Ensure sys is imported for __main__ scope if not already top-level for stdout redirection
    main() 