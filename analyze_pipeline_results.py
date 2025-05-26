import json
import pickle
import os
import matplotlib.pyplot as plt
import networkx as nx
import re
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict

def analyze_pipeline_results():
    # Base directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = Path(current_dir)
    
    # Paths to result files
    results_dir = base_dir / "Lent_Init" / "results"
    cache_dir = base_dir / "Lent_Init" / "cache"
    models_dir = base_dir / "Lent_Init" / "models"
    
    # 1. Analyze classifier models
    print("\nCLASSIFIER ANALYSIS:")
    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
    classifier_path = models_dir / "relevance_classifier.pkl"
    
    if vectorizer_path.exists() and classifier_path.exists():
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(classifier_path, 'rb') as f:
            classifier = pickle.load(f)
            
        # Get feature importance
        feature_importance = classifier.coef_[0]
        feature_names = vectorizer.get_feature_names_out()
        
        # Top and bottom features
        top_indices = feature_importance.argsort()[-20:]
        bottom_indices = feature_importance.argsort()[:20]
        
        print("Top Relevance Features (words that indicate relevance):")
        for idx in top_indices[::-1]:
            print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")
            
        print("\nTop Irrelevance Features (words that indicate irrelevance):")
        for idx in bottom_indices:
            print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")
            
        print(f"\nTotal vocabulary size: {len(feature_names)}")
        print(f"Classifier intercept: {classifier.intercept_[0]:.4f}")
        print(f"Classifier classes: {classifier.classes_}")
    else:
        print("Classifier models not found.")
    
    # 2. Analyze enriched triplets
    print("\nTRIPLET ANALYSIS:")
    triplets_path = results_dir / "enriched_triplets.json"
    
    if triplets_path.exists():
        with open(triplets_path, 'r') as f:
            triplets_data = json.load(f)
            
        triplets = triplets_data.get('triplets', [])
        taxonomic_info = triplets_data.get('taxonomic_info', {})
        
        # Statistics
        print(f"Total triplets: {len(triplets)}")
        
        # Unique counts
        subjects = [t['subject'] for t in triplets]
        predicates = [t['predicate'] for t in triplets]
        objects = [t['object'] for t in triplets]
        
        unique_subjects = set(subjects)
        unique_predicates = set(predicates)
        unique_objects = set(objects)
        
        print(f"Unique species (subjects): {len(unique_subjects)}")
        print(f"Unique impact mechanisms (predicates): {len(unique_predicates)}")
        print(f"Unique threats (objects): {len(unique_objects)}")
        
        # Most common subjects, predicates and objects
        print("\nTop 5 species:")
        for subject, count in Counter(subjects).most_common(5):
            print(f"  {subject}: {count} triplets")
            
        print("\nTop 5 impact mechanisms:")
        for predicate, count in Counter(predicates).most_common(5):
            print(f"  {predicate}: {count} triplets")
            
        # IUCN Analysis
        print("\nIUCN CLASSIFICATION ANALYSIS:")
        iucn_codes = []
        for obj in objects:
            match = re.search(r'\[IUCN:\s*([\d\.]+)', obj)
            if match:
                iucn_codes.append(match.group(1))
                
        if iucn_codes:
            # Count by top-level and specific codes
            top_level_codes = [code.split('.')[0] for code in iucn_codes]
            top_level_count = Counter(top_level_codes)
            specific_count = Counter(iucn_codes)
            
            print("IUCN Distribution by top-level category:")
            for code, count in sorted(top_level_count.items()):
                print(f"  {code}: {count} ({count/len(iucn_codes)*100:.1f}%)")
                
            print("\nTop 10 specific IUCN codes:")
            for code, count in specific_count.most_common(10):
                print(f"  {code}: {count} ({count/len(iucn_codes)*100:.1f}%)")
                
            # Checking for fallbacks
            fallback_count = specific_count.get("12.1", 0)
            print(f"\nFallback classifications ('12.1 Other threat'): {fallback_count} ({fallback_count/len(iucn_codes)*100:.1f}%)")
            
            # Visualize IUCN distribution
            plt.figure(figsize=(12, 6))
            plt.bar(top_level_count.keys(), top_level_count.values())
            plt.title("Distribution of IUCN Top-Level Categories")
            plt.xlabel("IUCN Category")
            plt.ylabel("Count")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(results_dir / "iucn_distribution.png", dpi=300)
            print(f"Saved IUCN distribution chart to {results_dir/'iucn_distribution.png'}")
        else:
            print("No IUCN codes found in triplets.")
            
        # Taxonomic Analysis
        print("\nTAXONOMIC ANALYSIS:")
        taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        level_values = defaultdict(set)
        
        # Count subjects with taxonomic data
        subjects_with_taxonomy = [s for s in unique_subjects if s in taxonomic_info and taxonomic_info[s]]
        
        print(f"Species with taxonomic data: {len(subjects_with_taxonomy)}/{len(unique_subjects)} ({len(subjects_with_taxonomy)/len(unique_subjects)*100:.1f}% if unique_subjects else 0.0%)")

        # Identify and list species without taxonomic data
        subjects_without_taxonomy = []
        for subject_name in unique_subjects:
            if subject_name not in taxonomic_info or not taxonomic_info[subject_name]:
                # Check for potential specific failure markers if your enrichment functions use them,
                # e.g., if taxonomic_info[subject_name] could be an empty dict or a dict with an error flag.
                # For now, just checking for presence and non-empty value.
                subjects_without_taxonomy.append(subject_name)
        
        if subjects_without_taxonomy:
            print(f"\nSpecies for which taxonomic data was NOT found ({len(subjects_without_taxonomy)} species):")
            for i, species_name in enumerate(subjects_without_taxonomy):
                print(f"  {i+1}. {species_name}")
                if i >= 19: # Print a sample if the list is too long
                    print(f"  ... and {len(subjects_without_taxonomy) - 20} more.")
                    break
        else:
            print("All species subjects have corresponding taxonomic information entries.")

        # Extract taxonomic hierarchy
        for subject in subjects_with_taxonomy:
            taxonomy = taxonomic_info.get(subject, {})
            if taxonomy:
                for level in taxonomy_levels:
                    if taxonomy.get(level):
                        level_values[level].add(taxonomy[level])
        
        # Print hierarchical counts
        for level in taxonomy_levels:
            if level in level_values:
                print(f"Unique {level}: {len(level_values[level])}")
                if level == 'class' and 'aves' in [v.lower() for v in level_values[level]]:
                    print("  ✓ Class Aves (birds) confirmed in data")
            
        # Generate graph for visualizing species-threat relationships
        print("\nGRAPH METRICS:")
        G = nx.DiGraph()
        for triplet in triplets:
            subject = triplet['subject']
            predicate = triplet['predicate']
            obj = triplet['object']
            
            # Extract base object name without IUCN tag
            obj_name = re.sub(r'\s*\[IUCN:.*?\]$', '', obj)
            
            G.add_node(subject, type='species')
            G.add_node(obj_name, type='threat')
            G.add_edge(subject, obj_name, relation=predicate)
            
        # Graph statistics
        print(f"Graph nodes: {G.number_of_nodes()}")
        print(f"Graph edges: {G.number_of_edges()}")
        
        # Hub nodes (most connected)
        species_nodes = [n for n, data in G.nodes(data=True) if data.get('type') == 'species']
        threat_nodes = [n for n, data in G.nodes(data=True) if data.get('type') == 'threat']
        
        species_degrees = {node: G.degree(node) for node in species_nodes}
        threat_degrees = {node: G.degree(node) for node in threat_nodes}
        
        print("\nTop 5 hub species (most connections):")
        for species, degree in sorted(species_degrees.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {species}: {degree} connections")
            
        print("\nTop 5 hub threats (most connections):")
        for threat, degree in sorted(threat_degrees.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {threat}: {degree} connections")
            
        # Calculate average connection statistics
        avg_species_connections = sum(species_degrees.values()) / len(species_degrees) if species_degrees else 0
        avg_threat_connections = sum(threat_degrees.values()) / len(threat_degrees) if threat_degrees else 0
        
        print(f"\nAverage connections per species: {avg_species_connections:.2f}")
        print(f"Average connections per threat: {avg_threat_connections:.2f}")
    else:
        print("Enriched triplets file not found.")
        
    # 3. Analyze Wikispecies data
    print("\nWIKISPECIES ANALYSIS:")
    wikispecies_path = results_dir / "wikispecies_results.json"
    
    if wikispecies_path.exists():
        with open(wikispecies_path, 'r') as f:
            wikispecies_data = json.load(f)
            
        total_queries = len(wikispecies_data)
        successful_queries = len([q for q in wikispecies_data if q.get('page_id')])
        
        print(f"Total Wikispecies queries: {total_queries}")
        print(f"Successful lookups: {successful_queries} ({successful_queries/total_queries*100:.1f}%)")
        print(f"Failed lookups: {total_queries - successful_queries} ({(total_queries - successful_queries)/total_queries*100:.1f}%)")
    else:
        print("Wikispecies results file not found.")

if __name__ == "__main__":
    analyze_pipeline_results()