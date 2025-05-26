import json
import pickle
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import re
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain
import matplotlib.cm as cm
import warnings
import logging
from typing import Optional, List

# Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class GraphAnalyzer:
    """
    A class for analyzing and visualizing knowledge graphs with advanced methods
    based on the paper 'Accelerating Scientific Discovery with Generative Knowledge 
    Extraction, Graph-Based Representation, and Multimodal Intelligent Graph Reasoning'.
    """
    
    def __init__(self, base_dir=None, results_dir=None, embedding_model_name='all-MiniLM-L6-v2'):
        """Initialize the analyzer with paths to required files."""
        if results_dir is not None:
            # Use the provided results directory
            self.results_dir = Path(results_dir)
            self.base_dir = self.results_dir.parent  # Parent of results dir
            self.figures_dir = self.results_dir / "figures"
            self.models_dir = self.base_dir / "models"  # Assuming models are at the same level as results
        else:
            # Use the old logic
            if base_dir is None:
                self.current_dir = os.path.dirname(os.path.abspath(__file__))
                self.base_dir = Path(self.current_dir)
            else:
                self.base_dir = Path(base_dir)
                
            # Define paths
            self.results_dir = self.base_dir / "Lent_Init" / "results"
            self.figures_dir = self.results_dir / "figures"
            self.models_dir = self.base_dir / "Lent_Init" / "models"
        
        # Create figures directory if it doesn't exist
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging() # Call setup logging here
        
        # Initialize graph
        self.graph = None
        self.embeddings = None
        self.threat_consolidation_map = {}
        
        # Initialize embedding model
        self.embedding_model = None
        try:
            logger.info(f"Initializing SentenceTransformer model: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info("SentenceTransformer model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer model: {e}. Some functionalities might be limited.")

        logger.info(f"GraphAnalyzer initialized with base directory: {self.base_dir}")
    
    def _setup_logging(self):
        """Sets up file and console logging for the analyzer."""
        logger.setLevel(logging.INFO)
        # Clear existing handlers to avoid duplicate logs if re-initialized
        if logger.hasHandlers():
            logger.handlers.clear()

        # File Handler for detailed logging
        log_file_path = self.results_dir / "graph_analysis_run.log"
        fh = logging.FileHandler(log_file_path, mode='w') # 'w' to overwrite each run
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(module)s - %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

        # Console Handler for general progress
        ch = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)

        logger.info(f"Logging initialized. Detailed logs will be written to: {log_file_path}")

    def _parse_iucn_tagged_string(self, object_str: str) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
        """
        Parses an IUCN-tagged string into its description and IUCN components.
        Returns: (full_original_string, description, iucn_code, iucn_name)
        If no IUCN tag, description is the full string, and code/name are None.
        """
        if not isinstance(object_str, str):
            s = str(object_str)
            return s, s, None, None # Full string as description if not a string

        pattern = r"^(.*?)\s*\[IUCN:\s*([\d\.]+)\s*(.*?)\]$" # Adjusted regex
        match = re.match(pattern, object_str, re.DOTALL)
        if match:
            description = match.group(1).strip()
            code = match.group(2).strip()
            name = match.group(3).strip()
            if re.match(r"^\d+(\.\d+)?$", code):
                return object_str, description, code, name if name else None
            else:
                # Invalid IUCN code format, treat as plain description
                return object_str, object_str.strip(), None, None
        else:
            # No IUCN tag found
            return object_str, object_str.strip(), None, None
    
    def _consolidate_threat_nodes(self, all_original_threat_strings: List[str], similarity_threshold=0.90):
        """
        Consolidates threat nodes based on semantic similarity of their descriptions.

        Args:
            all_original_threat_strings: A list of unique full threat strings (including IUCN tags).
            similarity_threshold: The cosine similarity threshold for grouping threats.

        Returns:
            A dictionary mapping original full threat strings to their canonical full threat string.
        """
        if not self.embedding_model:
            logger.warning("Embedding model not initialized. Skipping threat consolidation.")
            return {threat_str: threat_str for threat_str in all_original_threat_strings}

        if not all_original_threat_strings:
            logger.info("No threat strings to consolidate.")
            return {}

        logger.info(f"Starting threat consolidation for {len(all_original_threat_strings)} unique threat strings.")

        parsed_threats = [] # List of (original_full_string, clean_description)
        for original_str in all_original_threat_strings:
            _, clean_desc, _, _ = self._parse_iucn_tagged_string(original_str)
            if clean_desc: # Ensure we have a description to embed
                 parsed_threats.append((original_str, clean_desc))

        if not parsed_threats:
            logger.info("No valid threat descriptions found after parsing.")
            return {threat_str: threat_str for threat_str in all_original_threat_strings}

        original_strings_list = [pt[0] for pt in parsed_threats]
        clean_descriptions_list = [pt[1] for pt in parsed_threats]

        logger.info(f"Generating embeddings for {len(clean_descriptions_list)} clean threat descriptions...")
        embeddings = self.embedding_model.encode(clean_descriptions_list, show_progress_bar=True)

        # Build a similarity graph
        similarity_graph = nx.Graph()
        for i in range(len(clean_descriptions_list)):
            similarity_graph.add_node(i) # Node is index in clean_descriptions_list

        logger.info("Calculating similarity matrix and building similarity graph...")
        for i in range(len(clean_descriptions_list)):
            for j in range(i + 1, len(clean_descriptions_list)):
                sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
                if sim >= similarity_threshold:
                    similarity_graph.add_edge(i, j)

        # Find connected components (groups of similar threats)
        consolidation_map = {}
        processed_indices = set()
        
        logger.info("Identifying consolidation groups...")
        for component_indices in nx.connected_components(similarity_graph):
            if not component_indices:
                continue

            group_original_strings = [original_strings_list[i] for i in component_indices]
            group_clean_descriptions = [clean_descriptions_list[i] for i in component_indices]

            # Choose canonical representative: shortest clean description, then its original full string
            # This prioritizes conciseness while retaining original IUCN tagging for the canonical form.
            canonical_clean_desc = min(group_clean_descriptions, key=len)
            # Find the original full string that corresponds to this canonical clean description
            canonical_full_string = ""
            for i in component_indices:
                if clean_descriptions_list[i] == canonical_clean_desc:
                    canonical_full_string = original_strings_list[i]
                    break
            
            if not canonical_full_string: # Should not happen if logic is correct
                canonical_full_string = group_original_strings[0] # Fallback
                logger.warning(f"Fallback canonical representative for group: {group_clean_descriptions}")

            logger.info(f"  Consolidation Group Identified:")
            logger.info(f"    Canonical Full String: \"{canonical_full_string}\"")
            logger.info(f"    Represents the following {len(group_original_strings)} original strings (due to semantic similarity > {similarity_threshold}):")
            for original_str_in_group in group_original_strings:
                consolidation_map[original_str_in_group] = canonical_full_string
                processed_indices.add(original_strings_list.index(original_str_in_group))
                logger.info(f"      - Original: \"{original_str_in_group}\"")

        # Handle threats not in any group (singletons)
        for i, original_str in enumerate(original_strings_list):
            if i not in processed_indices:
                consolidation_map[original_str] = original_str
        
        # Ensure all input original_full_threat_strings have a mapping (even if to themselves)
        for original_threat_str in all_original_threat_strings:
            if original_threat_str not in consolidation_map:
                consolidation_map[original_threat_str] = original_threat_str
                logger.debug(f"  Adding singleton mapping for: {self._parse_iucn_tagged_string(original_threat_str)[1][:50]}...")

        logger.info(f"Threat consolidation complete. Mapped {len(all_original_threat_strings)} to {len(set(consolidation_map.values()))} canonical threats.")
        return consolidation_map
    
    def load_graph_from_triplets(self, triplets_path=None):
        """
        Load a graph from triplets file.
        
        Args:
            triplets_path: Path to the triplets JSON file
        
        Returns:
            NetworkX DiGraph
        """
        if triplets_path is None:
            triplets_path = self.results_dir / "enriched_triplets.json"
        
        logger.info(f"Loading graph from triplets: {triplets_path}")
        
        if not triplets_path.exists():
            logger.error(f"Triplets file not found at {triplets_path}")
            return None
        
        try:
            with open(triplets_path, 'r') as f:
                triplets_data = json.load(f)
                
            triplets = triplets_data.get('triplets', [])
            taxonomic_info = triplets_data.get('taxonomic_info', {})

            if not triplets:
                logger.warning("No triplets found in the JSON data.")
                self.graph = nx.DiGraph() # Initialize an empty graph
                return self.graph

            # --- Threat Consolidation Step ---
            all_original_threat_strings = list(set(triplet['object'] for triplet in triplets if triplet.get('object')))
            if all_original_threat_strings:
                self.threat_consolidation_map = self._consolidate_threat_nodes(all_original_threat_strings)
            else:
                logger.info("No threat objects found in triplets to consolidate.")
                self.threat_consolidation_map = {}
            # --- End Threat Consolidation ---
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes and edges using consolidated threat names
            for triplet in triplets:
                subject_node_original = triplet['subject'] # This should be already normalized by prior scripts
                predicate_relation = triplet['predicate']
                original_object_string = triplet.get('object')

                if not original_object_string:
                    logger.warning(f"Triplet missing object: {triplet}")
                    continue

                # Get the consolidated full threat string (includes IUCN tag)
                consolidated_full_threat_string = self.threat_consolidation_map.get(original_object_string, original_object_string)
                
                # Parse the consolidated string to get its parts
                _, clean_threat_description, iucn_code, iucn_name = self._parse_iucn_tagged_string(consolidated_full_threat_string)
                
                # Add species node (subject)
                species_taxonomy = taxonomic_info.get(subject_node_original, {})
                if species_taxonomy is None: species_taxonomy = {} # Ensure it's a dict
                G.add_node(subject_node_original, type='species', **species_taxonomy)
                
                # Add threat node (object) using the consolidated full string as the node ID
                # Attributes will store the clean description and IUCN details.
                G.add_node(consolidated_full_threat_string, 
                           type='threat', 
                           description=clean_threat_description, 
                           iucn_code=iucn_code, 
                           iucn_name=iucn_name)
                
                # Add edge
                G.add_edge(subject_node_original, consolidated_full_threat_string, relation=predicate_relation)
            
            self.graph = G
            logger.info(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges after threat consolidation.")
            return G
            
        except Exception as e:
            logger.error(f"Error loading graph: {e}", exc_info=True) # Added exc_info for more details
            self.graph = None # Ensure graph is None on error
            return None
    
    def analyze_graph_structure(self):
        """
        Analyze the structure of the graph and return key metrics.
        Inspired by the paper which analyzes graph properties in detail.
        
        Returns:
            dict: Dictionary of graph metrics
        """
        if self.graph is None:
            logger.warning("No graph loaded. Call load_graph_from_triplets() first.")
            return None
        
        G = self.graph
        logger.info("Analyzing graph structure...")
        
        results = {}
        
        # --- Analyze a given graph (can be global or giant component) ---
        def _analyze_specific_graph(graph_to_analyze, graph_name="Global"):
            logger.info(f"Starting analysis for: {graph_name} Graph")
            metrics = {}
            if not graph_to_analyze or graph_to_analyze.number_of_nodes() == 0:
                logger.warning(f"{graph_name} graph is empty or None. Skipping its analysis.")
                return metrics

            # Basic graph metrics
            metrics["num_nodes"] = graph_to_analyze.number_of_nodes()
            metrics["num_edges"] = graph_to_analyze.number_of_edges()
            metrics["density"] = nx.density(graph_to_analyze)
            metrics["is_directed"] = graph_to_analyze.is_directed()
            
            is_connected_metric_name = "is_connected" if not graph_to_analyze.is_directed() else "is_weakly_connected"
            if graph_to_analyze.number_of_nodes() > 0:
                 metrics[is_connected_metric_name] = nx.is_weakly_connected(graph_to_analyze) if graph_to_analyze.is_directed() else nx.is_connected(graph_to_analyze)
            else:
                metrics[is_connected_metric_name] = False # Or None, based on preference

            # Degree statistics
            in_degrees = [d for n, d in graph_to_analyze.in_degree()] if graph_to_analyze.is_directed() else []
            out_degrees = [d for n, d in graph_to_analyze.out_degree()] if graph_to_analyze.is_directed() else []
            total_degrees = [d for n, d in graph_to_analyze.degree()]
            
            metrics.update({
                "avg_degree": np.mean(total_degrees) if total_degrees else 0,
                "max_degree": max(total_degrees) if total_degrees else 0,
                "min_degree": min(total_degrees) if total_degrees else 0,
                "median_degree": np.median(total_degrees) if total_degrees else 0
            })
            if graph_to_analyze.is_directed():
                metrics.update({
                    "avg_in_degree": np.mean(in_degrees) if in_degrees else 0,
                    "avg_out_degree": np.mean(out_degrees) if out_degrees else 0,
                })
            
            # Power law fit (only for total degrees)
            try:
                degrees_array = np.array(total_degrees)
                degrees_array = degrees_array[degrees_array > 0]
                if len(degrees_array) > 10:
                    # Using powerlaw package if available, otherwise simple linregress
                    # For simplicity here, sticking to linregress as in original code
                    # counts = np.bincount(degrees_array)[1:] # Exclude degree 0
                    # unique_degrees = np.arange(1, len(counts) + 1)
                    # valid_indices = (counts > 0) # only fit where count > 0
                    
                    # Simpler approach from original, might need refinement for robust power-law fitting
                    # log_degrees = np.log(degrees_array)
                    # log_counts = np.log(np.arange(1, len(degrees_array) + 1)) # This seems to be rank-frequency
                    # alpha, intercept, r_value, p_value, stderr = stats.linregress(log_degrees, log_counts)
                    
                    # A more standard way to prep for fitting degree distribution p(k) ~ k^-alpha:
                    degree_counts = Counter(degrees_array)
                    distinct_degrees = sorted(degree_counts.keys())
                    counts_for_degrees = [degree_counts[d] for d in distinct_degrees]

                    if len(distinct_degrees) > 1: # Need at least 2 points to fit a line
                        log_distinct_degrees = np.log(distinct_degrees)
                        log_counts_for_degrees = np.log(counts_for_degrees)
                        slope, intercept, r_value, p_value, stderr = stats.linregress(log_distinct_degrees, log_counts_for_degrees)
                        alpha = -slope # Exponent is negative of slope for p(k) vs k
                        metrics.update({
                            "power_law_exponent": alpha,
                            "power_law_r_squared": r_value**2,
                            "power_law_p_value": p_value,
                            "power_law_fit_intercept": intercept,
                            "is_scale_free_candidate": alpha > 1 and alpha < 3.5 and r_value**2 > 0.7 # Adjusted criteria slightly
                        })
                        logger.info(f"{graph_name} - Power law exponent: {alpha:.4f} (R²: {r_value**2:.4f}) from {len(distinct_degrees)} unique degrees > 0")
                    else:
                        logger.info(f"{graph_name} - Not enough unique degrees > 0 to fit power law.")
            except Exception as e:
                logger.warning(f"{graph_name} - Could not fit power law: {e}")

            # Connected components
            if graph_to_analyze.is_directed():
                strongly_connected = list(nx.strongly_connected_components(graph_to_analyze))
                weakly_connected = list(nx.weakly_connected_components(graph_to_analyze))
                metrics.update({
                    "num_strongly_connected": len(strongly_connected),
                    "num_weakly_connected": len(weakly_connected),
                    "largest_strongly_connected_size": len(max(strongly_connected, key=len)) if strongly_connected else 0,
                    "largest_weakly_connected_size": len(max(weakly_connected, key=len)) if weakly_connected else 0,
                })
            else:
                connected_components_list = list(nx.connected_components(graph_to_analyze))
                metrics.update({
                    "num_connected_components": len(connected_components_list),
                    "largest_component_size": len(max(connected_components_list, key=len)) if connected_components_list else 0,
                })

            # Node type analysis (if 'type' attribute exists)
            if graph_to_analyze.number_of_nodes() > 0 and 'type' in next(iter(graph_to_analyze.nodes(data=True)))[1]:
                node_types = Counter([data.get('type', 'unknown') for _, data in graph_to_analyze.nodes(data=True)])
                metrics["node_type_distribution"] = dict(node_types)
            
            # IUCN category distribution for threat nodes (if relevant attributes exist)
            if graph_to_analyze.number_of_nodes() > 0 and any(d.get('type') == 'threat' for n,d in graph_to_analyze.nodes(data=True)):
                iucn_codes_counts = Counter()
                iucn_names_counts = Counter()
                num_threat_nodes_for_iucn = 0
                for node, data in graph_to_analyze.nodes(data=True):
                    if data.get('type') == 'threat':
                        num_threat_nodes_for_iucn +=1
                        iucn_code = data.get('iucn_code')
                        iucn_name = data.get('iucn_name')
                        if iucn_code:
                            iucn_codes_counts[iucn_code] += 1
                        if iucn_name:
                            iucn_names_counts[iucn_name.strip().lower()] += 1
                if num_threat_nodes_for_iucn > 0:
                    metrics["iucn_code_distribution"] = dict(iucn_codes_counts.most_common())
                    metrics["iucn_name_distribution"] = dict(iucn_names_counts.most_common())
                logger.info(f"{graph_name} - IUCN code/name distribution calculated for {num_threat_nodes_for_iucn} threat nodes.")

            logger.info(f"{graph_name} - Analysis complete. Nodes: {metrics.get('num_nodes', 0)}, Edges: {metrics.get('num_edges', 0)}.")
            return metrics

        # --- Analyze Global Graph --- 
        results["global_graph"] = _analyze_specific_graph(G, "Global")

        # --- Giant Component Analysis (Structural/Louvain) ---
        giant_component_nodes = None
        if G.is_directed():
            weakly_connected_components = list(nx.weakly_connected_components(G))
            if weakly_connected_components:
                giant_component_nodes = max(weakly_connected_components, key=len)
        else:
            connected_components = list(nx.connected_components(G))
            if connected_components:
                giant_component_nodes = max(connected_components, key=len)
        
        if giant_component_nodes and len(giant_component_nodes) > 0:
            self.giant_component_graph = G.subgraph(giant_component_nodes).copy()
            logger.info(f"Extracted Giant Component with {self.giant_component_graph.number_of_nodes()} nodes and {self.giant_component_graph.number_of_edges()} edges.")
            results["giant_component"] = _analyze_specific_graph(self.giant_component_graph, "Giant_Component")
            
            results["global_graph"]["giant_component_size"] = len(giant_component_nodes)
            results["global_graph"]["giant_component_ratio"] = len(giant_component_nodes) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0

            # --- Community Detection (Louvain) on Giant Component --- 
            # This must happen *after* giant_component_graph is defined.
            # Community detection is done on the UNDIRECTED version of the giant component.
            if self.giant_component_graph and self.giant_component_graph.number_of_nodes() > 0:
                try:
                    giant_G_undirected = self.giant_component_graph.to_undirected() if self.giant_component_graph.is_directed() else self.giant_component_graph
                    
                    # Ensure it's not an empty graph before community detection
                    if giant_G_undirected.number_of_nodes() > 0:
                        partition = community_louvain.best_partition(giant_G_undirected, random_state=42) # Added random_state for reproducibility
                        
                        # Store community IDs on the original directed giant component nodes and global graph nodes
                        # This is for structural communities, not semantic ones from embeddings.
                        communities_data = defaultdict(list)
                        for node, comm_id in partition.items():
                            communities_data[comm_id].append(node)
                            if node in self.giant_component_graph.nodes:
                                self.giant_component_graph.nodes[node]['louvain_community_id'] = comm_id
                            if node in G.nodes: # Also store on global graph for convenience
                                G.nodes[node]['louvain_community_id'] = comm_id
                        
                        modularity_score = community_louvain.modularity(partition, giant_G_undirected)
                        
                        if "giant_component" not in results: results["giant_component"] = {}
                        results["giant_component"].update({
                            "num_louvain_communities": len(communities_data),
                            "louvain_community_sizes": {comm: len(nodes) for comm, nodes in communities_data.items()},
                            "louvain_modularity": modularity_score
                        })
                        logger.info(f"Detected {len(communities_data)} Louvain communities in Giant Component with modularity {modularity_score:.4f}")
                        self.louvain_communities_partition = partition # Store for later use
                        self.louvain_communities_data = communities_data
                    else:
                        logger.warning("Giant component (undirected) is empty, skipping Louvain community detection.")

                except Exception as e:
                    logger.warning(f"Louvain community detection on Giant Component failed: {e}")
            else:
                 logger.info("Giant component is empty or not defined, skipping Louvain community detection.")

            # --- Centrality Measures on Giant Component ---
            # These are typically calculated on the giant component due to computational cost.
            if self.giant_component_graph and self.giant_component_graph.number_of_nodes() > 1: # Need more than 1 node for some centralities
                try:
                    # Betweenness centrality
                    # For larger graphs, consider k parameter for approximation if too slow: nx.betweenness_centrality(graph, k=min(100, graph.number_of_nodes()//10) )
                    betweenness = nx.betweenness_centrality(self.giant_component_graph) # Directed version is fine
                    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    # Closeness centrality (on the largest weakly connected component of the giant component if directed)
                    # Or, if we assume we operate on the main connected part, directly on giant_component_graph if it's strongly connected or undirected.
                    # For directed graphs, closeness needs reachability to all other nodes. Often computed on WCCs.
                    # For simplicity, if directed and not strongly connected, this might be problematic. Let's compute on the undirected version for robustness here.
                    closeness_graph_target = self.giant_component_graph.to_undirected() if self.giant_component_graph.is_directed() else self.giant_component_graph
                    closeness = nx.closeness_centrality(closeness_graph_target)
                    top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    # Eigenvector centrality (requires connected graph, typically on undirected or strongly connected component)
                    # Using the undirected version of giant component for robustness or largest SCC.
                    # For DiGraphs, nx.eigenvector_centrality can fail if not strongly connected. 
                    # If G is not connected, Eigenvector Centrality is computed for the largest connected component.
                    eigenvector_target_graph = self.giant_component_graph.to_undirected() if self.giant_component_graph.is_directed() else self.giant_component_graph
                    if nx.is_connected(eigenvector_target_graph): # Ensure it's connected for eigenvector
                        eigenvector = nx.eigenvector_centrality_numpy(eigenvector_target_graph, max_iter=500) # Switched to numpy version
                        top_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:10]
                    else:
                        logger.warning("Giant component (undirected) is not connected. Skipping eigenvector centrality or computing on largest sub-component.")
                        # Fallback: compute on largest connected component of the undirected giant graph
                        largest_cc_undir = max(nx.connected_components(eigenvector_target_graph), key=len)
                        sub_eigen_graph = eigenvector_target_graph.subgraph(largest_cc_undir)
                        if sub_eigen_graph.number_of_nodes() > 1:
                            eigenvector = nx.eigenvector_centrality_numpy(sub_eigen_graph, max_iter=500)
                            top_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:10]
                        else: top_eigenvector = []

                    if "giant_component" not in results: results["giant_component"] = {}
                    results["giant_component"].update({
                        "top_betweenness_nodes": dict(top_betweenness),
                        "top_closeness_nodes": dict(top_closeness),
                        "top_eigenvector_nodes": dict(top_eigenvector) if top_eigenvector else {}
                    })
                    logger.info("Centrality measures calculated for Giant Component.")
                except Exception as e:
                    logger.warning(f"Error calculating centrality measures for Giant Component: {e}", exc_info=True)
        else:
            logger.warning("No Giant Component found or it's empty. Skipping Giant Component specific analyses.")
            self.giant_component_graph = None # Ensure it's None

        # Global graph summary metrics related to communities (if detected on giant component)
        if 'louvain_community_sizes' in results.get("giant_component", {}):
            results["global_graph"]["num_louvain_communities_in_giant"] = results["giant_component"]["num_louvain_communities"]
            results["global_graph"]["louvain_modularity_of_giant"] = results["giant_component"]["louvain_modularity"]

        logger.info(f"Graph structure analysis complete.")
        self.graph_metrics = results # Store all metrics
        return results
    
    def calculate_embeddings(self, model_name='all-MiniLM-L6-v2'):
        """
        Calculate node embeddings using a sentence transformer model.
        Embeddings can be used for similarity search, clustering, and visualization.
        
        Args:
            model_name: Name of the sentence transformer model to use
        
        Returns:
            dict: Dictionary mapping node names to embedding vectors
        """
        if self.graph is None:
            logger.warning("No graph loaded. Call load_graph_from_triplets() first.")
            return None
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            # Ensure embedding model is initialized only once or if name changes
            if self.embedding_model is None or (hasattr(self.embedding_model, 'name') and self.embedding_model.name != model_name):
                try:
                    self.embedding_model = SentenceTransformer(model_name)
                    self.embedding_model.name = model_name # Add a name attribute for checking
                    logger.info(f"SentenceTransformer model '{model_name}' initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize SentenceTransformer model '{model_name}': {e}. Some functionalities might be limited.")
                    return None
            elif self.embedding_model is not None:
                logger.info(f"Using already initialized SentenceTransformer model: {getattr(self.embedding_model, 'name', 'unknown_model')}")

            if not self.embedding_model:
                logger.error("Embedding model could not be initialized. Aborting embedding calculation.")
                return None
            
            # Get all nodes from the main graph
            nodes = list(self.graph.nodes())
            
            logger.info(f"Calculating embeddings for {len(nodes)} nodes...")
            node_embeddings = self.embedding_model.encode(nodes)
            
            # Create dictionary mapping node names to embeddings
            self.embeddings = {node: embedding for node, embedding in zip(nodes, node_embeddings)}
            
            logger.info(f"Embeddings calculated for {len(self.embeddings)} nodes")
            return self.embeddings
        
        except Exception as e:
            logger.error(f"Error calculating embeddings: {e}")
            return None
    
    def find_similar_nodes(self, query_node, top_n=10):
        """
        Find the most similar nodes to a given query node based on embedding similarity.
        
        Args:
            query_node: Node to find similar nodes for
            top_n: Number of similar nodes to return
        
        Returns:
            list: List of (node, similarity_score) tuples
        """
        if self.embeddings is None:
            logger.warning("No embeddings calculated. Call calculate_embeddings() first.")
            return None
        
        if query_node not in self.embeddings:
            logger.warning(f"Query node '{query_node}' not found in embeddings")
            return None
        
        try:
            query_embedding = self.embeddings[query_node]
            
            # Calculate cosine similarity with all other nodes
            similarities = []
            for node, embedding in self.embeddings.items():
                if node != query_node:
                    similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                    similarities.append((node, similarity))
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Found {len(similarities[:top_n])} similar nodes to '{query_node}'")
            return similarities[:top_n]
        
        except Exception as e:
            logger.error(f"Error finding similar nodes: {e}")
            return None
    
    def find_shortest_paths(self, source, target, k=3):
        """
        Find k shortest paths between source and target nodes.
        Useful for knowledge graph reasoning by showing the connections between concepts.
        
        Args:
            source: Source node
            target: Target node
            k: Number of paths to find
        
        Returns:
            list: List of paths, where each path is a list of nodes
        """
        if self.graph is None:
            logger.warning("No graph loaded. Call load_graph_from_triplets() first.")
            return None
        
        G = self.graph
        
        if source not in G.nodes:
            logger.warning(f"Source node '{source}' not found in graph")
            return None
        
        if target not in G.nodes:
            logger.warning(f"Target node '{target}' not found in graph")
            return None
        
        try:
            # Try to find k shortest paths
            paths = list(nx.shortest_simple_paths(G, source, target))[:k]
            
            # Get edge attributes (relations) for each path
            paths_with_relations = []
            for path in paths:
                path_with_relations = []
                for i in range(len(path) - 1):
                    from_node = path[i]
                    to_node = path[i + 1]
                    relation = G.get_edge_data(from_node, to_node).get('relation', '')
                    path_with_relations.append((from_node, relation, to_node))
                paths_with_relations.append(path_with_relations)
            
            logger.info(f"Found {len(paths)} paths between '{source}' and '{target}'")
            return paths_with_relations
        
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between '{source}' and '{target}'")
            return []
        except Exception as e:
            logger.error(f"Error finding shortest paths: {e}")
            return None
    
    def visualize_graph(self, subset_nodes=None, layout='spring', filename=None, node_size_by_degree=False, base_node_size=50, degree_scale_factor=20):
        """
        Visualize the graph or a subset of it.
        
        Args:
            subset_nodes: List of nodes to include in the visualization (None for all)
            layout: Layout algorithm to use ('spring', 'circular', 'kamada_kawai')
            filename: File to save the visualization to (None for display only)
            node_size_by_degree: If True, node sizes are proportional to their degree.
            base_node_size: Base size for nodes if not scaling by degree, or minimum size if scaling.
            degree_scale_factor: Factor to scale node size by degree.
        
        Returns:
            matplotlib figure
        """
        if self.graph is None:
            logger.warning("No graph loaded. Call load_graph_from_triplets() first.")
            return None
        
        G = self.graph
        
        # Create subgraph if subset_nodes is provided
        if subset_nodes is not None:
            G = G.subgraph(subset_nodes).copy()
        
        # Limit size for visualization
        if G.number_of_nodes() > 200 and not node_size_by_degree: # Increased limit slightly
            logger.warning(f"Graph is large ({G.number_of_nodes()} nodes). Visualization may be cluttered.")
        elif G.number_of_nodes() > 500 and node_size_by_degree:
            logger.warning(f"Graph is large ({G.number_of_nodes()} nodes) even with degree-based sizing. Visualization may be slow or cluttered.")

        
        try:
            plt.figure(figsize=(15, 12)) # Increased figure size
            
            # Get node colors based on type
            node_colors = []
            for node in G.nodes():
                node_type = G.nodes[node].get('type', 'unknown')
                if node_type == 'species':
                    node_colors.append('skyblue')
                elif node_type == 'threat':
                    node_colors.append('salmon')
                else:
                    node_colors.append('lightgray')
            
            # Node sizes
            if node_size_by_degree:
                degrees = dict(G.degree())
                node_sizes = [base_node_size + degrees.get(node, 0) * degree_scale_factor for node in G.nodes()]
            else:
                node_sizes = base_node_size
            
            # Get layout
            if layout == 'spring':
                pos = nx.spring_layout(G)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spring_layout(G)
            
            # Draw graph
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, node_size=node_sizes)
            nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True, width=0.5) # Made arrows default, thinner edges
            
            # Add labels to nodes, limiting the length
            node_labels = {node: (node[:20] + '...' if len(node) > 20 else node) for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
            
            plt.title(f"Graph Visualization ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
            plt.axis('off')
            plt.tight_layout()
            
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Graph visualization saved to {filename}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
            return None
    
    def visualize_communities(self, filename=None):
        """
        Visualize the communities in the graph.
        
        Args:
            filename: File to save the visualization to
        
        Returns:
            matplotlib figure
        """
        if self.graph is None:
            logger.warning("No graph loaded. Call load_graph_from_triplets() first.")
            return None
        
        G = self.graph
        
        # Check if communities are already detected
        if not any('community' in G.nodes[node] for node in G.nodes()):
            logger.info("Communities not detected. Running community detection...")
            self.analyze_graph_structure()  # This will assign community attributes
        
        # Get giant component
        if G.is_directed():
            giant_component = max(nx.weakly_connected_components(G), key=len)
        else:
            giant_component = max(nx.connected_components(G), key=len)
        
        # Create subgraph of giant component
        giant_G = G.subgraph(giant_component).copy()
        
        # Convert to undirected for visualization
        if giant_G.is_directed():
            giant_G = giant_G.to_undirected()
        
        try:
            plt.figure(figsize=(14, 12))
            
            # Get node colors based on community
            communities = {}
            for node in giant_G.nodes():
                community = giant_G.nodes[node].get('community', 0)
                communities[node] = community
            
            # Get positions
            pos = nx.spring_layout(giant_G, k=0.3, iterations=50, seed=42)
            
            # Color map
            cmap = plt.cm.get_cmap('tab20', max(communities.values()) + 1)
            
            # Draw nodes
            for community, nodes in defaultdict(list, {comm: [node for node, c in communities.items() if c == comm] for comm in set(communities.values())}).items():
                nx.draw_networkx_nodes(giant_G, pos, nodelist=nodes, node_color=[cmap(community)], alpha=0.8, node_size=50, label=f"Community {community}")
            
            # Draw edges
            nx.draw_networkx_edges(giant_G, pos, alpha=0.2)
            
            plt.title(f"Community Structure ({len(set(communities.values()))} communities)")
            plt.axis('off')
            plt.legend(scatterpoints=1, loc='lower left', ncol=2, fontsize=8)
            plt.tight_layout()
            
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Community visualization saved to {filename}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error visualizing communities: {e}")
            return None
    
    def visualize_embeddings(self, method='pca', n_clusters=5, filename=None, target_graph=None):
        """
        Visualize node embeddings using dimensionality reduction and clustering.
        If target_graph is provided, embeddings are visualized only for nodes in that subgraph.
        
        Args:
            method: Dimensionality reduction method ('pca' or 'tsne')
            n_clusters: Number of clusters to detect
            filename: File to save the visualization to
            target_graph: Optional[nx.Graph]. If provided, visualize embeddings only for nodes in this graph.
        
        Returns:
            matplotlib figure, cluster representatives, and DataFrame
        """
        if self.embeddings is None:
            logger.warning("No embeddings calculated. Call calculate_embeddings() first.")
            return None
        
        try:
            # Convert embeddings to array
            if target_graph is not None:
                nodes = [n for n in list(self.embeddings.keys()) if n in target_graph]
                if not nodes:
                    logger.warning("No nodes from target_graph found in embeddings. Visualizing all embeddings.")
                    nodes = list(self.embeddings.keys())
            else:
                nodes = list(self.embeddings.keys())
            
            if not nodes:
                logger.warning("No nodes available for embedding visualization.")
                return None

            embeddings_array = np.array([self.embeddings[node] for node in nodes])
            
            # Reduce dimensions
            if method == 'pca':
                reducer = PCA(n_components=2)
                reduced_embeddings = reducer.fit_transform(embeddings_array)
                method_name = 'PCA'
            elif method == 'tsne':
                # Adjust perplexity based on sample size
                n_samples = len(embeddings_array)
                perplexity = min(30, max(5, n_samples - 1))
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                reduced_embeddings = reducer.fit_transform(embeddings_array)
                method_name = 't-SNE'
            else:
                logger.warning(f"Unknown method '{method}'. Using PCA.")
                reducer = PCA(n_components=2)
                reduced_embeddings = reducer.fit_transform(embeddings_array)
                method_name = 'PCA'
            
            # Cluster embeddings
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings_array)
            
            # Get node types for coloring
            node_types = []
            graph_source = target_graph if target_graph is not None else self.graph
            for node in nodes:
                if node in graph_source.nodes:
                    node_types.append(graph_source.nodes[node].get('type', 'unknown'))
                else:
                    node_types.append('unknown')
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'node': nodes,
                'x': reduced_embeddings[:, 0],
                'y': reduced_embeddings[:, 1],
                'cluster': clusters,
                'type': node_types
            })
            self.embeddings_df = df  # Store df as instance variable
            self.n_semantic_clusters = n_clusters # Store n_clusters as instance variable
            
            # Plot
            plt.figure(figsize=(12, 10))
            
            # Plot by cluster
            plt.subplot(1, 2, 1)
            sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='tab10', s=50, alpha=0.7)
            plt.title(f'Node Embeddings by Cluster ({method_name})')
            plt.xlabel(f'{method_name} Dimension 1')
            plt.ylabel(f'{method_name} Dimension 2')
            plt.legend(title='Cluster', loc='best')
            
            # Plot by node type
            plt.subplot(1, 2, 2)
            sns.scatterplot(data=df, x='x', y='y', hue='type', palette='Set2', s=50, alpha=0.7)
            plt.title(f'Node Embeddings by Type ({method_name})')
            plt.xlabel(f'{method_name} Dimension 1')
            plt.ylabel(f'{method_name} Dimension 2')
            plt.legend(title='Node Type', loc='best')
            
            plt.tight_layout()
            
            # Save visualization
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Embedding visualization saved to {filename}")
            
            # Find representative nodes for each cluster
            centroids = kmeans.cluster_centers_
            cluster_representatives = []
            
            for i in range(n_clusters):
                # Get nodes in this cluster
                cluster_nodes = df[df['cluster'] == i]['node'].tolist()
                cluster_embeddings = np.array([self.embeddings[node] for node in cluster_nodes])
                
                # Find node closest to centroid
                distances = np.linalg.norm(cluster_embeddings - centroids[i], axis=1)
                closest_idx = np.argmin(distances)
                representative = cluster_nodes[closest_idx]
                
                # Get a few examples from this cluster
                examples = np.random.choice(cluster_nodes, min(5, len(cluster_nodes)), replace=False).tolist()
                
                cluster_representatives.append({
                    'cluster': i,
                    'representative': representative,
                    'examples': examples,
                    'size': len(cluster_nodes)
                })
            
            logger.info("Cluster representatives:")
            for rep in cluster_representatives:
                logger.info(f"Cluster {rep['cluster']} (size: {rep['size']}): {rep['representative']}, examples: {', '.join(rep['examples'])}")
            
            # Store semantic cluster IDs on graph nodes
            logger.info("Storing K-Means based semantic cluster IDs on graph nodes...")
            for index, row in df.iterrows():
                node_name = row['node']
                cluster_id = row['cluster']
                if node_name in self.graph.nodes:
                    self.graph.nodes[node_name]['semantic_community_id'] = cluster_id
                else:
                    logger.warning(f"Node '{node_name}' from embeddings_df not found in self.graph.nodes while assigning semantic_community_id.")
            logger.info("Finished storing semantic cluster IDs.")
            
            # --- Enhanced Cluster Analysis ---
            logger.info("\n--- Detailed Semantic Cluster Analysis ---")
            for i in range(n_clusters):
                cluster_nodes_in_df = df[df['cluster'] == i]
                current_cluster_nodes = cluster_nodes_in_df['node'].tolist()
                logger.info(f"\nCluster {i} (Representative: {cluster_representatives[i]['representative']}, Size: {len(current_cluster_nodes)}):")

                # Node type distribution within the cluster
                species_count = 0
                threat_count = 0
                unknown_type_count = 0
                
                cluster_iucn_codes = Counter()
                cluster_iucn_names = Counter()

                for node_name in current_cluster_nodes:
                    if node_name in graph_source.nodes:
                        node_data = graph_source.nodes[node_name]
                        node_type = node_data.get('type', 'unknown')
                        if node_type == 'species':
                            species_count += 1
                        elif node_type == 'threat':
                            threat_count += 1
                            if node_data.get('iucn_code'):
                                cluster_iucn_codes[node_data['iucn_code']] += 1
                            if node_data.get('iucn_name'):
                                cluster_iucn_names[node_data['iucn_name'].strip().lower()] += 1
                        else:
                            unknown_type_count += 1
                    else:
                        unknown_type_count +=1 # Should not happen if embeddings are from graph nodes

                logger.info(f"  Node Types: Species: {species_count}, Threats: {threat_count}, Unknown: {unknown_type_count}")
                if threat_count > 0:
                    logger.info(f"  Top 5 IUCN Codes: {cluster_iucn_codes.most_common(5)}")
                    logger.info(f"  Top 5 IUCN Names: {cluster_iucn_names.most_common(5)}")

                # Identify and log outlier nodes
                dominant_type = 'species' if species_count >= threat_count else 'threat'
                if unknown_type_count > max(species_count, threat_count):
                    dominant_type = 'unknown' # Or handle as genuinely mixed
                
                outlier_nodes = []
                for node_name in current_cluster_nodes:
                    if node_name in graph_source.nodes:
                        node_data = graph_source.nodes[node_name]
                        node_type = node_data.get('type', 'unknown')
                        if node_type != dominant_type and node_type != 'unknown': # Don't list unknowns if dominant is species/threat
                            outlier_nodes.append(node_name)
                if outlier_nodes:
                    logger.info(f"  Outlier Nodes (dominant type: {dominant_type}): {outlier_nodes[:10]}") # Log up to 10 outliers

                # Intra-cluster and Inter-cluster edge predicate analysis (outgoing)
                intra_cluster_predicates = Counter()
                inter_cluster_predicates = defaultdict(Counter) # Key: other_cluster_id, Value: Counter of predicates

                for source_node in current_cluster_nodes:
                    if source_node not in self.graph: # Check against main graph for edges
                        continue
                    for target_node, edge_data in self.graph.adj[source_node].items():
                        predicate = edge_data.get('relation', 'unknown_relation')
                        
                        # Check if target_node is in the current df (i.e., part of the visualized set)
                        target_node_info = df[df['node'] == target_node]
                        if not target_node_info.empty:
                            target_cluster_id = target_node_info['cluster'].iloc[0]
                            if target_cluster_id == i: # Intra-cluster edge
                                intra_cluster_predicates[predicate] += 1
                            else: # Inter-cluster edge
                                inter_cluster_predicates[target_cluster_id][predicate] += 1
                
                if intra_cluster_predicates:
                    logger.info(f"  Top 3 Intra-cluster Predicates: {intra_cluster_predicates.most_common(3)}")
                else:
                    logger.info("  No intra-cluster edges found.")

                if inter_cluster_predicates:
                    logger.info("  Top Inter-cluster Predicates (to -> [target_cluster_id: (predicate, count)]):")
                    sorted_inter_cluster = sorted(inter_cluster_predicates.items(), key=lambda item: sum(item[1].values()), reverse=True)
                    for target_cls, pred_counts in sorted_inter_cluster[:3]: # Log top 3 target clusters by num of connections
                        logger.info(f"    To Cluster {target_cls}: {pred_counts.most_common(2)}")
                else:
                    logger.info("  No inter-cluster edges found originating from this cluster.")

                # Inter-cluster predicate analysis (incoming)
                incoming_cluster_predicates = defaultdict(Counter)
                for potential_source_node, source_data in self.graph.nodes(data=True): # Iterate all graph nodes
                    # Check if potential_source_node is in the current df (i.e., part of the visualized set)
                    source_node_info = df[df['node'] == potential_source_node]
                    if source_node_info.empty:
                        continue
                    source_cluster_id = source_node_info['cluster'].iloc[0]

                    if source_cluster_id == i: 
                        continue

                    if potential_source_node not in self.graph.adj:
                        continue 
                        
                    for potential_target_node, edge_data in self.graph.adj[potential_source_node].items():
                        if potential_target_node in current_cluster_nodes: 
                            predicate = edge_data.get('relation', 'unknown_relation')
                            incoming_cluster_predicates[source_cluster_id][predicate] += 1
                
                if incoming_cluster_predicates:
                    logger.info("  Top Incoming Inter-cluster Predicates (from -> [source_cluster_id: (predicate, count)]):")
                    sorted_incoming_cluster = sorted(incoming_cluster_predicates.items(), key=lambda item: sum(item[1].values()), reverse=True)
                    for source_cls, pred_counts in sorted_incoming_cluster[:3]: # Log top 3 source clusters by num of connections
                        logger.info(f"    From Cluster {source_cls}: {pred_counts.most_common(2)}")
                else:
                    logger.info("  No inter-cluster edges found pointing into this cluster.")

            # --- End Enhanced Cluster Analysis ---

            return plt.gcf(), cluster_representatives, df # Also return df for immediate use if needed
            
        except Exception as e:
            logger.error(f"Error visualizing embeddings: {e}")
            return None
    
    def analyze_degree_distribution(self, filename=None, target_graph=None, graph_name="Global"):
        """
        Analyze and visualize the degree distribution of the graph.
        
        Args:
            filename: File to save the visualization to
            target_graph: Optional[nx.Graph]. If provided, analyze this graph instead of self.graph.
            graph_name: Name of the graph being analyzed (for titles and logs).
        
        Returns:
            matplotlib figure
        """
        current_graph = target_graph if target_graph is not None else self.graph
        if current_graph is None:
            logger.warning(f"No graph ({graph_name}) available for degree distribution analysis.")
            return None
        
        G = current_graph
        logger.info(f"Analyzing degree distribution for {graph_name} graph...")
        
        try:
            # Get degree distribution
            degrees = [d for _, d in G.degree()]
            degree_count = Counter(degrees)
            
            # Convert to log-log for power law analysis
            x = np.array(sorted(degree_count.keys()))
            y = np.array([degree_count[k] for k in x])
            
            logger.info("Degree distribution data points (degree, count) for log-log plot:")
            for deg, count in zip(x, y):
                logger.info(f"Degree: {deg}, Count: {count}")
            
            # Plot degree distribution
            plt.figure(figsize=(12, 8))
            
            # Linear plot
            plt.subplot(2, 2, 1)
            plt.bar(degree_count.keys(), degree_count.values(), width=0.8, alpha=0.7)
            plt.title(f'Degree Distribution ({graph_name})')
            plt.xlabel('Degree')
            plt.ylabel('Count')
            
            # Log-log plot
            plt.subplot(2, 2, 2)
            plt.loglog(x, y, 'bo', markersize=5, alpha=0.7)
            plt.title(f'Degree Distribution ({graph_name} Log-Log)')
            plt.xlabel('Degree (log scale)')
            plt.ylabel('Count (log scale)')
            
            # Fit power law if enough data points
            if len(x) > 1: # Changed from >5 to >1 for trying fit more often
                # Filter out zeros for log-log fit
                mask = (x > 0) & (y > 0)
                x_fit = x[mask]
                y_fit = y[mask]
                
                if len(x_fit) > 1: # Need at least 2 points for linregress
                    # Fit power law: p(k) ~ k^(-alpha)
                    log_x = np.log(x_fit)
                    log_y = np.log(y_fit)
                    
                    # Power law fitting as per the paper's Table 2 and Figure 4f description
                    # The paper uses a more sophisticated tool (likely 'powerlaw' package)
                    # Here, we simulate the reported metrics using linregress on log-log data of degree counts.
                    # The slope of log(count) vs log(degree) is -alpha.
                    if len(log_x) >= 2: # Check if there are enough points to fit
                        slope, intercept, r_value, p_value, stderr = stats.linregress(log_x, log_y)
                        alpha = -slope
                        
                        # Plot fit line
                        fit_line_x_log = np.array([min(log_x), max(log_x)])
                        fit_line_y_log = intercept + slope * fit_line_x_log
                        plt.loglog(np.exp(fit_line_x_log), np.exp(fit_line_y_log), 'r--', linewidth=2, 
                                  label=f'Power Law Fit: α = {alpha:.2f}, R² = {r_value**2:.2f}')
                        plt.legend()
                        
                        logger.info(f"{graph_name} - Power law exponent (α): {alpha:.4f} (R²: {r_value**2:.4f}, p-value: {p_value:.2e}, stderr: {stderr:.4f})")
                        # Store these in graph_metrics if this function is called by run_full_analysis
                        if hasattr(self, 'graph_metrics') and self.graph_metrics:
                            target_metrics_key = "giant_component" if graph_name == "Giant Component" else "global_graph"
                            if target_metrics_key in self.graph_metrics:
                                self.graph_metrics[target_metrics_key]['power_law_exponent_fit'] = alpha
                                self.graph_metrics[target_metrics_key]['power_law_r_squared_fit'] = r_value**2
                                self.graph_metrics[target_metrics_key]['power_law_p_value_fit'] = p_value
                    else:
                        logger.warning(f"{graph_name} - Not enough data points ({len(log_x)}) after filtering for power law fit.")
            
            # Degree centrality distribution
            plt.subplot(2, 2, 3)
            if G.number_of_nodes() > 0 : # Check if graph is not empty
                degree_centrality = nx.degree_centrality(G)
                centrality_values = list(degree_centrality.values())
                plt.hist(centrality_values, bins=20, alpha=0.7)
            plt.title(f'Degree Centrality Distribution ({graph_name})')
            plt.xlabel('Degree Centrality')
            plt.ylabel('Count')
            
            # Node types by degree
            plt.subplot(2, 2, 4)
            node_types_for_plot = []
            node_degrees_for_plot = []
            
            if G.number_of_nodes() > 0 and 'type' in next(iter(G.nodes(data=True)))[1]:
                for node in G.nodes():
                    node_type = G.nodes[node].get('type', 'unknown')
                    node_types_for_plot.append(node_type)
                    node_degrees_for_plot.append(G.degree(node))
            
                df_degree_type = pd.DataFrame({'type': node_types_for_plot, 'degree': node_degrees_for_plot})
                sns.boxplot(x='type', y='degree', data=df_degree_type)
            plt.title(f'Degree by Node Type ({graph_name})')
            plt.xlabel('Node Type')
            plt.ylabel('Degree')
            
            plt.tight_layout()
            
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Degree distribution analysis for {graph_name} saved to {filename}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error analyzing degree distribution for {graph_name}: {e}", exc_info=True)
            return None
    
    def find_structural_patterns(self):
        """
        Find common structural patterns in the graph, such as motifs.
        
        Returns:
            dict: Dictionary of pattern statistics
        """
        if self.graph is None:
            logger.warning("No graph loaded. Call load_graph_from_triplets() first.")
            return None
        
        G = self.graph
        
        try:
            patterns = {}
            
            # Find hubs (high-degree nodes)
            node_degrees = dict(G.degree())
            hubs = {node: degree for node, degree in node_degrees.items() if degree > 5}
            patterns['hubs'] = hubs
            
            # Find bridges (edges that increase the number of connected components when removed)
            if not G.is_directed():
                bridges = list(nx.bridges(G))
                patterns['bridges'] = bridges
            
            # Find articulation points (nodes that increase the number of connected components when removed)
            articulation_points = list(nx.articulation_points(G.to_undirected()))
            patterns['articulation_points'] = articulation_points
            
            # Calculate clustering coefficient
            clustering = nx.clustering(G.to_undirected())
            patterns['clustering'] = clustering
            patterns['avg_clustering'] = sum(clustering.values()) / len(clustering) if clustering else 0
            
            # Find nodes with high betweenness centrality
            betweenness = nx.betweenness_centrality(G)
            high_betweenness = {node: bc for node, bc in betweenness.items() if bc > 0.01}
            patterns['high_betweenness'] = high_betweenness
            
            logger.info(f"Found {len(hubs)} hubs, {len(articulation_points)} articulation points, and {len(high_betweenness)} nodes with high betweenness")
            return patterns
            
        except Exception as e:
            logger.error(f"Error finding structural patterns: {e}")
            return None
    
    def summarize_analysis(self):
        """
        Summarize all analyses in a comprehensive report.
        
        Returns:
            dict: Summary of all analyses
        """
        if self.graph is None:
            logger.warning("No graph loaded. Call load_graph_from_triplets() first.")
            return None
        
        summary = {}
        
        # Run all analyses
        try:
            # Basic structure
            structure = self.analyze_graph_structure()
            summary['structure'] = structure
            
            # Degree distribution
            degrees = [d for _, d in self.graph.degree()]
            summary['degrees'] = {
                'mean': np.mean(degrees),
                'median': np.median(degrees),
                'max': max(degrees),
                'min': min(degrees),
                'std': np.std(degrees)
            }
            
            # Node type analysis
            node_types = Counter([data.get('type', 'unknown') for _, data in self.graph.nodes(data=True)])
            summary['node_types'] = dict(node_types)
            
            # Calculate structural patterns
            patterns = self.find_structural_patterns()
            if patterns:
                summary['patterns'] = {
                    'num_hubs': len(patterns.get('hubs', {})),
                    'num_articulation_points': len(patterns.get('articulation_points', [])),
                    'num_high_betweenness': len(patterns.get('high_betweenness', {})),
                    'avg_clustering': patterns.get('avg_clustering', 0)
                }
            
            # Get community statistics
            communities = {}
            for node, data in self.graph.nodes(data=True):
                community = data.get('community')
                if community is not None:
                    if community not in communities:
                        communities[community] = []
                    communities[community].append(node)
            
            if communities:
                summary['communities'] = {
                    'num_communities': len(communities),
                    'sizes': {comm: len(nodes) for comm, nodes in communities.items()},
                    'avg_size': sum(len(nodes) for nodes in communities.values()) / len(communities) if communities else 0
                }
            
            logger.info("Analysis summary complete")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing analysis: {e}")
            return None
    
    def analyze_paths_between_semantic_clusters(self, cluster_id_1, cluster_id_2, max_path_len=4, num_example_paths=3, sample_nodes_per_cluster=5):
        """
        Analyzes and logs a few example paths between nodes of two specified semantic clusters.

        Args:
            cluster_id_1: ID of the first semantic cluster.
            cluster_id_2: ID of the second semantic cluster.
            max_path_len: Maximum length for paths to consider.
            num_example_paths: Number of example paths to try and log/return.
            sample_nodes_per_cluster: Number of nodes to sample from each cluster to find paths between.
        Returns:
            List[str]: A list of string representations of the found paths.
        """
        if self.graph is None:
            logger.warning("Graph not loaded. Cannot analyze paths.")
            return []
        if not hasattr(self, 'embeddings_df') or self.embeddings_df is None or 'cluster' not in self.embeddings_df.columns:
            logger.warning("Embeddings DataFrame with cluster assignments not available. Run visualize_embeddings first.")
            return []
        if not hasattr(self, 'n_semantic_clusters') or self.n_semantic_clusters is None:
            logger.warning("Number of semantic clusters (n_semantic_clusters) not set. Run visualize_embeddings first.")
            return []

        # Use instance variables for n_clusters and embeddings_df
        n_clusters = self.n_semantic_clusters
        embeddings_df = self.embeddings_df

        if not (0 <= cluster_id_1 < n_clusters and 0 <= cluster_id_2 < n_clusters):
            logger.warning(f"Invalid cluster IDs provided. Must be between 0 and {n_clusters-1}.")
            return []
        if cluster_id_1 == cluster_id_2:
            logger.info(f"Skipping path analysis between a cluster and itself (Cluster {cluster_id_1}).")
            return []

        logger.info(f"\n--- Path Analysis Between Semantic Cluster {cluster_id_1} and Cluster {cluster_id_2} (max_len={max_path_len}) ---")

        nodes_c1 = embeddings_df[embeddings_df['cluster'] == cluster_id_1]['node'].tolist()
        nodes_c2 = embeddings_df[embeddings_df['cluster'] == cluster_id_2]['node'].tolist()

        if not nodes_c1 or not nodes_c2:
            logger.warning(f"One or both clusters ({cluster_id_1}, {cluster_id_2}) are empty or not found in embeddings_df.")
            return []

        # Sample a few nodes from each cluster to limit combinations
        sampled_nodes_c1 = np.random.choice(nodes_c1, min(len(nodes_c1), sample_nodes_per_cluster), replace=False)
        sampled_nodes_c2 = np.random.choice(nodes_c2, min(len(nodes_c2), sample_nodes_per_cluster), replace=False)

        found_paths_strings = []
        logged_paths_structures = set() # To avoid duplicate path structures

        for source_node in sampled_nodes_c1:
            if len(found_paths_strings) >= num_example_paths:
                break
            if source_node not in self.graph:
                continue
            for target_node in sampled_nodes_c2:
                if len(found_paths_strings) >= num_example_paths:
                    break
                if target_node not in self.graph or source_node == target_node:
                    continue
                
                try:
                    # Find all simple paths up to a certain length
                    for path_nodes in nx.all_simple_paths(self.graph, source=source_node, target=target_node, cutoff=max_path_len -1):
                        if len(path_nodes) > 1: # Ensure path has at least one edge
                            path_with_relations_str_parts = []
                            for i in range(len(path_nodes) - 1):
                                from_n, to_n = path_nodes[i], path_nodes[i+1]
                                relation = self.graph.get_edge_data(from_n, to_n).get('relation', '')
                                path_with_relations_str_parts.append(f"'{from_n}' -[{relation}]-> '{to_n}'")
                            
                            full_path_str = " --> ".join(path_with_relations_str_parts)
                            
                            # Log only if unique and we haven't found enough examples yet
                            if full_path_str not in logged_paths_structures:
                                logger.info(f"  Example Path: {full_path_str}")
                                found_paths_strings.append(full_path_str)
                                logged_paths_structures.add(full_path_str)
                                if len(found_paths_strings) >= num_example_paths:
                                    break # Break from inner loop (targets)
                    if len(found_paths_strings) >= num_example_paths:
                        break # Break from middle loop (sources)
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    logger.error(f"Error finding paths between {source_node} and {target_node}: {e}")
                    continue
        
        if len(found_paths_strings) == 0:
            logger.info(f"  No simple paths of length <= {max_path_len} found between sampled nodes of Cluster {cluster_id_1} and Cluster {cluster_id_2}.")
        else:
            logger.info(f"  Found {len(found_paths_strings)} example path(s) between Cluster {cluster_id_1} and Cluster {cluster_id_2}.")
        return found_paths_strings

    def automate_all_inter_cluster_path_analysis(self, output_filename="all_inter_cluster_paths.txt", **kwargs):
        """
        Automates the analysis of paths between all unique pairs of semantic clusters
        and saves the results to a file.

        Args:
            output_filename: Name of the file to save the paths to.
            **kwargs: Additional arguments to pass to analyze_paths_between_semantic_clusters 
                      (e.g., max_path_len, num_example_paths, sample_nodes_per_cluster).
        """
        if not hasattr(self, 'n_semantic_clusters') or self.n_semantic_clusters is None or self.n_semantic_clusters < 2:
            logger.warning("Not enough semantic clusters identified (or visualize_embeddings not run). Skipping automated path analysis.")
            return

        output_path = self.results_dir / output_filename
        all_paths_found = []
        logger.info(f"\n--- Starting Automated Inter-Cluster Path Analysis (output to: {output_path}) ---")

        for i in range(self.n_semantic_clusters):
            for j in range(i + 1, self.n_semantic_clusters): # Iterate unique pairs
                # Extract relevant kwargs or use defaults if not provided
                max_path_len = kwargs.get('max_path_len', 4)
                num_example_paths = kwargs.get('num_example_paths', 3)
                sample_nodes_per_cluster = kwargs.get('sample_nodes_per_cluster', 5)

                paths = self.analyze_paths_between_semantic_clusters(
                    i, j, 
                    max_path_len=max_path_len, 
                    num_example_paths=num_example_paths, 
                    sample_nodes_per_cluster=sample_nodes_per_cluster
                )
                if paths:
                    all_paths_found.append(f"\n=== Paths between Cluster {i} and Cluster {j} ===")
                    all_paths_found.extend(paths)
                
                # Also check paths in the opposite direction as the graph is directed
                # and sampling in analyze_paths_between_semantic_clusters might yield different results
                paths_reverse = self.analyze_paths_between_semantic_clusters(
                    j, i, 
                    max_path_len=max_path_len, 
                    num_example_paths=num_example_paths, 
                    sample_nodes_per_cluster=sample_nodes_per_cluster
                )
                if paths_reverse:
                    # Avoid duplicating section header if already processed in the other direction and paths were found
                    # However, the actual paths might be different due to sampling, so add them if new
                    if not any(f"=== Paths between Cluster {j} and Cluster {i} ===" in s for s in all_paths_found) and \
                       not any(f"=== Paths between Cluster {i} and Cluster {j} ===" in s for s in all_paths_found):
                        all_paths_found.append(f"\n=== Paths between Cluster {j} and Cluster {i} ===")
                    
                    unique_reverse_paths = [p for p in paths_reverse if p not in all_paths_found]
                    all_paths_found.extend(unique_reverse_paths)

        if all_paths_found:
            try:
                with open(output_path, 'w') as f:
                    for line in all_paths_found:
                        f.write(line + "\n")
                logger.info(f"Automated inter-cluster path analysis saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving inter-cluster paths to file: {e}")
        else:
            logger.info("No inter-cluster paths found during automated analysis.")

    def analyze_threats_within_iucn_categories(self, similarity_threshold=0.85):
        """
        Analyzes threat nodes, grouping them by IUCN code, and then assessing 
        semantic similarity of descriptions within each IUCN category.
        Logs the findings and returns a structured summary.

        Args:
            similarity_threshold: Cosine similarity threshold for considering descriptions similar.

        Returns:
            dict: A dictionary where keys are IUCN codes and values are lists of 
                  semantic clusters of threat descriptions within that IUCN code.
        """
        if self.graph is None:
            logger.warning("Graph not loaded. Cannot analyze IUCN threat categories.")
            return {}
        if not self.embedding_model:
            logger.warning("Embedding model not initialized. Cannot analyze IUCN threat similarities.")
            # Attempt to initialize it if not already done by other methods
            try:
                logger.info("Attempting to initialize default SentenceTransformer model for IUCN analysis.")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # Default model
                logger.info("SentenceTransformer model initialized successfully for IUCN analysis.")
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformer model for IUCN analysis: {e}. Aborting IUCN analysis.")
                return {}

        logger.info("\n--- Analyzing Threat Descriptions within IUCN Categories ---")
        iucn_grouped_threats = defaultdict(list) # IUCN_code -> list of (original_full_string, clean_description)

        for node_name, data in self.graph.nodes(data=True):
            if data.get('type') == 'threat':
                original_full_string, clean_desc, iucn_code, _ = self._parse_iucn_tagged_string(node_name)
                if iucn_code and clean_desc:
                    iucn_grouped_threats[iucn_code].append((original_full_string, clean_desc))

        if not iucn_grouped_threats:
            logger.info("No threat nodes with IUCN codes found to analyze.")
            return {}

        analysis_results = {}

        for iucn_code, threats_in_category in iucn_grouped_threats.items():
            if len(threats_in_category) <= 1:
                logger.info(f"IUCN Code {iucn_code}: Only one unique description: '{threats_in_category[0][1][:70]}...'")
                analysis_results[iucn_code] = [{'representative': threats_in_category[0][1], 'descriptions': [threats_in_category[0][1]], 'original_strings': [threats_in_category[0][0]]}]
                continue

            logger.info(f"\nIUCN Code {iucn_code} (Analyzing {len(threats_in_category)} descriptions):")
            
            original_strings_list = [t[0] for t in threats_in_category]
            clean_descriptions_list = [t[1] for t in threats_in_category]
            
            embeddings = self.embedding_model.encode(clean_descriptions_list, show_progress_bar=False) # Progress bar might be too much here

            # Build similarity graph for descriptions within this IUCN category
            similarity_graph = nx.Graph()
            for i in range(len(clean_descriptions_list)):
                similarity_graph.add_node(i) # Node is index in clean_descriptions_list

            for i in range(len(clean_descriptions_list)):
                for j in range(i + 1, len(clean_descriptions_list)):
                    sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
                    if sim >= similarity_threshold:
                        similarity_graph.add_edge(i, j)
            
            category_clusters = []
            processed_indices = set()
            for component_indices in nx.connected_components(similarity_graph):
                if not component_indices: continue

                group_original_strings = [original_strings_list[k] for k in component_indices]
                group_clean_descriptions = [clean_descriptions_list[k] for k in component_indices]
                
                # Choose canonical representative for this sub-cluster: shortest clean description
                canonical_desc_for_sub_cluster = min(group_clean_descriptions, key=len)
                
                category_clusters.append({
                    'representative': canonical_desc_for_sub_cluster,
                    'descriptions': group_clean_descriptions,
                    'original_strings': group_original_strings # Keep original strings for reference
                })
                for k_idx in component_indices:
                    processed_indices.add(k_idx)
            
            # Handle descriptions not in any semantic cluster (singletons within this IUCN category)
            for k_idx, desc_tuple in enumerate(threats_in_category):
                if k_idx not in processed_indices:
                    category_clusters.append({
                        'representative': desc_tuple[1],
                        'descriptions': [desc_tuple[1]],
                        'original_strings': [desc_tuple[0]]
                    })
            
            analysis_results[iucn_code] = category_clusters
            logger.info(f"  Found {len(category_clusters)} semantic group(s) of descriptions for IUCN {iucn_code}.")
            for idx, cluster_info in enumerate(category_clusters):
                logger.info(f"    Group {idx+1} (Rep: '{cluster_info['representative'][:70]}...'): {len(cluster_info['descriptions'])} description(s)")
                if len(cluster_info['descriptions']) > 1 and len(cluster_info['descriptions']) < 5: # Log few examples if diverse
                    for d_idx, d_text in enumerate(cluster_info['descriptions']):
                        if d_text != cluster_info['representative']:
                            logger.info(f"      - Variant: '{d_text[:70]}...'")
                            if d_idx > 2 : break # max 2 variants

        return analysis_results

    def get_hierarchical_threat_view(self, central_identifier: str,
                                     identifier_rank: Optional[str] = None,
                                     common_name_map: Optional[dict] = None):
        """
        Creates a subgraph view centered on a specific species or taxonomic group,
        showing all aggregated threats linked to the species within that group.

        Args:
            central_identifier (str): The name of the species or taxonomic group
                                      (e.g., "Anas platyrhynchos", "Anas", "Anatidae", "Aves", "Duck", "Bird").
            identifier_rank (Optional[str]): The taxonomic rank of the central_identifier
                                             (e.g., "species", "genus", "familia", "subclassis", "classis").
                                             Casing should match keys in rank_hierarchy (e.g., "Genus", "Familia").
                                             If None, and central_identifier is a known species, it's assumed to be "Species".
                                             Required if central_identifier is a higher taxon rank name.
            common_name_map (Optional[dict]): A map for common names to (taxon_name, taxon_rank),
                                              e.g., {"Bird": ("Aves", "Subclassis")}.

        Returns:
            nx.DiGraph: A new graph with the central_identifier as a hub connected to aggregated threats.
                        Returns an empty graph if no relevant species or threats are found.
        """
        if self.graph is None:
            logger.warning("Graph not loaded. Cannot create hierarchical view.")
            return nx.DiGraph()

        effective_identifier = central_identifier
        effective_rank = identifier_rank

        if common_name_map and central_identifier in common_name_map:
            effective_identifier, effective_rank = common_name_map[central_identifier]
            logger.info(f"Resolved common name '{central_identifier}' to taxon '{effective_identifier}' (Rank: {effective_rank})")

        if not effective_rank:
            if effective_identifier in self.graph and self.graph.nodes[effective_identifier].get('type') == 'species':
                effective_rank = "Species" # Default for known species
            else:
                logger.error(f"Identifier rank must be provided for higher taxon '{effective_identifier}' or if it's not a known species node.")
                return nx.DiGraph()
        
        # Ensure consistent casing for rank matching (e.g., "Genus", "Familia")
        # This might need adjustment based on the exact casing in your rank_hierarchy strings
        # For now, let's assume title case for ranks if they are not "Species"
        formatted_rank_for_search = effective_rank.capitalize() if effective_rank.lower() != "species" else "Species"


        target_species_nodes = []
        if formatted_rank_for_search == "Species":
            if effective_identifier in self.graph and self.graph.nodes[effective_identifier].get('type') == 'species':
                target_species_nodes.append(effective_identifier)
            else:
                logger.warning(f"Species '{effective_identifier}' not found in graph.")
        else:
            search_string_in_hierarchy = f"{formatted_rank_for_search}: {effective_identifier}"
            # Some hierarchies might just list the rank name like "Aves" without "Subclassis: Aves"
            # This is a simplification; a more robust parser for rank_hierarchy might be needed.
            # For common high ranks like "Aves", we might need specific handling if the rank prefix is missing.
            is_high_rank_direct_match = effective_identifier == "Aves" # Add other direct match cases if needed

            for node, data in self.graph.nodes(data=True):
                if data.get('type') == 'species':
                    rh = data.get('taxonomy', {}).get('rank_hierarchy', [])
                    if rh: # Ensure rank_hierarchy is not None or empty
                        for rank_entry in rh:
                            if not isinstance(rank_entry, str): continue # Skip non-string entries
                            if search_string_in_hierarchy in rank_entry or \
                               (is_high_rank_direct_match and effective_identifier in rank_entry and formatted_rank_for_search in rank_entry): # crude match for Aves etc.
                                target_species_nodes.append(node)
                                break
        
        if not target_species_nodes:
            logger.info(f"No species found belonging to '{effective_identifier}' (Rank: {effective_rank}).")
            return nx.DiGraph()

        logger.info(f"Found {len(target_species_nodes)} species under '{effective_identifier}' (Rank: {effective_rank}).")

        aggregated_threats = set()
        # We could also collect predicates if we want to summarize them later
        # For simplicity, we'll just link central hub to threats directly for now.

        for species_node in target_species_nodes:
            if self.graph.has_node(species_node):
                for successor in self.graph.successors(species_node):
                    if self.graph.nodes[successor].get('type') == 'threat':
                        aggregated_threats.add(successor)
        
        if not aggregated_threats:
            logger.info(f"No threats found for species under '{effective_identifier}'.")
            return nx.DiGraph()

        # Create the new hierarchical view graph
        hub_graph = nx.DiGraph()
        hub_node_label = f"{central_identifier} (Aggregated)" # Use original identifier for the hub
        hub_graph.add_node(hub_node_label, type='taxonomic_hub')

        for threat_node in aggregated_threats:
            # Add threat node with its original attributes
            original_threat_attributes = self.graph.nodes[threat_node]
            hub_graph.add_node(threat_node, **original_threat_attributes)
            hub_graph.add_edge(hub_node_label, threat_node, relation="aggregates_threat")
            
        logger.info(f"Created hierarchical view for '{hub_node_label}' with {len(aggregated_threats)} unique threats.")
        return hub_graph

    def analyze_louvain_community_properties(self):
        """
        Analyzes detailed properties of Louvain communities detected in the giant component.
        Calculates size, avg degree, avg clustering coeff, avg betweenness centrality,
        and intra/inter-community edge stats for each community.
        Stores results in self.graph_metrics['giant_component']['louvain_community_analysis']
        """
        if not hasattr(self, 'giant_component_graph') or self.giant_component_graph is None:
            logger.warning("Giant component graph not available. Run analyze_graph_structure() first.")
            return None
        if not hasattr(self, 'louvain_communities_data') or not self.louvain_communities_data:
            logger.warning("Louvain community data not available. Ensure analyze_graph_structure() ran successfully.")
            return None

        logger.info("--- Analyzing Detailed Louvain Community Properties (on Giant Component) ---")
        
        giant_component_undirected = self.giant_component_graph.to_undirected() if self.giant_component_graph.is_directed() else self.giant_component_graph
        community_analysis_results = {}
        all_nodes_betweenness = None

        # Calculate betweenness for all nodes in the giant component (undirected) once if needed
        # The paper mentions betweenness for nodes *within* a community, but also avg betweenness *of the community*.
        # Average betweenness of nodes *in each community* (Figure 4d)
        # This seems to imply calculating betweenness on the whole (giant component) graph first.
        try:
            logger.info("Calculating betweenness centrality for all nodes in the (undirected) giant component for community analysis...")
            all_nodes_betweenness = nx.betweenness_centrality(giant_component_undirected, normalized=True)
            logger.info("Betweenness centrality calculation complete.")
        except Exception as e:
            logger.error(f"Could not calculate betweenness centrality for giant component: {e}", exc_info=True)
            # Proceed without betweenness if it fails

        for comm_id, nodes_in_comm in self.louvain_communities_data.items():
            if not nodes_in_comm:
                logger.warning(f"Community {comm_id} has no nodes. Skipping.")
                continue

            comm_subgraph = giant_component_undirected.subgraph(nodes_in_comm)
            comm_metrics = {"size": len(nodes_in_comm)}

            # Average node degree within the community (based on edges within the community subgraph)
            if comm_subgraph.number_of_nodes() > 0:
                comm_degrees = [d for n, d in comm_subgraph.degree()]
                comm_metrics["avg_degree_internal"] = np.mean(comm_degrees) if comm_degrees else 0
            else:
                comm_metrics["avg_degree_internal"] = 0

            # Average clustering coefficient of nodes within the community
            # nx.average_clustering(comm_subgraph) computes it for the subgraph
            if comm_subgraph.number_of_nodes() > 0:
                try:
                    comm_metrics["avg_clustering_coefficient"] = nx.average_clustering(comm_subgraph)
                except Exception as e:
                    logger.warning(f"Could not calculate avg clustering for comm {comm_id}: {e}")
                    comm_metrics["avg_clustering_coefficient"] = 0.0
            else:
                comm_metrics["avg_clustering_coefficient"] = 0.0

            # Average betweenness centrality of nodes IN this community (using precomputed values on giant component)
            if all_nodes_betweenness:
                comm_node_betweenness_values = [all_nodes_betweenness.get(node, 0) for node in nodes_in_comm]
                comm_metrics["avg_betweenness_centrality"] = np.mean(comm_node_betweenness_values) if comm_node_betweenness_values else 0
            else:
                comm_metrics["avg_betweenness_centrality"] = 0.0 # Default if global calculation failed

            # Intra-community and Inter-community edges (from the perspective of this community)
            # Using the original giant_component_graph (potentially directed) to count edges
            num_intra_community_edges = 0
            num_inter_community_edges = 0
            
            for node_u in nodes_in_comm:
                if node_u not in self.giant_component_graph: continue
                for node_v in self.giant_component_graph.successors(node_u):
                    if node_v in nodes_in_comm: # Edge is within the community
                        num_intra_community_edges += 1
                    else: # Edge goes to a different community (or a node not in any community if that's possible)
                        num_inter_community_edges += 1
                # If graph is undirected, and we used directed successors, we need to count predecessors too for inter-community edges
                # However, louvain_communities_data comes from an undirected graph partition.
                # The paper's Fig 4e implies total edges within vs. total edges between.
                # For simplicity here, we count outgoing edges from the community.
                # A more robust definition would be: intra = edges (u,v) where u,v in C. inter = edges (u,v) where u in C, v not in C.

            comm_metrics["num_intra_community_edges"] = num_intra_community_edges
            comm_metrics["num_inter_community_edges_outgoing"] = num_inter_community_edges # Edges originating from this community to others
            
            # The paper also mentions avg number of edges. If it means edges per node:
            if len(nodes_in_comm) > 0:
                comm_metrics["avg_intra_community_edges_per_node"] = num_intra_community_edges / len(nodes_in_comm)
                comm_metrics["avg_inter_community_edges_per_node_outgoing"] = num_inter_community_edges / len(nodes_in_comm)
            else:
                comm_metrics["avg_intra_community_edges_per_node"] = 0
                comm_metrics["avg_inter_community_edges_per_node_outgoing"] = 0

            community_analysis_results[comm_id] = comm_metrics
            logger.info(f"  Analyzed Community {comm_id}: Size={comm_metrics['size']}, AvgDegInt={comm_metrics['avg_degree_internal']:.2f}, AvgClust={comm_metrics['avg_clustering_coefficient']:.2f}, AvgBetw={comm_metrics['avg_betweenness_centrality']:.4f}")

        if hasattr(self, 'graph_metrics') and 'giant_component' in self.graph_metrics:
            self.graph_metrics['giant_component']['louvain_community_details'] = community_analysis_results
            logger.info("Stored detailed Louvain community properties in graph_metrics.")
        else:
            logger.warning("graph_metrics or giant_component key not found. Detailed community analysis not stored globally.")
        
        return community_analysis_results

    def plot_community_size_distribution(self, filename=None):
        """
        Plots the distribution of Louvain community sizes (similar to Fig 4a).
        Assumes analyze_louvain_community_properties has been run and data is in graph_metrics.
        """
        if not hasattr(self, 'graph_metrics') or \
           not self.graph_metrics.get('giant_component', {}).get('louvain_community_details'):
            logger.warning("Louvain community details not found. Run analyze_louvain_community_properties first.")
            return None

        community_details = self.graph_metrics['giant_component']['louvain_community_details']
        sizes = sorted([details['size'] for details in community_details.values()], reverse=True)

        if not sizes:
            logger.warning("No community sizes found to plot.")
            return None

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sizes)), sizes, color='skyblue')
        plt.xlabel("Community Rank (by size)")
        plt.ylabel("Community Size (Number of Nodes)")
        plt.title("Distribution of Louvain Community Sizes in Giant Component")
        plt.yscale('log') # Figure 4a in paper seems to have y-axis potentially log-scaled or just skewed.
        plt.tight_layout()

        if filename:
            try:
                full_path = self.figures_dir / filename
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                logger.info(f"Community size distribution plot saved to {full_path}")
            except Exception as e:
                logger.error(f"Error saving community size distribution plot: {e}")
        return plt.gcf()

    def plot_community_size_vs_clustering_vs_degree(self, filename=None):
        """
        Plots community size vs. avg clustering coefficient, colored by avg internal degree (similar to Fig 5).
        Assumes analyze_louvain_community_properties has been run.
        """
        if not hasattr(self, 'graph_metrics') or \
           not self.graph_metrics.get('giant_component', {}).get('louvain_community_details'):
            logger.warning("Louvain community details not found for plotting. Run analyze_louvain_community_properties first.")
            return None

        community_details = self.graph_metrics['giant_component']['louvain_community_details']
        
        sizes = []
        avg_clustering_coeffs = []
        avg_degrees = []

        for comm_id, details in community_details.items():
            sizes.append(details.get('size', 0))
            avg_clustering_coeffs.append(details.get('avg_clustering_coefficient', 0))
            avg_degrees.append(details.get('avg_degree_internal', 0))

        if not sizes or not avg_clustering_coeffs or not avg_degrees:
            logger.warning("Not enough data to plot community size vs clustering vs degree.")
            return None

        df = pd.DataFrame({
            'size': sizes,
            'avg_clustering': avg_clustering_coeffs,
            'avg_degree': avg_degrees
        })
        # Filter out communities with size 0 if any (should not happen with proper community detection)
        df = df[df['size'] > 0]
        if df.empty:
            logger.warning("No valid community data after filtering for plotting.")
            return None

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df['size'], df['avg_clustering'], c=df['avg_degree'], cmap='viridis', s=50, alpha=0.7, edgecolors='w', linewidth=0.5)
        
        plt.xscale('log')
        # Figure 5 in paper has y-axis (avg clustering coeff) also log-scaled. Let's check if values are suitable.
        # If avg_clustering_coeffs can be 0, log scale will fail. We might need to add a small epsilon or use symlog.
        # For now, let's try linear y-axis, and adjust if needed based on typical data values.
        # plt.yscale('log') 
        plt.xlabel("Community Size (Number of Nodes) - Log Scale")
        plt.ylabel("Average Clustering Coefficient") # Potentially Log Scale if data suits
        plt.title("Community Size vs. Avg Clustering Coefficient (Color: Avg Internal Degree)")
        
        cbar = plt.colorbar(scatter, label='Average Internal Degree')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()

        if filename:
            try:
                full_path = self.figures_dir / filename
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                logger.info(f"Community size vs clustering plot saved to {full_path}")
            except Exception as e:
                logger.error(f"Error saving community size vs clustering plot: {e}")
        return plt.gcf()

    def analyze_top_nodes_in_louvain_communities(self, num_top_nodes=5, specific_community_ids: Optional[List[int]] = None, filename=None):
        """
        Analyzes and optionally plots the degree distribution of the top N nodes 
        within specified Louvain communities (or a selection if specific_community_ids is None).
        Similar to Figure 6 from the paper.

        Args:
            num_top_nodes: Number of top degree nodes to analyze/plot per community.
            specific_community_ids: A list of community IDs to analyze. If None, a sample of communities might be chosen.
            filename: If provided, saves the plot to this file in the figures_dir.

        Returns:
            A dictionary containing the top nodes and their degrees for each analyzed community.
            Also returns the matplotlib figure if plotted.
        """
        if not hasattr(self, 'giant_component_graph') or self.giant_component_graph is None:
            logger.warning("Giant component graph not available. Run analyze_graph_structure() first.")
            return None, None
        if not hasattr(self, 'louvain_communities_data') or not self.louvain_communities_data:
            logger.warning("Louvain community data not available. Ensure analyze_graph_structure() ran successfully.")
            return None, None

        logger.info(f"--- Analyzing Top {num_top_nodes} Nodes in Louvain Communities ---")
        
        target_community_ids = []
        if specific_community_ids:
            target_community_ids = [cid for cid in specific_community_ids if cid in self.louvain_communities_data]
            if len(target_community_ids) != len(specific_community_ids):
                logger.warning("Some specified community IDs were not found.")
        else:
            # If no specific IDs, pick a few (e.g., largest or a sample)
            # For now, let's pick up to 6 largest communities if not specified
            sorted_communities = sorted(self.louvain_communities_data.items(), key=lambda item: len(item[1]), reverse=True)
            target_community_ids = [cid for cid, nodes in sorted_communities[:6]]
            logger.info(f"No specific community IDs provided, analyzing up to 6 largest: {target_community_ids}")

        if not target_community_ids:
            logger.warning("No target communities to analyze.")
            return None, None

        analysis_results = {}
        num_communities_to_plot = len(target_community_ids)
        
        # Determine subplot layout (e.g., 2x3 or 3x2 for 6 communities)
        if num_communities_to_plot == 0 : return {}, None # Should be caught by earlier check
        
        # Simplified: Aim for a roughly square layout, or max 3 columns
        ncols = min(3, num_communities_to_plot) 
        nrows = (num_communities_to_plot + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        plot_idx = 0

        for comm_id in target_community_ids:
            nodes_in_comm = self.louvain_communities_data.get(comm_id)
            if not nodes_in_comm:
                logger.warning(f"Community {comm_id} not found or is empty. Skipping.")
                if plot_idx < len(axes_flat):
                    axes_flat[plot_idx].axis('off') # Turn off empty subplot
                    axes_flat[plot_idx].set_title(f"Community {comm_id}\n(Not found or empty)", fontsize=10)
                plot_idx += 1
                continue

            # Get degrees of nodes within this community (degree in the context of the giant component)
            node_degrees_in_comm = []
            for node_name in nodes_in_comm:
                if node_name in self.giant_component_graph:
                    degree = self.giant_component_graph.degree(node_name)
                    node_degrees_in_comm.append((node_name, degree))
            
            # Sort by degree and get top N
            top_nodes = sorted(node_degrees_in_comm, key=lambda x: x[1], reverse=True)[:num_top_nodes]
            analysis_results[comm_id] = {'top_nodes': dict(top_nodes)}
            logger.info(f"  Community {comm_id} (Size: {len(nodes_in_comm)}): Top {len(top_nodes)} nodes: {top_nodes}")

            # Plotting for this community
            if plot_idx < len(axes_flat):
                ax = axes_flat[plot_idx]
                node_labels = [item[0][:20] + '...' if len(item[0]) > 20 else item[0] for item in top_nodes] # Truncate labels
                degrees = [item[1] for item in top_nodes]
                
                ax.bar(node_labels, degrees, color=plt.cm.viridis(plot_idx / num_communities_to_plot)) # Different color per community plot
                ax.set_title(f"Community {comm_id} (Top {len(top_nodes)} Nodes)", fontsize=10)
                ax.set_ylabel("Degree in Giant Component", fontsize=8)
                ax.tick_params(axis='x', rotation=45, labelsize=7)
                ax.tick_params(axis='y', labelsize=8)
            plot_idx += 1
        
        # Turn off any unused subplots
        for i in range(plot_idx, len(axes_flat)):
            axes_flat[i].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        fig.suptitle(f"Degree Distribution of Top Nodes in Selected Louvain Communities", fontsize=14, y=0.99)

        if filename:
            try:
                full_path = self.figures_dir / filename
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                logger.info(f"Top nodes in communities plot saved to {full_path}")
            except Exception as e:
                logger.error(f"Error saving top nodes in communities plot: {e}")
        
        return analysis_results, fig

    def run_full_analysis(self, output_dir=None):
        """
        Run a complete analysis pipeline with visualizations.
        
        Args:
            output_dir: Directory to save output files (None for default)
        
        Returns:
            dict: Analysis results
        """
        if output_dir is None:
            output_dir = self.figures_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Load graph
        if self.graph is None:
            self.load_graph_from_triplets()
        
        if self.graph is None:
            logger.error("Failed to load graph. Aborting analysis.")
            return None
        
        try:
            # 1. Analyze graph structure
            logger.info("1. Analyzing graph structure...")
            structure = self.analyze_graph_structure()
            results['structure'] = structure
            
            # 2. Visualize graph
            logger.info("2. Visualizing graph...")
            self.visualize_graph(filename=os.path.join(output_dir, "graph_visualization.png"))
            
            # 3. Analyze degree distribution
            logger.info("3. Analyzing degree distribution...")
            self.analyze_degree_distribution(filename=os.path.join(output_dir, "degree_distribution.png"))
            
            # 4. Visualize communities
            logger.info("4. Visualizing communities...")
            self.visualize_communities(filename=os.path.join(output_dir, "community_structure.png"))
            
            # 5. Calculate embeddings and visualize
            logger.info("5. Calculating and visualizing embeddings...")
            if self.embeddings is None:
                self.calculate_embeddings()
            
            if self.embeddings is not None:
                # PCA visualization (Global Graph)
                logger.info("5a. Visualizing Embeddings (PCA) for Global Graph...")
                pca_global_result = self.visualize_embeddings(method='pca', n_clusters=5, filename=os.path.join(output_dir, "embeddings_pca_global.png"), target_graph=self.graph)
                if pca_global_result:
                    _, cs_pca_g, df_pca_g = pca_global_result
                    results['clusters_pca_global'] = cs_pca_g
                
                # t-SNE visualization (Global Graph)
                logger.info("5b. Visualizing Embeddings (t-SNE) for Global Graph...")
                tsne_global_result = self.visualize_embeddings(method='tsne', n_clusters=5, filename=os.path.join(output_dir, "embeddings_tsne_global.png"), target_graph=self.graph)
                if tsne_global_result:
                    _, cs_tsne_g, df_tsne_g = tsne_global_result
                    results['clusters_tsne_global'] = cs_tsne_g

                # PCA visualization (Giant Component if available)
                if hasattr(self, 'giant_component_graph') and self.giant_component_graph and self.giant_component_graph.number_of_nodes() > 0:
                    logger.info("5c. Visualizing Embeddings (PCA) for Giant Component...")
                    pca_gc_result = self.visualize_embeddings(method='pca', n_clusters=5, filename=os.path.join(output_dir, "embeddings_pca_giant_component.png"), target_graph=self.giant_component_graph)
                    if pca_gc_result:
                        _, cs_pca_gc, _ = pca_gc_result # df not stored here to avoid too much data in results
                        results['clusters_pca_giant_component'] = cs_pca_gc
                    
                    logger.info("5d. Visualizing Embeddings (t-SNE) for Giant Component...")
                    tsne_gc_result = self.visualize_embeddings(method='tsne', n_clusters=5, filename=os.path.join(output_dir, "embeddings_tsne_giant_component.png"), target_graph=self.giant_component_graph)
                    if tsne_gc_result:
                        _, cs_tsne_gc, _ = tsne_gc_result
                        results['clusters_tsne_giant_component'] = cs_tsne_gc
                else:
                    logger.info("Giant component not available or empty, skipping its embedding visualization.")
            else:
                logger.warning("Embeddings not available, skipping embedding visualizations.")
            
            # 6. Find structural patterns (typically on global graph)
            logger.info("6. Finding structural patterns...")
            patterns = self.find_structural_patterns()
            results['patterns'] = patterns
            
            # 7. Create summary report
            logger.info("7. Creating summary report...")
            summary = self.summarize_analysis() # summarize_analysis should use self.graph_metrics if populated
            results['summary'] = summary
            
            # 7b. Perform and plot detailed Louvain community analysis from giant component (Figures 4a, 5, 6)
            if hasattr(self, 'giant_component_graph') and self.giant_component_graph and \
               hasattr(self, 'louvain_communities_data') and self.louvain_communities_data:
                logger.info("7b. Performing detailed Louvain community analysis and plotting...")
                # Calculate properties (results stored in self.graph_metrics['giant_component']['louvain_community_details'])
                self.analyze_louvain_community_properties() 
                
                # Plot Fig 4a style: Community size distribution
                self.plot_community_size_distribution(filename=os.path.join(output_dir, "louvain_community_size_dist_giant.png"))
                
                # Plot Fig 5 style: Community size vs clustering vs degree
                self.plot_community_size_vs_clustering_vs_degree(filename=os.path.join(output_dir, "louvain_community_size_vs_clust_vs_degree_giant.png"))
                
                # Plot Fig 6 style: Top nodes in some communities
                # Use default (up to 6 largest) communities for the plot
                self.analyze_top_nodes_in_louvain_communities(num_top_nodes=5, filename=os.path.join(output_dir, "louvain_top_nodes_degree_dist_giant.png"))
            else:
                logger.warning("Giant component or Louvain community data not ready for detailed community plots.")

            # 7c. Example: Finding paths between concepts using embeddings (Paper section 2.2)
            # This is an example of an ad-hoc query. Replace concepts with those relevant to your specific investigation.
            logger.info("7c. Example: Finding paths between bird species and a threat type using embeddings...")
            concept1_example = "Buff-breasted Paradise-kingfisher"
            concept2_example = "predation"
            example_paths = self.find_paths_between_concepts_via_embeddings(concept1_example, concept2_example, k_shortest_paths=2, max_path_len=4)
            if example_paths:
                results[f'example_paths_{concept1_example}_to_{concept2_example}'] = example_paths
                # Log the found paths for quick review
                for idx, path in enumerate(example_paths):
                    path_str_parts = [f"'{path[0][0]}'"] # Start with the first node
                    for (from_n, rel, to_n) in path:
                        path_str_parts.append(f"--[{rel}]--> '{to_n}'")
                    logger.info(f"  Example Path {idx+1} ({concept1_example} to {concept2_example}): {' '.join(path_str_parts)}")
            else:
                logger.info(f"Could not find paths between '{concept1_example}' and '{concept2_example}' for the example.")

            # 7d. Semantic K-Means based inter-cluster path analysis (from visualize_embeddings)
            if hasattr(self, 'embeddings_df') and self.embeddings_df is not None and \
               hasattr(self, 'n_semantic_clusters') and self.n_semantic_clusters is not None and self.n_semantic_clusters > 1:
                logger.info("7d. Automating inter-semantic-cluster path analysis...")
                # Example: Analyze paths between cluster 0 and cluster 1
                # self.analyze_paths_between_semantic_clusters(0, 1) # n_clusters and embeddings_df are now instance vars
                # self.analyze_paths_between_semantic_clusters(0, 4)
                
                # Run automated analysis for all pairs
                self.automate_all_inter_cluster_path_analysis(
                    max_path_len=3, 
                    num_example_paths=2, 
                    sample_nodes_per_cluster=3
                )
            else:
                logger.warning("Embeddings data not fully available, skipping automated inter-cluster path analysis.")

            # 8. Analyze threats within IUCN categories
            logger.info("\n8. Analyzing threats within IUCN categories...") # Corrected step number in log
            iucn_threat_analysis = self.analyze_threats_within_iucn_categories()
            results['iucn_threat_analysis'] = iucn_threat_analysis # Add to main results

            # 9. Save results to JSON
            results_file = os.path.join(output_dir, "graph_analysis_results.json")
            with open(results_file, 'w') as f:
                # Convert non-serializable objects to strings
                serializable_results = json.loads(json.dumps(results, default=str))
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Analysis complete. Results saved to {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error running full analysis: {e}", exc_info=True)
            return None

    def find_paths_between_concepts_via_embeddings(self, concept1_text: str, concept2_text: str, k_shortest_paths=1, max_path_len=5):
        """
        Finds paths in the graph between two concepts described by free text.
        It first finds the closest matching nodes in the graph for the input concept texts
        using sentence embeddings, then finds shortest paths between these nodes.

        Args:
            concept1_text (str): Text describing the first concept.
            concept2_text (str): Text describing the second concept.
            k_shortest_paths (int): Number of shortest paths to find.
            max_path_len (int): Maximum length of paths to consider (for all_simple_paths if shortest_simple_paths yields nothing long enough).

        Returns:
            List[List[tuple]]: A list of paths, where each path is a list of (from_node, relation, to_node) tuples.
                               Returns None if embeddings are not available or critical errors occur.
                               Returns an empty list if no paths are found.
        """
        if self.graph is None:
            logger.warning("Graph not loaded. Cannot find paths.")
            return None
        if self.embeddings is None:
            logger.warning("Node embeddings not calculated. Call calculate_embeddings() first.")
            # Try to calculate if not present, using default model
            logger.info("Attempting to calculate embeddings with default model...")
            self.calculate_embeddings()
            if self.embeddings is None:
                logger.error("Failed to calculate embeddings. Cannot find paths between concepts.")
                return None
        if not self.embedding_model:
             logger.error("Embedding model not available. Cannot find paths between concepts.")
             return None

        logger.info(f"Attempting to find paths between concepts: '{concept1_text}' and '{concept2_text}'")

        try:
            # Get embeddings for the input concept texts
            concept1_embedding = self.embedding_model.encode([concept1_text])[0]
            concept2_embedding = self.embedding_model.encode([concept2_text])[0]

            graph_node_list = list(self.embeddings.keys())
            graph_node_embeddings = np.array([self.embeddings[node] for node in graph_node_list])

            # Find closest node in graph to concept1
            similarities1 = cosine_similarity(concept1_embedding.reshape(1, -1), graph_node_embeddings)[0]
            closest_node1_idx = np.argmax(similarities1)
            closest_node1 = graph_node_list[closest_node1_idx]
            similarity_score1 = similarities1[closest_node1_idx]
            logger.info(f"Closest node to '{concept1_text}': '{closest_node1}' (Similarity: {similarity_score1:.4f})")

            # Find closest node in graph to concept2
            similarities2 = cosine_similarity(concept2_embedding.reshape(1, -1), graph_node_embeddings)[0]
            closest_node2_idx = np.argmax(similarities2)
            closest_node2 = graph_node_list[closest_node2_idx]
            similarity_score2 = similarities2[closest_node2_idx]
            logger.info(f"Closest node to '{concept2_text}': '{closest_node2}' (Similarity: {similarity_score2:.4f})")

            if closest_node1 == closest_node2:
                logger.info(f"Both concepts map to the same node: '{closest_node1}'. No path to find between a node and itself in this context.")
                return []
            
            if closest_node1 not in self.graph or closest_node2 not in self.graph:
                logger.warning("One or both mapped concept nodes are not in the graph (this shouldn't happen if embeddings are from graph nodes).")
                return []

            # Use the existing find_shortest_paths method, but it returns list of tuples.
            # The paper describes: graphene --> improves --> strength --> ... --> silk
            # This implies a sequence of nodes and the relations on edges.
            
            found_paths_detailed = []
            try:
                # Using shortest_simple_paths which gives node lists
                paths_nodes_only = list(nx.shortest_simple_paths(self.graph, source=closest_node1, target=closest_node2))[:k_shortest_paths]
                
                if not paths_nodes_only:
                    logger.info(f"No direct shortest path found up to k={k_shortest_paths}. Trying all_simple_paths with cutoff={max_path_len}.")
                    # Fallback to all_simple_paths if no direct path or if we want more variety
                    paths_nodes_only = list(nx.all_simple_paths(self.graph, source=closest_node1, target=closest_node2, cutoff=max_path_len))[:k_shortest_paths]

                for path_nodes in paths_nodes_only:
                    path_with_relations = []
                    path_str_parts = [f"'{path_nodes[0]}'"]
                    for i in range(len(path_nodes) - 1):
                        from_node = path_nodes[i]
                        to_node = path_nodes[i+1]
                        relation = self.graph.get_edge_data(from_node, to_node).get('relation', '[unknown_relation]')
                        path_with_relations.append((from_node, relation, to_node))
                        path_str_parts.append(f"--[{relation}]--> '{to_node}'")
                    found_paths_detailed.append(path_with_relations)
                    logger.info(f"  Found Path: {' '.join(path_str_parts)}")
            
            except nx.NetworkXNoPath:
                logger.warning(f"No path found between '{closest_node1}' and '{closest_node2}' in the graph.")
                return []
            except Exception as e:
                logger.error(f"Error finding paths between '{closest_node1}' and '{closest_node2}': {e}", exc_info=True)
                return []

            if not found_paths_detailed:
                logger.info(f"No paths found between mapped nodes '{closest_node1}' and '{closest_node2}'.")

            return found_paths_detailed

        except Exception as e:
            logger.error(f"Error in find_paths_between_concepts_via_embeddings: {e}", exc_info=True)
            return None

def main():
    """Main function to run the graph analysis for all run directories."""
    # Find all run/*/results directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = Path(current_dir)
    run_dir = base_dir / "Lent_Init" / "runs"
    
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return
    
    # Find all subdirectories under runs/
    run_subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
    
    if not run_subdirs:
        print("No subdirectories found under runs/")
        return
    
    for run_subdir in run_subdirs:
        results_dir = run_subdir / "results"
        if results_dir.exists():
            print(f"\n{'='*60}")
            print(f"Processing: {results_dir}")
            print(f"{'='*60}")
            
            # Create analyzer for this specific results directory
            analyzer = GraphAnalyzer(results_dir=results_dir)
            graph = analyzer.load_graph_from_triplets()
            
            if graph is None:
                print(f"Failed to load graph from {results_dir}. Skipping.")
                continue
            
            print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
            
            # Run full analysis
            print("Running full graph analysis...")
            # Update run_full_analysis call to use the correct output_dir derived from results_dir
            analysis_output_dir = analyzer.figures_dir # figures_dir is already set up correctly in __init__
            results = analyzer.run_full_analysis(output_dir=analysis_output_dir)
            
            if results:
                print(f"Analysis complete for {run_subdir.name}. Check the figures directory for visualizations.")
            else:
                print(f"Analysis failed for {run_subdir.name}.")
        else:
            print(f"Results directory not found in {run_subdir}")

if __name__ == "__main__":
    main() 