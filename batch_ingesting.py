import json
import time
import polars as pl
import nltk
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
import hashlib
import pickle
from pathlib import Path
import os
from dotenv import load_dotenv
import requests
import http.server
import socketserver
import webbrowser
from threading import Timer
import socket
from bs4 import BeautifulSoup
from collections import defaultdict
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from thefuzz import fuzz
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import asyncio # Added for asynchronous operations
import argparse # For command-line arguments for the verification step
import sys # To exit after verification step is run
import aiohttp # Added for asynchronous HTTP requests
from openai import OpenAI  # Added for OpenRouter integration
import logging # Added for logging

# === BATCH PROCESSING CONFIGURATIONS ===
# Default BATCH_CONFIG, can be overridden by command-line argument
BATCH_CONFIG = {
    'summary_batch_size': 5, 
    'triplet_batch_size': 3,   
    'max_summary_workers': 20, 
    'max_triplet_workers': 15, 
    'max_enrichment_workers': 10, 
    'enable_batch_processing': True, # Default to True, can be overridden
    'processing_batch_size': 500  # New: Size of batches for overall processing to prevent memory issues
}

# === ENHANCED BATCH SEMAPHORE MANAGER ===
class BatchSemaphoreManager:
    """Manages semaphores for different types of batch workers"""
    def __init__(self, config=None):
        current_config = config or BATCH_CONFIG
        self.summary_semaphore = asyncio.Semaphore(current_config['max_summary_workers'])
        self.triplet_semaphore = asyncio.Semaphore(current_config['max_triplet_workers'])
        self.enrichment_semaphore = asyncio.Semaphore(current_config['max_enrichment_workers'])
        # Add a general verification semaphore if needed for verify_triplets or other general async tasks
        self.verification_semaphore = asyncio.Semaphore(current_config.get('max_verification_workers', 10)) # Default to 10


# --- Global Logger Setup ---
# This will be configured by the main execution function (run_main_pipeline_logic or standalone stage functions)
logger = logging.getLogger("pipeline")
# Prevent duplicate handlers if script is re-run or functions are called multiple times in some contexts
if not logger.handlers:
    logger.setLevel(logging.DEBUG) # Default level, can be overridden by handlers
    # Basic console handler for initial setup, will be replaced by specific config
    ch_setup = logging.StreamHandler()
    ch_setup.setLevel(logging.INFO)
    formatter_setup = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch_setup.setFormatter(formatter_setup)
    logger.addHandler(ch_setup)

# Try to import sentence-transformers, handle if not available
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("sentence-transformers not installed. Some embedding features will be unavailable.")
    print("Install with: pip install sentence-transformers")
    EMBEDDINGS_AVAILABLE = False

def setup_pipeline_logging(log_file_path: Path):
    """Configures the global 'pipeline' logger to write to a file and console."""
    # Remove any existing handlers to prevent duplication if called multiple times
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG) # Capture all levels for the file

    # File Handler - for detailed logs
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # Console Handler - for general info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    logger.info(f"Logging configured. Log file: {log_file_path}")

def get_dynamic_run_base_path(model_name: str, max_r_val: Optional[Union[int, str]], current_script_dir: Path) -> Path:
    """
    Generates the base path for a specific run's outputs.
    Pattern: current_script_dir/runs/sanitized_model_name_max_r_str/
    Example: .../Lent_Init/runs/google_gemini-2-0-flash-001_1000/
    If max_r_val is None or "all", it uses "all".
    """
    model_name_sanitized = model_name.replace("/", "_").replace(":", "_").replace(".", "-")
    max_r_str = str(max_r_val) if max_r_val is not None and str(max_r_val).lower() != "all" else "all"
    
    run_folder_name = f"{model_name_sanitized}_{max_r_str}"
    # Place the 'runs' directory directly inside the current_script_dir (e.g., Lent_Init/runs/...)
    return current_script_dir / "runs" / run_folder_name

def load_data(file_name, max_rows=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path_to_load = None

    if os.path.isabs(file_name):
        file_path_to_load = file_name
    elif file_name == "all_abstracts.parquet":
        # Enhanced path checking for all_abstracts.parquet, similar to load_data_with_offset
        possible_paths = [
            os.path.join("/app", file_name),  # Docker typical path
            os.path.join(os.path.dirname(current_dir), file_name), # Path relative to parent (e.g. /app if current_dir is /app/Lent_Init)
            os.path.join(current_dir, file_name), # Path relative to current script dir
            os.path.join(os.path.expanduser("~"), "Desktop", file_name), # User's Desktop
            # Add any other common locations if necessary, like a direct absolute path for local dev
            # For example: "/Users/your_username/path_to_data/all_abstracts.parquet"
        ]
        for path_option in possible_paths:
            if os.path.exists(path_option):
                file_path_to_load = path_option
                break
        if file_path_to_load is None:
            # Try one level up from current_dir as a common project structure
            # e.g. if script is in Lent_Init and file is in project root
            parent_dir_path = os.path.join(os.path.dirname(current_dir), file_name)
            if os.path.exists(parent_dir_path):
                file_path_to_load = parent_dir_path
            else:
                # Last resort: if running in Docker and it's copied to /app
                docker_app_path = f"/app/{file_name}"
                if os.path.exists(docker_app_path):
                    file_path_to_load = docker_app_path
                else:
                    raise FileNotFoundError(f"Could not find '{file_name}' in any of the expected locations: {possible_paths + [parent_dir_path, docker_app_path]}")
    else:
        # For other files, assume they are relative to the script directory
        file_path_to_load = os.path.join(current_dir, file_name)

    if not os.path.exists(file_path_to_load):
         raise FileNotFoundError(f"Specified file '{file_name}' (resolved to '{file_path_to_load}') does not exist.")

    logger.info(f"Loading data from: {file_path_to_load}")
    
    # Determine file type and read accordingly
    if file_name.endswith(".parquet"):
        logger.info("Reading as parquet file...")
        if max_rows is not None:
            # Polars read_parquet doesn't have n_rows directly like read_csv.
            # Read full then take head, or use pyarrow for more complex partial reads if performance is critical for large files.
            df = pl.read_parquet(file_path_to_load)
            df = df.head(max_rows)
        else:
            df = pl.read_parquet(file_path_to_load)
    elif file_name.endswith(".csv"):
        logger.info("Reading as CSV file...")
        if max_rows is not None:
            df = pl.read_csv(file_path_to_load, n_rows=max_rows)
        else:
            df = pl.read_csv(file_path_to_load)
    else:
        raise ValueError(f"Unsupported file type for {file_name}. Please use .parquet or .csv")

    df = df.drop_nulls(["title", "abstract", "doi"])
    logger.info(f"Loaded {len(df)} rows after dropping nulls from {file_path_to_load}")
    return df

def load_data_with_offset(file_name, skip_rows=0, max_rows=1000):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if file is on desktop (for the all_abstracts.parquet case)
    if file_name == "all_abstracts.parquet":
        possible_paths = [
            os.path.join("/app", file_name),
            os.path.join(os.path.expanduser("~"), "Desktop", file_name),
            os.path.join(current_dir, file_name),
            os.path.join(os.path.dirname(current_dir), file_name),
            os.path.join("/Users/kittsonhamill/Desktop", file_name)
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
                
        if file_path is None:
            raise FileNotFoundError(f"Could not find {file_name} in any of the expected locations: {possible_paths}")
    else:
        file_path = os.path.join(current_dir, file_name)
    
    print(f"Loading from: {file_path} (Skipping {skip_rows} rows, loading {max_rows} rows)")
    
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
        
        # Open the parquet file
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        
        if skip_rows >= total_rows:
            return pl.DataFrame()  # Return an empty DataFrame if we skip past EOF
        
        # Use a fixed batch size for iteration (e.g., 1024 rows)
        batches = parquet_file.iter_batches(batch_size=1024)
        rows_needed = max_rows
        rows_skipped = 0
        arrow_batches = []
        
        # Iterate over the batches until we've skipped the first skip_rows rows and then collected max_rows rows
        for batch in batches:
            batch_len = batch.num_rows
            if rows_skipped + batch_len <= skip_rows:
                rows_skipped += batch_len
                continue  # Skip this entire batch
            
            # Determine the starting row in the current batch
            start_in_batch = max(0, skip_rows - rows_skipped)
            available = batch_len - start_in_batch
            to_take = min(available, rows_needed)
            sliced_batch = batch.slice(start_in_batch, to_take)
            arrow_batches.append(sliced_batch)
            rows_needed -= to_take
            rows_skipped += batch_len
            
            if rows_needed <= 0:
                break
        
        if arrow_batches:
            # Create a table directly from the record batches
            table = pa.Table.from_batches(arrow_batches)
            df = pl.from_arrow(table)
        else:
            df = pl.DataFrame()
            
    except Exception as e:
        print(f"Error with PyArrow: {e}")
        print("Falling back to basic Polars read...")
        df = pl.read_parquet(file_path)
        if skip_rows >= len(df):
            return pl.DataFrame()  # Return empty if skip_rows is beyond file length
        end_idx = min(skip_rows + max_rows, len(df))
        df = df[skip_rows:end_idx]
    
    # Drop rows with null values in "title" or "abstract"
    df = df.drop_nulls(["title", "abstract", "doi"])
    print(f"Loaded {len(df)} rows after dropping nulls")
    return df


# follows the: https://thedataschool.com/salome-grasland/using-the-isbndb-api-with-python/
# Uses the same idea/setup of rate limiting, but per minute instead of calls_per_secpnd like in the example I followed
class RateLimiter:
    def __init__(self, rpm: int = 2, is_ollama: bool = False):
        self.rpm = rpm 
        self.last_call = 0 
        self.interval = 60.0 / self.rpm
        self.backoff_time = 0
        self.is_ollama = is_ollama
        # Remove the 2-second minimum wait for non-Ollama APIs, rely purely on RPM-calculated interval
        self.min_wait = 0
        
    def wait(self):
        # No waiting needed for Ollama
        if self.is_ollama:
            return
            
        now = time.time()
        elapsed = now - self.last_call
        
        # Calculate base wait time
        wait_time = max(self.interval - elapsed, self.min_wait)
        
        # Add backoff if we've hit rate limits
        if self.backoff_time > 0:
            wait_time = max(wait_time, self.backoff_time)
            self.backoff_time *= 0.5  # Gradually reduce backoff
        
        if wait_time > 0:
            print(f"Waiting {wait_time:.1f} seconds before next request...")
            time.sleep(wait_time)
            self.last_call = time.time()  # Update last_call after the wait
    
    def handle_rate_limit(self):
        """Called when a rate limit is hit to increase backoff"""
        self.backoff_time = max(60, self.backoff_time * 2)
        # Force an immediate wait when rate limit is hit
        self.wait()

    async def async_wait(self):
        """Asynchronous wait for rate limiting."""
        if self.is_ollama: # Assuming this might be relevant for future async ollama
            return

        now = time.time()
        elapsed = now - self.last_call
        wait_time = max(self.interval - elapsed, self.min_wait)

        if self.backoff_time > 0:
            wait_time = max(wait_time, self.backoff_time)
            # Backoff reduction can happen after a successful call or be managed by handle_async_rate_limit

        if wait_time > 0:
            print(f"AsyncRateLimiter: Waiting {wait_time:.1f} seconds...")
            await asyncio.sleep(wait_time)
            self.last_call = time.time()
    
    def handle_async_rate_limit(self):
        """Called when a rate limit is hit to increase backoff in an async context."""
        self.backoff_time = max(60, self.backoff_time * 2)
        print(f"AsyncRateLimiter: Rate limit hit. Backoff increased to {self.backoff_time}s. Next async_wait will use this.")

#caches the responses from the api so I don't have to keep calling with the same prompt when testing other things
class Cache:
    def __init__(self, cache_dir: str = "cache"):
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the full path to the cache directory
        self.cache_dir = Path(current_dir) / cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        print(f"Cache directory: {self.cache_dir}")
        
    def clear(self):
        for file in self.cache_dir.glob("*.pkl"):
            try:
                file.unlink()
            except Exception as e:
                print(f"ERROR deleting {file}: {e}")
        print("Cache cleared successfully")
    
    def clear_invalid(self):
        for file in self.cache_dir.glob("*.pkl"):
            try:
                with open(file, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # Check for empty or invalid responses
                if isinstance(cached_data, list) and not cached_data:  # Empty list
                    file.unlink()
                elif isinstance(cached_data, str) and not cached_data.strip():  # Empty string
                    file.unlink()
                    
            except Exception as e:
                print(f"ERROR checking {file}: {e}")
                file.unlink()  # Delete files that can't be loaded
        print("Invalid cache entries cleared")
        
    # makes a unique hash key to store then get each abstract and generated summary
    def make_hash_key(self, abstract: str, gen_summary: str) -> str:
        # Ensure both strings are properly encoded before hashing
        if abstract is None:
            abstract = ""
        if gen_summary is None:
            gen_summary = ""
            
        # Convert to string if not already
        abstract_str = str(abstract)
        gen_summary_str = str(gen_summary)
        
        # Create the combined string and encode it
        combined = f"{gen_summary_str}:{abstract_str}"
        encoded = combined.encode('utf-8', errors='replace')
        
        return hashlib.md5(encoded).hexdigest()
        
    def get(self, abstract: str, gen_summary: str):
        # uses the abstract and gen summary to make a unique hash
        cache_key = self.make_hash_key(abstract, gen_summary)
        # sets the dir for the cache_file to be in the generated unique hash .pkl file
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # if that .pkl file exists, then that means there exists the exact same abstract in the cache
        if cache_file.exists():
            try:
                #loads the cache data 
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                # should return either summary or triplets depending on what we are pulling
                print(f"Cache file found for {gen_summary}")
                return cached_data
            except Exception as e:
                print(f"ERROR reading the cache: {e}")
                return None
        return None
        
    def set(self, abstract: str, gen_summary: str, result):
        # does the same process as the getting, but instead creates the file
        cache_key = self.make_hash_key(abstract, gen_summary)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            # Opens the file name it made in writing binary mode so we can pickle dump
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"ERROR writing the cache: {e}")

# --- Refinement Script Cache --- 
class SimpleCache:
    """Simplified cache for refinement task."""
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Refinement cache directory: {self.cache_dir}")

    def get(self, key_text: str):
        cache_key = hashlib.md5(key_text.encode('utf-8')).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"ERROR reading refinement cache: {e}")
        return None

    def set(self, key_text: str, result):
        cache_key = hashlib.md5(key_text.encode('utf-8')).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"ERROR writing refinement cache: {e}")
# --- End Refinement Cache --- 

# --- IUCN Classification Constants and Functions --- 
# Define IUCN categories as a separate list for clarity
IUCN_CATEGORIES_TEXT = """
**IUCN THREAT CATEGORIES:**
1 Residential & commercial development (1.1 Housing & urban areas, 1.2 Commercial & industrial areas, 1.3 Tourism & recreation areas)
2 Agriculture & aquaculture (2.1 Annual & perennial non-timber crops, 2.2 Wood & pulp plantations, 2.3 Livestock farming & ranching, 2.4 Marine & freshwater aquaculture)
3 Energy production & mining (3.1 Oil & gas drilling, 3.2 Mining & quarrying, 3.3 Renewable energy)
4 Transportation & service corridors (4.1 Roads & railroads, 4.2 Utility & service lines, 4.3 Shipping lanes, 4.4 Flight paths)
5 Biological resource use (5.1 Hunting & collecting terrestrial animals, 5.2 Gathering terrestrial plants, 5.3 Logging & wood harvesting, 5.4 Fishing & harvesting aquatic resources)
6 Human intrusions & disturbance (6.1 Recreational activities, 6.2 War, civil unrest & military exercises, 6.3 Work & other activities)
7 Natural system modifications (7.1 Fire & fire suppression, 7.2 Dams & water management/use, 7.3 Other ecosystem modifications)
8 Invasive & other problematic species, genes & diseases (8.1 Invasive non-native/alien species/diseases, 8.2 Problematic native species/diseases, 8.3 Introduced genetic material, 8.4 Problematic species/diseases of unknown origin, 8.5 Viral/prion-induced diseases, 8.6 Diseases of unknown cause)
9 Pollution (9.1 Domestic & urban waste water, 9.2 Industrial & military effluents, 9.3 Agricultural & forestry effluents, 9.4 Garbage & solid waste, 9.5 Air-borne pollutants, 9.6 Excess energy)
10 Geological events (10.1 Volcanoes, 10.2 Earthquakes/tsunamis, 10.3 Avalanches/landslides)
11 Climate change & severe weather (11.1 Habitat shifting & alteration, 11.2 Droughts, 11.3 Temperature extremes, 11.4 Storms & flooding, 11.5 Other impacts)
12 Other options (12.1 Other threat)
"""

IUCN_THREAT_PROMPT_SYSTEM = f"""
You are an expert ecological threat classifier. Your task is to assign the single most appropriate IUCN threat category to a given threat description, considering the context of the species and the impact mechanism.

{IUCN_CATEGORIES_TEXT}

**Instructions:**
1. Analyze the provided Threat Description in the context of the Subject (Species) and Predicate (Impact Mechanism).
2. Identify the *underlying cause* of the threat.
3. Select the single *most specific and relevant* IUCN category code and name from the list above that best represents the *underlying cause*.
4. **Avoid 12.1 Other threat** unless no other category is remotely applicable. Think critically about the root cause.
5. Return ONLY a valid JSON object containing the selected code and name.
"""

def parse_and_validate_object(object_str: str) -> tuple[str, Optional[str], Optional[str], bool]:
    """
    Parses the object string, validates the IUCN format.
    Returns: (description, code, name, is_valid)
    """
    if not isinstance(object_str, str):
        return str(object_str), None, None, False
    # Corrected regex to handle potential missing names better and avoid issues with brackets in description
    pattern = r"^(.*?)\s*\[IUCN:\s*([\d\.]+)\s*(.*?)\]$"
    match = re.match(pattern, object_str, re.DOTALL)
    if match:
        description = match.group(1).strip()
        code = match.group(2).strip()
        name = match.group(3).strip() # Name can be empty if only code is present
        # Validate code format
        if re.match(r"^\d+(\.\d+)?$", code):
            # If name is empty, it's still potentially valid format, just incomplete
            return description, code, name if name else None, True 
        else:
            # Code format is wrong, treat whole thing as description
            return object_str.strip(), None, None, False
    else:
        # Doesn't match the [IUCN: ...] pattern at all
        return object_str.strip(), None, None, False

async def get_iucn_classification_json(subject: str, predicate: str, threat_desc: str, llm_setup, cache: SimpleCache) -> tuple[str, str]: 
    """Uses Ollama with JSON format (enforced by schema) to get IUCN classification asynchronously."""
    cache_key = f"iucn_classify_json_schema:{threat_desc}|context:{subject}|{predicate}" # Updated key for schema version
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"  (IUCN JSON Schema Cache Hit for: '{threat_desc[:50]}...')") # Use logger
        return cached_result

    logger.info(f"  Asking LLM (JSON schema) to classify threat: '{threat_desc[:50]}...'") # Use logger
    
    # Define the expected JSON structure for Ollama's response
    iucn_schema = {
        "type": "object",
        "properties": {
            "iucn_code": {"type": "string", "description": "The specific IUCN numerical code, e.g., '5.3' or '11.1'"},
            "iucn_name": {"type": "string", "description": "The corresponding IUCN category name, e.g., 'Logging & wood harvesting'"}
        },
        "required": ["iucn_code", "iucn_name"]
    }

    # Simplified prompt, relies on schema for formatting
    prompt = f"""
Context:
Subject (Species): {subject}
Predicate (Impact Mechanism): {predicate}

Threat Description to Classify:
{threat_desc}

Based ONLY on the Threat Description and its context, determine the single most appropriate IUCN category from the list provided in the system prompt.
Focus on the underlying cause of the threat described.
"""
    
    response_str = await llm_generate(
        prompt=prompt,
        system=IUCN_THREAT_PROMPT_SYSTEM, # System prompt still contains the list and core instructions
        model=llm_setup['model'], 
        temperature=0.0, 
        format=iucn_schema, # Pass the schema to enforce structure
        llm_setup=llm_setup
    )

    if response_str:
        try:
            # Response should now be valid JSON matching the schema
            result_json = json.loads(response_str) 
            code = result_json.get("iucn_code")
            name = result_json.get("iucn_name")
            
            # Basic validation of content type and code format
            if isinstance(code, str) and isinstance(name, str) and code.strip() and name.strip() and re.match(r"^\d+(\.\d+)?$", code.strip()):
                code = code.strip()
                name = name.strip()
                logger.info(f"    -> LLM Schema classified as: {code} - {name}") # Use logger
                result = (code, name)
                cache.set(cache_key, result)
                return result
            else:
                 logger.warning(f"    -> LLM Schema response invalid content: Code='{code}', Name='{name}' - Using fallback.") # Use logger

        except json.JSONDecodeError as e:
            logger.error(f"    -> Failed to decode LLM JSON Schema response: {e}") # Use logger
            logger.error(f"    -> Received: '{response_str}' - Using fallback.") # Use logger
        except Exception as e:
             logger.error(f"    -> Error processing LLM JSON Schema response: {e} - Using fallback.", exc_info=True) # Use logger
             
    else:
        logger.warning(f"    -> LLM call failed or returned empty. Using fallback.") # Use logger

    # Fallback if Ollama fails or returns invalid JSON/content
    result = ("12.1", "Other threat")
    cache.set(cache_key, result) # Cache the fallback too
    return result
# --- End IUCN Classification --- 

# initializes the api client
def setup_llm():
    load_dotenv()
    # Setup models to use OpenRouter with Gemini
    return {
        'cache': Cache(),
        # Corrected default model ID for OpenRouter for Gemini 1.5 Flash
        'model': "google/gemini-flash-1.5", # Placeholder, confirm if different
        'api_rate_limiter': RateLimiter(rpm=30, is_ollama=False),  # Rate limiter for API calls
        # Assuming these specialized models also use OpenRouter and need correct IDs
        'species_model': "google/gemini-flash-1.5", 
        'threat_model': "google/gemini-flash-1.5",   
        'impact_model': "google/gemini-flash-1.5",   
        'use_openrouter': True  # Flag to indicate we're using OpenRouter
    }

#################################### NOTES ####################################
### Paper Used:
##The analyses presented here are conducted with different generative models, including fine-tuned models targeted for
#materials and biological materials:
#• X-LoRA, a fine-tuned, dynamic dense mixture-of-experts large language model with strong biological materials,
#math, chemistry, logic and reasoning capabilities [24] that uses two forward passes (details see reference and
#discussion in main text)
#• BioinspiredLLM-Mixtral, a fine-tuned mixture-of-experts (MoE) model based on the original BioinspiredLLM
#model [11] but using a mixture-of-expert approach basde on the Mixtral model [50]
#We also use general-purpose models, including:
#• Mistral-7B-OpenOrca [51, 52, 53] (used for text distillation into a heading, summary and bulleted list of
#detailed mechanisms and reasoning)
#• Zephyr-7B-β [54] built on top of the Mistral-7B model[5] (used for original graph generation due efficient
#compute and local hosting)
#• GPT-4 (gpt-4-0125-preview), at the time of the writing of this paper, this is the latest GPT model by
#OpenAI [3] (for some less complex tasks, specificall graph augmentation, we use GPT 3.5)
#• GPT-4V (gpt-4-vision-preview), a multimodal vision-text model by OpenAI [55, 56], for some use cases
#accessed via https://chat.openai.com/
#• Claude-3 Opus and Sonnet [57], accessed via https://claude.ai/chats

################################################################################


# Based on the knowledge graph paper, they did a multi step process of getting a summary with key facts and a title then extracting triples
# Following that, this asks the llm to generate a summary, though more basic, and send it off to generate triples
async def convert_to_summary(abstract: str, llm_setup) -> str:
    cached = llm_setup["cache"].get(abstract, "summary")
    if cached:
        return cached

    # Maybe a bit much, could refine and see if results are still similar with llama:8b to run quicker or smaller datasets
    system_prompt = """
    You are a scientific knowledge summarizer. Convert the following text into a structured summary that:
    1. Focuses on species-specific impacts and threats
    2. Clearly states causal mechanisms and relationships
    3. Includes quantitative data when available
    4. Emphasizes HOW impacts occur, not just WHAT happened
    5. Use scientific names (Latin binomial) when mentioned in the abstract
    6. If a group of species is mentioned, look for any specific examples in the abstract
    7. If no specific species are named, use the most specific taxonomic group mentioned
    8. Never use vague terms like "birds", "larger species", or "# bird species"
    9. Do not include phrases like "spp." or number of species
    10. Each species or taxonomic group should not be a phrase
    Summarize this scientific abstract focusing on specific species and their threats. 
    
    Format the summary with clear sections:
    - Species Affected
    - Threat Mechanisms
    - Quantitative Findings
    - Causal Relationships
    """

    try:
        # Combine system prompt and abstract
        full_prompt = f"Text to summarize:\\n{abstract}\\n\\nStructured Summary:"
        
        # Use the new LLM wrapper function
        summary = await llm_generate(
            prompt=full_prompt,
            system=system_prompt,
            model=llm_setup["model"],
            temperature=0.1,
            timeout=120,  # Shorter timeout for summaries
            llm_setup=llm_setup
        )
        
        # logger.info(f"\nGenerated Summary:\n{summary}\n") # Use logger
        
        # Basic validation based on length
        if len(summary) < 50:
            logger.warning("WARNING: Summary seems too short, might not be valid") # Use logger
            return ""
            
        llm_setup["cache"].set(abstract, "summary", summary)
        return summary
        
    except Exception as e:
        logger.error(f"ERROR generating summary: {e}", exc_info=True) # Use logger
        return ""

# checks if the terms are similar using fuzzy matching over threshold of 85%
def are_terms_similar(term1: str, term2: str, threshold: int = 85) -> bool:
    ratio = fuzz.ratio(term1.lower(), term2.lower())
    return ratio >= threshold

# consolidates similar triplets using fuzzy matching results
def consolidate_triplets(triplets: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
    if not triplets:
        return []
    
    # Group similar triplets
    consolidated = []
    used_indices = set()
    
    for i, (subj1, pred1, obj1, doi1) in enumerate(triplets):
        if i in used_indices:
            continue
            
        similar_group = [(subj1, pred1, obj1, doi1)]
        used_indices.add(i)
        
        # Look for similar triplets
        for j, (subj2, pred2, obj2, doi2) in enumerate(triplets[i+1:], i+1):
            if j in used_indices:
                continue
                
            # Check if subjects and objects are similar
            if (are_terms_similar(subj1, subj2) and 
                are_terms_similar(obj1, obj2)):
                # For consolidation, we assume DOI similarity is not a primary factor, 
                # but we need to carry one DOI forward. We'll use the first one.
                similar_group.append((subj2, pred2, obj2, doi2))
                used_indices.add(j)
        
        # If we found similar triplets, combine them
        if len(similar_group) > 1:
            # Use the first subject, object, and DOI (since they're similar enough on S,O)
            combined_subj = similar_group[0][0]
            combined_obj = similar_group[0][2]
            combined_doi = similar_group[0][3] # Carry forward the DOI of the first triplet in the group
            
            # Combine unique predicates
            predicates = list(set(t[1] for t in similar_group))
            if len(predicates) > 1:
                combined_pred = " and ".join(predicates)
            else:
                combined_pred = predicates[0]
            
            consolidated.append((combined_subj, combined_pred, combined_obj, combined_doi))
            print(f"\nConsolidated these triplets:")
            for t in similar_group:
                print(f"  {t[0]} | {t[1]} | {t[2]} (DOI: {t[3]})")
            print(f"Into: {combined_subj} | {combined_pred} | {combined_obj} (DOI: {combined_doi})\n")
        else:
            # If no similar triplets found, just add the original
            consolidated.append((subj1, pred1, obj1, doi1))
    
    return consolidated

# ollama structured output link: https://ollama.com/blog/structured-outputs#:~:text=Ollama%20now%20supports%20structured%20outputs,Parsing%20data%20from%20documents
async def extract_triplets(summary: str, llm_setup, doi: str) -> List[Tuple[str, str, str, str]]:
    # Cache check commented out to force regeneration (as per previous request)
    # cached = llm_setup["cache"].get(summary, "triplets")
    # if cached:
    #     return cached

    logger.info("\nGenerating triplets (bypassing triplet cache)...") # Use logger

    try:
        # STAGE 1: Extract species mentioned in the summary
        logger.info("\nSTAGE 1: Extracting species from summary...") # Use logger
        
        # Define schema for species extraction
        species_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "scientific_name": {"type": "string"},
                    "confidence": {"type": "string"}
                },
                "required": ["name", "confidence"]
            }
        }
        
        species_system_prompt = """
        Extract all specific species or taxonomic groups mentioned in the text.

    Rules:
        1. Only include species or taxonomic groups that are DIRECTLY mentioned in the text
        2. Keep scientific names exactly as written
        3. Each entry must be a single species or specific taxonomic group
        4. Never combine multiple species into one entry (e.g., not "# bird species")
        5. Remove any qualifiers like "spp." or species counts
        6. If a scientific name is provided in the text, include it
        7. Assign a confidence level (high, medium, low) based on how clearly the species is mentioned
        """
        
        species_prompt = f"Extract all species or taxonomic groups mentioned in this text:\n\n{summary}"
        
        # Stage 1: Species extraction with schema-based formatting
        species_response = await llm_generate(
            prompt=species_prompt,
            system=species_system_prompt,
            model=llm_setup["species_model"],
            temperature=0.1,
            format=species_schema, # This tells the LLM the schema we want for its *output value*
            llm_setup=llm_setup
        )
        
        species_list = []
        try:
            parsed_json = json.loads(species_response)
            # Check if the response is the schema-plus-value structure
            if isinstance(parsed_json, dict) and "value" in parsed_json and isinstance(parsed_json["value"], list):
                species_data_actual = parsed_json["value"]
            # Check if the response is directly a list (ideal case)
            elif isinstance(parsed_json, list):
                species_data_actual = parsed_json
            else:
                logger.error(f"Unexpected JSON structure for species. Expected list or dict with 'value' key. Got: {type(parsed_json)}. Raw response was: {species_response}") # Log the raw response here
                species_data_actual = [] # Proceed with empty to avoid further errors

            for s_item in species_data_actual:
                if isinstance(s_item, dict) and s_item.get('confidence', '').lower() != 'low':
                    species_list.append(s_item['name'])
                elif not isinstance(s_item, dict):
                    logger.warning(f"Skipping non-dict item in species_data: {s_item}")
                    
        except json.JSONDecodeError as e_json:
            logger.error(f"Error parsing species JSON (JSONDecodeError): {e_json}. Raw response: '{species_response}'")
        except Exception as e_general: # Catch other potential errors like AttributeError if parsing was wrong
            logger.error(f"Error processing species data: {e_general}. Raw response: '{species_response}'")
            # Fallback (already present but good to be aware of)
            json_start = species_response.find('[')
            json_end = species_response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                try:
                    species_json = species_response[json_start:json_end]
                    species_data = json.loads(species_json)
                    
                    # Filter out low confidence species
                    species_list = []
                    for s in species_data:
                        if isinstance(s, dict) and 'name' in s and s.get('confidence', '').lower() != 'low':
                            species_list.append(s['name'])
                except Exception:
                    # Final fallback: simple text parsing
                    species_list = []
                    for line in species_response.split('\n'):
                        if '*' in line:
                            species = line.split('*')[1].strip()
                            if species and len(species) > 2:
                                species_list.append(species)
            else:
                # Fallback: simple text parsing
                species_list = []
                for line in species_response.split('\n'):
                    if ':' in line and 'species' not in line.lower() and 'name' not in line.lower():
                        species = line.split(':')[1].strip()
                        if species and len(species) > 2:
                            species_list.append(species)
        
        if not species_list:
            logger.info("No species found in the summary.") # Use logger
            return []
        
        logger.info(f"Extracted {len(species_list)} species:") # Use logger
        for i, species in enumerate(species_list, 1):
            logger.info(f"{i}. {species}") # Use logger
        
        # STAGE 2: Identify threats for each species (Simplified - NO IUCN HERE)
        logger.info("\nSTAGE 2: Identifying threats for each species (Description Only)...") # Use logger
        
        # Define simplified schema for threats (description only)
        threats_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "species_name": {"type": "string"},
                    "threats": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "threat_description": {"type": "string"},
                                "confidence": {"type": "string"}
                            },
                            "required": ["threat_description", "confidence"]
                        }
                    }
                },
                "required": ["species_name", "threats"]
            }
        }
        
        # Simplified system prompt for Stage 2
        threats_system_prompt = """
        For each species mentioned in the text, identify the specific NEGATIVE threats, stressors, or CAUSES OF HARM described as impacting them.

        **Rules:**
        1. Focus ONLY on factors that HARM or NEGATIVELY impact the species.
        2. Extract the *specific description of the threat or stressor* (e.g., "drowning in oil pits", "habitat loss from logging", "increasing shoreline development", "competition from invasive species").
        3. **DO NOT extract protective factors or beneficial conditions** (e.g., do not extract "protected by vegetated shorelines").
        4. Only include threats DIRECTLY mentioned as impacting the species in the text.
        5. Do NOT attempt to classify the threat using IUCN categories here.
        6. Assign a confidence level (high, medium, low) based on how clearly the text links the threat description to the species.

        **Output Format:** Respond with ONLY a valid JSON array matching the required schema.
        """
        
        threats_prompt = f"Identify threats for each species mentioned in this text:\n\n{summary}\n\nSpecies list: {json.dumps(species_list)}" # Ensure species_list is properly formatted for prompt
        
        # Stage 2: Threat identification with simplified schema
        threats_response = await llm_generate(
            prompt=threats_prompt,
            system=threats_system_prompt,
            model=llm_setup["threat_model"],
            temperature=0.1,
            format=threats_schema, # Pass the actual schema object instead of just "json"
            llm_setup=llm_setup
        )
        
        # Parse the JSON response - should be direct JSON with schema
        threats_data_parsed = None # Use a temporary variable
        try:
            threats_data_parsed = json.loads(threats_response)
        except Exception as e:
            logger.error(f"Error parsing simplified threats JSON with schema: {e}. Raw response: '{threats_response}'") # Log raw response
            # Fallback logic might need adjustment if JSON structure is critical downstream
            # Consider adding more robust text-based fallback if needed

        # Prepare species-threat pairs (description only) for impact analysis
        species_threat_pairs = []
        threats_list_to_process = [] # Initialize empty list

        # Check if parsed data is a list (expected) or a single dict (handle this case)
        if isinstance(threats_data_parsed, list):
            threats_list_to_process = threats_data_parsed
            logger.info(f"  Received list of {len(threats_list_to_process)} species entries.") # Use logger
        elif isinstance(threats_data_parsed, dict):
            logger.warning("  Warning: Received single dict instead of list for threats. Wrapping in list.") # Use logger
            
            # Check if the dictionary has a "species" key containing a list of species
            if "species" in threats_data_parsed and isinstance(threats_data_parsed["species"], list):
                logger.info(f"  Found alternative format with 'species' key containing {len(threats_data_parsed['species'])} species.") # Use logger
                # Convert the alternative format to the expected format
                converted_list = []
                for species_item in threats_data_parsed["species"]:
                    if isinstance(species_item, dict):
                        converted_species = {
                            "species_name": species_item.get("name", ""),
                            "threats": []
                        }
                        # Convert the threats array too
                        for threat_entry in species_item.get("threats", []): # Renamed to avoid conflict
                            if isinstance(threat_entry, dict):
                                converted_threat = {
                                    "threat_description": threat_entry.get("description", ""),
                                    "confidence": threat_entry.get("confidence", "low")
                                }
                                converted_species["threats"].append(converted_threat)
                            elif isinstance(threat_entry, str) and threat_entry.strip():
                                # Handle case where threat is a string
                                threat_text = threat_entry.strip()
                                # Don't add if it's just "unknown"
                                if threat_text.lower() != "unknown":
                                    converted_threat = {
                                        "threat_description": threat_text,
                                        "confidence": "medium"  # Default confidence for string threats
                                    }
                                    converted_species["threats"].append(converted_threat)
                        
                        # Only add species with valid threats - no defaults
                        if converted_species["threats"]:
                            converted_list.append(converted_species)
                
                threats_list_to_process = converted_list
                logger.info(f"  Converted to {len(threats_list_to_process)} standardized species entries.") # Use logger
                
                # Skip this abstract if no valid species-threat pairs
                if not threats_list_to_process:
                    logger.info("  No valid species with threats found. Skipping this abstract.") # Use logger
                    return []
            else:
                # Original wrapping behavior for non-species format dictionaries
                threats_list_to_process = [threats_data_parsed]
        else:
             logger.warning(f"  Warning: Expected threats_data to be a list or dict, but got {type(threats_data_parsed)}. Cannot process threats.") # Use logger
             if threats_data_parsed is not None:
                  logger.warning(f"  Unparseable content: {str(threats_data_parsed)[:200]}") # Use logger
             
        # Process the list (which might contain the wrapped dict)
        for species_threat in threats_list_to_process:
            # Add check: Ensure item is a dictionary
            if not isinstance(species_threat, dict):
                logger.warning(f"  Warning: Expected dict in threats_data list, got {type(species_threat)}. Skipping item: {species_threat}") # Use logger
                continue 
                
            species_name = species_threat.get("species_name", "")
            threats_inner_list = species_threat.get("threats", [])
            
            # Add check: Ensure threats_inner_list is actually a list
            if not isinstance(threats_inner_list, list):
                logger.warning(f"  Warning: Expected list for 'threats' key, got {type(threats_inner_list)}. Skipping threats for {species_name}") # Use logger
                continue
                
            # Skip if threats list is empty - no defaults
            if not threats_inner_list:
                logger.info(f"  Empty threats list for species: {species_name}. Skipping.") # Use logger
                continue
            
            threats_found = 0
            
            for threat_detail in threats_inner_list: # Renamed to avoid conflict
                # Existing check is good here
                if isinstance(threat_detail, dict):
                    confidence = threat_detail.get("confidence", "").lower()
                    if confidence == "low":
                        continue
                        
                    threat_desc = threat_detail.get("threat_description")
                    if not threat_desc:
                        continue
                        
                    if species_name and threat_desc:
                        threats_found += 1
                        species_threat_pairs.append({
                             "species": species_name,
                             "threat": threat_desc, 
                        })
                else:
                    logger.warning(f"  Warning: Expected dict for threat, got {type(threat_detail)}: {str(threat_detail)[:50]}") # Use logger
                    
        # Extra diagnostics after processing
        if not species_threat_pairs:
            logger.info("  No valid species-threat pairs were identified. Skipping this abstract.") # Use logger
            return []
        
        # Print valid pairs found
        logger.info(f"Found {len(species_threat_pairs)} potential species-threat pairs (description only):") # Use logger
        for i, pair in enumerate(species_threat_pairs, 1):
            logger.info(f"{i}. {pair['species']} potentially affected by '{pair['threat']}'") # Use logger
        
        # STAGE 3: Determine impact mechanisms for each species-threat pair
        logger.info("\nSTAGE 3: Determining impact mechanisms...") # Use logger
        
        # Define schema for impact mechanisms
        impacts_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "species_name": {"type": "string"},
                    "threat_name": {"type": "string"},
                    "mechanisms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "confidence": {"type": "string"}
                            },
                            "required": ["description", "confidence"]
                        }
                    }
                },
                "required": ["species_name", "threat_name", "mechanisms"]
            }
        }
        
        impacts_system_prompt = """
        For each species-threat pair provided, identify the specific NEGATIVE impact mechanism described in the text. Focus on HOW the threat DIRECTLY HARMS the species.

        Rules:
        1. Describe the harmful consequence CAUSED BY the threat. Do NOT describe the benefits of habitat or resources that are lost or affected.
        2. Focus ONLY on the negative impact mechanism (e.g., 'reduces nesting success', 'causes poisoning', 'increases predation risk', 'blocks migration route').
        3. Include specific biological, physiological, or ecological processes involved in the harm.
        4. Include quantitative measures of the negative impact when available (e.g., "reduces breeding success by 45%").
        5. Provide direct evidence or strong inference from the text for the mechanism.
        6. Assign a confidence level (high, medium, low) based on how clearly the negative impact mechanism is described.
        7. If multiple distinct negative mechanisms exist for the same species-threat pair, list them separately.

        Example:
        - Text mentions: "Shoreline development leads to loss of vegetated nesting sites crucial for Wood Ducks."
        - Threat (from Stage 2): Shoreline development
        - Species: Wood Ducks
        - Correct Mechanism: "loss of crucial vegetated nesting sites" or "reduces availability of nesting habitat"
        - Incorrect Mechanism: "benefit from vegetated nesting sites"
        """
        
        # Prepare the pairs for the prompt
        pair_strings = []
        for pair in species_threat_pairs:
            pair_strings.append(f"{pair['species']} - {pair['threat']}")
            
        # Define the impacts prompt
        impacts_prompt = f"Identify how each threat affects each species in this text:\n\n{summary}\n\nPairs to analyze: {json.dumps(pair_strings)}"
        
        # Stage 3: Impact analysis with schema-based formatting
        impacts_response = await llm_generate(
            prompt=impacts_prompt,
            system=impacts_system_prompt,
            model=llm_setup["impact_model"],
            temperature=0.1,
            format=impacts_schema, # This tells the LLM the schema we want for its *output value*
            llm_setup=llm_setup
        )
        
        impacts_data_parsed_list = [] # Ensure this is always a list
        try:
            parsed_json = json.loads(impacts_response)
            # Scenario 1: LLM returns the schema definition AND the data array under a duplicate "items" key
            # json.loads will typically take the *last* "items" key, which should be the data array.
            if isinstance(parsed_json, dict) and "items" in parsed_json and isinstance(parsed_json["items"], list):
                logger.info("Impacts JSON is a dict with an 'items' key containing the data list.")
                impacts_data_parsed_list = parsed_json["items"]
            # Scenario 2: LLM returns a dict with a "value" key containing the data list (like species sometimes does)
            elif isinstance(parsed_json, dict) and "value" in parsed_json and isinstance(parsed_json["value"], list):
                logger.info("Impacts JSON is a dict with a 'value' key containing the data list.")
                impacts_data_parsed_list = parsed_json["value"]
            # Scenario 3: LLM returns the data list directly (ideal)
            elif isinstance(parsed_json, list):
                logger.info("Impacts JSON is a direct list of data.")
                impacts_data_parsed_list = parsed_json
            else:
                logger.error(f"Unexpected JSON structure for impacts. Expected list or dict with 'items' or 'value' key. Got: {type(parsed_json)}. Raw response: {impacts_response}")
        except json.JSONDecodeError as e_json:
            logger.error(f"Error parsing impacts JSON (JSONDecodeError): {e_json}. Raw response: '{impacts_response}'")
        except Exception as e_general: # Catch other potential errors
            logger.error(f"Error processing impacts data (General Exception): {e_general}. Raw response: '{impacts_response}'")

        if not impacts_data_parsed_list and species_threat_pairs: 
            logger.warning("Impacts data parsing failed or yielded empty, creating fallback structure.")
            temp_fallback_list = [] # Use a temporary list for fallback
            for pair in species_threat_pairs:
                temp_fallback_list.append({
                    "species_name": pair["species"],
                    "threat_name": pair["threat"],
                    "mechanisms": [
                        {
                            "description": f"negatively impacts {pair['species']} population",
                            "confidence": "medium"
                        }
                    ]
                })
            impacts_data_parsed_list = temp_fallback_list # Assign after full creation
        
        logger.info("\nFINAL STAGE: Assembling raw triplets (without IUCN)...") 
        
        raw_triplets = []
        for impact_item in impacts_data_parsed_list: 
            if not isinstance(impact_item, dict):
                logger.warning(f"Skipping non-dict item during triplet assembly: {impact_item}")
                continue 
            species = impact_item.get("species_name", "")
            threat_obj_desc_only = impact_item.get("threat_name", "") 
            
            # Create a triplet for each mechanism
            for mechanism in impact_item.get("mechanisms", []):
                if isinstance(mechanism, dict) and mechanism.get("confidence", "").lower() != "low":
                    predicate = mechanism.get("description", "")
                    # Object is JUST the threat description now
                    if species and predicate and threat_obj_desc_only: # Ensure all parts are non-empty
                        raw_triplets.append((species, predicate, threat_obj_desc_only, doi)) # Add DOI here
        
        # Print extracted raw triplets
        logger.info("\nExtracted Raw Triplets (before refinement/consolidation):") # Use logger
        for i, (subject, predicate, obj, d) in enumerate(raw_triplets, 1):
            logger.info(f"{i}. {subject} | {predicate} | {obj} | DOI: {d}") # Use logger
        
        # Consolidate similar triplets (based on S, P, O where O is just threat description)
        consolidated_triplets = consolidate_triplets(raw_triplets)
        
        # Print final consolidated RAW triplets
        logger.info("\nConsolidated Raw Triplets (before IUCN refinement in main loop):") # Use logger
        for subject, predicate, obj, d in consolidated_triplets:
            logger.info(f"• {subject} | {predicate} | {obj} (DOI: {d})") # Use logger
        logger.info(f"Number of raw triplets: {len(consolidated_triplets)}") # Use logger
        
        # NO CACHING HERE - Caching should happen AFTER refinement in main loop if desired
        # llm_setup["cache"].set(summary, "triplets", consolidated_triplets) 
        
        # Return the list of (Subject, Predicate, ThreatDescription) tuples
        # IUCN refinement will happen in the main loop
        return consolidated_triplets 
        
    except Exception as e:
        logger.error(f"ERROR extracting triplets: {e}", exc_info=True) # Use logger
        return []

def build_global_graph(all_triplets: list) -> nx.DiGraph:
    # builds directed graph
    global_graph = nx.DiGraph()
    for triplet in all_triplets:
        subject, predicate, obj, _doi = triplet # DOI is available if needed for edge attributes
        global_graph.add_node(subject)
        global_graph.add_node(obj)
        ######## Merge multiple relationships between the same nodes.
        # Get more works with related species in different papers and see i can link and zoom into either threats 
        # on species or species on threats
        global_graph.add_edge(subject, obj, relation=predicate)
    return global_graph

def analyze_graph_detailed(graph: nx.DiGraph, 
                           figures_dir: Path # Expects direct Path to "figures" subdir
                           ):
    # Get current directory and create figures path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_path = figures_dir
    figures_path.mkdir(exist_ok=True)
    
    # Convert to undirected graph for certain analyses
    undirected_graph = graph.to_undirected()
    
    # Figure 2 equivalent - Graph Visualization
    plt.figure(figsize=(15, 5))
    
    # Full graph
    plt.subplot(131)
    pos = nx.spring_layout(graph) # force directed layout? fruchterman-reingold algorithm
    # it calculates node positions based on simulated attractive and repulsive forces which leads to a visual clustering of nodes.
    nx.draw(graph, pos, node_size=20, alpha=0.6, with_labels=False)
    plt.title("Global Graph")
    
    # Zoomed section similar to paper
    plt.subplot(132)
    subset_nodes = list(graph.nodes())[:10]  # takes the first 10 nodes to zoom into the graph
    subgraph = graph.subgraph(subset_nodes)
    pos_sub = nx.spring_layout(subgraph)
    nx.draw(subgraph, pos_sub, with_labels=True, node_size=500)
    plt.title("Zoomed Section")
    
    # Print triplets for the zoomed section
    print("\n--- Triplets in Zoomed Section --- (First 10 nodes)")
    for u, v, data in subgraph.edges(data=True):
        relation = data.get('relation', 'related_to')
        print(f"Triplet: {u} | {relation} | {v}")
    print("---------------------------------")
    
    # Single node highlight
    plt.subplot(133)
    # Find most connected node
    if not graph.nodes():
        print("Graph is empty, cannot find most connected node.")
        plt.title("Empty Graph")
    else:
        max_degree_node = max(graph.degree, key=lambda x: x[1])[0]
        neighbors = list(graph.neighbors(max_degree_node)) + [max_degree_node]
        highlight_graph = graph.subgraph(neighbors)
        pos_highlight = nx.spring_layout(highlight_graph)
        
        nx.draw(highlight_graph, pos_highlight, node_color='lightgrey', with_labels=True, node_size=500)
        nx.draw_networkx_nodes(highlight_graph, pos_highlight, 
                            nodelist=[max_degree_node], 
                            node_color='red', 
                            node_size=700)
        plt.title(f"Connections of '{max_degree_node}'")
        
        # Print triplets for the highlighted node section
        print(f"\n--- Triplets involving Highlighted Node ({max_degree_node}) --- ")
        for u, v, data in highlight_graph.edges(data=True):
            relation = data.get('relation', 'related_to')
            print(f"Triplet: {u} | {relation} | {v}")
        print("---------------------------------")
    
    # Save the figure
    plt.savefig(figures_path / "graph_analysis.png", 
                bbox_inches='tight', 
                dpi=300,
                format='png')
    plt.close()
    print(f"\nGraph analysis visualization saved to {figures_path / 'graph_analysis.png'}")

    # Graph stats from paper 
    plt.figure(figsize=(15, 5))
    
    # Need to update
    plt.subplot(131)
    degrees = [d for n, d in graph.degree()]
    if not degrees:
        print("Graph is empty, cannot plot degree distribution.")
    else:
        degree_count = {}
        for d in degrees:
            degree_count[d] = degree_count.get(d, 0) + 1
        
        plt.loglog(list(degree_count.keys()), list(degree_count.values()), 'bo-')
        plt.xlabel('Degree (log)')
        plt.ylabel('Frequency (log)')
        plt.title('Degree Distribution')
    
    # Print statistics similar to table 1 in the knowledge graph paper
    print("\nGraph Statistics:")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    if degrees:
        print(f"Average node degree: {sum(degrees)/len(degrees):.2f}")
        print(f"Maximum node degree: {max(degrees)}")
        print(f"Minimum node degree: {min(degrees)}")
        print(f"Median node degree: {sorted(degrees)[len(degrees)//2]}")
    else:
        print("Average node degree: N/A (empty graph)")
    print(f"Density: {nx.density(graph):.5f}")
    
    # Similarly do the same with the giant_component that ignores the sparsely connected nodes
    if graph.number_of_nodes() > 0:
        try:
            # Find connected components in the undirected version
            undirected_graph_for_components = graph.to_undirected()
            if undirected_graph_for_components.number_of_edges() > 0 or undirected_graph_for_components.number_of_nodes() > 0:
                 connected_components = list(nx.connected_components(undirected_graph_for_components))
                 if connected_components:
                     giant_component_nodes = max(connected_components, key=len)
                     giant_subgraph = graph.subgraph(giant_component_nodes)
                     print("\nGiant Component Statistics:")
                     gc_degrees = [d for n, d in giant_subgraph.degree()]
                     print(f"Number of nodes: {giant_subgraph.number_of_nodes()}")
                     print(f"Number of edges: {giant_subgraph.number_of_edges()}")
                     if gc_degrees:
                         print(f"Average node degree: {sum(gc_degrees)/len(gc_degrees):.2f}")
                         print(f"Maximum node degree: {max(gc_degrees)}")
                         print(f"Minimum node degree: {min(gc_degrees)}")
                         print(f"Median node degree: {sorted(gc_degrees)[len(gc_degrees)//2]}")
                     else:
                         print("Average node degree: N/A (isolated nodes in giant component?)")
                     print(f"Density: {nx.density(giant_subgraph):.5f}")
                 else:
                    print("\nGiant Component Statistics: No connected components found.")
            else:
                print("\nGiant Component Statistics: Graph is empty or has no edges.")
        except Exception as e:
            print(f"Error calculating giant component: {e}")
    else:
        print("\nGiant Component Statistics: Graph is empty.")

def analyze_hub_node(graph: nx.DiGraph, 
                       figures_dir: Path # Expects direct Path to "figures" subdir
                       ):
    # Get current directory and create figures path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_path = figures_dir
    figures_path.mkdir(exist_ok=True)

    if not graph.nodes():
        print("Graph is empty, cannot perform hub node analysis.")
        return

    # finds the node with the highest degree by iterating over node and degree and using the lambda to compare and return the second element of the tuple, degree, for each node
    max_degree_node = max(graph.degree, key=lambda x: x[1])[0] # [0] gets the node itself to pass to degree
    degree = graph.degree[max_degree_node]
    
    # Get all neighbors of the degree node
    neighbors = list(graph.neighbors(max_degree_node))
    predecessors = list(graph.predecessors(max_degree_node))
    
    print(f"\n--- Hub Node Analysis: '{max_degree_node}' (Degree: {degree}) ---")
    print("Outgoing Connections (Triplets):")
    if not neighbors:
        print("  None")
    else:
        for neighbor in neighbors:
            edge_data = graph.get_edge_data(max_degree_node, neighbor)
            relation = edge_data.get('relation', 'related_to')
            print(f"  Triplet: {max_degree_node} | {relation} | {neighbor}")
    
    print("\nIncoming Connections (Triplets):")
    if not predecessors:
        print("  None")
    else:
        for predecessor in predecessors:
            edge_data = graph.get_edge_data(predecessor, max_degree_node)
            relation = edge_data.get('relation', 'related_to')
            print(f"  Triplet: {predecessor} | {relation} | {max_degree_node}")
    print("---------------------------------")
        
    # Output snippet (remains illustrative):
    # ...

    # plotting for the hub and its connections
    plt.figure(figsize=(12, 8))
    hub_nodes_to_plot = list(set([max_degree_node] + neighbors + predecessors))
    hub_subgraph = graph.subgraph(hub_nodes_to_plot)
    pos = nx.spring_layout(hub_subgraph, k=1, iterations=50)
    
    nx.draw(hub_subgraph, pos, 
           node_color='lightblue',
           node_size=2000,
           with_labels=True,
           font_size=8,
           font_weight='bold')
    
    nx.draw_networkx_nodes(hub_subgraph, pos,
                           nodelist=[max_degree_node],
                           node_color='red',
                           node_size=3000)
    
    edge_labels = nx.get_edge_attributes(hub_subgraph, 'relation')
    nx.draw_networkx_edge_labels(hub_subgraph, pos, 
                                edge_labels=edge_labels,
                                font_size=6)
    
    plt.title(f"Hub Node: {max_degree_node} and its connections")
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(figures_path / "hub_node_analysis.png", 
                bbox_inches='tight', 
                dpi=300,
                format='png')
    plt.close()
    print(f"\nHub node visualization saved to {figures_path / 'hub_node_analysis.png'}")

#Wikispecies api verification for texonomic enrichment
# Guidelines for rpm and usage: https://species.wikimedia.org/robots.txt
class WikispeciesClient:
    def __init__(self):
        self.base_url = "https://species.wikimedia.org/w/api.php"
        self.rate_limiter = RateLimiter(rpm=30) # rpm=60 is too high for wikimedia
        self.query_results = []
        
    # Search for a species and get its page info
    def search_species(self, name: str) -> Optional[Dict]:
        self.rate_limiter.wait()
        
        result = {
            'query_name': name,
            'found_name': None,
            'page_id': None,
            'taxonomy': None,
            'error': None,
            'raw_response': None
        }
        
        try:
            # First search for the page
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': name,
                'format': 'json'
            }
            
            print(f"\nSearching Wikispecies for: {name}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Save raw for debugging
            result['raw_response'] = data
            
            if data.get('query', {}).get('search'):
                # Get the first result, could maybe get multiple and then use smthn to compare with results against triplet or species or abstract key species?
                first_result = data['query']['search'][0]
                result['found_name'] = first_result['title']
                result['page_id'] = first_result['pageid']
                
                # Get the page content
                page_data = self.get_page_content(first_result['title'])
                if page_data:
                    result['taxonomy'] = self.parse_taxonomy(page_data)
                
                print(f"Found Wikispecies page: {result['found_name']}")
            else:
                result['error'] = "No results found"
                print(f"No Wikispecies results for: {name}")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"ERROR searching Wikispecies for {name}: {e}")
        
        self.query_results.append(result)
        return result
    
    def get_page_content(self, title: str) -> Optional[str]:
        self.rate_limiter.wait()
        
        params = {
            'action': 'parse',
            'page': title,
            'format': 'json',
            'prop': 'text'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Get the text content of the page
            if data.get('parse', {}).get('text', {}).get('*'):
                return data['parse']['text']['*']
            return None
            
        except Exception as e:
            print(f"ERROR getting page content for {title}: {e}")
            return None
    
    def parse_taxonomy(self, html_content: str) -> Dict:
        taxonomy = {
            'kingdom': None,
            'phylum': None,
            'class': None,
            'order': None,
            'family': None,
            'genus': None,
            'species': None,
            'rank_hierarchy': []  # Keep full hierarchy separately possibly for later
        }
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find taxonomic hierarchy
            for p in soup.find_all('p'):
                text = p.text.strip()
                for line in text.split('\n'):
                    # Store full hierarchy
                    taxonomy['rank_hierarchy'].append(line.strip())
                    
                    # Extract main ranks
                    line = line.lower()
                    if 'regnum: ' in line and not taxonomy['kingdom']:
                        taxonomy['kingdom'] = line.split('regnum: ')[1].strip()
                    elif 'phylum: ' in line and not taxonomy['phylum']:
                        taxonomy['phylum'] = line.split('phylum: ')[1].strip()
                    elif 'classis: ' in line and not taxonomy['class']:
                        taxonomy['class'] = line.split('classis: ')[1].strip()
                    elif 'ordo: ' in line and not taxonomy['order']:
                        taxonomy['order'] = line.split('ordo: ')[1].strip()
                    elif 'familia: ' in line and not taxonomy['family']:
                        taxonomy['family'] = line.split('familia: ')[1].strip()
                    elif 'genus: ' in line and not taxonomy['genus']:
                        taxonomy['genus'] = line.split('genus: ')[1].strip()
                    elif 'species: ' in line and not taxonomy['species']:
                        taxonomy['species'] = line.split('species: ')[1].strip()
            
        except Exception as e:
            print(f"ERROR parsing taxonomy: {e}")
        
        return taxonomy
    
    def save_results(self, output_dir: str = "results"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = Path(current_dir) / output_dir
        output_path.mkdir(exist_ok=True)
        
        # Save all results including raw responses for debug
        with open(output_path / "wikispecies_results.json", "w", encoding='utf-8') as f:
            json.dump(self.query_results, f, indent=2)
        
        # Save a summary CSV for eaasier viewing
        with open(output_path / "wikispecies_summary.csv", "w", encoding='utf-8') as f:
            f.write("query_name,found_name,page_id,error\n")
            for result in self.query_results:
                f.write(f"{result['query_name']},{result['found_name'] or ''},")
                f.write(f"{result['page_id'] or ''},{result['error'] or ''}\n")
        
        print(f"\nSaved Wikispecies results to:")
        print(f"### Full data: {output_path}/wikispecies_results.json")
        print(f"### Summary: {output_path}/wikispecies_summary.csv")

    async def search_species_async(self, name: str, session: aiohttp.ClientSession) -> Dict:
        await self.rate_limiter.async_wait() 
        
        result = {
            'query_name': name, 'found_name': None, 'page_id': None,
            'taxonomy': None, 'error': None, 'raw_response': None
        }
        
        try:
            params = {
                'action': 'query', 'list': 'search',
                'srsearch': name, 'format': 'json'
            }
            print(f"Async Searching Wikispecies for: {name}")
            async with session.get(self.base_url, params=params, timeout=30) as response: # Added timeout
                response.raise_for_status() 
                data = await response.json()
            
            result['raw_response'] = data
            
            if data.get('query', {}).get('search'):
                first_result = data['query']['search'][0]
                result['found_name'] = first_result['title']
                result['page_id'] = first_result['pageid']
                
                page_html_content = await self.get_page_content_async(first_result['title'], session)
                if page_html_content:
                    # parse_taxonomy is synchronous but operates on local data
                    result['taxonomy'] = self.parse_taxonomy(page_html_content) 
                
                print(f"Async Found Wikispecies page: {result['found_name']}")
            else:
                result['error'] = "No results found"
                print(f"Async No Wikispecies results for: {name}")
                
        except aiohttp.ClientResponseError as http_err:
            error_message = f"HTTP error: {http_err.status} {http_err.message}"
            result['error'] = error_message
            print(f"ERROR async searching Wikispecies for {name}: {error_message}")
            if http_err.status == 429: 
                self.rate_limiter.handle_async_rate_limit()
        except asyncio.TimeoutError: # Specific timeout error
            error_message = "Request timed out"
            result['error'] = error_message
            print(f"ERROR async searching Wikispecies for {name}: {error_message}")
        except Exception as e:
            result['error'] = str(e)
            print(f"ERROR async searching Wikispecies for {name}: {e}")
        
        return result

    async def get_page_content_async(self, title: str, session: aiohttp.ClientSession) -> Optional[str]:
        await self.rate_limiter.async_wait()
        
        params = {
            'action': 'parse', 'page': title,
            'format': 'json', 'prop': 'text'
        }
        
        try:
            async with session.get(self.base_url, params=params, timeout=30) as response: # Added timeout
                response.raise_for_status()
                data = await response.json()
            
            if data.get('parse', {}).get('text', {}).get('*'):
                return data['parse']['text']['*']
            return None
            
        except aiohttp.ClientResponseError as http_err:
            print(f"ERROR async getting page content for {title}: {http_err.status} {http_err.message}")
            if http_err.status == 429:
                self.rate_limiter.handle_async_rate_limit()
            return None
        except asyncio.TimeoutError:
            print(f"ERROR async getting page content for {title}: Request timed out")
            return None
        except Exception as e:
            print(f"ERROR async getting page content for {title}: {e}")
            return None

def cache_enriched_triples(triplets: List[Tuple[str, str, str, str]], 
                         llm_taxonomy_map_by_original_name: Dict[str, Dict],
                         output_dir: Path # Now expects the direct Path to the "results" subdirectory
                         ) -> None:
    # Cache the enriched triplets with their taxonomic data for easier visualization refinement
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = output_dir
    output_path.mkdir(exist_ok=True)
    
    # 1. Prepare a map from canonical bird names to their full LLM taxonomy data.
    #    This is used for embedding the correct taxonomy into each triplet object.
    #    It's derived from the comprehensive map keyed by original names.
    canonical_to_llm_taxonomy_for_birds = {}
    for _original_name, tax_data in llm_taxonomy_map_by_original_name.items():
        if tax_data.get('is_bird', False):
            canonical_form = tax_data.get('canonical_form')
            # If multiple original names map to the same canonical bird form,
            # this will store the one processed last. This is generally fine as taxonomy should be consistent.
            if canonical_form: 
                canonical_to_llm_taxonomy_for_birds[canonical_form] = tax_data

    output_triplets_for_json = []
    for canonical_subject, predicate, obj, doi_val in triplets: # Subjects are already canonical bird names, now with DOI
        # Fetch the full LLM taxonomy for this canonical bird subject
        subject_taxonomy_for_triplet = canonical_to_llm_taxonomy_for_birds.get(canonical_subject, {
            'error': f'LLM Taxonomy (bird) not found for canonical name: {canonical_subject}',
            'is_bird': True # Implied, as 'triplets' should only contain birds
        })
        
        output_triplets_for_json.append({
            'subject': canonical_subject,
            'predicate': predicate,
            'object': obj,
            'doi': doi_val, # Include DOI
            'taxonomy': subject_taxonomy_for_triplet # Embed LLM taxonomy
        })
    
    # 2. Filter the input llm_taxonomy_map_by_original_name for the top-level 'taxonomic_info' field.
    #    This ensures the top-level info dump only contains bird taxonomies, keyed by their original names.
    filtered_taxonomic_info_for_json_top_level = {
        original_name: tax_data
        for original_name, tax_data in llm_taxonomy_map_by_original_name.items()
        if tax_data.get('is_bird', False)
    }

    enriched_data = {
        'triplets': output_triplets_for_json,
        'taxonomic_info': filtered_taxonomic_info_for_json_top_level
    }
    
    # Save to JSON
    with open(output_path / "enriched_triplets.json", "w", encoding='utf-8') as f:
        # Save the full structure with "triplets" key
        json.dump(enriched_data, f, indent=2)
    
    print("\nEnriched triplets (with embedded LLM bird taxonomy) saved to enriched_triplets.json")

    # Save the separate, comprehensive LLM bird taxonomy map to its own file
    if filtered_taxonomic_info_for_json_top_level:
        with open(output_path / "llm_bird_taxonomies.json", "w", encoding='utf-8') as f:
            json.dump(filtered_taxonomic_info_for_json_top_level, f, indent=2)
        print("Comprehensive LLM-derived bird taxonomies saved to llm_bird_taxonomies.json")
    else:
        print("No LLM-derived bird taxonomies to save separately.")

# For easier debugging, checking how information was enriched to ensure structure manually 
def print_enriched_triple(subject: str, predicate: str, obj: str, doi:str, 
                         taxonomy: Dict) -> None:
    print("\nEnriched Triple:")
    print(f"Subject: {subject}")
    if taxonomy:
        print("Taxonomy:")
        for level in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']:
            if taxonomy.get(level):
                print(f"  {level.title()}: {taxonomy[level]}")
    print(f"Predicate: {predicate}")
    print(f"Object: {obj}")
    print(f"DOI: {doi}")
    print("-" * 50)

def build_threat_hierarchy(triplets: List[Tuple[str, str, str, str]], 
                         species_taxonomy_map: Dict[str, Dict],
                         results_dir_path: Path # Expects the direct Path to the "results" subdirectory
                         ) -> nx.DiGraph: # species_taxonomy_map is llm_taxonomy_map_by_original_name
    G = nx.DiGraph()
    
    # Track threats at each taxonomic level
    threat_groups = {
        'kingdom': defaultdict(set),
        'phylum': defaultdict(set),
        'class': defaultdict(set),
        'order': defaultdict(set),
        'family': defaultdict(set),
        'genus': defaultdict(set)
    }
    
    # The species_taxonomy_map (derived from LLM via normalize_species_names) 
    # is now the primary source for taxonomy in this function.
    # This map is also what will be used for caching via wiki_data_for_cache.
    wiki_data_for_cache = species_taxonomy_map # This is llm_taxonomy_map_by_original_name
    
    print("\nBuilding threat hierarchy using provided LLM-derived taxonomy data:")
    
    # Create a lookup for canonical bird names to their taxonomy for internal use
    canonical_to_llm_taxonomy_for_birds = {}
    for _original_name, tax_data in species_taxonomy_map.items(): # Iterate the received map
        if tax_data.get('is_bird', False):
            canonical_form = tax_data.get('canonical_form')
            if canonical_form and canonical_form not in canonical_to_llm_taxonomy_for_birds:
                canonical_to_llm_taxonomy_for_birds[canonical_form] = tax_data
    
    # First pass: use provided taxonomic data and group threats
    for subject, predicate, obj, doi_val in triplets: # 'subject' here is the canonical bird name from normalized_triplets, now with DOI
        # Retrieve taxonomy for the canonical subject from our derived map
        taxonomy = canonical_to_llm_taxonomy_for_birds.get(subject) 
        
        if taxonomy: # Check if taxonomy data exists for this canonical bird subject
            # print_enriched_triple is called with the LLM-derived taxonomy
            print_enriched_triple(subject, predicate, obj, doi_val, taxonomy) # Pass DOI
            
            # Add threat relationships at each taxonomic level based on LLM data
            for level in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']:
                taxon_name = taxonomy.get(level) # LLM data should have these keys
                if taxon_name:
                    threat_groups[level][obj].add(subject)
        else:
            print(f"\nNo pre-fetched taxonomic data found for subject: {subject} in the provided map.")
            # Ensure an entry for caching, even if it's None or minimal
            if subject not in wiki_data_for_cache: # Should already be there if map is comprehensive
                 wiki_data_for_cache[subject] = {'source': 'Not found in provided map'}

    # Cache the enriched triplets using the LLM-derived taxonomy map
    # (or whatever was in species_taxonomy_map)
    cache_enriched_triples(triplets, wiki_data_for_cache, results_dir_path)
    
    # Build the graph (rest of the function remains the same)
    for level in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']:
        for threat, species_set in threat_groups[level].items():
            if len(species_set) > 1:
                taxon_name = level.title()
                
                if not G.has_node(taxon_name):
                    G.add_node(taxon_name, 
                             type='taxonomic_level',
                             level=level)
                
                threat_node = f"{threat} ({len(species_set)} species)"
                G.add_node(threat_node,
                          type='threat',
                          affected_species=list(species_set))
                
                G.add_edge(taxon_name, threat_node, 
                          relation='affected_by',
                          species_count=len(species_set))
                
                for species in species_set:
                    G.add_node(species, type='species')
                    G.add_edge(threat_node, species,
                             relation='affects')
    
    return G

def save_threat_hierarchy_viz(graph: nx.DiGraph, 
                              visualization_dir_path: Path, # Expects direct Path to "visualization" subdir
                              use_3d: bool = True):
    """Save hierarchical threat visualization."""
    # current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    # output_dir = get_dynamic_output_path("visualization", model_name_for_path, max_r_for_path, current_script_dir)
    # Note: The original request used "figures" as base, but this function used "visualization". 
    # For consistency with other figure-saving functions, let's change "visualization" to "figures"
    # output_dir = get_dynamic_output_path("figures", model_name_for_path, max_r_for_path, current_script_dir)
    visualization_dir_path.mkdir(parents=True, exist_ok=True) # visualization_dir_path is already the specific viz path
    
    # Convert to D3 hierarchical format
    hierarchy_data = {
        "name": "Threats",
        "children": []
    }
    
    # Group by taxonomic level
    for level in ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']:
        level_threats = []
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'threat':
                # Get connected species at this level
                species = [n for n in graph.successors(node)]
                if species:
                    level_threats.append({
                        "name": node,
                        "size": len(species),
                        "species": species,
                        "type": "threat"
                    })
        
        if level_threats:
            hierarchy_data["children"].append({
                "name": level,
                "children": level_threats
            })
    
    # Save the data
    with open(visualization_dir_path / "threat_hierarchy.json", "w") as f:
        json.dump(hierarchy_data, f, indent=2)
    
    # Create HTML with collapsible tree visualization
    if use_3d:
        # Use 3D visualization template
        html_template = """<!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>3D Threat Network</title>
            <script src="https://unpkg.com/3d-force-graph"></script>
            <script src="https://unpkg.com/d3"></script>
            <style>
                body { margin: 0; }
                #graph { width: 100vw; height: 100vh; }
            </style>
        </head>
        <body>
            <div id="graph"></div>
            <script>
                // 3D visualization code
            </script>
        </body>
        </html>
        """
    else:
        # Use the existing 2D template
        html_template = """<!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Shared Threats Hierarchy</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                .node circle {
                    fill: #fff;
                    stroke: steelblue;
                    stroke-width: 3px;
                }
                .node text { 
                    font: 12px sans-serif; 
                }
                .threat circle {
                    fill: #ff7f7f;
                }
                .species circle {
                    fill: #7fbf7f;
                }
                .link {
                    fill: none;
                    stroke: #ccc;
                    stroke-width: 2px;
                }
            </style>
        </head>
        <body>
            <div id="tree"></div>
            <script>
                // D3 visualization code will go here
            </script>
        </body>
        </html>
        """
    
    with open(visualization_dir_path / "threat_hierarchy.html", "w") as f:
        f.write(html_template)

# Verify triplets against original abstract using Ollama
async def verify_triplets( # Make it async
    triplets: List[Tuple[str, str, str, str]], 
    abstract: str, 
    llm_setup,
    # semaphore: asyncio.Semaphore, # Semaphore might be managed by the caller or not needed if calls are already rate-limited
    verification_cutoff: float = 0.75
) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, int]]:
    """Verify triplets against original abstract using LLM calls concurrently."""
    verified_triplets_for_abstract = []
    counts = {
        'submitted': len(triplets),
        'verified_yes': 0,
        'verified_no': 0,
        'errors': 0
    }

    if not abstract:
        counts['errors'] = len(triplets)
        return [], counts

    verification_schema = {
        "type": "object",
        "properties": {
            "verification": {"type": "string", "enum": ["YES", "NO"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["verification", "confidence"]
    }
    system_prompt = (
        "You are a precise scientific fact checker. "
        "Based on the provided abstract, verify if the relationship is true. "
        "ADDITIONALLY, answer NO if the species in the relationship is not a type of bird. "
        "Respond ONLY with a valid JSON object matching the specified schema. "
        "The JSON object must contain two keys: 'verification' (string: \"YES\" or \"NO\") "
        "and 'confidence' (float: 0.0 to 1.0 representing your confidence in the verification)."
    )

    # --- Caching logic (remains synchronous, checks before creating tasks) --- 
    # Construct a more robust cache key that is less likely to cause issues if abstract is huge
    abstract_hash_part = hashlib.md5(abstract.encode('utf-8', errors='replace')).hexdigest()[:16]
    cache_key_text = f"verify_json_confidence_batch_async:{abstract_hash_part}:{verification_cutoff}:{len(triplets)}" 
    cache_key_hash = hashlib.md5(cache_key_text.encode('utf-8', errors='replace')).hexdigest()
    cache_file_path = Path(llm_setup['cache'].cache_dir) / f"{cache_key_hash}.pkl" 

    if cache_file_path.exists():
        try:
            with open(cache_file_path, 'rb') as f:
                cached_data = pickle.load(f)
            if isinstance(cached_data, tuple) and len(cached_data) == 2:
                cached_triplets_list, cached_counts_dict = cached_data
                if isinstance(cached_triplets_list, list) and isinstance(cached_counts_dict, dict):
                    logger.info(f"  (Verification Cache Hit for abstract hash: '{abstract_hash_part}')") # Use logger
                    return cached_triplets_list, cached_counts_dict
        except Exception as e:
            logger.warning(f"  Error reading verification cache for abstract hash '{abstract_hash_part}': {e}. Re-verifying.") # Use logger
            if cache_file_path.exists(): cache_file_path.unlink(missing_ok=True)
    # --- End Caching Logic --- 

    async def verify_single_triplet_task(subject, predicate, obj, doi_val, p_llm_setup, p_system_prompt, p_verification_schema):
        # async with semaphore: # Semaphore removed as llm_generate handles its own rate limiting/concurrency
        prompt = f"""Abstract:
{abstract}

Relationship to verify:
Subject: "{subject}"
Predicate: "{predicate}"
Object: "{obj}"

Is this relationship true based on the abstract (and is the subject a bird)? Provide your answer in the specified JSON format."""
        
        response_str = None
        try:
            response_str = await llm_generate( # Await the async llm_generate
                prompt=prompt, 
                system=p_system_prompt, 
                model=p_llm_setup["model"], # Ensure this is the correct model for verification
                temperature=0.0, 
                format=p_verification_schema, 
                llm_setup=p_llm_setup
            )
            if not response_str:
                return (subject, predicate, obj, doi_val), "ERROR_EMPTY_RESPONSE", 0.0

            result_json = json.loads(response_str)
            verification_decision = result_json.get("verification")
            confidence_score = result_json.get("confidence")

            if isinstance(verification_decision, str) and isinstance(confidence_score, (float, int)):
                return (subject, predicate, obj, doi_val), verification_decision.upper(), confidence_score
            else:
                return (subject, predicate, obj, doi_val), "ERROR_INVALID_JSON_CONTENT", 0.0
        
        except json.JSONDecodeError:
            logger.error(f"JSONDecodeError for triplet: {subject}|{predicate}|{obj}. Response: {response_str}") # Use logger
            return (subject, predicate, obj, doi_val), "ERROR_JSON_DECODE", 0.0
        except Exception as e:
            logger.error(f"Exception in verify_single_triplet_task for {subject}|{predicate}|{obj}: {e}", exc_info=True) # Use logger
            return (subject, predicate, obj, doi_val), f"ERROR_LLM_CALL: {str(e)[:50]}", 0.0

    tasks = []
    for subject, predicate, obj, doi_val in triplets:
        tasks.append(verify_single_triplet_task(subject, predicate, obj, doi_val, llm_setup, system_prompt, verification_schema))
    
    if not tasks: return [], counts

    logger.info(f"    Starting concurrent verification for {len(tasks)} triplets for DOI: {triplets[0][3] if triplets else 'N/A'}...") # Use logger
    verification_results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(f"    Finished concurrent verification for {len(tasks)} triplets.") # Use logger

    for i, res_tuple_or_exc in enumerate(verification_results):
        original_triplet = triplets[i]
        subject, predicate, obj, doi_val = original_triplet

        if isinstance(res_tuple_or_exc, Exception):
            counts['errors'] += 1
            logger.error(f"    ERROR during single triplet verification task: {res_tuple_or_exc} for triplet {original_triplet}", exc_info=True) # Use logger
            continue
        
        # Ensure res_tuple_or_exc is not None and is a tuple before unpacking
        if res_tuple_or_exc is None or not isinstance(res_tuple_or_exc, tuple) or len(res_tuple_or_exc) != 3:
            counts['errors'] += 1
            logger.error(f"    ERROR: Unexpected result format from verify_single_triplet_task for {original_triplet}. Result: {res_tuple_or_exc}")
            continue

        _triplet_data, decision, confidence = res_tuple_or_exc

        if "ERROR" in decision:
            counts['errors'] += 1
            logger.warning(f"    Rejected ({decision}, Confidence: {confidence:.2f}): {subject} | {predicate} | {obj} (DOI: {doi_val})") # Use logger
        elif decision == "YES" and confidence >= verification_cutoff:
            verified_triplets_for_abstract.append(original_triplet)
            counts['verified_yes'] += 1
            logger.info(f"    Verified (Confidence: {confidence:.2f}): {subject} | {predicate} | {obj} (DOI: {doi_val})") # Use logger
        else:
            counts['verified_no'] += 1
            logger.warning(f"    Rejected (Decision: {decision}, Confidence: {confidence:.2f}, Cutoff: {verification_cutoff}): {subject} | {predicate} | {obj} (DOI: {doi_val})") # Use logger

    try:
        with open(cache_file_path, 'wb') as f:
            pickle.dump((verified_triplets_for_abstract, counts), f)
    except Exception as e:
        logger.error(f"  Error caching batch verification results for abstract hash '{abstract_hash_part}': {e}", exc_info=True) # Use logger

    return verified_triplets_for_abstract, counts

def clear_cached_results(output_dir: str = "results") -> None:
    if os.getenv('CLEAR_CACHE', 'false').lower() == 'true':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = Path(current_dir) / output_dir
        
        if output_path.exists():
            print("\nClearing cached results---")
            for file in output_path.glob('*'):
                file.unlink()
            print("Cache cleared")

def ollama_generate(prompt, model="llama3:8b", system="", temperature=0.1, timeout=120, format=None):
    # Use host.docker.internal to access host machine from Docker https://forum.weaviate.io/t/issue-connecting-to-ollama-llama3-instance-in-docker-need-help/4006
    url = "http://host.docker.internal:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "temperature": temperature,
        "stream": False
    }
    
    # Add format parameter if provided for structured output
    if format:
        payload["format"] = format
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
    except requests.exceptions.Timeout:
        print(f"⚠️ Ollama request timed out after {timeout} seconds")
        return ""
    except requests.exceptions.ConnectionError as e:
        raise Exception(f"Cannot connect to Ollama at {url} ERROR: {e}")

def load_classifier_components(vectorizer_path, classifier_path):
    """Loads the TF-IDF vectorizer and Logistic Regression classifier if they exist."""
    vectorizer = None
    classifier = None
    if vectorizer_path.exists() and classifier_path.exists():
        try:
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(classifier_path, 'rb') as f:
                classifier = pickle.load(f)
            print("Loaded existing relevance classifier and vectorizer.")
            return vectorizer, classifier
        except Exception as e:
            print(f"Error loading classifier components: {e}")
            # Delete potentially corrupted files
            try:
                vectorizer_path.unlink(missing_ok=True)
                classifier_path.unlink(missing_ok=True)
            except Exception as del_e:
                print(f"Error deleting corrupted classifier files: {del_e}")
            return None, None
    return None, None

def train_and_save_classifier(training_data, vectorizer_path, classifier_path):
    """Trains a TF-IDF + Logistic Regression classifier and saves it."""
    if not training_data:
        return None, None
        
    print(f"\nTraining relevance classifier with {len(training_data)} examples...")
    texts = [item['text'] for item in training_data]
    # Convert boolean labels to integers (0 or 1)
    labels = [1 if item['label'] else 0 for item in training_data]
    
    try:
        # Create a pipeline: TF-IDF vectorizer followed by Logistic Regression
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
            ('clf', LogisticRegression(solver='liblinear', random_state=42))
        ])
        
        # Train the pipeline
        pipeline.fit(texts, labels)
        
        # Save the vectorizer and classifier separately
        vectorizer = pipeline.named_steps['tfidf']
        classifier = pipeline.named_steps['clf']
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(classifier_path, 'wb') as f:
            pickle.dump(classifier, f)
            
        print(f"Classifier trained and saved to {classifier_path.parent}")
        return vectorizer, classifier
        
    except Exception as e:
        print(f"ERROR training classifier: {e}")
        return None, None

def predict_relevance_local(abstract, vectorizer, classifier):
    """Predicts relevance using the loaded local classifier."""
    try:
        transformed_abstract = vectorizer.transform([abstract])
        # predict_proba gives [[prob_false, prob_true]]
        prediction_prob = classifier.predict_proba(transformed_abstract)[0][1]
        # Use a threshold (e.g., 0.5) to decide relevance
        is_relevant = prediction_prob > 0.5
        print(f"Local prediction: Relevant={is_relevant} (Prob: {prediction_prob:.2f})")
        return is_relevant
    except Exception as e:
        print(f"Error predicting with local classifier: {e}")
        return False # Fallback to not relevant on error

# Ensure this function is async as it calls await llm_generate
async def classify_abstract_relevance_ollama(title: str, abstract: str, llm_setup) -> bool:
    """Determine if an abstract contains relevant species threat information using Ollama (or OpenRouter via llm_generate)."""
    
    cache_key_text = f"relevance:{title[:50]}:{abstract[:100]}" # Shorter key components
    cache_key = hashlib.md5(cache_key_text.encode('utf-8', errors='replace')).hexdigest()
    cache_file = llm_setup['cache'].cache_dir / f"{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
                logger.info(f"(Cache hit for relevance check: {title[:30]}...)")
                return cached_result
        except Exception as e:
            logger.warning(f"ERROR reading relevance cache for {title[:30]}...: {e}")
    
    relevance_schema = {
        "type": "object",
        "properties": {
            "is_relevant": {
                "type": "boolean",
                "description": "Whether the abstract contains specific species threat information"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation for the decision"
            }
        },
        "required": ["is_relevant"]
    }
    
    system_prompt = """
    Determine if the scientific abstract contains information about threats to specific species or taxonomic groups.

    An abstract is RELEVANT if it:
    1. Mentions specific species or taxonomic groups (e.g., "Adelie penguins", "coral reefs")
    2. Describes specific threats or stressors (e.g., "climate change", "habitat loss")
    3. Explains how these threats affect the species (mechanisms, impacts)

    An abstract is NOT RELEVANT if it:
    1. Only discusses methodology without species impacts
    2. Focuses on conservation solutions without describing threats
    3. Is about species distribution without mentioning threats
    4. Discusses only human impacts without specific species effects

    Respond with a JSON object containing 'is_relevant' (boolean) and optional 'reasoning'.
    """
    
    response_json_str = "" # Initialize to ensure it's defined in case of early exit
    try:
        full_text = f"Title: {title}\n\nAbstract: {abstract}"
        logger.info(f"Asking LLM for structured relevance check for: {title[:30]}...")
        response_json_str = await llm_generate( # await the call
            prompt=full_text,
            system=system_prompt,
            model=llm_setup["model"], 
            temperature=0.1,
            timeout=120,
            format=relevance_schema, 
            llm_setup=llm_setup
        )
        
        if not response_json_str:
            logger.warning(f"LLM returned empty for relevance check: {title[:30]}... Defaulting to NOT relevant.")
            return False

        result = json.loads(response_json_str) 
        is_relevant = result.get("is_relevant", False)
        reasoning = result.get("reasoning", "")
        logger.info(f"LLM relevance check for '{title[:30]}...': Relevant={is_relevant}. Reasoning: {reasoning[:50]}...")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(is_relevant, f)
        except Exception as e_cache:
            logger.error(f"ERROR caching relevance result for {title[:30]}...: {e_cache}")
        
        return is_relevant
            
    except json.JSONDecodeError as e_json_decode:
        logger.error(f"Failed to parse JSON response for relevance: {e_json_decode}. Response: '{response_json_str}'. Title: {title[:30]}...")
        return False 
    except Exception as e_main:
        logger.error(f"ERROR in classify_abstract_relevance_ollama for {title[:30]}...: {e_main}", exc_info=True)
        return False

# This async wrapper is now redundant if classify_abstract_relevance_ollama is itself async.
# async def classify_abstract_relevance_ollama_async(title: str, abstract: str, llm_setup) -> bool:
#     return await classify_abstract_relevance_ollama(title, abstract, llm_setup)

def enrich_species_data(species_name: str, wiki_client: WikispeciesClient, llm_setup) -> Dict:
    """Enrich species data with taxonomic information and verification."""
    
    # First try exact match in Wikispecies
    wiki_result = wiki_client.search_species(species_name)
    
    if wiki_result and wiki_result.get('taxonomy'):
        return wiki_result['taxonomy']
    
    # Define JSON schema for taxonomic classification
    taxonomy_schema = {
        "type": "object",
        "properties": {
            "scientific_name": {
                "type": "string",
                "description": "The scientific name (binomial nomenclature) of the species"
            },
            "common_name": {
                "type": "string",
                "description": "The common name of the species"
            },
            "kingdom": {
                "type": "string",
                "description": "The kingdom taxonomic rank"
            },
            "phylum": {
                "type": "string", 
                "description": "The phylum taxonomic rank"
            },
            "class": {
                "type": "string",
                "description": "The class taxonomic rank (e.g., Aves for birds)"
            },
            "order": {
                "type": "string",
                "description": "The order taxonomic rank"
            },
            "family": {
                "type": "string",
                "description": "The family taxonomic rank"
            },
            "genus": {
                "type": "string",
                "description": "The genus taxonomic rank"
            },
            "confidence": {
                "type": "string",
                "description": "Confidence level (high, medium, low)"
            }
        },
        "required": ["scientific_name", "class", "confidence"]
    }
    
    system_prompt = """
    You are a taxonomic expert. For the given species or taxonomic group name, provide detailed taxonomic classification.
    
    Some common examples:
    - "Mallard" → Scientific name: Anas platyrhynchos, Class: Aves
    - "Wood Duck" → Scientific name: Aix sponsa, Class: Aves
    - "Canada Goose" → Scientific name: Branta canadensis, Class: Aves
    
    Be precise and provide as much taxonomic information as you can confidently determine.
    If scientific name is unknown, use best approximation and indicate lower confidence.
    """
    
    try:
        # Query LLM for taxonomic information with structured output
        response_json = llm_generate(
            prompt=f"Provide taxonomic classification for: {species_name}",
            system=system_prompt,
            model=llm_setup["model"],
            temperature=0.1,
            timeout=120,
            format=taxonomy_schema,
            llm_setup=llm_setup
        )
        
        # Parse the JSON response
        try:
            taxonomy_data = json.loads(response_json)
            
            # Get the scientific name from LLM response
            scientific_name = taxonomy_data.get("scientific_name")
            
            # Try searching Wikispecies again with the scientific name
            if scientific_name and scientific_name != species_name:
                wiki_result = wiki_client.search_species(scientific_name)
                if wiki_result and wiki_result.get('taxonomy'):
                    print(f"Found Wikispecies entry using scientific name: {scientific_name}")
                    return wiki_result['taxonomy']
            
            # If Wikispecies still fails, create taxonomy from LLM response
            llm_taxonomy = {
                'kingdom': taxonomy_data.get('kingdom'),
                'phylum': taxonomy_data.get('phylum'),
                'class': taxonomy_data.get('class'),
                'order': taxonomy_data.get('order'),
                'family': taxonomy_data.get('family'),
                'genus': taxonomy_data.get('genus'),
                'species': scientific_name,
                'common_name': taxonomy_data.get('common_name') or species_name,
                'rank_hierarchy': [],
                'llm_enriched': True,
                'confidence': taxonomy_data.get('confidence', 'medium')
            }
            
            print(f"Created LLM-based taxonomy for {species_name} (scientific name: {scientific_name})")
            return llm_taxonomy
            
        except json.JSONDecodeError as e:
            print(f"Error decoding taxonomy JSON: {e}")
            
            # Extract scientific name from text if JSON parsing fails
            scientific_name = None
            if "scientific name:" in response_json.lower():
                scientific_name = response_json.lower().split("scientific name:")[1].split("\n")[0].strip()
                wiki_result = wiki_client.search_species(scientific_name)
                if wiki_result and wiki_result.get('taxonomy'):
                    return wiki_result['taxonomy']
        
        # Fallback to basic taxonomy
        return {
            'kingdom': None,
            'phylum': None,
            'class': None,
            'order': None,
            'family': None,
            'genus': None,
            'species': species_name,
            'rank_hierarchy': [],
            'llm_enriched': True,
            'confidence': 'low'
        }
        
    except Exception as e:
        print(f"ERROR enriching species data: {e}")
        return None

def filter_abstracts_by_taxonomy(df, taxonomic_term="duck", max_results=None):
    """
    Filter abstracts by taxonomic term using a stricter pattern matching.
    
    Args:
        df: DataFrame containing abstracts
        taxonomic_term: Taxonomic term to filter by (e.g., "duck")
        max_results: Maximum number of results to consider per batch
    
    Returns:
        Filtered DataFrame
    """
    if not taxonomic_term:
        return df
        
    # Create a pattern that matches the term as a whole word, not as a substring
    # This helps avoid matching "duck" in words like "duckling" or "productive"
    pattern = f"(?i)\\b{taxonomic_term}s?\\b"
    
    # First attempt stricter filtering
    filtered_df = df.filter(
        pl.col("title").str.contains(pattern) | 
        pl.col("abstract").str.contains(pattern) | 
        pl.col("doi").str.contains(pattern) # Example if DOI was also a filter criterion, not requested here
    )
    
    # If we get too few results, fall back to a more permissive filter
    if len(filtered_df) < 5:
        print(f"Only found {len(filtered_df)} abstracts with strict filtering for '{taxonomic_term}'")
        print(f"Falling back to more permissive filtering...")
        pattern = f"(?i){taxonomic_term}"
        filtered_df = df.filter(
            pl.col("title").str.contains(pattern) | 
            pl.col("abstract").str.contains(pattern) | 
            pl.col("doi").str.contains(pattern) # Example if DOI was also a filter criterion, not requested here
        )
    
    # If max_results specified, limit the DataFrame but ensure we have enough to process
    if max_results is not None:
        # Allow some buffer to find relevant abstracts (2x requested)
        buffer_limit = min(len(filtered_df), max_results * 2)
        filtered_df = filtered_df.head(buffer_limit)
    
    print(f"Found {len(filtered_df)} abstracts mentioning '{taxonomic_term}'")
    return filtered_df

async def normalize_species_names(triplets: List[Tuple[str, str, str, str]], llm_setup) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, Dict]]:
    """ # Make sure this is async def
    Normalizes species names in triplets asynchronously and returns only bird species with their taxonomy.
    """
    logger.info(f"\n=== Starting Species Normalization for {len(triplets)} triplets ===") # Use logger
    
    unique_subjects = sorted(list(set(t[0] for t in triplets)))
    logger.info(f"Found {len(unique_subjects)} unique subjects to normalize.") # Use logger

    # Schema for normalization
    normalization_schema = {
        "type": "object",
        "properties": {
            "canonical_form": {"type": "string"},
            "scientific_name": {"type": "string"},
            "kingdom": {"type": "string"},
            "phylum": {"type": "string"},
            "class": {"type": "string"},
            "order": {"type": "string"},
            "family": {"type": "string"},
            "genus": {"type": "string"},
            "is_bird": {"type": "boolean"}
        },
        "required": ["canonical_form", "is_bird"]
    }

    system_prompt = """You are a taxonomic expert. For the given species or group name:
1. Provide the canonical form (standard, singular common name, e.g., "Mallard" for "mallards", "Bird" for "birds").
2. Provide the scientific name if available. For specific species, this is the Latin binomial. For broader groups, it's the taxon name (e.g., "Aves" for birds).
3. Provide the taxonomic classification (Kingdom, Phylum, Class, Order, Family, Genus) as specifically as possible based on the input.
4. Determine if the input refers to a bird (i.e., belongs to Class Aves) and set 'is_bird' to true or false.

Important: Only set 'is_bird' to true if the species/group belongs to Class Aves (birds).
Respond with valid JSON matching the required schema."""

    species_taxonomy_cache = {}
    
    # Create tasks for all unique subjects
    tasks = []
    for subject in unique_subjects:
        species_for_llm = subject
        if subject.lower() == "birds":
            species_for_llm = "Bird"
        
        # Define the coroutine for each subject
        async def get_taxonomy_for_subject(s_name, s_llm_name):
            try:
                response_json_str = await llm_generate(
                    prompt=f"Normalize this species name: {s_llm_name}",
                    system=system_prompt,
                    model=llm_setup["model"],
                    temperature=0.1,
                    format=normalization_schema,
                    llm_setup=llm_setup
                )
                if not response_json_str:
                    logger.error(f"  Error normalizing '{s_name}': LLM returned empty response.")
                    return s_name, {
                        'original_query': s_name,
                        'canonical_form': s_name,
                        'is_bird': False,
                        'source': 'Fallback_empty_llm_response'
                    }
                norm_data = json.loads(response_json_str)
                is_bird = norm_data.get("is_bird", False) or (
                    norm_data.get("class") and "aves" in norm_data.get("class", "").lower()
                )
                
                return s_name, {
                    'original_query': s_name,
                    'canonical_form': norm_data.get("canonical_form", s_llm_name),
                    'scientific_name': norm_data.get("scientific_name"),
                    'kingdom': norm_data.get("kingdom"),
                    'phylum': norm_data.get("phylum"),
                    'class': norm_data.get("class"),
                    'order': norm_data.get("order"),
                    'family': norm_data.get("family"),
                    'genus': norm_data.get("genus"),
                    'species': norm_data.get("scientific_name") if is_bird else None,
                    'is_bird': is_bird,
                    'rank_hierarchy': [],
                    'llm_enriched': True,
                    'source': 'LLM_normalization'
                }
            except json.JSONDecodeError as e_json:
                logger.error(f"  Error normalizing '{s_name}' (JSONDecodeError): {e_json}. Raw response: '{response_json_str}'") # Log raw response
                return s_name, {
                    'original_query': s_name,
                    'canonical_form': s_name,
                    'is_bird': False,
                    'source': 'Fallback_json_decode_error'
                }
            except Exception as e:
                logger.error(f"  Error normalizing '{s_name}' (General Exception): {e}", exc_info=True) # Log raw response
                return s_name, {
                    'original_query': s_name,
                    'canonical_form': s_name,
                    'is_bird': False,
                    'source': 'Fallback_general_exception'
                }
        tasks.append(get_taxonomy_for_subject(subject, species_for_llm))

    # Run all normalization tasks concurrently
    if tasks:
        logger.info(f"Starting concurrent normalization for {len(tasks)} unique subjects...")
        results = await asyncio.gather(*tasks)
        for subject_name, tax_data in results:
            species_taxonomy_cache[subject_name] = tax_data
            logger.info(f"  Normalized '{subject_name}' -> '{tax_data.get('canonical_form', subject_name)}' (Bird: {tax_data.get('is_bird', False)})") # Use logger
        logger.info("Concurrent normalization finished.")
    else:
        logger.info("No unique subjects to normalize.")


    # Filter triplets to only include birds
    normalized_triplets = []
    llm_taxonomy_map = {}
    
    for subject, predicate, obj, doi in triplets:
        tax_data = species_taxonomy_cache.get(subject)
        if tax_data and tax_data.get('is_bird', False):
            normalized_triplets.append((tax_data['canonical_form'], predicate, obj, doi))
        
        if tax_data: # This ensures we populate the map even for non-birds that were processed
            llm_taxonomy_map[subject] = tax_data

    logger.info(f"=== Species Normalization Complete ===") # Use logger
    logger.info(f"  - Original triplets: {len(triplets)}") # Use logger
    logger.info(f"  - Normalized bird triplets: {len(normalized_triplets)}") # Use logger
    logger.info(f"  - Taxonomy entries (including non-birds): {len(llm_taxonomy_map)}") # Use logger
    
    return normalized_triplets, llm_taxonomy_map

def setup_embedding_classifier(models_path=None):
    """
    Set up and return the embedding model for classification tasks.
    
    Args:
        models_path: Optional path to models directory. If None, uses default.
    
    Returns:
        The embedding model or None if setup fails
    """
    if not EMBEDDINGS_AVAILABLE:
        print("Cannot setup embedding classifier: sentence-transformers not installed")
        return None, None
        
    try:
        # Load a pre-trained sentence transformer model
        model_name = "all-mpnet-base-v2"  # Good balance of performance and efficiency
        model = SentenceTransformer(model_name)
        
        # Define paths
        if models_path is not None:
            models_dir = Path(models_path)
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = Path(current_dir) / "models"
        
        models_dir.mkdir(exist_ok=True)
        classifier_path = models_dir / "embedding_classifier.pkl"
        
        # Load existing classifier if available
        classifier = None
        if classifier_path.exists():
            try:
                with open(classifier_path, 'rb') as f:
                    classifier = pickle.load(f)
                print("Loaded existing embedding-based classifier.")
            except Exception as e:
                print(f"Error loading embedding classifier: {e}")
        
        return model, classifier
    except Exception as e:
        print(f"Error setting up embedding model: {e}")
        return None, None

def generate_embedding(text, embedding_model):
    """
    Generate an embedding vector for the given text.
    
    Args:
        text: Text to generate embedding for
        embedding_model: The embedding model to use

    Returns:
        Embedding vector or None if generation fails
    """
    if not text or embedding_model is None:
        return None
    
    try:
        embedding = embedding_model.encode(text)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Similarity score between -1 and 1
    """
    if vec1 is None or vec2 is None:
        return 0.0
    
    try:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0.0

def train_embedding_classifier(training_data, model):
    """
    Trains an embedding-based classifier from training data.
    
    Args:
        training_data: List of {'text': str, 'label': bool} training examples
        model: SentenceTransformer model for generating embeddings
        
    Returns:
        Trained classifier or None if training fails
    """
    if not EMBEDDINGS_AVAILABLE or model is None:
        print("Cannot train embedding classifier: Model not available")
        return None
        
    try:
        print(f"Training embedding classifier with {len(training_data)} examples...")
        texts = [item['text'] for item in training_data]
        labels = [1 if item['label'] else 0 for item in training_data]
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=True)
        
        # Train classifier
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(embeddings, labels)
        
        # Save the classifier
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = Path(current_dir) / "models"
        models_dir.mkdir(exist_ok=True)
        classifier_path = models_dir / "embedding_classifier.pkl"
        
        with open(classifier_path, 'wb') as f:
            pickle.dump(classifier, f)
        
        print("Trained and saved embedding-based classifier")
        return classifier
    except Exception as e:
        print(f"Error training embedding classifier: {e}")
        return None

def predict_relevance_embeddings(abstract, model, classifier):
    """
    Predicts relevance using embeddings and the classifier.
    
    Args:
        abstract: Text to classify
        model: SentenceTransformer model
        classifier: Trained classifier
        
    Returns:
        Boolean indicating if the abstract is relevant
    """
    if not EMBEDDINGS_AVAILABLE or model is None or classifier is None:
        print("Cannot predict with embeddings: Components not available")
        return False
        
    try:
        embedding = model.encode([abstract])[0].reshape(1, -1)
        prediction_prob = classifier.predict_proba(embedding)[0][1]
        is_relevant = prediction_prob > 0.5
        print(f"Embedding prediction: Relevant={is_relevant} (Prob: {prediction_prob:.2f})")
        return is_relevant
    except Exception as e:
        print(f"Error predicting with embedding classifier: {e}")
        return False

def classify_abstract_relevance_zeroshot(title, abstract, model):
    """
    Determine if an abstract contains relevant species threat information using zero-shot embedding comparison.
    
    Args:
        title: Abstract title
        abstract: Abstract text
        model: SentenceTransformer model
        
    Returns:
        Boolean indicating if the abstract is relevant
    """
    if not EMBEDDINGS_AVAILABLE or model is None:
        print("Cannot perform zero-shot classification: Model not available")
        return False
    
    # Check cache
    cache_key = hashlib.md5(f"zeroshot:{title}:{abstract[:100]}".encode('utf-8')).hexdigest()
    cache_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
                print("(Cache hit for zero-shot relevance check)")
                return cached_result
        except Exception as e:
            print(f"ERROR reading zero-shot cache: {e}")
    
    try:
        # Define reference descriptions
        references = [
            "This paper discusses specific threats to bird species and describes how these threats affect the species.",
            "This paper describes conservation efforts for protecting bird species from specific threats.",
            "This paper focuses on methodology, techniques, or general theories without specific information about bird threats."
        ]
        
        # Generate embeddings
        full_text = f"Title: {title}\n\nAbstract: {abstract}"
        text_embedding = model.encode([full_text])[0]
        reference_embeddings = model.encode(references)
        
        # Calculate similarities
        similarities = sklearn_cosine_similarity([text_embedding], reference_embeddings)[0]
        
        # Decide if relevant (more similar to references 1-2 than reference 3)
        relevant_score = max(similarities[0], similarities[1])
        irrelevant_score = similarities[2]
        is_relevant = relevant_score > irrelevant_score
        
        print(f"Zero-shot classification: Relevant={is_relevant} (Scores: Relevant={relevant_score:.3f}, Irrelevant={irrelevant_score:.3f})")
        
        # Cache result
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(is_relevant, f)
        except Exception as e:
            print(f"ERROR caching zero-shot result: {e}")
        
        return is_relevant
    except Exception as e:
        print(f"Error in zero-shot classification: {e}")
        return False

def cluster_threats(triplets, model, 
                      figures_dir: Path, # Expects direct Path to "figures" subdir
                      n_clusters=8):
    """
    Cluster threats based on semantic similarity to identify patterns.
    
    Args:
        triplets: List of (subject, predicate, object) triplets
        model: SentenceTransformer model
        n_clusters: Number of clusters to create
        
    Returns:
        Dictionary with cluster information
    """
    if not EMBEDDINGS_AVAILABLE or model is None:
        print("Cannot cluster threats: Model not available")
        return {}
        
    # Extract all threat descriptions
    threat_texts = []
    for _, _, obj, _doi in triplets: # Adjusted to unpack 4 elements, though _doi is not used here
        # Extract just the description part from objects that may contain IUCN tags
        description = obj
        if "[IUCN:" in obj:
            description = obj.split("[IUCN:")[0].strip()
        threat_texts.append(description)
    
    if not threat_texts:
        print("No threats to cluster")
        return {}
        
    print(f"Clustering {len(threat_texts)} threats...")
    
    try:
        # Generate embeddings
        threat_embeddings = model.encode(threat_texts, show_progress_bar=True)
        
        # Import KMeans
        from sklearn.cluster import KMeans
        
        # Cluster
        kmeans = KMeans(n_clusters=min(n_clusters, len(threat_texts)))
        clusters = kmeans.fit_predict(threat_embeddings)
        
        # Organize threats by cluster
        clustered_threats = defaultdict(list)
        for i, (text, cluster_id) in enumerate(zip(threat_texts, clusters)):
            clustered_threats[int(cluster_id)].append((i, text))
        
        # Analyze clusters
        cluster_summaries = {}
        for cluster_id, threats in clustered_threats.items():
            # Get representative threats (closest to cluster center)
            cluster_center = kmeans.cluster_centers_[cluster_id]
            threat_distances = []
            
            for idx, text in threats:
                dist = np.linalg.norm(threat_embeddings[idx] - cluster_center)
                threat_distances.append((idx, text, dist))
            
            # Sort by distance to center
            sorted_threats = sorted(threat_distances, key=lambda x: x[2])
            
            # Get the 3 most representative examples
            examples = [text for _, text, _ in sorted_threats[:3]]
            
            # Save summary
            cluster_summaries[cluster_id] = {
                "count": len(threats),
                "examples": examples,
                "threat_indices": [idx for idx, _text in threats] # Corrected unpacking
            }
            
            print(f"\nCluster {cluster_id} ({len(threats)} threats):")
            for ex in examples:
                print(f"  - {ex[:100]}...")
        
        # Save visualization
        try:
            # Try to import UMAP
            import umap
            
            # Create UMAP projection for visualization
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(threat_embeddings)
            
            # Plot
            plt.figure(figsize=(12, 10))
            
            # Create color map
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            
            # Plot points
            for cluster_id in range(n_clusters):
                cluster_points = [i for i, c in enumerate(clusters) if c == cluster_id]
                plt.scatter(
                    embedding_2d[cluster_points, 0], 
                    embedding_2d[cluster_points, 1],
                    s=50, 
                    c=[colors[cluster_id]],
                    label=f'Cluster {cluster_id} ({len(cluster_points)})'
                )
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title('Threat Clusters Visualization')
            
            # current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            # figures_dir = get_dynamic_output_path("figures", model_name_for_path, max_r_for_path, current_script_dir)
            figures_dir.mkdir(parents=True, exist_ok=True) # figures_dir is already the specific path
            output_figure_path = figures_dir / "threat_clusters.png"

            if output_figure_path: # This will always be true if figures_dir is valid
                plt.savefig(output_figure_path, bbox_inches='tight')
                plt.close()
                print(f"\nThreat cluster visualization saved to {output_figure_path}")
            # No else needed as output_figure_path is now always defined if this block is reached

        except ImportError:
            print("Could not create cluster visualization: UMAP not installed (pip install umap-learn)")
        except Exception as e:
            print(f"Could not create cluster visualization: {e}")
        
        return cluster_summaries
    except Exception as e:
        print(f"Error clustering threats: {e}")
        return {}

def setup_vector_search(abstracts, model):
    """
    Set up a vector database for similarity search.
    
    Args:
        abstracts: List of abstract texts
        model: SentenceTransformer model
        
    Returns:
        Dictionary with vector store data
    """
    if not EMBEDDINGS_AVAILABLE or model is None:
        print("Cannot setup vector search: Model not available")
        return None
        
    try:
        # Create embeddings
        print("Generating abstract embeddings...")
        abstract_embeddings = model.encode(abstracts, show_progress_bar=True)
        
        # Create in-memory vector store
        vector_store = {
            'embeddings': abstract_embeddings,
            'abstracts': abstracts
        }
        
        # Save for reuse
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = Path(current_dir) / "models"
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "vector_store.pkl", 'wb') as f:
            pickle.dump(vector_store, f)
            
        print(f"Vector store created with {len(abstracts)} abstracts")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def find_related_research(query_text, vector_store, model, top_k=5):
    """
    Find related abstracts using semantic similarity.
    
    Args:
        query_text: Text to find related abstracts for
        vector_store: Vector store from setup_vector_search
        model: SentenceTransformer model
        top_k: Number of results to return
        
    Returns:
        List of dictionaries with abstract and similarity score
    """
    if not EMBEDDINGS_AVAILABLE or model is None:
        print("Cannot find related research: Model not available")
        return []
        
    if not vector_store:
        print("Vector store not available")
        return []
    
    try:
        # Encode query
        query_vector = model.encode([query_text])[0]
        
        # Calculate similarities
        similarities = sklearn_cosine_similarity([query_vector], vector_store['embeddings'])[0]
        
        # Get top matches
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return results
        results = []
        for idx in top_indices:
            results.append({
                'abstract': vector_store['abstracts'][idx],
                'similarity': similarities[idx]
            })
        
        return results
    except Exception as e:
        print(f"Error finding related research: {e}")
        return []

def enrich_graph_with_embeddings(graph, model, 
                                 results_dir: Path # Expects direct Path to "results" subdir
                                 ):
    """
    Add embedding vectors as node attributes to enhance graph analysis.
    
    Args:
        graph: NetworkX graph
        model: SentenceTransformer model
        
    Returns:
        List of potential semantic connections
    """
    if not EMBEDDINGS_AVAILABLE or model is None:
        print("Cannot enrich graph: Model not available")
        return []
        
    print(f"Enriching graph with embeddings ({len(graph.nodes())} nodes)...")
    
    try:
        # Generate embeddings for all nodes
        node_texts = list(graph.nodes())
        node_embeddings = model.encode(node_texts, show_progress_bar=True)
        
        # Add embeddings as node attributes
        for i, node in enumerate(node_texts):
            graph.nodes[node]['embedding'] = node_embeddings[i]
        
        # Find potential missing connections based on similarity
        print("Analyzing potential semantic connections...")
        potential_connections = []
        
        # Sample a subset of nodes if graph is very large
        if len(graph.nodes()) > 1000:
            import random
            sample_nodes = random.sample(list(graph.nodes()), 1000)
        else:
            sample_nodes = list(graph.nodes())
        
        # For each node, find semantically similar nodes not already connected
        for i, node1 in enumerate(sample_nodes):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(sample_nodes)} nodes")
                
            embed1 = graph.nodes[node1]['embedding'].reshape(1, -1)
            
            for node2 in sample_nodes:
                if node1 != node2 and not graph.has_edge(node1, node2) and not graph.has_edge(node2, node1):
                    embed2 = graph.nodes[node2]['embedding'].reshape(1, -1)
                    similarity = sklearn_cosine_similarity(embed1, embed2)[0][0]
                    
                    if similarity > 0.85:  # High similarity threshold
                        potential_connections.append((node1, node2, similarity))
        
        # Sort by similarity (highest first)
        potential_connections.sort(key=lambda x: x[2], reverse=True)
        
        # Return the top potentials
        print(f"Found {len(potential_connections)} potential semantic connections")
        return potential_connections[:100]  # Return top 100
    except Exception as e:
        print(f"Error enriching graph: {e}")
        return []

def create_embedding_visualization(graph, model, 
                                   figures_dir: Path # Expects direct Path to "figures" subdir
                                   ):
    """
    Create a visualization of the graph based on embeddings instead of force-directed layout.
    
    Args:
        graph: NetworkX graph
        model: SentenceTransformer model
        
    Returns:
        Boolean indicating success
    """
    if not EMBEDDINGS_AVAILABLE or model is None:
        logger.warning("Cannot create embedding visualization: Model or sentence-transformers not available.")
        return False
        
    logger.info("Creating graph visualization based on node embeddings...")
    try:
        # Create a list of all nodes
        nodes = list(graph.nodes())
        
        # Get or create embeddings
        embeddings = []
        for node in nodes:
            if 'embedding' in graph.nodes[node]:
                embeddings.append(graph.nodes[node]['embedding'])
            else:
                # Generate embedding if not yet added
                embeddings.append(model.encode([node])[0])
        
        embeddings = np.array(embeddings)
        
        # Reduce to 2D
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            node_positions = reducer.fit_transform(embeddings)
            
            # Create visualization
            plt.figure(figsize=(20, 20))
            
            # Draw edges
            for u, v in graph.edges():
                i = nodes.index(u)
                j = nodes.index(v)
                plt.plot([node_positions[i, 0], node_positions[j, 0]],
                         [node_positions[i, 1], node_positions[j, 1]],
                         'k-', alpha=0.1, linewidth=0.5)
            
            # Draw nodes
            plt.scatter(node_positions[:, 0], node_positions[:, 1], s=10, alpha=0.8)
            
            # Draw labels for a subset of nodes (the ones with highest degree)
            degrees = dict(graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:100]
            
            for node, _ in top_nodes:
                i = nodes.index(node)
                plt.text(node_positions[i, 0], node_positions[i, 1], node, 
                        fontsize=8, alpha=0.7)
            
            figures_dir.mkdir(parents=True, exist_ok=True) # figures_dir is already specific path
            png_path = figures_dir / "embedding_graph.png"
            html_path = figures_dir / "embedding_graph.html"

            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close() # Close the plot to free memory
            
            logger.info(f"Embedding visualization (static PNG) saved to {png_path}")
            
            # Also create interactive HTML version if pyvis is available
            try:
                from pyvis.network import Network
                
                net = Network(height="800px", width="100%", notebook=False)
                
                # Add nodes
                for i, node_label in enumerate(nodes): # Use node_label to avoid conflict with node variable in outer scope
                    # Convert numpy.float32 to standard Python float for JSON serialization
                    x_coord = float(node_positions[i, 0])
                    y_coord = float(node_positions[i, 1])
                    net.add_node(i, label=node_label, x=x_coord*100, y=y_coord*100)
                    
                # Add edges
                for u, v in graph.edges():
                    i = nodes.index(u)
                    j = nodes.index(v)
                    net.add_edge(i, j)
                    
                net.save_graph(str(html_path)) 
                logger.info(f"Interactive embedding visualization (HTML) saved to {html_path}")
                
            except ImportError:
                logger.warning("Could not create interactive embedding visualization: pyvis not installed (pip install pyvis)")
            except Exception as e:
                logger.exception(f"Could not create interactive embedding visualization with pyvis: {e}")
                
        except ImportError:
            logger.warning("Could not create embedding visualization: UMAP not installed (pip install umap-learn)")
            return False
        except Exception as e:
            logger.exception(f"Error during UMAP/plotting for embedding visualization: {e}")
            return False
            
        return True
    except Exception as e:
        logger.exception(f"General error creating embedding visualization: {e}")
        return False

def analyze_related_research(query, vector_store=None, embedding_model=None):
    """
    Find abstracts related to a specific query.
    
    Args:
        query: Query text
        vector_store: Vector store from setup_vector_search
        embedding_model: SentenceTransformer model
        
    Returns:
        List of related abstracts
    """
    if not EMBEDDINGS_AVAILABLE:
        print("Cannot analyze related research: sentence-transformers not installed")
        return []
        
    try:
        # Initialize model if not provided
        if embedding_model is None:
            print("Initializing embedding model for analysis...")
            embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Load vector store if not provided
        if vector_store is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            vector_store_path = Path(current_dir) / "models" / "vector_store.pkl"
            
            if vector_store_path.exists():
                with open(vector_store_path, 'rb') as f:
                    vector_store = pickle.load(f)
            else:
                print("Vector store not found. Run setup_vector_search first.")
                return []
        
        # Find related abstracts
        results = find_related_research(query, vector_store, embedding_model)
        
        # Print results
        print(f"\nTop {len(results)} abstracts related to: '{query}'\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. Similarity: {result['similarity']:.3f}")
            print(f"   Abstract: {result['abstract'][:200]}...\n")
        
        return results
    except Exception as e:
        print(f"Error analyzing related research: {e}")
        return []

def parse_wikispecies_rank_hierarchy(rank_hierarchy_list: List[str]) -> Dict[str, Optional[str]]:
    """
    Parses the rank_hierarchy list from Wikispecies data into a standardized dictionary.
    """
    parsed_ranks = {
        "kingdom": None, "phylum": None, "class": None, "order": None,
        "family": None, "genus": None, "species": None
    }
    ws_to_standard_keys = {
        "superregnum": "kingdom", "regnum": "kingdom", "phylum": "phylum",
        "subphylum": "phylum", "classis": "class", "subclassis": "class",
        "ordo": "order", "familia": "family", "genus": "genus", "species": "species"
    }
    temp_class_values = []
    temp_kingdom_values = {}

    for item_str in rank_hierarchy_list:
        if ':' not in item_str:
            continue
        try:
            rank_label_ws, rank_value_ws = item_str.split(':', 1)
        except ValueError:
            continue # Skip lines not in key:value format
        rank_label_ws_lower = rank_label_ws.strip().lower()
        rank_value_ws_stripped = rank_value_ws.strip()

        standard_key = ws_to_standard_keys.get(rank_label_ws_lower)
        if standard_key:
            if standard_key == "class":
                temp_class_values.append(rank_value_ws_stripped)
            elif standard_key == "kingdom":
                temp_kingdom_values[rank_label_ws_lower] = rank_value_ws_stripped
            elif standard_key == "phylum":
                if rank_label_ws_lower == "phylum" or not parsed_ranks.get(standard_key):
                    parsed_ranks[standard_key] = rank_value_ws_stripped
            elif standard_key == "species":
                # Species entry in WS hierarchy is often "Genus species_epithet"
                # We want to store the full string, e.g., "Tanysiptera sylvia"
                parsed_ranks[standard_key] = rank_value_ws_stripped
            else:
                parsed_ranks[standard_key] = rank_value_ws_stripped
    
    if 'regnum' in temp_kingdom_values:
        parsed_ranks["kingdom"] = temp_kingdom_values['regnum']
    elif 'superregnum' in temp_kingdom_values:
        parsed_ranks["kingdom"] = temp_kingdom_values['superregnum']

    aves_class_value = None
    for cv in temp_class_values:
        if "aves" in cv.lower():
            aves_class_value = cv 
            break
    if aves_class_value:
        parsed_ranks["class"] = aves_class_value
    elif temp_class_values:
        parsed_ranks["class"] = temp_class_values[-1]
        
    return parsed_ranks

def compare_and_log_taxonomy_discrepancies(
    enriched_triplets_path: Path, 
    wikispecies_log_path: Path,
    output_log_path: Path
):
    """
    Compares taxonomic data from LLM-enriched triplets and Wikispecies verification log.
    Logs discrepancies to a specified output file and prints a summary.
    """
    try:
        with open(enriched_triplets_path, 'r', encoding='utf-8') as f:
            enriched_data = json.load(f)
        with open(wikispecies_log_path, 'r', encoding='utf-8') as f:
            wikispecies_data_list = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: One or both input files not found for taxonomy comparison: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from one of the files for taxonomy comparison: {e}")
        return

    # Handle both wrapped and unwrapped formats
    if isinstance(enriched_data, dict) and 'triplets' in enriched_data:
        enriched_data_list = enriched_data['triplets']
    elif isinstance(enriched_data, list):
        enriched_data_list = enriched_data
    else:
        print(f"Error: Unexpected data format in {enriched_triplets_path}")
        return

    llm_taxonomy_map = {}
    for item in enriched_data_list:
        taxonomy_info = item.get('taxonomy', {})
        scientific_name = taxonomy_info.get('scientific_name') # LLM output uses this for scientific name
        common_name = item.get('subject') 
        if not scientific_name and taxonomy_info.get('species'): 
            scientific_name = taxonomy_info.get('species')

        # Ensure entries are added even if scientific_name is an empty string ""
        if isinstance(scientific_name, str):
            key_to_use = scientific_name.lower()
            llm_taxonomy_map[key_to_use] = {
                'taxonomy_data': taxonomy_info,
                'common_name_llm': common_name or taxonomy_info.get('canonical_form')
            }

    wikispecies_taxonomy_map = {}
    for item in wikispecies_data_list:
        ws_tax_data = item.get('taxonomy')
        if not ws_tax_data: 
            continue

        key_name_ws = None
        if ws_tax_data.get('species') and isinstance(ws_tax_data.get('species'), str):
            key_name_ws = ws_tax_data.get('species')
        elif item.get('found_name') and isinstance(item.get('found_name'), str):
            key_name_ws = item.get('found_name')
        
        if key_name_ws:
            ws_tax_data_augmented = ws_tax_data.copy()
            ws_tax_data_augmented['original_wikispecies_query_name'] = item.get('query_name', 'N/A')
            # Process with rank_hierarchy parser before storing in map
            if 'rank_hierarchy' in ws_tax_data_augmented and ws_tax_data_augmented['rank_hierarchy']:
                parsed_from_hierarchy = parse_wikispecies_rank_hierarchy(ws_tax_data_augmented['rank_hierarchy'])
                
                # Debug: Print what the parser returned for the specific problematic case or any case
                #print(f"  [Debug MapBuild] For WS Log Entry Query: '{item.get('query_name', 'N/A')}' (becomes map key: '{key_name_ws.lower() if key_name_ws else 'None'}')")
                #print(f"    Parser returned: kingdom='{parsed_from_hierarchy.get('kingdom')}', phylum='{parsed_from_hierarchy.get('phylum')}', class='{parsed_from_hierarchy.get('class')}', species='{parsed_from_hierarchy.get('species')}'")

                for rank_key, rank_value in parsed_from_hierarchy.items():
                    if rank_value is not None: # Only overwrite if parser provided a value
                        ws_tax_data_augmented[rank_key] = rank_value
                
                # Debug: Print the state of ws_tax_data_augmented after attempting update
                print(f"    ws_tax_data_augmented after update: kingdom='{ws_tax_data_augmented.get('kingdom')}', phylum='{ws_tax_data_augmented.get('phylum')}', class='{ws_tax_data_augmented.get('class')}', species='{ws_tax_data_augmented.get('species')}'")
            else:
                print(f"  [Debug MapBuild] For WS Log Entry Query: '{item.get('query_name', 'N/A')}' (becomes map key: '{key_name_ws.lower() if key_name_ws else 'None'}') - No rank_hierarchy to parse or hierarchy was empty.")
            
            wikispecies_taxonomy_map[key_name_ws.lower()] = ws_tax_data_augmented

    discrepancies = []
    taxonomic_ranks_to_compare = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    print(f"\n--- Comparing Taxonomies (LLM-enriched vs. Wikispecies) ---")
    print(f"Found {len(llm_taxonomy_map)} unique scientific names in LLM-enriched data.")
    print(f"Found {len(wikispecies_taxonomy_map)} unique scientific names in Wikispecies log with taxonomy.")

    for llm_sci_name_lower, llm_entry in llm_taxonomy_map.items():
        llm_tax_data = llm_entry['taxonomy_data']
        llm_common_name = llm_entry['common_name_llm']
        ws_tax_data_for_comparison = wikispecies_taxonomy_map.get(llm_sci_name_lower)
        match_type = "scientific_name"

        if not ws_tax_data_for_comparison:
            # Try common name fallback
            for ws_item_original_log in wikispecies_data_list:
                original_query_name_from_log = ws_item_original_log.get('query_name', '')
                found_name_from_log = ws_item_original_log.get('found_name', '')
                potential_ws_taxonomy_raw = ws_item_original_log.get('taxonomy', {})

                # Ensure llm_common_name is a string first for its .lower() call
                if isinstance(llm_common_name, str):
                    llm_common_name_lower = llm_common_name.lower()
                    
                    # Check against original_query_name_from_log
                    match_in_original = False
                    if isinstance(original_query_name_from_log, str):
                        match_in_original = llm_common_name_lower in original_query_name_from_log.lower()
                    
                    # Check against found_name_from_log
                    match_in_found = False
                    if isinstance(found_name_from_log, str):
                        match_in_found = llm_common_name_lower in found_name_from_log.lower()

                    if match_in_original or match_in_found:
                        # This block is entered if a match is found in either the original query name or the found name from the log
                        
                        # potential_ws_taxonomy_raw is from: ws_item_original_log.get('taxonomy', {})
                        # This means potential_ws_taxonomy_raw can be:
                        # 1. A dictionary with data (if 'taxonomy' key had data)
                        # 2. An empty dictionary {} (if 'taxonomy' key was missing)
                        # 3. None (if 'taxonomy' key existed with value null/None)

                        if potential_ws_taxonomy_raw is None: # Handles case 3
                            print(f"  [Debug] Fallback common name match for LLM:'{llm_common_name}' (SciName: '{llm_sci_name_lower}') found WS entry query:'{original_query_name_from_log}', but its taxonomy data in log was explicitly null. Continuing search...")
                            continue # To the next ws_item_original_log in the inner loop

                        # Now, potential_ws_taxonomy_raw is guaranteed to be a dictionary (either populated or empty).
                        processed_fallback_taxonomy = potential_ws_taxonomy_raw.copy() # Line 3387 - Now safe.
                        
                        # Check for 'rank_hierarchy' existence AND it being non-empty list
                        if processed_fallback_taxonomy.get('rank_hierarchy'): # .get() handles missing key, and then check if list is not empty
                            parsed_from_hierarchy = parse_wikispecies_rank_hierarchy(processed_fallback_taxonomy['rank_hierarchy'])
                            for rank_key, rank_value in parsed_from_hierarchy.items():
                                if rank_value is not None:
                                    processed_fallback_taxonomy[rank_key] = rank_value
                            
                            matched_wikispecies_query_name = original_query_name_from_log
                            ws_tax_data_for_comparison = processed_fallback_taxonomy
                            match_type = "common_name_fallback"
                            print(f"  [Debug] Fallback common name match for LLM:'{llm_common_name}' (SciName: '{llm_sci_name_lower}') using WS query:'{original_query_name_from_log}'. Parsed hierarchy.")
                            break 
                        else: # rank_hierarchy missing or empty in the found WS log entry
                            print(f"  [Debug] Fallback common name match for LLM:'{llm_common_name}' (SciName: '{llm_sci_name_lower}') found WS entry query:'{original_query_name_from_log}', but its rank_hierarchy was missing or empty. Continuing search...")
                            # ws_tax_data_for_comparison is NOT set here for THIS ws_item_original_log.
                            # The loop will continue to search for other common name matches.

        if not ws_tax_data_for_comparison:
            discrepancies.append({
                "scientific_name_llm": llm_sci_name_lower,
                "common_name_llm": llm_common_name or "N/A",
                "discrepancy_type": "Wikispecies_Entry_Missing_Or_No_Taxonomy",
                "details": f"No taxonomy found in Wikispecies log for LLM scientific name '{llm_sci_name_lower}' Fallback by common name ('{llm_common_name}') also failed or not applicable."
            })
            continue

        species_rank_discrepancies = []
        for rank in taxonomic_ranks_to_compare:
            llm_rank_value = None
            if rank == 'species': # This refers to the scientific name itself
                llm_rank_value = llm_tax_data.get('scientific_name') or llm_tax_data.get('species')
            else:
                llm_rank_value = llm_tax_data.get(rank)
            
            ws_rank_value = ws_tax_data_for_comparison.get(rank)
            
            llm_norm = str(llm_rank_value).lower().strip() if llm_rank_value is not None else None
            ws_norm = str(ws_rank_value).lower().strip() if ws_rank_value is not None else None
            
            if llm_norm != ws_norm:
                # Special handling for 'class' to allow for prefixes like 'subclassis:' from Wikispecies
                if rank == 'class' and ws_norm and llm_norm:
                    # If one is a substring of the other (e.g., "aves" in "subclassis: aves"), consider match
                    if llm_norm in ws_norm or ws_norm in llm_norm: 
                        continue
                 
                species_rank_discrepancies.append({
                    "rank": rank,
                    "llm_value": llm_rank_value,
                    "wikispecies_value": ws_rank_value,
                })
        
        if species_rank_discrepancies:
            discrepancies.append({
                "scientific_name_llm": llm_sci_name_lower,
                "common_name_llm": llm_common_name or "N/A",
                "wikispecies_original_query": ws_tax_data_for_comparison.get('original_wikispecies_query_name', "N/A"),
                "discrepancy_type": "Taxonomic_Rank_Mismatch",
                "mismatches": species_rank_discrepancies,
                "match_type": match_type
            })

    if discrepancies:
        print(f"Found discrepancies for {len(discrepancies)} species entries.")
        for disc_entry in discrepancies:
            if disc_entry["discrepancy_type"] == "Wikispecies_Entry_Missing_Or_No_Taxonomy":
                print(f"  - Missing Wikispecies data for: {disc_entry['scientific_name_llm']} (Common: {disc_entry['common_name_llm']})")
            else:
                print(f"  - Mismatches for: {disc_entry['scientific_name_llm']} (Common: {disc_entry['common_name_llm']})")
                for mismatch in disc_entry['mismatches']:
                    print(f"    Rank '{mismatch['rank']}': LLM='{mismatch['llm_value']}', WS='{mismatch['wikispecies_value']}'")
    else:
        print("No taxonomic discrepancies found between LLM-enriched data and Wikispecies log for matched scientific names.")

    try:
        with open(output_log_path, 'w', encoding='utf-8') as f:
            json.dump(discrepancies, f, indent=2)
        print(f"Taxonomy comparison log saved to: {output_log_path}")
    except IOError as e:
        print(f"Error writing taxonomy comparison log: {e}")

    return discrepancies

# Ensure this helper function is defined globally
def parse_wikispecies_rank_hierarchy(rank_hierarchy_list: List[str]) -> Dict[str, Optional[str]]:
    """
    Parses the rank_hierarchy list from Wikispecies data into a standardized dictionary.
    """
    parsed_ranks = {
        "kingdom": None, "phylum": None, "class": None, "order": None,
        "family": None, "genus": None, "species": None
    }
    ws_to_standard_keys = {
        "superregnum": "kingdom", "regnum": "kingdom", "phylum": "phylum",
        "subphylum": "phylum", "classis": "class", "subclassis": "class",
        "ordo": "order", "familia": "family", "genus": "genus", "species": "species"
    }
    temp_class_values = []
    temp_kingdom_values = {}

    for item_str in rank_hierarchy_list:
        if ':' not in item_str:
            continue
        try:
            rank_label_ws, rank_value_ws = item_str.split(':', 1)
        except ValueError:
            continue # Skip lines not in key:value format
        rank_label_ws_lower = rank_label_ws.strip().lower()
        rank_value_ws_stripped = rank_value_ws.strip()

        standard_key = ws_to_standard_keys.get(rank_label_ws_lower)
        if standard_key:
            if standard_key == "class":
                temp_class_values.append(rank_value_ws_stripped)
            elif standard_key == "kingdom":
                temp_kingdom_values[rank_label_ws_lower] = rank_value_ws_stripped
            elif standard_key == "phylum":
                if rank_label_ws_lower == "phylum" or not parsed_ranks.get(standard_key):
                    parsed_ranks[standard_key] = rank_value_ws_stripped
            elif standard_key == "species":
                # Species entry in WS hierarchy is often "Genus species_epithet"
                # We want to store the full string, e.g., "Tanysiptera sylvia"
                parsed_ranks[standard_key] = rank_value_ws_stripped
            else:
                parsed_ranks[standard_key] = rank_value_ws_stripped
    
    if 'regnum' in temp_kingdom_values:
        parsed_ranks["kingdom"] = temp_kingdom_values['regnum']
    elif 'superregnum' in temp_kingdom_values:
        parsed_ranks["kingdom"] = temp_kingdom_values['superregnum']

    aves_class_value = None
    for cv in temp_class_values:
        if "aves" in cv.lower():
            aves_class_value = cv 
            break
    if aves_class_value:
        parsed_ranks["class"] = aves_class_value
    elif temp_class_values:
        parsed_ranks["class"] = temp_class_values[-1]
        
    return parsed_ranks

async def verify_species_with_wikispecies_concurrently(species_list: List[str], 
                                                 run_results_path: Path # Pass the target results path directly
                                                 ):
    """
    Verifies a list of species names concurrently using Wikispecies API and aiohttp.
    Uses a persistent Wikispecies lookup file shared across all runs, plus saves a run-specific log.
    """
    wiki_client = WikispeciesClient()
    
    # Persistent lookup file path (shared across all runs, relative to this script's directory)
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    persistent_lookup_path = script_dir / "wikispecies_taxonomy_lookup.json"
    
    # Run-specific log file path (for this run's documentation)
    run_results_path.mkdir(parents=True, exist_ok=True)
    run_log_filepath = run_results_path / "wikispecies_verification_log.json"
    
    print(f"Using persistent Wikispecies lookup: {persistent_lookup_path}")
    print(f"Run-specific log will be saved to: {run_log_filepath}")

    final_log_entries_map = {} # Stores all entries: query_name.lower() -> entry_dict

    # 1. Load existing data from persistent lookup file
    if persistent_lookup_path.exists():
        try:
            with open(persistent_lookup_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                for entry in loaded_data:
                    if isinstance(entry, dict) and 'query_name' in entry and isinstance(entry['query_name'], str):
                        final_log_entries_map[entry['query_name'].lower()] = entry
            print(f"Loaded {len(final_log_entries_map)} entries from persistent lookup: {persistent_lookup_path}")
        except FileNotFoundError:
            print(f"Persistent lookup file {persistent_lookup_path} not found. Will create a new one.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {persistent_lookup_path}: {e}. Proceeding as if lookup is empty.")
            final_log_entries_map = {} # Reset if corrupted
        except Exception as e:
            print(f"Error loading {persistent_lookup_path}: {e}. Proceeding as if lookup is empty.")
            final_log_entries_map = {} # Reset on other errors
    else:
        print(f"Persistent lookup file {persistent_lookup_path} does not exist. Will create it.")

    species_to_fetch_live = []
    # Use a set to track processed species from the input list to avoid duplicate fetches if input has duplicates
    input_species_processed_this_run = set()

    # 2. Identify species from input_list that need live fetching
    for species_name_from_input in species_list:
        if not isinstance(species_name_from_input, str) or not species_name_from_input.strip():
            print(f"Skipping invalid species name in input: {species_name_from_input}")
            continue

        species_name_lower = species_name_from_input.lower()
        
        if species_name_lower in input_species_processed_this_run:
            continue # Already decided to fetch or use cache for this, based on first encounter in list
        input_species_processed_this_run.add(species_name_lower)

        if species_name_lower not in final_log_entries_map:
            print(f"Species '{species_name_from_input}' not in persistent lookup. Will fetch live.")
            species_to_fetch_live.append(species_name_from_input) # Use original casing for API query
        else:
            print(f"Using cached Wikispecies result for: {species_name_from_input}")
            # Entry is already in final_log_entries_map from initial load

    # 3. Fetch live data for new species
    if species_to_fetch_live:
        print(f"Fetching {len(species_to_fetch_live)} new species from Wikispecies live...")
        timeout = aiohttp.ClientTimeout(total=60) 
        connector = aiohttp.TCPConnector(limit_per_host=5, ssl=False) 

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [wiki_client.search_species_async(name, session) for name in species_to_fetch_live]
            
            if not tasks:
                print("No new species identified to process live after checking cache.") # Should not happen if species_to_fetch_live is populated
            else:
                gathered_new_results = await asyncio.gather(*tasks, return_exceptions=True)

                # 4. Add newly fetched results to final_log_entries_map
                for i, res_or_exc in enumerate(gathered_new_results):
                    # The query_name used for the API call (original casing)
                    original_query_name_for_this_task = species_to_fetch_live[i]
                    entry_for_map = None

                    if isinstance(res_or_exc, Exception):
                        print(f"Error processing '{original_query_name_for_this_task}' live: {res_or_exc}")
                        entry_for_map = {
                            'query_name': original_query_name_for_this_task,
                            'found_name': None, 'page_id': None, 'taxonomy': None,
                            'error': f"Task failed: {str(res_or_exc)}",
                            'raw_response': None
                        }
                    elif isinstance(res_or_exc, dict):
                        # Ensure the result from search_species_async has query_name matching input
                        if res_or_exc.get('query_name') != original_query_name_for_this_task:
                            print(f"Warning: Mismatch in query_name for {original_query_name_for_this_task}. Result was for {res_or_exc.get('query_name')}. Adjusting.")
                            res_or_exc['query_name'] = original_query_name_for_this_task # Ensure consistency
                        entry_for_map = res_or_exc
                    else: 
                        print(f"Unknown result type for '{original_query_name_for_this_task}': {type(res_or_exc)}")
                        entry_for_map = {
                            'query_name': original_query_name_for_this_task,
                            'error': f"Unknown result type from task: {type(res_or_exc)}",
                            'raw_response': None, 'taxonomy': None, 'page_id': None, 'found_name': None
                        }
                    
                    # Add/update in the map using the original query name's lowercase as key
                    if entry_for_map and 'query_name' in entry_for_map:
                        final_log_entries_map[entry_for_map['query_name'].lower()] = entry_for_map
    else:
        print("All species from input list were already found in the persistent lookup. No live fetching needed.")
    
    # 5. Save the complete map's values to both the persistent lookup file and run-specific log
    try:
        all_entries_to_save = list(final_log_entries_map.values())
        
        # Save to persistent lookup file
        with open(persistent_lookup_path, "w", encoding='utf-8') as f:
            json.dump(all_entries_to_save, f, indent=2)
        print(f"Updated persistent Wikispecies lookup with {len(all_entries_to_save)} total entries: {persistent_lookup_path}")
        
        # Save to run-specific log file
        with open(run_log_filepath, "w", encoding='utf-8') as f:
            json.dump(all_entries_to_save, f, indent=2)
        print(f"Run-specific Wikispecies log with {len(all_entries_to_save)} total entries saved to {run_log_filepath}")
        
    except Exception as e:
        print(f"ERROR writing Wikispecies files: {e}")

    # Update the client's internal list if needed for other consumers (optional, as primary state is the file)
    wiki_client.query_results = all_entries_to_save

# Define a safety constant for scanning the parquet file if no max_results is set
MAX_PARQUET_ROWS_TO_SCAN_IF_NO_MAX_RESULTS = 50000 # Adjust as needed

async def process_abstract_chunk(
    relevant_abstract_data_chunk: List[Dict], 
    llm_setup, 
    refinement_cache
) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, Dict]]:
    """Processes a chunk of relevant abstracts through summary, P2, IUCN, and normalization."""
    logger.info(f"--- Starting process_abstract_chunk for {len(relevant_abstract_data_chunk)} abstracts ---")
    chunk_dois = [d.get('doi', 'N/A') for d in relevant_abstract_data_chunk]
    logger.debug(f"Chunk DOIs: {chunk_dois}")

    # --- Abstract Processing (Summaries, Pillar 2 Raw Triplets) for Chunk ---
    summary_extraction_tasks_chunk = []
    abstract_details_for_p2_chunk = [] 

    for abstract_data in relevant_abstract_data_chunk:
        summary_extraction_tasks_chunk.append(convert_to_summary(abstract_data['abstract'], llm_setup))
        abstract_details_for_p2_chunk.append({
            'abstract_text': abstract_data['abstract'],
            'doi': abstract_data['doi'],
            'title': abstract_data['title']
        })
    
    chunk_all_raw_triplets_flat = []
    if summary_extraction_tasks_chunk:
        logger.info(f"P2 Chunk: Starting summary generation for {len(summary_extraction_tasks_chunk)} abstracts...")
        generated_summaries_chunk = await asyncio.gather(*summary_extraction_tasks_chunk)
        logger.info("P2 Chunk: Summary generation finished.")

        p2_processing_tasks_chunk = []
        for i, summary_text in enumerate(generated_summaries_chunk):
            if i < len(abstract_details_for_p2_chunk):
                current_abstract_detail = abstract_details_for_p2_chunk[i]
                abs_text = current_abstract_detail['abstract_text']
                current_doi = current_abstract_detail['doi']

                if summary_text: 
                    async def process_single_abstract_p2_in_chunk(abstract_content, doi_val, llm_s): # Renamed for clarity
                        logger.info(f"P2 Chunk: Stage 2.1 - Extracting entities for DOI {doi_val}...")
                        entities = await extract_entities_concurrently(abstract_content, llm_s)
                        if entities and entities.get("species") and entities.get("threats"):
                            logger.info(f"P2 Chunk: Stage 2.2 - Generating relationships for DOI {doi_val} using {len(entities['species'])} species and {len(entities['threats'])} threats...")
                            raw_trips = await generate_relationships_concurrently(abstract_content, entities["species"], entities["threats"], llm_s, doi_val)
                            return raw_trips
                        else:
                            logger.warning(f"P2 Chunk: Entity extraction failed or yielded no entities for DOI {doi_val}. Abstract: {abstract_content[:50]}...")
                            return []
                    p2_processing_tasks_chunk.append(process_single_abstract_p2_in_chunk(abs_text, current_doi, llm_setup))
                else:
                    logger.warning(f"P2 Chunk: Skipping Pillar 2 processing for DOI {current_doi} due to empty summary.")
            else:
                logger.error(f"P2 Chunk: Mismatch index {i} for summaries vs abstract_details_for_p2. Critical error.")
        
        if p2_processing_tasks_chunk:
            logger.info(f"P2 Chunk: Starting consolidated entity extraction & relationship generation for {len(p2_processing_tasks_chunk)} abstracts...")
            results_from_p2_tasks_chunk = await asyncio.gather(*p2_processing_tasks_chunk)
            logger.info("P2 Chunk: Consolidated entity extraction & relationship generation finished.")
            for result_triplet_list in results_from_p2_tasks_chunk:
                if result_triplet_list: 
                    chunk_all_raw_triplets_flat.extend(result_triplet_list)
    
    logger.info(f"P2 Chunk: Extracted {len(chunk_all_raw_triplets_flat)} raw triplets in this chunk.")

    if not chunk_all_raw_triplets_flat:
        logger.warning("P2 Chunk: No raw triplets extracted. Returning empty from chunk.")
        return [], {}

    # --- IUCN Enrichment for Chunk ---
    iucn_classification_input_items_chunk = [] 
    pre_enriched_triplets_map_chunk = {} 

    for original_idx, (s, p, original_o, d) in enumerate(chunk_all_raw_triplets_flat):
        desc, code, name, is_valid = parse_and_validate_object(original_o)
        final_desc = desc if desc else original_o
        needs_iucn_call = not is_valid or not (code and name) or code == "12.1"

        if needs_iucn_call:
            cache_key_iucn = f"iucn_classify_json_schema:{final_desc}|context:{s}|{p}"
            cached_iucn = refinement_cache.get(cache_key_iucn)
            if cached_iucn:
                cached_code, cached_name = cached_iucn
                refined_o_str = f"{final_desc} [IUCN: {cached_code} {cached_name}]"
                pre_enriched_triplets_map_chunk[original_idx] = (s, p, refined_o_str, d)
            else:
                iucn_classification_input_items_chunk.append((s, p, final_desc, original_o, original_idx))
        else:
            refined_o_str = f"{final_desc} [IUCN: {code} {name}]"
            pre_enriched_triplets_map_chunk[original_idx] = (s, p, refined_o_str, d)
    
    iucn_tasks_chunk = [
        get_iucn_classification_json(item[0], item[1], item[2], llm_setup, refinement_cache) 
        for item in iucn_classification_input_items_chunk
    ]
    
    chunk_final_enriched_triplets_for_verification = [None] * len(chunk_all_raw_triplets_flat)

    if iucn_tasks_chunk:
        logger.info(f"Chunk IUCN: Starting classification for {len(iucn_tasks_chunk)} items...")
        iucn_results_chunk = await asyncio.gather(*iucn_tasks_chunk)
        logger.info("Chunk IUCN: Classification finished.")
        for i, (classified_code, classified_name) in enumerate(iucn_results_chunk):
            s_iucn, p_iucn, desc_iucn, _original_o_iucn, original_idx_iucn = iucn_classification_input_items_chunk[i]
            refined_o_str = f"{desc_iucn} [IUCN: {classified_code} {classified_name}]"
            chunk_final_enriched_triplets_for_verification[original_idx_iucn] = (s_iucn, p_iucn, refined_o_str, chunk_all_raw_triplets_flat[original_idx_iucn][3])
    
    for idx, triplet_data in pre_enriched_triplets_map_chunk.items():
        chunk_final_enriched_triplets_for_verification[idx] = triplet_data
    
    for idx_fallback in range(len(chunk_all_raw_triplets_flat)):
        if chunk_final_enriched_triplets_for_verification[idx_fallback] is None:
            s_fallback, p_fallback, o_fallback, d_fallback = chunk_all_raw_triplets_flat[idx_fallback]
            logger.warning(f"Chunk IUCN: Triplet at flat index {idx_fallback} ({s_fallback[:20]}) missed IUCN, using original: {o_fallback[:30]}")
            chunk_final_enriched_triplets_for_verification[idx_fallback] = (s_fallback, p_fallback, o_fallback, d_fallback)
            
    chunk_final_enriched_triplets_for_verification = [t for t in chunk_final_enriched_triplets_for_verification if t is not None]
    logger.info(f"Chunk IUCN: Enrichment complete: {len(chunk_final_enriched_triplets_for_verification)} triplets.")

    if not chunk_final_enriched_triplets_for_verification:
        logger.warning("Chunk: No enriched triplets found after IUCN. Returning empty from chunk.")
        return [], {}

    # --- Species Name Normalization for Chunk ---
    logger.info(f"Chunk Norm: Normalizing species names for {len(chunk_final_enriched_triplets_for_verification)} enriched triplets...")
    chunk_normalized_triplets, chunk_llm_taxonomy_map = await normalize_species_names(
        chunk_final_enriched_triplets_for_verification, llm_setup
    )
    logger.info(f"Chunk Norm: Normalization complete. {len(chunk_normalized_triplets)} triplets, {len(chunk_llm_taxonomy_map)} taxonomy entries.")
    
    logger.info(f"--- Finished process_abstract_chunk. Returning {len(chunk_normalized_triplets)} normalized triplets and {len(chunk_llm_taxonomy_map)} taxonomy entries. ---")
    return chunk_normalized_triplets, chunk_llm_taxonomy_map


async def run_main_pipeline_logic(args):
    """Run the main (sequential but internally asynchronous) pipeline logic with batched data loading and chunked processing."""
    
    # --- Determine Run-Specific Paths Early & Setup Logging ---
    current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    llm_system_setup = setup_llm() 
    model_name_for_path = os.getenv('MODEL_NAME_FOR_RUN', llm_system_setup["model"])

    max_r_from_args = getattr(args, 'max', None) 
    max_r_env_str = os.getenv('MAX_RESULTS', 'all')
    max_r_from_env = None
    if str(max_r_env_str).lower() == 'all':
        max_r_from_env = "all"
    elif str(max_r_env_str).isdigit():
        max_r_from_env = int(max_r_env_str)
    
    max_r_for_path_setup = max_r_from_args if max_r_from_args is not None else max_r_from_env
    if max_r_for_path_setup is None: 
        max_r_for_path_setup = "all"

    # Define max_results_limit from max_r_for_path_setup before it's used in logging
    max_results_limit = float('inf')
    if isinstance(max_r_for_path_setup, int):
        max_results_limit = max_r_for_path_setup
    elif str(max_r_for_path_setup).lower() != 'all': 
        try:
            max_results_limit = int(max_r_for_path_setup)
        except ValueError:
            # This log will go to the console or a previously configured handler if setup_pipeline_logging hasn't run yet.
            # It's acceptable here as it's a config issue noted before full file logging might be active.
            logging.warning(f"Invalid value for MAX_RESULTS or --max: '{max_r_for_path_setup}'. Defaulting to processing all relevant abstracts.")
            # max_r_for_path_setup for path naming will remain as is (e.g., potentially problematic string if not "all" or int-like)
            # max_results_limit remains float('inf')

    dynamic_run_base = get_dynamic_run_base_path(model_name_for_path, max_r_for_path_setup, current_script_dir)
    logs_path = dynamic_run_base / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_path / "main_pipeline.log"
    setup_pipeline_logging(log_file_path) 

    # Now use the logger for all subsequent messages
    logger.info("--- Starting Main Pipeline Logic (with async sub-tasks & Batched Data Loading & Chunked Processing) ---")
    logger.info(f"Run specific logs will be saved to: {log_file_path}")
    logger.info(f"Run base directory: {dynamic_run_base}")
    logger.info(f"Processing configuration: Max relevant abstracts = {max_results_limit if max_results_limit != float('inf') else 'all'}, Processing chunk size = {BATCH_CONFIG['processing_batch_size']}")

    llm_setup = llm_system_setup

    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e_main_pipeline:
        logger.critical(f"CRITICAL ERROR in run_main_pipeline_logic: {e_main_pipeline}", exc_info=True)
        return dynamic_run_base if 'dynamic_run_base' in locals() else current_script_dir

    embedding_model, embedding_classifier = None, None
    if EMBEDDINGS_AVAILABLE:
        logger.info("Setting up embedding model...")
        embedding_model, embedding_classifier = setup_embedding_classifier()
    
    results_path = dynamic_run_base / "results"
    figures_path = dynamic_run_base / "figures"
    cache_path = dynamic_run_base / "cache"
    models_path = dynamic_run_base / "models" 

    for p in [results_path, figures_path, cache_path, models_path]:
        p.mkdir(parents=True, exist_ok=True)

    # --- Load Pre-trained Classifiers ---
    # TF-IDF based classifier (legacy, if still used or preferred for some scenarios)
    vectorizer_path = models_path / "tfidf_vectorizer.pkl"
    legacy_classifier_path = models_path / "relevance_classifier.pkl"
    vectorizer, legacy_classifier = load_classifier_components(vectorizer_path, legacy_classifier_path)
    classifier_ready = bool(vectorizer and legacy_classifier)
    if classifier_ready:
        logger.info("Loaded pre-trained TF-IDF based relevance classifier.")
    else:
        logger.info("Pre-trained TF-IDF based relevance classifier not found or failed to load.")

    # Embedding-based classifier (primary relevance classifier)
    # setup_embedding_classifier now expects the models_path to load from
    if EMBEDDINGS_AVAILABLE:
        logger.info(f"Attempting to load embedding model and classifier from: {models_path}")
        # embedding_model is the SentenceTransformer, embedding_classifier is the LogisticRegression model
        embedding_model, embedding_classifier = setup_embedding_classifier(models_path) 
        if embedding_model and embedding_classifier:
            logger.info("Successfully loaded embedding model and pre-trained embedding-based relevance classifier.")
        elif embedding_model:
            logger.info("Embedding model loaded, but pre-trained embedding-based classifier not found/loaded. Relevance will rely on LLM or zero-shot.")
        else:
            logger.warning("Failed to load embedding model. Embedding-based relevance classification will be unavailable.")
    else:
        embedding_model, embedding_classifier = None, None
        logger.warning("Sentence-transformers not installed. Embedding features (including relevance classifier) are unavailable.")

    llm_setup['cache'] = Cache(cache_dir=str(cache_path))
    refinement_cache_dir = cache_path / "refinement_cache"
    refinement_cache = SimpleCache(refinement_cache_dir)
    
    taxonomic_filter = args.taxonomy if hasattr(args, 'taxonomy') and args.taxonomy else os.getenv('TAXONOMY_FILTER', '')
    VERIFICATION_THRESHOLD = 0.75

    # --- Batched Data Loading and Initial Filtering (NEW BLOCK) --- 
    all_relevant_abstract_data = [] # This will collect all relevant abstract data for vector store at the end
    overall_normalized_triplets = []
    overall_llm_taxonomy_map = {}
    
    temp_relevant_abstract_chunk = [] # Accumulates relevant abstracts for the current processing chunk
    
    file_batch_size = 1000 
    current_file_skip_rows = 0
    processed_relevant_abstract_count = 0 
    total_rows_scanned_from_parquet = 0

    logger.info(f"Starting batched data loading from all_abstracts.parquet (file batch size: {file_batch_size}, max relevant abstracts: {max_results_limit if max_results_limit != float('inf') else 'all'}, processing chunk size: {BATCH_CONFIG['processing_batch_size']}).")

    async def classify_relevance_task(title, abstract, p_llm_setup, p_embedding_model, p_embedding_classifier, p_vectorizer, p_legacy_classifier):
        # Priority: 1. Pre-trained embedding classifier, 2. Pre-trained legacy TF-IDF classifier, 3. LLM, 4. Zero-shot
        if p_embedding_classifier and p_embedding_model and EMBEDDINGS_AVAILABLE:
            logger.debug(f"Classifying '{title[:30]}...' using pre-trained embedding classifier.")
            return predict_relevance_embeddings(abstract, p_embedding_model, p_embedding_classifier)
        elif p_legacy_classifier and p_vectorizer: # Check for TF-IDF classifier next
            logger.debug(f"Classifying '{title[:30]}...' using pre-trained TF-IDF classifier.")
            return predict_relevance_local(abstract, p_vectorizer, p_legacy_classifier)
        # Fallback to LLM-based classification if no pre-trained models are effective
        logger.debug(f"Classifying '{title[:30]}...' using LLM relevance check as no suitable pre-trained classifier was loaded/effective.")
        return await classify_abstract_relevance_ollama(title, abstract, p_llm_setup)
        # Zero-shot is removed as a primary option here, as LLM is more robust generally if no trained model.
        # If classify_abstract_relevance_ollama itself fails, it returns False.

    while True:
        if processed_relevant_abstract_count >= max_results_limit:
            logger.info(f"Reached max relevant abstracts limit ({max_results_limit}). Stopping data loading.")
            break

        logger.info(f"Loading next file batch from all_abstracts.parquet: skip_rows={current_file_skip_rows}, max_rows={file_batch_size}")
        df_file_batch = load_data_with_offset("all_abstracts.parquet", current_file_skip_rows, file_batch_size)
        
        if len(df_file_batch) == 0:
            logger.info("No more data in all_abstracts.parquet.")
            break
        
        actual_rows_in_batch = len(df_file_batch)
        total_rows_scanned_from_parquet += actual_rows_in_batch
        
        batch_abstract_details = []
        for i, row_data in enumerate(df_file_batch.iter_rows(named=True)):
            abstract_text = row_data["abstract"]
            title_text = row_data["title"]
            doi_text = row_data.get("doi")
            if not doi_text: continue
            batch_abstract_details.append({'title': title_text, 'abstract': abstract_text, 'doi': doi_text, 'original_file_index': current_file_skip_rows + i})

        current_file_skip_rows += actual_rows_in_batch

        if taxonomic_filter:
            logger.info(f"Applying taxonomic filter '{taxonomic_filter}' to current batch of {len(batch_abstract_details)} items.")
            filtered_batch_details = []
            for detail in batch_abstract_details:
                if (taxonomic_filter.lower() in detail['title'].lower() or 
                    taxonomic_filter.lower() in detail['abstract'].lower()):
                    filtered_batch_details.append(detail)
            batch_abstract_details = filtered_batch_details
            logger.info(f"Batch size after taxonomic filter: {len(batch_abstract_details)}")

        if not batch_abstract_details:
            logger.info("No abstracts remaining in current batch after filtering. Continuing to next file batch.")
            if max_results_limit == float('inf') and total_rows_scanned_from_parquet >= MAX_PARQUET_ROWS_TO_SCAN_IF_NO_MAX_RESULTS:
                 logger.warning(f"Scanned {total_rows_scanned_from_parquet} rows from parquet without a max_results_limit and no relevant data found recently. Stopping.")
                 break
            continue
        
        relevance_classification_tasks = []
        for detail in batch_abstract_details:
            relevance_classification_tasks.append(
                classify_relevance_task(detail['title'], detail['abstract'], llm_setup, embedding_model, embedding_classifier, vectorizer, legacy_classifier)
            )
        
        if relevance_classification_tasks:
            logger.info(f"Starting relevance classification for {len(relevance_classification_tasks)} abstracts from current file batch...")
            relevance_results = await asyncio.gather(*relevance_classification_tasks)
            logger.info("Relevance classification for file batch finished.")

            for i, is_relevant in enumerate(relevance_results):
                if is_relevant:
                    relevant_detail = batch_abstract_details[i]
                    temp_relevant_abstract_chunk.append(relevant_detail)
                    processed_relevant_abstract_count += 1 

                    if len(temp_relevant_abstract_chunk) >= BATCH_CONFIG['processing_batch_size'] or \
                       processed_relevant_abstract_count >= max_results_limit:
                        
                        logger.info(f"Processing a chunk of {len(temp_relevant_abstract_chunk)} relevant abstracts. Total relevant identified so far: {processed_relevant_abstract_count}")
                        
                        chunk_normalized_triplets, chunk_llm_taxonomy_map = await process_abstract_chunk(
                            temp_relevant_abstract_chunk, 
                            llm_setup, 
                            refinement_cache
                        )
                        logger.info(f"Chunk processing returned {len(chunk_normalized_triplets)} normalized triplets and {len(chunk_llm_taxonomy_map)} taxonomy entries.")
                        
                        overall_normalized_triplets.extend(chunk_normalized_triplets)
                        overall_llm_taxonomy_map.update(chunk_llm_taxonomy_map) 
                        all_relevant_abstract_data.extend(temp_relevant_abstract_chunk) # Collect for vector store
                        logger.info(f"After aggregation: {len(overall_normalized_triplets)} total normalized triplets, {len(overall_llm_taxonomy_map)} total taxonomy entries.")

                        temp_relevant_abstract_chunk = [] # Reset for next chunk
                    
                    if processed_relevant_abstract_count >= max_results_limit:
                        logger.info(f"Inner loop: Reached max relevant abstracts limit ({max_results_limit}).")
                        break 
        
        if processed_relevant_abstract_count >= max_results_limit:
            logger.info(f"Outer loop: Reached max relevant abstracts limit ({max_results_limit}). Stopping data loading.")
            break # Break from outer while True data loading loop
        
        if len(df_file_batch) == 0: # Check if the file ended
            logger.info("No more data in all_abstracts.parquet.")
            # Process any remaining items in temp_relevant_abstract_chunk before exiting
            if temp_relevant_abstract_chunk:
                logger.info(f"Processing final remaining chunk of {len(temp_relevant_abstract_chunk)} relevant abstracts as file ended.")
                chunk_normalized_triplets, chunk_llm_taxonomy_map = await process_abstract_chunk(
                    temp_relevant_abstract_chunk, llm_setup, refinement_cache
                )
                logger.info(f"Final chunk processing returned {len(chunk_normalized_triplets)} normalized triplets and {len(chunk_llm_taxonomy_map)} taxonomy entries.")
                overall_normalized_triplets.extend(chunk_normalized_triplets)
                overall_llm_taxonomy_map.update(chunk_llm_taxonomy_map)
                all_relevant_abstract_data.extend(temp_relevant_abstract_chunk)
                logger.info(f"After final aggregation: {len(overall_normalized_triplets)} total normalized triplets, {len(overall_llm_taxonomy_map)} total taxonomy entries.")
                temp_relevant_abstract_chunk = []
            break # Break from outer while True data loading loop

        if max_results_limit == float('inf') and total_rows_scanned_from_parquet >= MAX_PARQUET_ROWS_TO_SCAN_IF_NO_MAX_RESULTS:
            logger.warning(f"Scanned {total_rows_scanned_from_parquet} rows from parquet without a max_results_limit. Processing remaining chunk and stopping.")
            if temp_relevant_abstract_chunk:
                logger.info(f"Processing final chunk ({len(temp_relevant_abstract_chunk)}) due to scan limit.")
                chunk_normalized_triplets, chunk_llm_taxonomy_map = await process_abstract_chunk(
                    temp_relevant_abstract_chunk, llm_setup, refinement_cache
                )
                logger.info(f"Scan limit chunk processing returned {len(chunk_normalized_triplets)} normalized triplets and {len(chunk_llm_taxonomy_map)} taxonomy entries.")
                overall_normalized_triplets.extend(chunk_normalized_triplets)
                overall_llm_taxonomy_map.update(chunk_llm_taxonomy_map)
                all_relevant_abstract_data.extend(temp_relevant_abstract_chunk)
                logger.info(f"After scan limit aggregation: {len(overall_normalized_triplets)} total normalized triplets, {len(overall_llm_taxonomy_map)} total taxonomy entries.")
                temp_relevant_abstract_chunk = []
            break
            
    logger.info(f"Collected {processed_relevant_abstract_count} relevant abstracts in total. {len(overall_normalized_triplets)} normalized triplets generated across all chunks.")

    if not overall_normalized_triplets: # Changed from all_relevant_abstract_data
        logger.warning("No normalized triplets generated after all file batches and processing chunks.")
        return dynamic_run_base
    # --- END OF MODIFIED BATCH LOADING AND CHUNK PROCESSING BLOCK ---

    # --- Downstream processes now use aggregated results ---
    
    # --- Enhanced Results Caching ---
    logger.info(f"Caching {len(overall_normalized_triplets)} enriched triplets...")
    cache_enriched_triples(overall_normalized_triplets, overall_llm_taxonomy_map, results_path)

    # --- Enhanced Vector Search Setup ---
    if EMBEDDINGS_AVAILABLE and embedding_model and all_relevant_abstract_data: # Use all_relevant_abstract_data
        logger.info("Setting up vector search for semantic abstract retrieval...")
        print("\nSetting up vector search for semantic abstract retrieval...")
        all_abstracts_text_for_store = [item['abstract'] for item in all_relevant_abstract_data] # Use collected abstracts
        vector_store = setup_vector_search(all_abstracts_text_for_store, embedding_model) # Pass correct data
    
    # --- Enhanced Threat Clustering ---
    if EMBEDDINGS_AVAILABLE and embedding_model and overall_normalized_triplets:
        logger.info("Clustering threats using embeddings...")
        print("\nClustering threats using embeddings...")
        cluster_results = cluster_threats(overall_normalized_triplets, embedding_model, figures_path)

    # --- Enhanced Graph Building and Analysis ---
    logger.info("Building graphs and performing enhanced analysis...")
    print("\nBuilding graphs and performing enhanced analysis...")
    basic_graph = build_global_graph(overall_normalized_triplets)
    
    # Enhanced embedding visualization
    if EMBEDDINGS_AVAILABLE and embedding_model:
        logger.info("Enriching graph with embeddings...")
        print("\nEnriching graph with embeddings...")
        potential_connections = enrich_graph_with_embeddings(basic_graph, embedding_model, results_path)
        
        if potential_connections:
            with open(results_path / "potential_connections.txt", 'w') as f:
                f.write("Potential semantic connections not currently in graph:\n\n")
                for node1, node2, similarity in potential_connections:
                    f.write(f"{node1} -- {node2} (similarity: {similarity:.3f})\\n")
            
        create_embedding_visualization(basic_graph, embedding_model, figures_path)

    # Enhanced threat hierarchy building
    logger.info("Building threat hierarchy...")
    threat_graph = build_threat_hierarchy(overall_normalized_triplets, overall_llm_taxonomy_map, results_path)

    # Enhanced visualizations
    logger.info("Creating visualizations...")
    save_threat_hierarchy_viz(threat_graph, figures_path, use_3d=True)
    analyze_graph_detailed(basic_graph, figures_path)
    analyze_hub_node(basic_graph, figures_path)

    # --- Enhanced Species List for Verification ---
    # This list should be derived from the subjects of the *final, overall* triplets before normalization,
    # or if subjects are canonical after normalization, then from `overall_llm_taxonomy_map` keys.
    # The original logic used `final_enriched_triplets_for_verification`.
    # We need to reconstruct a similar list or adapt.
    # For now, let's assume `overall_llm_taxonomy_map` keys (original names from normalization input) are suitable.
    
    original_subject_names_for_verification = sorted(list(overall_llm_taxonomy_map.keys()))

    if original_subject_names_for_verification: # Check this list
        verification_list_filepath = results_path / "species_to_verify_with_wikispecies.txt"
        with open(verification_list_filepath, 'w', encoding='utf-8') as f:
            for species_name in original_subject_names_for_verification: # Iterate this list
                f.write(f"{species_name}\\n")
        logger.info(f"Species list for verification saved to: {verification_list_filepath}")
        print(f"\\nSpecies list for verification saved to: {verification_list_filepath}")
        print(f"Total species to verify: {len(original_subject_names_for_verification)}")
        relative_lookup_path = Path(os.path.dirname(os.path.abspath(__file__))) / "wikispecies_taxonomy_lookup.json"
        print(f"Verification results will be stored in persistent lookup: {relative_lookup_path}")
    else:
        logger.warning("No species found for Wikispecies verification based on overall taxonomy map.")
        print("\\nNo species found for Wikispecies verification.")


    # triplets_by_doi_for_verification = defaultdict(list) # This was unused, can be removed if not planned for reintroduction

    print("\\nEnhanced main pipeline processing complete!")
    print(f"Results saved to: {results_path}")
    print(f"Figures saved to: {figures_path}")
    print("\nNext steps:")
    print("1. Run Wikispecies verification")
    print("2. Run taxonomy comparison after verification completes")
    print(f"\nEnhanced features used:")
    print(f"- Batch processing: ✓ ({file_batch_size} records per batch)") # Changed batch_size to file_batch_size
    # print(f"- Adaptive classifier training: ✓ ({target_per_class} samples per class initially targeted)") # REMOVED
    print(f"- Pre-trained classifier loading: ✓ (Attempted)")
    print(f"- Multi-tier relevance filtering: ✓")
    print(f"- IUCN refinement loop: ✓")
    print(f"- Configurable verification threshold: ✓ ({VERIFICATION_THRESHOLD})")

def openrouter_generate(prompt, model="google/gemini-2.0-flash-001", system="", temperature=0.1, timeout=120, format=None):
    """
    Generate text using OpenRouter API with Google Gemini 2.0 Flash model.
    Supports structured outputs using JSON Schema.
    """
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    try:
        # Initialize OpenAI client for OpenRouter
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        # Prepare messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout
        }
        
        # Add structured output format if provided
        if format:
            request_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "strict": True,
                    "schema": format
                }
            }
        
        # Make the API call
        response = client.chat.completions.create(**request_params)
        
        # Extract response content
        response_content = response.choices[0].message.content
        
        return response_content
        
    except Exception as e:
        logger.exception(f"OpenRouter API error: {e}")
        return ""

def strip_markdown_json(response_text: str) -> str:
    """Strips ```json ... ``` markdown from a string if present."""
    if response_text is None:
        return ""
    stripped_text = response_text.strip()
    if stripped_text.startswith("```json") and stripped_text.endswith("```"):
        stripped_text = stripped_text[7:-3].strip()
    elif stripped_text.startswith("```") and stripped_text.endswith("```"):
        # Generic markdown block
        stripped_text = stripped_text[3:-3].strip()
    return stripped_text

async def llm_generate(prompt: str, system: str, model: str, temperature: float = 0.1, 
                timeout: int = 120, format=None, llm_setup=None) -> str:
    """
    Unified LLM generation function that handles both OpenRouter and Ollama calls asynchronously.
    """
    raw_response_content = "" # Initialize to ensure it's always a string
    try:
        if llm_setup and llm_setup.get('use_openrouter', False):
            load_dotenv()
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            if llm_setup.get('api_rate_limiter'):
                await llm_setup['api_rate_limiter'].async_wait()

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            
            extra_kwargs = {}
            # For OpenRouter, if format is a dict (JSON schema), it implies JSON mode.
            # The OpenAI client for OpenRouter might handle response_format differently or expect it within model params.
            # Forcing JSON mode for OpenRouter often involves specific model suffixes like -json or specific request params.
            # The generic client.chat.completions.create might not use `format` in the same way Ollama does.
            # We will rely on the prompt to ask for JSON and then strip markdown.
            if format and isinstance(format, dict):
                 # Update system prompt to strongly request JSON according to schema
                 enhanced_system = f"{system}\n\nPlease respond ONLY with a valid JSON object matching this schema (do not include any other explanatory text or markdown): {json.dumps(format)}"
                 messages[0]["content"] = enhanced_system
            elif format == "json": # General request for json
                 enhanced_system = f"{system}\n\nPlease respond ONLY with a valid JSON object (do not include any other explanatory text or markdown)."
                 messages[0]["content"] = enhanced_system

            loop = asyncio.get_running_loop()
            response_obj = await loop.run_in_executor(
                None, 
                lambda: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=4090, # Slightly less than common max to be safe
                    timeout=timeout, 
                    **extra_kwargs
                )
            )
            raw_response_content = response_obj.choices[0].message.content
            
        else: # Ollama path
            ollama_url = "http://localhost:11434/api/generate"
            payload = {
                "model": model,
                "prompt": prompt, # Ollama often takes the full prompt here, system can be part of it or separate field in some versions
                "system": system, # Added system parameter for Ollama
                "stream": False,
                "options": {
                "temperature": temperature,
                }
            }
            if format: # Ollama specific format handling for JSON
                payload["format"] = "json" # if format == "json" or isinstance(format, dict) else format

            if llm_setup and llm_setup.get('api_rate_limiter'):
                 await llm_setup['api_rate_limiter'].async_wait()

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(ollama_url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                        response.raise_for_status()
                        result = await response.json()
                        raw_response_content = result.get("response", "")
                except aiohttp.ClientResponseError as http_err:
                    logger.error(f"Ollama API HTTP error: {http_err.status} {http_err.message} for model {model}... Raw text: {await http_err.text() if hasattr(http_err, 'text') else 'N/A'}")
                    if http_err.status == 429 and llm_setup and llm_setup.get('api_rate_limiter'):
                        llm_setup['api_rate_limiter'].handle_async_rate_limit()
                    raw_response_content = "" 
                except asyncio.TimeoutError:
                    logger.error(f"Ollama API request timed out after {timeout}s for model {model}...")
                    raw_response_content = ""
                except Exception as e:
                    logger.error(f"Ollama API general error: {e} for model {model}...", exc_info=True)
                    raw_response_content = ""

    except Exception as e:
        logger.error(f"Outer error in async_llm_generate for model {model}...: {e}", exc_info=True)
        raw_response_content = "" # Ensure it's a string
    
    return strip_markdown_json(raw_response_content)

def run_wikispecies_verification_logic(args):
    # --- Determine Run-Specific Paths Early & Setup Logging ---
    current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    target_model_name = args.target_model_name if hasattr(args, 'target_model_name') and args.target_model_name else os.getenv('MODEL_NAME_FOR_RUN', "google/gemini-flash-1.5")
    target_max_results_str = args.target_max_results if hasattr(args, 'target_max_results') and args.target_max_results else os.getenv('MAX_RESULTS', "all")
    
    target_max_r_for_path = "all"
    if str(target_max_results_str).lower() == 'all':
        target_max_r_for_path = "all"
    elif str(target_max_results_str).isdigit():
        target_max_r_for_path = int(target_max_results_str)

    target_run_base = get_dynamic_run_base_path(target_model_name, target_max_r_for_path, current_script_dir)
    logs_path = target_run_base / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_path / "wikispecies_verification.log"
    setup_pipeline_logging(log_file_path) 

    logger.info("--- Starting Wikispecies Verification ---")
    logger.info(f"Run specific logs will be saved to: {log_file_path}")
    
    try:
        species_filepath = Path(args.verify_species_wikispecies)
        if not species_filepath.is_file():
            logger.error(f"Error: Species file for verification not found at {species_filepath}")
            sys.exit(1)
        
        with open(species_filepath, 'r', encoding='utf-8') as f:
            species_to_verify = [line.strip() for line in f if line.strip()]
        
        if not species_to_verify:
            logger.error("No species found in the verification file. Exiting.")
            sys.exit(1)
        
        print(f"Starting asynchronous Wikispecies verification for {len(species_to_verify)} species from {species_filepath}...")
        print(f"Will save results to: {target_run_base / 'results'}")
        
        try:
            asyncio.run(verify_species_with_wikispecies_concurrently(species_to_verify, target_run_base / 'results'))
        except Exception as e:
            logger.error(f"ERROR during Wikispecies verification run: {e}", exc_info=True)
            sys.exit(1)
        logger.info("--- Wikispecies Verification Finished ---")
    except Exception as e:
        logger.error(f"ERROR during Wikispecies verification run: {e}", exc_info=True)
        sys.exit(1)

def run_taxonomy_comparison_logic(args):
    # --- Determine Run-Specific Paths Early & Setup Logging ---
    current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    target_model_name = args.target_model_name if hasattr(args, 'target_model_name') and args.target_model_name else os.getenv('MODEL_NAME_FOR_RUN', "google/gemini-flash-1.5")
    target_max_results_str = args.target_max_results if hasattr(args, 'target_max_results') and args.target_max_results else os.getenv('MAX_RESULTS', "all")

    target_max_r_for_path = "all"
    if str(target_max_results_str).lower() == 'all':
        target_max_r_for_path = "all"
    elif str(target_max_results_str).isdigit():
        target_max_r_for_path = int(target_max_results_str)
            
    target_run_base = get_dynamic_run_base_path(target_model_name, target_max_r_for_path, current_script_dir)
    logs_path = target_run_base / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_path / "taxonomy_comparison.log"
    setup_pipeline_logging(log_file_path) 

    logger.info("--- Starting Taxonomy Comparison ---")
    logger.info(f"Run specific logs will be saved to: {log_file_path}")

    try:
        enriched_triplets_file = target_run_base / "results" / "enriched_triplets.json" 
        persistent_wikispecies_lookup = current_script_dir / "wikispecies_taxonomy_lookup.json"
        discrepancy_output_log_path = target_run_base / "results" / "taxonomy_discrepancy_details.log.json" 

        if not enriched_triplets_file.exists():
            logger.error(f"Error: Enriched triplets file not found at {enriched_triplets_file}. Run main pipeline first.")
            return
        if not persistent_wikispecies_lookup.exists():
            logger.error(f"Error: Persistent Wikispecies lookup not found at {persistent_wikispecies_lookup}. Run --verify-species-wikispecies first.")
            return

        print(f"Using enriched triplets: {enriched_triplets_file}")
        print(f"Using persistent Wikispecies lookup: {persistent_wikispecies_lookup}")
        compare_and_log_taxonomy_discrepancies(
            enriched_triplets_file,
            persistent_wikispecies_lookup,
            discrepancy_output_log_path
        )
        print("--- Taxonomy Comparison Finished ---")
    except Exception as e_tax_compare:
        logger.critical(f"CRITICAL ERROR in run_taxonomy_comparison_logic: {e_tax_compare}", exc_info=True)

async def execute_all_core_async_tests(llm_setup_for_tests): 
    # --- Determine Run-Specific Paths Early & Setup Logging ---
    current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    model_name_for_path = llm_setup_for_tests.get("model", "test_model").replace("/", "_").replace(":", "-")
    max_r_for_path = "all_tests" 
    
    dynamic_run_base = get_dynamic_run_base_path(model_name_for_path, max_r_for_path, current_script_dir)
    logs_path = dynamic_run_base / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_path / "core_async_tests.log"
    setup_pipeline_logging(log_file_path) 

    logger.info("--- Running All Core Async Function Tests (Pillar 1 & Pillar 2 components) ---")
    logger.info(f"Test run logs will be saved to: {log_file_path}")

    try:
        logger.info("\\n--- Test 1: async_llm_generate ---")
        test_capital_prompt = "What is the capital of France? Reply in one word."
        test_capital_system = "You are a helpful assistant."
        test_openrouter_model_id = llm_setup_for_tests.get("model", "test_model")

        if llm_setup_for_tests.get('use_openrouter', True):
            logger.info(f"Testing llm_generate with OpenRouter model: {test_openrouter_model_id}...")
            or_response = await llm_generate(test_capital_prompt, test_capital_system, test_openrouter_model_id, llm_setup=llm_setup_for_tests)
            logger.info(f"OpenRouter Response ({test_openrouter_model_id}): {or_response}")
        
        if not llm_setup_for_tests.get('use_openrouter', True): 
            logger.info("Testing llm_generate with Ollama (e.g., llama3:8b)...")
            ollama_test_model = llm_setup_for_tests.get('ollama_model_for_test', 'llama3:8b') 
            ollama_response = await llm_generate(test_capital_prompt, test_capital_system, ollama_test_model, llm_setup=llm_setup_for_tests)
            logger.info(f"Ollama Response ({ollama_test_model}): {ollama_response}")

        # Test 2: async_convert_to_summary
        logger.info("\n--- Test 2: async_convert_to_summary ---")
        sample_abstract_summ = "The quick brown fox jumps over the lazy dog. This study observed foxes and dogs in their natural habitat. Foxes often prey on smaller animals, while dogs are domesticated."
        summary_for_triplet_test = "" 
        try:
            summary_for_triplet_test = await convert_to_summary(sample_abstract_summ, llm_setup_for_tests)
            logger.info(f"Test Summary: {summary_for_triplet_test}")
        except Exception as e:
            logger.error(f"Error in convert_to_summary test: {e}", exc_info=True)

        # Test 3: async_extract_triplets (Original multi-stage version for baseline)
        logger.info("\n--- Test 3: async_extract_triplets (Original Multi-Stage) ---")
        sample_summary_for_triplets = summary_for_triplet_test if summary_for_triplet_test else "Foxes (Vulpes vulpes) prey on rabbits. Habitat loss is a threat to rabbits."
        sample_doi_extractT = "test-doi-extractT-789"
        try:
            original_triplets = await extract_triplets(sample_summary_for_triplets, llm_setup_for_tests, sample_doi_extractT)
            logger.info(f"Test Triplets (Original Method): {original_triplets}")
        except Exception as e:
            logger.error(f"Error in original extract_triplets test: {e}", exc_info=True)

        # Test 4: async_get_iucn_classification_json
        logger.info("\n--- Test 4: async_get_iucn_classification_json ---")
        test_cache_dir_iucn = Path(os.path.dirname(os.path.abspath(__file__))) / "cache" / "test_iucn_cache"
        test_iucn_cache = SimpleCache(test_cache_dir_iucn)
        try:
            iucn_result = await get_iucn_classification_json("Anas platyrhynchos", "impacted by", "oil spills in marine environments", llm_setup_for_tests, test_iucn_cache)
            logger.info(f"Test IUCN Classification: {iucn_result}")
        except Exception as e:
            logger.error(f"Error in get_iucn_classification_json test: {e}", exc_info=True)

        # Test 5: async_normalize_species_names
        logger.info("\n--- Test 5: async_normalize_species_names ---")
        sample_triplets_for_norm = [
            ("ducks", "affected by", "pollution", "doi-norm1"),
            ("Anas platyrhynchos", "eats", "insects", "doi-norm2"),
            ("Grizzly Bear", "lives in", "forest", "doi-norm3") 
        ]
        try:
            norm_triplets, tax_map = await normalize_species_names(sample_triplets_for_norm, llm_setup_for_tests)
            logger.info(f"Test Normalized Bird Triplets: {norm_triplets}")
            logger.info(f"Test Taxonomy Map: {json.dumps(tax_map, indent=2)}")
        except Exception as e:
            logger.error(f"Error in normalize_species_names test: {e}", exc_info=True)
        
        # --- Pillar 2 Component Tests ---
        logger.info("\n--- Pillar 2 Component Function Tests --- ")
        await test_extract_entities(llm_setup_for_tests) 
        await test_generate_relationships(llm_setup_for_tests)
        
        logger.info("--- All Core Async Function Tests (Pillar 1 & Pillar 2 components) Finished ---")
    except Exception as e_core_async_tests:
        logger.critical(f"CRITICAL ERROR in execute_all_core_async_tests: {e_core_async_tests}", exc_info=True)

async def extract_entities_concurrently(abstract_text: str, llm_setup) -> Optional[Dict[str, List[str]]]:
    """Extracts species and threat descriptions from an abstract in a single async LLM call, using combined original prompt logic."""
    logger.info(f"P2.1 (Revised): Extracting entities (species & threats) for abstract starting: {abstract_text[:50]}...")
    entity_extraction_schema = {
        "type": "object",
        "properties": {
            "species": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific species or most specific taxonomic groups mentioned, adhering to species extraction rules."
            },
            "threats": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of distinct phrases describing threats or negative impacts found in the abstract, adhering to general threat extraction rules."
            }
        },
        "required": ["species", "threats"]
    }
    original_species_system_prompt = """
        Extract all specific species or taxonomic groups mentioned in the text.

    Rules:
        1. Only include species or taxonomic groups that are DIRECTLY mentioned in the text
        2. Keep scientific names exactly as written
        3. Each entry must be a single species or specific taxonomic group
        4. Never combine multiple species into one entry (e.g., not "# bird species")
        5. Remove any qualifiers like "spp." or species counts
        6. If a scientific name is provided in the text, include it
        7. Assign a confidence level (high, medium, low) based on how clearly the species is mentioned
        """
    general_threat_extraction_rules = """
        Based on the abstract, identify all distinct phrases describing specific NEGATIVE threats, stressors, or CAUSES OF HARM.
        **Rules for Threat Extraction:**
        1. Focus ONLY on factors that HARM or NEGATIVELY impact species generally described in the abstract.
        2. Extract the *specific description of the threat or stressor* (e.g., "drowning in oil pits", "habitat loss from logging", "increasing shoreline development", "competition from invasive species").
        3. **DO NOT extract protective factors or beneficial conditions.**
        4. Only include threats DIRECTLY mentioned in the text.
        5. Do NOT attempt to classify the threat using IUCN categories here.
        6. Do not try to link these threats to specific species *yet*. That will be a subsequent step.
        """
    combined_system_prompt = f"""You are a scientific entity extraction expert. Perform the following two tasks based on the provided abstract:

TASK 1: SPECIES EXTRACTION
---
{original_species_system_prompt}
---
List the species found under the "species" key in your JSON output (as a list of strings).

TASK 2: THREAT EXTRACTION (General from Abstract)
---
{general_threat_extraction_rules}
---
List these general threat descriptions under the "threats" key in your JSON output (as a list of strings).

Provide your complete output *only* as a single valid JSON object matching this schema:
{json.dumps(entity_extraction_schema)}
Do not include any other explanatory text or markdown around the JSON object.
"""
    user_prompt = abstract_text
    try:
        response_str = await llm_generate(
            prompt=user_prompt,
            system=combined_system_prompt,
            model=llm_setup.get("model"), 
            temperature=0.0, 
            format=entity_extraction_schema, 
            llm_setup=llm_setup
        )
        if not response_str:
            logger.error(f"P2.1: LLM returned empty response for entity extraction. Abstract: {abstract_text[:50]}...")
            return None
        entities_data = json.loads(response_str)
        if isinstance(entities_data, dict) and \
           isinstance(entities_data.get("species"), list) and \
           isinstance(entities_data.get("threats"), list):
            if all(isinstance(s, str) for s in entities_data.get("species")) and \
               all(isinstance(t, str) for t in entities_data.get("threats")):
                logger.info(f"P2.1: Successfully extracted {len(entities_data['species'])} species and {len(entities_data['threats'])} threats.")
                return entities_data
            else:
                logger.error(f"P2.1: Extracted species/threats lists contain non-string elements. Raw: '{response_str}'.")
                return None
        elif isinstance(entities_data, dict) and "value" in entities_data and isinstance(entities_data["value"], dict):
            actual_data = entities_data["value"]
            if isinstance(actual_data.get("species"), list) and \
               isinstance(actual_data.get("threats"), list) and \
               all(isinstance(s, str) for s in actual_data.get("species")) and \
               all(isinstance(t, str) for t in actual_data.get("threats")):
                logger.info(f"P2.1: Successfully extracted {len(actual_data['species'])} species and {len(actual_data['threats'])} threats (from 'value' key).")
                return actual_data
            else:
                logger.error(f"P2.1: Unexpected structure or non-string elements in 'value' key. Raw: '{response_str}'.")
                return None
        logger.error(f"P2.1: Unexpected JSON structure from entity extraction. Raw: '{response_str}'. Abstract: {abstract_text[:50]}...")
        return None
    except json.JSONDecodeError as e_json:
        logger.error(f"P2.1: JSONDecodeError in entity extraction: {e_json}. Raw response: '{response_str}'. Abstract: {abstract_text[:50]}...")
        return None
    except Exception as e:
        logger.error(f"P2.1: Error in extract_entities_concurrently: {e}. Abstract: {abstract_text[:50]}...", exc_info=True)
        return None

async def generate_relationships_concurrently(abstract_text: str, species_list: List[str], threats_list: List[str], llm_setup, doi: str) -> List[Tuple[str, str, str, str]]:
    """Generates (Subject, Predicate, Object-Threat) triplets based on pre-extracted entities and an abstract."""
    logger.info(f"P2.2: Generating relationships for DOI: {doi}, {len(species_list)} species, {len(threats_list)} threats. Abstract: {abstract_text[:50]}...")
    if not species_list or not threats_list:
        logger.warning(f"P2.2: Missing species or threats list for DOI {doi}. Skipping relationship generation.")
        return []
    relationship_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Species name from the provided species list."},
                "predicate": {"type": "string", "description": "The relationship/impact mechanism linking subject and object, detailing how and why it's important."},
                "object": {"type": "string", "description": "Threat description from the provided threats list."}
            },
            "required": ["subject", "predicate", "object"]
        }
    }
    system_prompt = (
        "You are a precise relational analyst. Below is a scientific abstract, followed by pre-identified lists of species "
        "and threats relevant to this abstract.\n\n"
        "Your task is to generate (Subject, Predicate, Object) triplets that describe the causal relationships "
        "*between the provided species and the provided threats* as detailed *only* "
        "in the abstract text.\n\n"
        "GUIDELINES FOR TRIPLETS:\n"
        "1. The Subject of a triplet MUST be a species name exactly as it appears in the 'Identified Species' list.\n"
        "2. The Object of a triplet MUST be a threat description exactly as it appears in the 'Identified Threats' list.\n"
        "3. The Predicate is CRITICAL. It must clearly and concisely describe:\n"
        "    a. HOW the threat specifically impacts the subject (the mechanism of interaction or effect).\n"
        "    b. WHY this impact is significant for the species, if mentioned (e.g., 'leading to population decline', 'reducing foraging success', 'causing habitat unsuitability').\n"
        "    c. Use strong, descriptive verbs. Avoid vague predicates like 'is affected by' or 'is related to' unless no more detail is available in the abstract.\n"
        "    d. Example Predicates: 'experiences reduced breeding success due to', 'suffers mortality from direct strikes with', 'shows altered migration patterns because of increased', 'faces habitat degradation from'.\n"
        "4. Focus on extracting direct causal links and significant negative impacts mentioned in the abstract.\n"
        "5. If the abstract does not provide enough detail for a rich predicate connecting a listed species to a listed threat, do not create a triplet for that pair.\n"
        "6. Return an empty list if no valid, detailed relationships can be confidently extracted.\n\n"
        "Provide your output *only* as a single valid JSON array of objects matching the schema."
    )
    user_prompt = f"""Abstract:
{abstract_text}

Identified Species:
{json.dumps(species_list)}

Identified Threats:
{json.dumps(threats_list)}

Extract relationship triplets based on the abstract, linking species to threats (ensure output is ONLY the JSON array):
"""
    raw_triplets = []
    try:
        response_str = await llm_generate(
            prompt=user_prompt,
            system=system_prompt,
            model=llm_setup.get("model"), 
            temperature=0.0,
            format=relationship_schema,
            llm_setup=llm_setup
        )
        if not response_str:
            logger.error(f"P2.2: LLM returned empty response for relationship generation. DOI: {doi}")
            return []
        relationships_data = json.loads(response_str)
        if isinstance(relationships_data, list):
            for rel in relationships_data:
                if isinstance(rel, dict):
                    subject = rel.get("subject")
                    predicate = rel.get("predicate")
                    obj_threat = rel.get("object")
                    if subject and predicate and obj_threat and subject in species_list and obj_threat in threats_list:
                        raw_triplets.append((subject, predicate, obj_threat, doi))
                    else:
                        logger.warning(f"P2.2: Dropping invalid/incomplete triplet or subject/object not in provided lists: {rel}. DOI: {doi}")
                else:
                    logger.warning(f"P2.2: Expected dict in relationships list, got {type(rel)}. Data: {rel}. DOI: {doi}")
            logger.info(f"P2.2: Successfully parsed {len(raw_triplets)} relationships for DOI: {doi}.")
        elif isinstance(relationships_data, dict) and "value" in relationships_data and isinstance(relationships_data["value"], list):
            logger.info("P2.2: Relationships JSON is a dict with a 'value' key containing the data list.")
            actual_data_list = relationships_data["value"]
            for rel in actual_data_list:
                if isinstance(rel, dict):
                    subject = rel.get("subject")
                    predicate = rel.get("predicate")
                    obj_threat = rel.get("object")
                    if subject and predicate and obj_threat and subject in species_list and obj_threat in threats_list:
                        raw_triplets.append((subject, predicate, obj_threat, doi))
                    else:
                        logger.warning(f"P2.2: Dropping invalid triplet from 'value' list: {rel}. DOI: {doi}")
                else:
                    logger.warning(f"P2.2: Expected dict in 'value' relationships list, got {type(rel)}. Data: {rel}. DOI: {doi}")
            logger.info(f"P2.2: Successfully parsed {len(raw_triplets)} relationships from 'value' key for DOI: {doi}.")
        else:
            logger.error(f"P2.2: Unexpected JSON structure for relationships. Expected list or dict with 'value'. Got {type(relationships_data)}. Raw: '{response_str}'. DOI: {doi}")
    except json.JSONDecodeError as e_json:
        logger.error(f"P2.2: JSONDecodeError in relationship generation: {e_json}. Raw: '{response_str}'. DOI: {doi}")
    except Exception as e:
        logger.error(f"P2.2: Error in generate_relationships_concurrently for DOI {doi}: {e}", exc_info=True)
    return raw_triplets

async def test_extract_entities(llm_setup):
    logger.info("\n--- Pillar 2 - Test 2.1.2: async_extract_entities_concurrently ---")
    sample_abstract_for_entities = "The majestic Blue Whale (Balaenoptera musculus) faces threats from vessel strikes and ocean noise pollution. Entanglement in fishing gear also poses a significant risk. Humpback whales (Megaptera novaeangliae) are also affected by noise."
    try:
        entities = await extract_entities_concurrently(sample_abstract_for_entities, llm_setup)
        if entities:
            logger.info(f"Extracted Entities: Species: {entities.get('species')}, Threats: {entities.get('threats')}")
        else:
            logger.warning("Entity extraction returned None or empty.")
    except Exception as e:
        logger.error(f"Error in extract_entities_concurrently test: {e}", exc_info=True)

async def test_generate_relationships(llm_setup):
    logger.info("\n--- Pillar 2 - Test 2.2.1: async_generate_relationships_concurrently ---")
    sample_abstract_for_relations = "The majestic Blue Whale (Balaenoptera musculus) faces threats from vessel strikes and ocean noise pollution. Entanglement in fishing gear also poses a significant risk. Humpback whales (Megaptera novaeangliae) are also affected by noise."
    sample_species = ["Blue Whale (Balaenoptera musculus)", "Humpback whales (Megaptera novaeangliae)"]
    sample_threats = ["vessel strikes", "ocean noise pollution", "Entanglement in fishing gear", "noise"]
    sample_doi_relations = "test-doi-relations-456"
    try:
        triplets = await generate_relationships_concurrently(sample_abstract_for_relations, sample_species, sample_threats, llm_setup, sample_doi_relations)
        logger.info(f"Generated Relationship Triplets: {triplets}")
    except Exception as e:
        logger.error(f"Error in generate_relationships_concurrently test: {e}", exc_info=True)

async def execute_all_core_async_tests(llm_setup):
    logger.info("--- Running All Core Async Function Tests (Pillar 1 & Pillar 2 components) ---")

    # Test 1: async_llm_generate
    logger.info("\n--- Test 1: async_llm_generate ---")
    try:
        test_capital_prompt = "What is the capital of France? Reply in one word."
        test_capital_system = "You are a helpful assistant."
        test_openrouter_model_id = llm_setup.get("model", "google/gemini-flash-1.5")

        if llm_setup.get('use_openrouter', True):
            logger.info(f"Testing llm_generate with OpenRouter model: {test_openrouter_model_id}...")
            or_response = await llm_generate(test_capital_prompt, test_capital_system, test_openrouter_model_id, llm_setup=llm_setup)
            logger.info(f"OpenRouter Response ({test_openrouter_model_id}): {or_response}")
        
        if not llm_setup.get('use_openrouter', True): 
            logger.info("Testing llm_generate with Ollama (e.g., llama3:8b)...")
            ollama_test_model = llm_setup.get('ollama_model_for_test', 'llama3:8b') 
            ollama_response = await llm_generate(test_capital_prompt, test_capital_system, ollama_test_model, llm_setup=llm_setup)
            logger.info(f"Ollama Response ({ollama_test_model}): {ollama_response}")
    except Exception as e:
        logger.error(f"Error in llm_generate test: {e}", exc_info=True)

    # Test 2: async_convert_to_summary
    logger.info("\n--- Test 2: async_convert_to_summary ---")
    sample_abstract_summ = "The quick brown fox jumps over the lazy dog. This study observed foxes and dogs in their natural habitat. Foxes often prey on smaller animals, while dogs are domesticated."
    summary_for_triplet_test = "" 
    try:
        summary_for_triplet_test = await convert_to_summary(sample_abstract_summ, llm_setup)
        logger.info(f"Test Summary: {summary_for_triplet_test}")
    except Exception as e:
        logger.error(f"Error in convert_to_summary test: {e}", exc_info=True)

    # Test 3: async_extract_triplets (Original multi-stage version for baseline)
    logger.info("\n--- Test 3: async_extract_triplets (Original Multi-Stage) ---")
    sample_summary_for_triplets = summary_for_triplet_test if summary_for_triplet_test else "Foxes (Vulpes vulpes) prey on rabbits. Habitat loss is a threat to rabbits."
    sample_doi_extractT = "test-doi-extractT-789"
    try:
        original_triplets = await extract_triplets(sample_summary_for_triplets, llm_setup, sample_doi_extractT)
        logger.info(f"Test Triplets (Original Method): {original_triplets}")
    except Exception as e:
        logger.error(f"Error in original extract_triplets test: {e}", exc_info=True)

    # Test 4: async_get_iucn_classification_json
    logger.info("\n--- Test 4: async_get_iucn_classification_json ---")
    test_cache_dir_iucn = Path(os.path.dirname(os.path.abspath(__file__))) / "cache" / "test_iucn_cache"
    test_iucn_cache = SimpleCache(test_cache_dir_iucn)
    try:
        iucn_result = await get_iucn_classification_json("Anas platyrhynchos", "impacted by", "oil spills in marine environments", llm_setup, test_iucn_cache)
        logger.info(f"Test IUCN Classification: {iucn_result}")
    except Exception as e:
        logger.error(f"Error in get_iucn_classification_json test: {e}", exc_info=True)

    # Test 5: async_normalize_species_names
    logger.info("\n--- Test 5: async_normalize_species_names ---")
    sample_triplets_for_norm = [
        ("ducks", "affected by", "pollution", "doi-norm1"),
        ("Anas platyrhynchos", "eats", "insects", "doi-norm2"),
        ("Grizzly Bear", "lives in", "forest", "doi-norm3") 
    ]
    try:
        norm_triplets, tax_map = await normalize_species_names(sample_triplets_for_norm, llm_setup)
        logger.info(f"Test Normalized Bird Triplets: {norm_triplets}")
        logger.info(f"Test Taxonomy Map: {json.dumps(tax_map, indent=2)}")
    except Exception as e:
        logger.error(f"Error in normalize_species_names test: {e}", exc_info=True)
    
    # --- Pillar 2 Component Tests ---
    logger.info("\n--- Pillar 2 Component Function Tests --- ")
    await test_extract_entities(llm_setup) 
    await test_generate_relationships(llm_setup)
    
    logger.info("--- All Core Async Function Tests (Pillar 1 & Pillar 2 components) Finished ---")

async def run_batch_pipeline_logic(args):
    """Placeholder for actual batch-wise concurrent pipeline. 
       For now, it calls the sequential-async main pipeline logic.
    """
    logger.info("RUNNING PLACEHOLDER run_batch_pipeline_logic: This will execute run_main_pipeline_logic.")
    # In a future step, this function will be filled with true batch processing logic 
    # (e.g., loading N abstracts, running extract_entities_concurrently on all, then 
    #  generate_relationships_concurrently on all results, etc.)
    return await run_main_pipeline_logic(args) # Call the already refactored main logic

def run_batch_enabled_pipeline(args):
    """Entry point for batch-enabled pipeline. Calls the async batch orchestrator."""
    logger.info("run_batch_enabled_pipeline called. Running async batch pipeline logic...")
    # BATCH_CONFIG is used by BatchSemaphoreManager, and effective_batch_processing_enabled
    # is determined in __main__ before this is called.
    return asyncio.run(run_batch_pipeline_logic(args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process abstracts to extract and analyze species-threat information, or run specific pipeline stages.")
    # Arguments for main pipeline run
    parser.add_argument(
        '--run-main-pipeline', 
        action='store_true', 
        help='Run the main data processing pipeline (extracts triplets, generates species list for verification). This is the default if no other stage is specified.'
    )
    # Batch processing arguments
    parser.add_argument(
        '--enable-batch-processing',
        action='store_true',
        help='Enable batch processing for significant performance improvements. Overrides BATCH_CONFIG if set.'
    )
    parser.add_argument(
        '--disable-batch-processing',
        action='store_true',
        help='Disable batch processing and use the original sequential pipeline. Overrides BATCH_CONFIG if set.'
    )
    parser.add_argument(
        '--batch-config',
        type=str,
        help='JSON string to override BATCH_CONFIG processing configurations (e.g., \'{"summary_batch_size": 3, "max_summary_workers": 10}\')'
    )
    parser.add_argument(
        '--taxonomy',
        type=str,
        help='Filter abstracts by a taxonomic term (e.g., "duck") for the main pipeline run.'
    )
    parser.add_argument(
        '--max',
        type=str, # Keep as string to handle "all" or int for path generation consistency
        help='Maximum number of relevant abstracts to process for the main pipeline run (e.g., 100 or "all").'
    )
    # Argument for Wikispecies verification stage
    parser.add_argument(
        '--verify-species-wikispecies',
        type=str,
        metavar='FILEPATH',
        help='Run only the asynchronous Wikispecies verification for species listed in the provided filepath. Each species should be on a new line.'
    )
    # Argument for Taxonomy comparison stage
    parser.add_argument(
        '--compare-taxonomies', 
        action='store_true', 
        help='Run only the taxonomy comparison step. Requires prior completion of main pipeline and Wikispecies verification.'
    )
    parser.add_argument(
        '--target_model_name',
        type=str,
        help='Model name used for the target run (e.g., "google/gemini-2.0-flash-001"), for --verify-species-wikispecies and --compare-taxonomies.'
    )
    parser.add_argument(
        '--target_max_results',
        type=str, # Use str to handle "all" or integer
        help='Max results value (or "all") of the target run, for --verify-species-wikispecies and --compare-taxonomies.'
    )

    # Argument for semantic query (remains separate as it's a distinct utility)
    parser.add_argument(
        '--query',
        type=str,
        help='Run a semantic search for related research based on this query text (embedding features must be available).'
    )

    parser.add_argument(
        '--test-pillar1',
        action='store_true',
        help='Run pillar 1 core async function tests.'
    )

    args = parser.parse_args()

    # Basic console logger setup - this will be overridden by stage-specific file loggers
    # if a stage function that calls setup_pipeline_logging runs.
    # This ensures at least console logging if no specific stage configures a file log.
    if not logger.handlers or all(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        # Remove any handlers that might have been added by a previous basicConfig or initial setup
        for handler in logger.handlers[:]: 
            logger.removeHandler(handler)
        
        logger.setLevel(logging.DEBUG) # Set root logger level
        
        ch_main_fallback = logging.StreamHandler() # Console handler
        ch_main_fallback.setLevel(logging.INFO) # Console shows INFO and above
        formatter_main_fallback = logging.Formatter('%(asctime)s - %(levelname)s - MAIN_ENTRY - %(message)s')
        ch_main_fallback.setFormatter(formatter_main_fallback)
        logger.addHandler(ch_main_fallback)
        logger.info("Main script entry point. Stage-specific file logging will be configured if a stage is run.")

    # Determine effective batch processing state
    if args.enable_batch_processing:
        effective_batch_processing_enabled = True
    elif args.disable_batch_processing:
        effective_batch_processing_enabled = False
    else:
        effective_batch_processing_enabled = BATCH_CONFIG.get('enable_batch_processing', True)

    # Override BATCH_CONFIG with command-line JSON if provided
    if args.batch_config:
        try:
            override_config = json.loads(args.batch_config)
            BATCH_CONFIG.update(override_config)
            logger.info(f"BATCH_CONFIG updated with command-line JSON: {BATCH_CONFIG}")
            # If batch_config is provided, it might imply enabling batch processing unless explicitly disabled
            if not args.disable_batch_processing:
                 effective_batch_processing_enabled = BATCH_CONFIG.get('enable_batch_processing', True)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid --batch-config JSON: {e}. Using existing BATCH_CONFIG.")
    
    # Update the global BATCH_CONFIG based on effective state, primarily for other modules if they import it.
    # The effective_batch_processing_enabled variable will gate the pipeline choice.
    BATCH_CONFIG['enable_batch_processing'] = effective_batch_processing_enabled

    # Determine which stage to run
    if args.test_pillar1:
        logger.info("Pillar 1 Test Mode Activated.")
        llm_setup_for_tests = setup_llm()
        asyncio.run(execute_all_core_async_tests(llm_setup_for_tests))
    elif args.verify_species_wikispecies:
        run_wikispecies_verification_logic(args)
    elif args.compare_taxonomies:
        run_taxonomy_comparison_logic(args)
    elif args.query:
        if EMBEDDINGS_AVAILABLE:
            analyze_related_research(args.query) 
        else:
            logger.warning("Cannot run query: sentence-transformers not installed")
    else: 
        logger.info(f"Defaulting to pipeline execution from __main__. Effective batch processing: {effective_batch_processing_enabled}")
        if effective_batch_processing_enabled:
             logger.info("Batch processing is enabled. Calling run_batch_enabled_pipeline.")
             # run_batch_enabled_pipeline should internally handle asyncio.run for run_batch_pipeline_logic
             run_batch_enabled_pipeline(args) 
        else:
             logger.info("Batch processing is disabled. Calling run_main_pipeline_logic (will be run with asyncio.run).")
             asyncio.run(run_main_pipeline_logic(args))

    # Old main() call removed, logic is now within the if/else block above.
    # main() 