import argparse
import re
import time
import math
import sys
import os # For path expansion

import numpy as np
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors, Matrix, Matrices # Note: Matrix for local, Matrices for utility
from pyspark.mllib.linalg.distributed import RowMatrix

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords

# --- Paths (These will be used if not overridden by command-line args) ---
WIKIPEDIA_DATA_PATH_PATTERN = os.path.expanduser("~/big_data_assignment1/Wikipedia-En-41784-Articles/Wikipedia-En-41784-Articles/*/*")
STOPWORDS_FILE_PATH = os.path.expanduser("~/big_data_assignment1/assignment3/stopwords.txt")


# --- Helper: Setup Spark Session and Logging ---
def get_spark_session(app_name="LSAPipeline"):
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

# --- Wikipedia Parsing Logic ---
def parse_header_py(line):
    try:
        s = line[line.find("id=\"") + 4:]
        _ = s[:s.find("\"")] # doc_id
        s = s[s.find("url=\"") + 5:]
        _ = s[:s.find("\"")] # url
        s = s[s.find("title=\"") + 7:]
        title = s[:s.find("\"")]
        return title
    except Exception:
        return ""

def parse_xml_doc_py(file_content_tuple):
    _, text_content = file_content_tuple
    lines = text_content.split("\n")
    docs = []
    current_title = ""
    current_content = ""
    in_doc = False
    for line in lines:
        try:
            if line.startswith("<doc "):
                current_title = parse_header_py(line)
                current_content = ""
                in_doc = True
            elif line.startswith("</doc>"):
                if current_title and current_content and in_doc:
                    docs.append((current_title, current_content.strip()))
                current_title = ""
                current_content = ""
                in_doc = False
            elif in_doc:
                current_content += line + " "
        except Exception:
            current_title = ""
            current_content = ""
            in_doc = False
    return docs

# --- Tokenizer Helper ---
def is_only_letters_py(token_str):
    return re.match(r'^[a-zA-Z]+$', token_str) is not None

# --- NLTK based Lemmatizer (NLP Pipeline) ---
def lemmatize_nltk(text, lemmatizer_instance, stop_words_set):
    tokens = nltk.word_tokenize(text.lower())
    lemmas = []
    for token in tokens:
        lemma = lemmatizer_instance.lemmatize(token)
        if len(lemma) > 2 and lemma not in stop_words_set and is_only_letters_py(lemma):
            lemmas.append(lemma)
    return lemmas

# --- Simple Regex Tokenizer ---
def simple_tokenize_py(text, stop_words_set):
    word_regex = r"\b[a-zA-Z]{3,}\b"
    tokens = re.findall(word_regex, text.lower())
    return [token for token in tokens if token not in stop_words_set]

# --- LSA Core Logic ---
def run_lsa_pipeline(sc, config): # config is a dictionary
    global_start_time = time.time()
    pipeline_successful = False # Flag to track success

    print(f"Configuration: {config}")

    # 1. Load and Parse Data
    print("Loading and parsing data...")
    parse_time_start = time.time()
    
    input_path = config.get('input_path', WIKIPEDIA_DATA_PATH_PATTERN)
    text_files_rdd = sc.wholeTextFiles(input_path)

    if config.get('sample_fraction', 1.0) < 1.0:
        text_files_rdd = text_files_rdd.sample(withReplacement=False, fraction=config['sample_fraction'], seed=42)
    
    parsed_docs_rdd = text_files_rdd.flatMap(parse_xml_doc_py)
    parsed_docs_rdd.cache()
    total_docs_initial = parsed_docs_rdd.count()
    if total_docs_initial == 0:
        print(f"No documents parsed from {input_path}. Check path and parsing. Exiting.")
        return None, time.time() - global_start_time
    print(f"Parsed {total_docs_initial} documents in {time.time() - parse_time_start:.2f}s")

    stopwords_path = config.get('stopwords_path', STOPWORDS_FILE_PATH)
    try:
        with open(stopwords_path, 'r') as f:
            stop_words_list = {line.strip().lower() for line in f}
    except FileNotFoundError:
        print(f"Stopwords file not found at {stopwords_path}. Exiting.")
        return None, time.time() - global_start_time
    b_stop_words = sc.broadcast(stop_words_list)

    # 2. Tokenize
    print(f"Tokenizing using '{config['tokenizer_type']}'...")
    tokenize_time_start = time.time()
    if config['tokenizer_type'] == "nlp":
        def tokenize_partition_nlp(iterator):
            try:
                lemmatizer = WordNetLemmatizer()
                # Optional: test lemmatizer to force NLTK data loading if needed
                # lemmatizer.lemmatize("test") 
            except Exception as e:
                print(f"CRITICAL ERROR ON EXECUTOR: Could not initialize WordNetLemmatizer. NLTK resource 'wordnet' and 'omw-1.4' might be missing or inaccessible to NLTK on this worker node.")
                print(f"Original NLTK error: {e}")
                print(f"Please ensure NLTK data is correctly installed and NLTK_DATA environment variable is set if necessary for Spark executors.")
                raise RuntimeError(f"NLTK Lemmatizer initialization failed on worker: {e}") from e
            
            processed_docs = []
            stop_words_val = b_stop_words.value # Access broadcasted value once per partition
            for title, content in iterator:
                processed_docs.append((title, lemmatize_nltk(content, lemmatizer, stop_words_val)))
            return iter(processed_docs)
        tokenized_docs_rdd = parsed_docs_rdd.mapPartitions(tokenize_partition_nlp)
    elif config['tokenizer_type'] == "simple":
        tokenized_docs_rdd = parsed_docs_rdd.map(
            lambda doc: (doc[0], simple_tokenize_py(doc[1], b_stop_words.value))
        )
    else:
        print(f"Unknown tokenizer type: {config['tokenizer_type']}. Exiting.")
        return None, time.time() - global_start_time
    
    tokenized_docs_rdd = tokenized_docs_rdd.filter(lambda x: len(x[1]) > 0)
    tokenized_docs_rdd.cache()
    final_doc_count = tokenized_docs_rdd.count() # Action to trigger tokenization
    print(f"Tokenized {final_doc_count} documents (after filtering empty ones) in {time.time() - tokenize_time_start:.2f}s")
    if final_doc_count == 0:
        print("No documents left after tokenization. Exiting.")
        return None, time.time() - global_start_time

    # 3. TF-IDF Calculation
    print("Calculating TF-IDF...")
    tfidf_time_start = time.time()
    doc_term_freqs_rdd = tokenized_docs_rdd.map(lambda doc: (doc[0], nltk.FreqDist(doc[1])))
    doc_term_freqs_rdd.cache()

    doc_title_to_id_rdd = tokenized_docs_rdd.map(lambda x: x[0]).zipWithUniqueId()
    # Collect for local use and broadcast for search results display
    id_to_doc_title_map_local = dict(doc_title_to_id_rdd.map(lambda x: (x[1], x[0])).collect())
    b_id_to_doc_title_map = sc.broadcast(id_to_doc_title_map_local)

    term_doc_freqs_rdd = doc_term_freqs_rdd.flatMap(lambda x: [(term, 1) for term in x[1].keys()]) \
                                     .reduceByKey(lambda a, b: a + b)

    top_terms_by_freq_list = term_doc_freqs_rdd.takeOrdered(config['num_freq_terms'], key=lambda x: -x[1])
    if not top_terms_by_freq_list:
        print("No terms found for vocabulary. Check tokenization or num_freq_terms. Exiting.")
        return None, time.time() - global_start_time
    print(f"Selected top {len(top_terms_by_freq_list)} terms for vocabulary.")

    b_final_doc_count = sc.broadcast(float(final_doc_count))
    idfs_map_local = {term: math.log((b_final_doc_count.value +1.0) / (count + 1.0)) for term, count in top_terms_by_freq_list}
    
    term_to_id_map_local = {term: i for i, term in enumerate(idfs_map_local.keys())}
    id_to_term_map_local = {i: term for term, i in term_to_id_map_local.items()}
    b_term_to_id = sc.broadcast(term_to_id_map_local)
    b_id_to_term = sc.broadcast(id_to_term_map_local) # For topTermsInTopConcepts
    vocab_size = len(term_to_id_map_local)
    if vocab_size == 0:
        print("Vocabulary size is 0. Exiting.")
        return None, time.time() - global_start_time

    def create_tf_idf_vector(doc_freq_tuple, term_to_id_val, idfs_val, vocab_size_val):
        _, term_freq_dist = doc_freq_tuple
        doc_total_terms = sum(term_freq_dist.values())
        if doc_total_terms == 0: return Vectors.sparse(vocab_size_val, {})
        term_scores = {term_to_id_val[term]: (freq / doc_total_terms) * idfs_val.get(term, 0.0)
                       for term, freq in term_freq_dist.items() if term in term_to_id_val}
        return Vectors.sparse(vocab_size_val, term_scores)

    tf_idf_vectors_rdd = doc_term_freqs_rdd.map(
        lambda dt_freq: create_tf_idf_vector(dt_freq, b_term_to_id.value, idfs_map_local, vocab_size)
    )
    tf_idf_vectors_rdd.cache()
    tf_idf_vectors_rdd.count() # Action to materialize vectors
    print(f"TF-IDF calculation and vector creation took {time.time() - tfidf_time_start:.2f}s")

    # 4. SVD
    print(f"Computing SVD with k={config['k_svd']}...")
    svd_time_start = time.time()
    term_doc_matrix = RowMatrix(tf_idf_vectors_rdd)
    try:
        svd_model = term_doc_matrix.computeSVD(config['k_svd'], computeU=True)
    except Exception as e:
        print(f"Error during SVD computation: {e}")
        print(f"Details: NumDocs={final_doc_count}, VocabSize={vocab_size}, k={config['k_svd']}")
        return None, time.time() - global_start_time
        
    U_matrix_rdd = svd_model.U
    s_vector = svd_model.s
    V_matrix_local = svd_model.V
    print(f"SVD computation took {time.time() - svd_time_start:.2f}s")

    if U_matrix_rdd is None:
        print("SVD.U is None. This can happen if k is too large or documents are too few/sparse. Exiting.")
        return None, time.time() - global_start_time
    
    print(f"U matrix (RowMatrix): {U_matrix_rdd.numRows()} rows (docs), {U_matrix_rdd.numCols()} cols (concepts)")
    print(f"s vector (local): {s_vector.size} singular values")
    print(f"V matrix (local): {V_matrix_local.numRows} rows (terms), {V_matrix_local.numCols} cols (concepts)")

    pipeline_successful = True # Mark as successful if we reach here

    if config.get('perform_part_a', False):
        print("\n--- Part (a) Style LSA Output (Top Terms/Docs for current run) ---")
        num_concepts_to_show = min(config['k_svd'], 25)
        num_items_per_concept = 25

        V_np = V_matrix_local.toArray()
        print(f"\nTop {num_items_per_concept} Terms in Top {num_concepts_to_show} Concepts (Tokenizer: {config['tokenizer_type']}):")
        for i in range(num_concepts_to_show):
            term_weights = sorted([(V_np[j, i], j) for j in range(V_np.shape[0])], key=lambda x: x[0], reverse=True)
            top_terms_for_concept = [b_id_to_term.value.get(tid, f"ID_{tid}") for _, tid in term_weights[:num_items_per_concept]]
            print(f"Concept {i+1}: {', '.join(top_terms_for_concept)}")

        doc_unique_ids_rdd = doc_title_to_id_rdd.map(lambda p: p[1]) # RDD[unique_id]
        u_rows_with_id_rdd = U_matrix_rdd.rows.zip(doc_unique_ids_rdd).map(lambda p: (p[1], p[0])) # RDD[(unique_id, concept_vector)]
        u_rows_with_id_rdd.cache()
        print(f"\nTop {num_items_per_concept} Documents in Top {num_concepts_to_show} Concepts (Tokenizer: {config['tokenizer_type']}):")
        for i in range(num_concepts_to_show):
            try:
                # Get (score_for_concept_i, original_doc_id)
                concept_doc_scores_rdd = u_rows_with_id_rdd.map(lambda id_vec_tuple: (id_vec_tuple[1][i], id_vec_tuple[0]))
                top_docs_for_concept_ids = concept_doc_scores_rdd.top(num_items_per_concept) 
                top_docs_titles = [f"{b_id_to_doc_title_map.value.get(doc_id, f'ID_{doc_id}')} ({score:.3f})" for score, doc_id in top_docs_for_concept_ids]
                print(f"Concept {i+1}: {', '.join(top_docs_titles)}")
            except IndexError: # If k_svd < num_concepts_to_show for some reason, U might not have that many columns
                print(f"Warning: Concept index {i} out of bounds for U matrix columns ({U_matrix_rdd.numCols()}). Skipping.")
            except Exception as e_concept:
                 print(f"Error processing concept {i} for top docs: {e_concept}")
        u_rows_with_id_rdd.unpersist()

    lsa_components = {
        "U_matrix_rdd": U_matrix_rdd,
        "s_vector": s_vector,
        "V_matrix_local": V_matrix_local,
        "term_to_id_map_local": term_to_id_map_local,
        "idfs_map_local": idfs_map_local,
        "b_id_to_doc_title_map": b_id_to_doc_title_map,
        "vocab_size": vocab_size,
        "tokenizer_type": config['tokenizer_type'],
        "b_stop_words": b_stop_words,
        "doc_title_to_id_rdd": doc_title_to_id_rdd
    }

    total_pipeline_time = time.time() - global_start_time
    print(f"\nTotal LSA pipeline (build) time: {total_pipeline_time:.2f}s")
    
    return lsa_components, total_pipeline_time


# --- Search Engine Logic ---
def run_search_queries(sc, components, queries_list):
    if not queries_list:
        print("No queries provided for search.")
        return
    if not components: # Should be checked before calling, but good for robustness
        print("LSA components are missing. Cannot run search.")
        return

    print("\n--- Part (c) Search Engine Queries ---")
    
    U_matrix_rdd = components["U_matrix_rdd"]
    s_vector_np = components["s_vector"].toArray()
    V_matrix_np = components["V_matrix_local"].toArray()
    term_to_id_map = components["term_to_id_map_local"]
    idfs_map = components["idfs_map_local"]
    b_id_to_doc_title_map = components["b_id_to_doc_title_map"]
    vocab_size = components["vocab_size"]
    tokenizer_type = components["tokenizer_type"]
    b_stop_words = components["b_stop_words"]
    doc_title_to_id_rdd = components["doc_title_to_id_rdd"]

    US_rows_rdd = U_matrix_rdd.rows.map(lambda u_vec: Vectors.dense([u_vec[j] * s_vector_np[j] for j in range(len(s_vector_np))]))
    US_matrix = RowMatrix(US_rows_rdd)
    US_matrix.rows.cache()
    US_matrix.rows.count() # Action to materialize US matrix

    lemmatizer_query_instance = WordNetLemmatizer() if tokenizer_type == "nlp" else None

    for query_str in queries_list:
        print(f"\nQuery: '{query_str}'")
        query_start_time = time.time()

        if tokenizer_type == "nlp":
            tokenized_query = lemmatize_nltk(query_str, lemmatizer_query_instance, b_stop_words.value)
        else:
            tokenized_query = simple_tokenize_py(query_str, b_stop_words.value)

        if not tokenized_query:
            print("Query tokenized to empty list. No results.")
            continue
        
        query_freq_dist = nltk.FreqDist(tokenized_query)
        query_total_terms = len(tokenized_query)
        if query_total_terms == 0:
            print("Query has no terms after effective tokenization. No results.")
            continue

        query_tfidf_scores = {}
        for term, freq in query_freq_dist.items():
            if term in term_to_id_map:
                term_id = term_to_id_map[term]
                tf = freq / query_total_terms
                idf = idfs_map.get(term, 0.0)
                query_tfidf_scores[term_id] = tf * idf
        
        if not query_tfidf_scores:
            print("Query terms not in vocabulary or all have zero TF-IDF. No results.")
            continue

        query_tfidf_vector_mllib = Vectors.sparse(vocab_size, query_tfidf_scores)
        query_tfidf_vector_np = query_tfidf_vector_mllib.toArray()

        q_concept_np = V_matrix_np.transpose().dot(query_tfidf_vector_np)
        q_concept_mllib_matrix = Matrices.dense(len(q_concept_np), 1, q_concept_np)

        scores_row_matrix = US_matrix.multiply(q_concept_mllib_matrix)
        scores_rdd = scores_row_matrix.rows.map(lambda vec: vec[0])

        doc_unique_ids_rdd = doc_title_to_id_rdd.map(lambda p: p[1])
        scores_with_ids_rdd = scores_rdd.zip(doc_unique_ids_rdd)
        
        top_n_results = 10
        results_list = scores_with_ids_rdd.top(top_n_results, key=lambda x: x[0])

        print(f"Top {len(results_list)} results (found in {time.time() - query_start_time:.3f}s):")
        for i, (score, doc_id) in enumerate(results_list):
            doc_title = b_id_to_doc_title_map.value.get(doc_id, f"UNKNOWN_DOC_ID_{doc_id}")
            print(f"  {i+1}. {doc_title} (Score: {score:.4f})")
            
    US_matrix.rows.unpersist()


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSA on Wikipedia data.")
    parser.add_argument("--input_path", type=str, default=WIKIPEDIA_DATA_PATH_PATTERN,
                        help=f"Path to Wikipedia articles (default: {WIKIPEDIA_DATA_PATH_PATTERN})")
    parser.add_argument("--stopwords_path", type=str, default=STOPWORDS_FILE_PATH,
                        help=f"Path to stopwords.txt file (default: {STOPWORDS_FILE_PATH})")
    
    parser.add_argument("--tokenizer_type", type=str, choices=['simple', 'nlp'], required=True,
                        help="Tokenizer type: 'simple' or 'nlp'")
    parser.add_argument("--num_freq_terms", type=int, default=5000,
                        help="Number of frequent terms for vocabulary")
    parser.add_argument("--k_svd", type=int, default=25,
                        help="Number of latent dimensions (k for SVD)")
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="Fraction of data to sample (0.0 to 1.0)")
    parser.add_argument("--perform_part_a", action='store_true',
                        help="Perform Part (a) style analysis (top terms/docs for current run)")
    parser.add_argument("--queries", type=str, nargs='*',
                        help="Keyword queries for search engine (Part c)")

    args = parser.parse_args()

    config_dict = {
        'input_path': args.input_path,
        'stopwords_path': args.stopwords_path,
        'tokenizer_type': args.tokenizer_type,
        'num_freq_terms': args.num_freq_terms,
        'k_svd': args.k_svd,
        'sample_fraction': args.sample_fraction,
        'perform_part_a': args.perform_part_a
    }

    if args.tokenizer_type == 'nlp':
        print("--- NLTK Resource Check (Informational for Driver) ---")
        missing_resources = []
        resources_to_check = {
            'wordnet': 'corpora/wordnet.zip',
            'punkt': 'tokenizers/punkt',
            'stopwords': 'corpora/stopwords',
            'omw-1.4': 'corpora/omw-1.4.zip' # NLTK often looks for the .zip for omw-1.4
        }
        for name, path in resources_to_check.items():
            try:
                nltk.data.find(path)
            except LookupError:
                # Try finding non-zipped version for omw-1.4 if zip not found
                if name == 'omw-1.4':
                    try:
                        nltk.data.find('corpora/omw-1.4')
                    except LookupError:
                        missing_resources.append(name)
                else:
                    missing_resources.append(name)
        
        if missing_resources:
            print(f"WARNING: The following NLTK resources might be missing on the DRIVER or not locatable by NLTK: {', '.join(missing_resources)}")
            print("If the NLP tokenizer fails during Spark execution, it means these resources are not accessible on the SPARK EXECUTOR nodes.")
            print("Please ensure they are downloaded (e.g., `python -m nltk.downloader resource_name`) and that NLTK on executor nodes can find them (e.g., via NLTK_DATA environment variable).")
        else:
            print("NLTK resources (wordnet, punkt, stopwords, omw-1.4) appear to be locatable by NLTK on the driver.")
        print("----------------------------------------------------")

    spark_app_name = f"LSA-{args.tokenizer_type}-f{args.num_freq_terms}-k{args.k_svd}"
    if args.queries:
        spark_app_name += "-Search"
    
    spark = get_spark_session(spark_app_name)
    sc = spark.sparkContext

    print(f"\n--- Running LSA Pipeline ---")
    print(f"Input Path: {args.input_path}")
    print(f"Stopwords Path: {args.stopwords_path}")
    print(f"Parameters - Tokenizer: {args.tokenizer_type}, NumFreqTerms: {args.num_freq_terms}, kSVD: {args.k_svd}, Sample: {args.sample_fraction}")

    lsa_components = None # Initialize to None
    pipeline_time = 0   # Initialize to 0
    try:
        lsa_components, pipeline_time = run_lsa_pipeline(sc, config_dict)

        if args.queries:
            if lsa_components:
                run_search_queries(sc, lsa_components, args.queries)
            else:
                print("LSA model components not built successfully. Cannot run search queries.")
        
        print(f"\n--- Run Summary ---")
        print(f"Final Configuration Used:")
        print(f"  Tokenizer: {args.tokenizer_type}")
        print(f"  Num Frequent Terms: {args.num_freq_terms}")
        print(f"  K for SVD: {args.k_svd}")
        print(f"  Sample Fraction: {args.sample_fraction}")

        if lsa_components is not None:
            print(f"Total LSA Pipeline Build Time: {pipeline_time:.2f} seconds")
        else:
             print(f"LSA Pipeline did NOT complete successfully. Total time recorded: {pipeline_time:.2f} seconds (this may be time until failure).")

        if args.queries:
            print(f"Search queries processed: {' '.join(args.queries) if args.queries else 'None'}")

    except Exception as e_main:
        print(f"An unhandled exception occurred in the main script execution: {e_main}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping Spark session.")
        spark.stop()
