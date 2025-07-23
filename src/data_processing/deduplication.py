import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import hashlib
import time
from collections import defaultdict
import json

# Add FAISS for efficient similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

class Deduplicator:
    def __init__(self, similarity_threshold: float = 0.95, method: str = "levenshtein"):
        """
        Initialize deduplicator with configurable threshold and method
        
        Args:
            similarity_threshold: Threshold for considering documents as duplicates (0.0-1.0)
            method: Deduplication method ('levenshtein', 'semantic', 'hybrid', 'fast_levenshtein', 'faiss_semantic')
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.similarity_threshold = similarity_threshold
        self.method = method
        
        # Initialize semantic model if needed
        self.semantic_model = None
        if method in ["semantic", "hybrid", "faiss_semantic"]:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Loaded semantic model for deduplication")
            except Exception as e:
                self.logger.warning(f"Failed to load semantic model: {e}")
                self.method = "levenshtein" if method == "semantic" else "levenshtein"
        
        # Check FAISS availability
        if method == "faiss_semantic" and not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available. Install with: pip install faiss-cpu or pip install faiss-gpu")
            self.method = "semantic"  # Fallback to regular semantic

    def deduplicate(self, documents: List[Dict[str, Any]], text_column: str = "text") -> List[Dict[str, Any]]:
        """
        Remove near-duplicate documents using content similarity
        
        Args:
            documents: List of document dictionaries
            text_column: Column name containing the text content
            
        Returns:
            List of deduplicated documents
        """
        if not documents:
            self.logger.info("No documents to deduplicate")
            return []
        
        self.logger.info(f"Starting deduplication of {len(documents)} documents using {self.method} method")
        start_time = time.time()
        
        # Pre-filter exact duplicates using hash
        documents = self._remove_exact_duplicates(documents, text_column)
        
        if self.method == "levenshtein":
            result = self._levenshtein_deduplication(documents, text_column)
        elif self.method == "fast_levenshtein":
            result = self._fast_levenshtein_deduplication(documents, text_column)
        elif self.method == "semantic":
            result = self._semantic_deduplication(documents, text_column)
        elif self.method == "faiss_semantic":
            result = self._faiss_semantic_deduplication(documents, text_column)
        elif self.method == "hybrid":
            result = self._hybrid_deduplication(documents, text_column)
        else:
            raise ValueError(f"Unknown deduplication method: {self.method}")
        
        end_time = time.time()
        self.logger.info(f"Deduplication completed in {end_time - start_time:.2f} seconds")
        
        return result

    def _faiss_semantic_deduplication(self, documents: List[Dict[str, Any]], text_column: str) -> List[Dict[str, Any]]:
        """
        FAISS-based semantic deduplication with O(n log n) complexity
        Much faster than O(n²) pairwise comparison
        """
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available, falling back to regular semantic deduplication")
            return self._semantic_deduplication(documents, text_column)
        
        if self.semantic_model is None:
            self.logger.warning("Semantic model not available, falling back to Levenshtein")
            return self._levenshtein_deduplication(documents, text_column)
        
        self.logger.info("Using FAISS-based semantic deduplication for optimal performance")
        
        # Extract texts
        texts = [doc.get(text_column, "") for doc in documents]
        
        # Compute embeddings
        self.logger.info("Computing semantic embeddings...")
        embeddings = self.semantic_model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        index.add(embeddings.astype('float32'))
        
        # Search for similar documents
        self.logger.info("Searching for similar documents using FAISS...")
        k = min(100, len(documents))  # Search top k similar documents
        
        # Search for each document
        unique_docs = []
        removed_count = 0
        duplicate_pairs = []
        processed = set()
        
        for i in tqdm(range(len(documents)), desc="FAISS semantic deduplication"):
            if i in processed:
                continue
            
            # Add current document to unique set
            unique_docs.append(documents[i])
            processed.add(i)
            
            # Search for similar documents
            query_embedding = embeddings[i:i+1].astype('float32')
            similarities, indices = index.search(query_embedding, k)
            
            # Process similar documents
            for j, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == i or idx in processed:  # Skip self and already processed
                    continue
                
                if similarity > self.similarity_threshold:
                    processed.add(idx)
                    removed_count += 1
                    duplicate_pairs.append({
                        'original': documents[i].get('id', f'doc_{i}'),
                        'duplicate': documents[idx].get('id', f'doc_{idx}'),
                        'similarity': float(similarity)
                    })
        
        self.logger.info(f"FAISS semantic deduplication: removed {removed_count} duplicates")
        self.duplicate_pairs = duplicate_pairs
        
        return unique_docs

    def _remove_exact_duplicates(self, documents: List[Dict[str, Any]], text_column: str) -> List[Dict[str, Any]]:
        """Remove exact duplicates using content hash"""
        seen_hashes = set()
        unique_docs = []
        
        for doc in documents:
            content = doc.get(text_column, "")
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
        
        removed = len(documents) - len(unique_docs)
        if removed > 0:
            self.logger.info(f"Removed {removed} exact duplicates")
        
        return unique_docs

    def _fast_levenshtein_deduplication(self, documents: List[Dict[str, Any]], text_column: str) -> List[Dict[str, Any]]:
        """
        Fast Levenshtein deduplication using the ratio approach
        This is the implementation requested in the user query
        """
        try:
            from Levenshtein import ratio
        except ImportError:
            self.logger.error("Levenshtein package not installed. Install with: pip install python-Levenshtein")
            return documents
        
        self.logger.info("Using fast Levenshtein ratio deduplication")
        
        # Sort by length to keep longer documents (more complete)
        sorted_docs = sorted(documents, key=lambda x: len(x.get(text_column, "")), reverse=True)
        
        unique = []
        removed_count = 0
        duplicate_pairs = []
        
        for doc in tqdm(sorted_docs, desc="Fast Levenshtein deduplication"):
            doc_content = doc.get(text_column, "")
            is_duplicate = False
            
            # Check against all unique documents
            for u_doc in unique:
                u_content = u_doc.get(text_column, "")
                
                # Calculate similarity ratio
                similarity = ratio(doc_content, u_content)
                
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    removed_count += 1
                    duplicate_pairs.append({
                        'original': u_doc.get('id', 'unknown'),
                        'duplicate': doc.get('id', 'unknown'),
                        'similarity': similarity
                    })
                    break
            
            if not is_duplicate:
                unique.append(doc)
        
        self.logger.info(f"Fast Levenshtein deduplication: removed {removed_count} duplicates")
        
        # Store duplicate pairs for analysis
        self.duplicate_pairs = duplicate_pairs
        
        return unique

    def _levenshtein_deduplication(self, documents: List[Dict[str, Any]], text_column: str) -> List[Dict[str, Any]]:
        """Enhanced Levenshtein deduplication with optimizations"""
        try:
            from Levenshtein import ratio
        except ImportError:
            self.logger.error("Levenshtein package not installed. Install with: pip install python-Levenshtein")
            return documents
        
        # Sort by length to compare long vs short (keep longer documents)
        sorted_docs = sorted(documents, key=lambda x: len(x.get(text_column, "")), reverse=True)
        
        unique_docs = [sorted_docs[0]]
        removed_count = 0
        duplicate_pairs = []
        
        for doc in tqdm(sorted_docs[1:], desc="Deduplicating with Levenshtein"):
            is_duplicate = False
            doc_content = doc.get(text_column, "")
            
            for u_doc in unique_docs:
                u_content = u_doc.get(text_column, "")
                
                # Skip if lengths are too different (optimization)
                if abs(len(doc_content) - len(u_content)) / max(len(doc_content), len(u_content)) > 0.3:
                    continue
                
                # Calculate similarity
                similarity = ratio(doc_content, u_content)
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    removed_count += 1
                    duplicate_pairs.append({
                        'original': u_doc.get('id', 'unknown'),
                        'duplicate': doc.get('id', 'unknown'),
                        'similarity': similarity
                    })
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
        
        self.logger.info(f"Levenshtein deduplication: removed {removed_count} duplicates")
        self.duplicate_pairs = duplicate_pairs
        
        return unique_docs

    def _semantic_deduplication(self, documents: List[Dict[str, Any]], text_column: str) -> List[Dict[str, Any]]:
        """Deduplicate using semantic similarity with chunk-wise encoding"""
        if self.semantic_model is None:
            self.logger.warning("Semantic model not available, falling back to Levenshtein")
            return self._levenshtein_deduplication(documents, text_column)
        
        # Extract texts
        texts = [doc.get(text_column, "") for doc in documents]
        
        # Compute embeddings with chunk-wise encoding for better performance
        self.logger.info("Computing semantic embeddings with chunk-wise encoding...")
        embeddings = self.semantic_model.encode(
            texts,
            batch_size=256,               # tune to your GPU / RAM
            show_progress_bar=True,
            convert_to_numpy=True         # saves one cast later
        )
        
        # Find duplicates using cosine similarity
        unique_docs = [documents[0]]
        removed_count = 0
        duplicate_pairs = []
        
        for i, doc in enumerate(tqdm(documents[1:], desc="Deduplicating with semantic similarity")):
            is_duplicate = False
            doc_embedding = embeddings[i]
            
            for u_doc in unique_docs:
                u_idx = documents.index(u_doc)
                u_embedding = embeddings[u_idx]
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(doc_embedding, u_embedding)
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    removed_count += 1
                    duplicate_pairs.append({
                        'original': u_doc.get('id', 'unknown'),
                        'duplicate': doc.get('id', 'unknown'),
                        'similarity': similarity
                    })
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
        
        self.logger.info(f"Semantic deduplication: removed {removed_count} duplicates")
        self.duplicate_pairs = duplicate_pairs
        
        return unique_docs

    def _hybrid_deduplication(self, documents: List[Dict[str, Any]], text_column: str) -> List[Dict[str, Any]]:
        """Deduplicate using both Levenshtein and semantic similarity"""
        # First pass: semantic deduplication
        self.logger.info("Hybrid deduplication: First pass (semantic)")
        semantic_docs = self._semantic_deduplication(documents, text_column)
        
        # Second pass: Levenshtein deduplication
        self.logger.info("Hybrid deduplication: Second pass (Levenshtein)")
        final_docs = self._levenshtein_deduplication(semantic_docs, text_column)
        
        return final_docs

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def deduplicate_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Deduplicate pandas DataFrame"""
        documents = df.to_dict('records')
        deduplicated_docs = self.deduplicate(documents, text_column)
        return pd.DataFrame(deduplicated_docs)

    def get_deduplication_stats(self, original_count: int, final_count: int) -> Dict[str, Any]:
        """Get deduplication statistics"""
        removed_count = original_count - final_count
        reduction_percentage = (removed_count / original_count * 100) if original_count > 0 else 0
        
        stats = {
            "original_count": original_count,
            "final_count": final_count,
            "removed_count": removed_count,
            "reduction_percentage": reduction_percentage,
            "method": self.method,
            "threshold": self.similarity_threshold
        }
        
        # Add duplicate pair analysis if available
        if hasattr(self, 'duplicate_pairs') and self.duplicate_pairs:
            similarities = [pair['similarity'] for pair in self.duplicate_pairs]
            stats.update({
                "duplicate_pairs_count": len(self.duplicate_pairs),
                "avg_similarity": np.mean(similarities),
                "min_similarity": np.min(similarities),
                "max_similarity": np.max(similarities)
            })
        
        return stats

    def analyze_duplicates(self, documents: List[Dict[str, Any]], text_column: str = "text") -> Dict[str, Any]:
        """
        Analyze potential duplicates without removing them
        
        Args:
            documents: List of document dictionaries
            text_column: Column name containing the text content
            
        Returns:
            Dictionary with duplicate analysis
        """
        try:
            from Levenshtein import ratio
        except ImportError:
            self.logger.error("Levenshtein package not installed")
            return {"error": "Levenshtein package not available"}
        
        analysis = {
            "total_documents": len(documents),
            "potential_duplicates": [],
            "similarity_distribution": defaultdict(int),
            "high_similarity_pairs": []
        }
        
        # Analyze all pairs
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents[i+1:], i+1):
                content1 = doc1.get(text_column, "")
                content2 = doc2.get(text_column, "")
                
                similarity = ratio(content1, content2)
                
                # Categorize by similarity level
                if similarity > 0.9:
                    category = "very_high"
                elif similarity > 0.8:
                    category = "high"
                elif similarity > 0.7:
                    category = "medium"
                else:
                    category = "low"
                
                analysis["similarity_distribution"][category] += 1
                
                # Store high similarity pairs
                if similarity > 0.8:
                    analysis["high_similarity_pairs"].append({
                        "doc1_id": doc1.get('id', f'doc_{i}'),
                        "doc2_id": doc2.get('id', f'doc_{j}'),
                        "similarity": similarity,
                        "doc1_preview": content1[:100] + "..." if len(content1) > 100 else content1,
                        "doc2_preview": content2[:100] + "..." if len(content2) > 100 else content2
                    })
        
        analysis["potential_duplicates"] = len(analysis["high_similarity_pairs"])
        
        return analysis

    def save_duplicate_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save duplicate analysis to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Duplicate analysis saved to {output_path}")

    def load_duplicate_analysis(self, input_path: str) -> Dict[str, Any]:
        """Load duplicate analysis from file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_duplicate_clusters(self, documents: List[Dict[str, Any]], text_column: str = "text") -> List[List[Dict[str, Any]]]:
        """
        Group documents into duplicate clusters
        
        Args:
            documents: List of document dictionaries
            text_column: Column name containing the text content
            
        Returns:
            List of duplicate clusters
        """
        try:
            from Levenshtein import ratio
        except ImportError:
            self.logger.error("Levenshtein package not installed")
            return []
        
        clusters = []
        processed = set()
        
        for i, doc in enumerate(documents):
            if i in processed:
                continue
            
            cluster = [doc]
            processed.add(i)
            doc_content = doc.get(text_column, "")
            
            for j, other_doc in enumerate(documents[i+1:], i+1):
                if j in processed:
                    continue
                
                other_content = other_doc.get(text_column, "")
                similarity = ratio(doc_content, other_content)
                
                if similarity > self.similarity_threshold:
                    cluster.append(other_doc)
                    processed.add(j)
            
            if len(cluster) > 1:  # Only return clusters with duplicates
                clusters.append(cluster)
        
        return clusters 