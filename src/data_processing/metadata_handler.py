import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

class MetadataHandler:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_metadata(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and standardize metadata from different sources"""
        self.logger.info("Adding metadata to documents...")
        processed = []
        
        for doc in documents:
            # Extract source-specific metadata
            meta = self._extract_source_metadata(doc)
            
            # Add processing metadata
            meta.update({
                "processed_date": datetime.now().isoformat(),
                "pipeline_version": "1.0"
            })
            
            processed.append({**doc, "metadata": meta})
        
        self.logger.info(f"Added metadata to {len(processed)} documents")
        return processed

    def _extract_source_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata based on document source type"""
        source = doc.get("source", "")
        doc_type = doc.get("type", "unknown")
        
        if doc_type == "pdf" or source.endswith('.pdf'):
            return {
                "source_type": "pdf",
                "file_name": source,
                "page_numbers": doc.get("metadata", {}).get("page_number", 1),
                "extraction_method": doc.get("metadata", {}).get("extractor", "pdfplumber")
            }
        elif doc_type == "web" or source.startswith(('http://', 'https://')):
            return {
                "source_type": "web",
                "url": source,
                "scraped_date": doc.get("metadata", {}).get("timestamp", datetime.now().isoformat()),
                "status_code": doc.get("metadata", {}).get("status_code", 200)
            }
        else:
            return {
                "source_type": "unknown",
                "source": source,
                "original_metadata": doc.get("metadata", {})
            }

    def propagate_metadata_to_qa(self, qa_pairs: List[Dict[str, Any]], source_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Propagate source metadata to QA pairs"""
        self.logger.info("Propagating metadata to QA pairs...")
        
        # Create source lookup
        source_lookup = {}
        for doc in source_docs:
            text_hash = self._hash_text(doc.get("text", ""))
            source_lookup[text_hash] = doc.get("metadata", {})
        
        # Add metadata to QA pairs
        for qa in qa_pairs:
            context = qa.get("context", "")
            context_hash = self._hash_text(context)
            
            if context_hash in source_lookup:
                qa["source_metadata"] = source_lookup[context_hash]
            else:
                # Try to find partial matches
                qa["source_metadata"] = self._find_best_source_match(context, source_lookup)
        
        self.logger.info(f"Propagated metadata to {len(qa_pairs)} QA pairs")
        return qa_pairs

    def _hash_text(self, text: str) -> str:
        """Simple hash for text matching"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def _find_best_source_match(self, context: str, source_lookup: Dict[str, Dict]) -> Dict[str, Any]:
        """Find the best matching source for a context"""
        # Simple substring matching for now
        for text_hash, metadata in source_lookup.items():
            # This is a simplified approach - in production you might want more sophisticated matching
            if len(context) > 50 and any(keyword in context.lower() for keyword in ["ev", "charging", "electric"]):
                return metadata
        return {"source_type": "unknown", "confidence": "low"}

    def add_attribution_to_qa(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add attribution information to QA pairs"""
        self.logger.info("Adding attribution to QA pairs...")
        
        for qa in qa_pairs:
            source_meta = qa.get("source_metadata", {})
            
            # Create attribution text
            if source_meta.get("source_type") == "web":
                attribution = f"Source: {source_meta.get('url', 'Unknown URL')}"
            elif source_meta.get("source_type") == "pdf":
                attribution = f"Source: {source_meta.get('file_name', 'Unknown PDF')}"
            else:
                attribution = "Source: Unknown"
            
            qa["attribution"] = attribution
            qa["source_tracking"] = {
                "has_source": bool(source_meta.get("source_type") != "unknown"),
                "source_type": source_meta.get("source_type", "unknown"),
                "confidence": source_meta.get("confidence", "unknown")
            }
        
        self.logger.info(f"Added attribution to {len(qa_pairs)} QA pairs")
        return qa_pairs

    def validate_metadata(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate metadata completeness"""
        self.logger.info("Validating metadata...")
        
        stats = {
            "total_documents": len(documents),
            "with_source": 0,
            "with_timestamp": 0,
            "with_url": 0,
            "with_filename": 0,
            "missing_metadata": 0
        }
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            
            if metadata.get("source_type") != "unknown":
                stats["with_source"] += 1
            
            if metadata.get("scraped_date") or metadata.get("processed_date"):
                stats["with_timestamp"] += 1
            
            if metadata.get("url"):
                stats["with_url"] += 1
            
            if metadata.get("file_name"):
                stats["with_filename"] += 1
            
            if not metadata:
                stats["missing_metadata"] += 1
        
        self.logger.info(f"Metadata validation complete: {stats}")
        return stats 