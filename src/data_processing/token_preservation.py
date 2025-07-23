#!/usr/bin/env python3
"""
Token Preservation for EV-Specific Terminology
Ensures technical terms and domain-specific vocabulary are properly tokenized
"""

import re
import json
import logging
from typing import List, Dict, Any, Set, Optional, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizer
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)

class TokenPreservation:
    """Handles preservation of domain-specific tokens during tokenization"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, domain: str = "electric_vehicles"):
        self.tokenizer = tokenizer
        self.domain = domain
        self.special_terms = self._load_domain_terms(domain)
        self.term_mappings = {}
        self.preserved_tokens = set()
        
        # Initialize token preservation
        self._setup_token_preservation()
    
    def _load_domain_terms(self, domain: str) -> Dict[str, List[str]]:
        """Load domain-specific terminology"""
        
        # EV-specific terminology
        ev_terms = {
            "charging_standards": [
                "CCS1", "CCS2", "CHAdeMO", "Type1", "Type2", "Type3",
                "GB/T", "Tesla Supercharger", "NACS", "Mennekes"
            ],
            "protocols": [
                "OCPP", "OCPP1.6", "OCPP2.0.1", "ISO15118", "DIN70121",
                "Plug&Charge", "ISO15118-20", "DIN70122"
            ],
            "power_ratings": [
                "3.7kW", "7.4kW", "11kW", "22kW", "50kW", "150kW", "350kW",
                "400kW", "800kW", "1MW", "3.6kW", "7.2kW", "10.5kW"
            ],
            "connector_types": [
                "CCS Combo", "CHAdeMO", "Type1", "Type2", "GB/T", "Tesla",
                "Mennekes", "Schuko", "CEE", "IEC62196"
            ],
            "battery_terms": [
                "kWh", "Ah", "V", "A", "DC", "AC", "PHEV", "BEV", "HEV",
                "SOC", "SOH", "BMS", "thermal_management"
            ],
            "network_terms": [
                "smart_grid", "V2G", "V2H", "V2X", "bidirectional",
                "load_balancing", "peak_shaving", "frequency_regulation"
            ],
            "environmental_terms": [
                "carbon_footprint", "CO2_emissions", "renewable_energy",
                "green_electricity", "sustainability", "carbon_neutral"
            ],
            "technical_specs": [
                "efficiency", "power_factor", "power_quality", "harmonics",
                "voltage_drop", "cable_losses", "thermal_rating"
            ]
        }
        
        # General technical terms
        general_terms = {
            "measurements": [
                "kW", "kWh", "V", "A", "Hz", "W", "J", "C", "F", "Ω",
                "m/s", "km/h", "bar", "psi", "°C", "°F", "K"
            ],
            "abbreviations": [
                "API", "HTTP", "HTTPS", "JSON", "XML", "REST", "SOAP",
                "TCP", "UDP", "IP", "DNS", "SSL", "TLS", "OAuth"
            ],
            "units": [
                "percent", "percentage", "milliseconds", "seconds", "minutes",
                "hours", "days", "weeks", "months", "years"
            ]
        }
        
        # Domain-specific term mappings
        domain_mappings = {
            "electric_vehicles": ev_terms,
            "general": general_terms,
            "automotive": {
                **ev_terms,
                "vehicle_terms": [
                    "MPG", "MPGe", "range", "efficiency", "torque", "horsepower",
                    "acceleration", "top_speed", "weight", "payload"
                ]
            }
        }
        
        return domain_mappings.get(domain, general_terms)
    
    def _setup_token_preservation(self):
        """Set up token preservation for special terms"""
        
        # Collect all special terms
        all_terms = []
        for category, terms in self.special_terms.items():
            all_terms.extend(terms)
        
        # Add special tokens to tokenizer
        added_tokens = self.tokenizer.add_tokens(all_terms)
        logger.info(f"Added {added_tokens} special tokens for {self.domain} domain")
        
        # Create term mappings for verification
        for term in all_terms:
            token_ids = self.tokenizer.encode(term, add_special_tokens=False)
            if len(token_ids) == 1:
                # Single token - perfect preservation
                self.term_mappings[term] = {
                    "token_ids": token_ids,
                    "preserved": True,
                    "token_count": 1
                }
                self.preserved_tokens.add(term)
            else:
                # Multiple tokens - check if we can improve
                self.term_mappings[term] = {
                    "token_ids": token_ids,
                    "preserved": False,
                    "token_count": len(token_ids)
                }
        
        # Log preservation statistics
        preserved_count = len(self.preserved_tokens)
        total_count = len(all_terms)
        logger.info(f"Token preservation: {preserved_count}/{total_count} terms preserved ({preserved_count/total_count*100:.1f}%)")
    
    def tokenize_with_preservation(self, text: str, **kwargs) -> Dict[str, Any]:
        """Tokenize text while preserving special terms"""
        
        # Pre-process text to protect special terms
        protected_text = self._protect_special_terms(text)
        
        # Tokenize
        tokens = self.tokenizer(
            protected_text,
            **kwargs
        )
        
        # Post-process to restore special terms
        restored_tokens = self._restore_special_terms(tokens, text)
        
        return restored_tokens
    
    def _protect_special_terms(self, text: str) -> str:
        """Protect special terms from being split during tokenization"""
        
        protected_text = text
        
        # Replace special terms with placeholders
        for term in self.special_terms:
            if term in protected_text:
                placeholder = f"__{term.upper()}__"
                protected_text = protected_text.replace(term, placeholder)
        
        return protected_text
    
    def _restore_special_terms(self, tokens: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Restore special terms in tokenized output"""
        
        # Convert token IDs back to tokens
        token_list = self.tokenizer.convert_ids_to_tokens(tokens["input_ids"])
        
        # Restore special terms
        restored_tokens = []
        for token in token_list:
            if token.startswith("__") and token.endswith("__"):
                # Extract original term
                term = token[2:-2].lower()
                if term in self.special_terms:
                    restored_tokens.append(term)
                else:
                    restored_tokens.append(token)
            else:
                restored_tokens.append(token)
        
        # Convert back to token IDs
        restored_ids = self.tokenizer.convert_tokens_to_ids(restored_tokens)
        
        return {
            "input_ids": restored_ids,
            "attention_mask": tokens["attention_mask"][:len(restored_ids)],
            "token_type_ids": tokens.get("token_type_ids", [0] * len(restored_ids))[:len(restored_ids)]
        }
    
    def verify_token_preservation(self, text: str) -> Dict[str, Any]:
        """Verify that special terms are properly preserved"""
        
        verification_results = {
            "text": text,
            "terms_found": [],
            "preservation_status": {},
            "overall_score": 0.0
        }
        
        # Find all special terms in text
        found_terms = []
        for category, terms in self.special_terms.items():
            for term in terms:
                if term.lower() in text.lower():
                    found_terms.append({
                        "term": term,
                        "category": category,
                        "count": text.lower().count(term.lower())
                    })
        
        verification_results["terms_found"] = found_terms
        
        # Check preservation for each found term
        preserved_count = 0
        for term_info in found_terms:
            term = term_info["term"]
            
            # Tokenize the term
            tokenized = self.tokenize_with_preservation(term)
            token_ids = tokenized["input_ids"]
            
            # Check if term is preserved as single token
            is_preserved = len(token_ids) == 1 and term in self.preserved_tokens
            
            verification_results["preservation_status"][term] = {
                "preserved": is_preserved,
                "token_count": len(token_ids),
                "token_ids": token_ids,
                "category": term_info["category"]
            }
            
            if is_preserved:
                preserved_count += 1
        
        # Calculate overall preservation score
        if found_terms:
            verification_results["overall_score"] = preserved_count / len(found_terms)
        
        return verification_results
    
    def get_preservation_statistics(self) -> Dict[str, Any]:
        """Get statistics about token preservation"""
        
        stats = {
            "domain": self.domain,
            "total_terms": len(self.term_mappings),
            "preserved_terms": len(self.preserved_tokens),
            "preservation_rate": len(self.preserved_tokens) / len(self.term_mappings) if self.term_mappings else 0,
            "categories": {},
            "term_details": self.term_mappings
        }
        
        # Category-wise statistics
        for category, terms in self.special_terms.items():
            category_preserved = sum(1 for term in terms if term in self.preserved_tokens)
            stats["categories"][category] = {
                "total": len(terms),
                "preserved": category_preserved,
                "rate": category_preserved / len(terms) if terms else 0
            }
        
        return stats
    
    def add_custom_terms(self, terms: List[str], category: str = "custom") -> int:
        """Add custom terms to the preservation system"""
        
        # Add to special terms
        if category not in self.special_terms:
            self.special_terms[category] = []
        
        self.special_terms[category].extend(terms)
        
        # Add to tokenizer
        added_count = self.tokenizer.add_tokens(terms)
        
        # Update mappings
        for term in terms:
            token_ids = self.tokenizer.encode(term, add_special_tokens=False)
            self.term_mappings[term] = {
                "token_ids": token_ids,
                "preserved": len(token_ids) == 1,
                "token_count": len(token_ids)
            }
            
            if len(token_ids) == 1:
                self.preserved_tokens.add(term)
        
        logger.info(f"Added {added_count} custom terms to category '{category}'")
        return added_count
    
    def export_preservation_config(self, filepath: str):
        """Export preservation configuration to file"""
        
        config = {
            "domain": self.domain,
            "special_terms": self.special_terms,
            "term_mappings": self.term_mappings,
            "preserved_tokens": list(self.preserved_tokens),
            "statistics": self.get_preservation_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Preservation config exported to {filepath}")
    
    def load_preservation_config(self, filepath: str):
        """Load preservation configuration from file"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.special_terms = config["special_terms"]
        self.term_mappings = config["term_mappings"]
        self.preserved_tokens = set(config["preserved_tokens"])
        
        logger.info(f"Preservation config loaded from {filepath}")

def create_ev_tokenizer(base_model: str = "microsoft/DialoGPT-medium") -> Tuple[PreTrainedTokenizer, TokenPreservation]:
    """Create tokenizer with EV-specific token preservation"""
    
    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create token preservation system
    preservation = TokenPreservation(tokenizer, domain="electric_vehicles")
    
    return tokenizer, preservation

def tokenize_ev_documents(documents: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, preservation: TokenPreservation) -> List[Dict[str, Any]]:
    """Tokenize EV documents with term preservation"""
    
    tokenized_docs = []
    
    for doc in documents:
        # Tokenize with preservation
        tokens = preservation.tokenize_with_preservation(
            doc["content"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Verify preservation
        verification = preservation.verify_token_preservation(doc["content"])
        
        # Add to results
        tokenized_doc = {
            "id": doc.get("id", len(tokenized_docs)),
            "content": doc["content"],
            "tokens": tokens,
            "preservation_verification": verification,
            "preservation_score": verification["overall_score"]
        }
        
        tokenized_docs.append(tokenized_doc)
    
    return tokenized_docs

def analyze_token_preservation(documents: List[Dict[str, Any]], preservation: TokenPreservation) -> Dict[str, Any]:
    """Analyze token preservation across documents"""
    
    analysis = {
        "total_documents": len(documents),
        "total_terms_found": 0,
        "total_terms_preserved": 0,
        "overall_preservation_rate": 0.0,
        "category_analysis": defaultdict(lambda: {"found": 0, "preserved": 0}),
        "document_analysis": []
    }
    
    for doc in documents:
        verification = preservation.verify_token_preservation(doc["content"])
        
        # Update overall statistics
        analysis["total_terms_found"] += len(verification["terms_found"])
        analysis["total_terms_preserved"] += sum(
            1 for term, status in verification["preservation_status"].items()
            if status["preserved"]
        )
        
        # Update category analysis
        for term_info in verification["terms_found"]:
            category = term_info["category"]
            analysis["category_analysis"][category]["found"] += term_info["count"]
            
            if verification["preservation_status"][term_info["term"]]["preserved"]:
                analysis["category_analysis"][category]["preserved"] += term_info["count"]
        
        # Document-level analysis
        doc_analysis = {
            "id": doc.get("id", "unknown"),
            "terms_found": len(verification["terms_found"]),
            "terms_preserved": sum(
                1 for status in verification["preservation_status"].values()
                if status["preserved"]
            ),
            "preservation_score": verification["overall_score"]
        }
        analysis["document_analysis"].append(doc_analysis)
    
    # Calculate overall preservation rate
    if analysis["total_terms_found"] > 0:
        analysis["overall_preservation_rate"] = analysis["total_terms_preserved"] / analysis["total_terms_found"]
    
    # Calculate category preservation rates
    for category, stats in analysis["category_analysis"].items():
        if stats["found"] > 0:
            stats["preservation_rate"] = stats["preserved"] / stats["found"]
        else:
            stats["preservation_rate"] = 0.0
    
    return analysis 