import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from tqdm import tqdm
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class QAGenerationConfig:
    """Configuration for QA generation"""
    model: str = "gpt-4-turbo"
    temperature: float = 0.3
    max_tokens: int = 1000
    chunk_size: int = 1000
    chunk_overlap: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 10
    max_qa_per_chunk: int = 2
    include_source: bool = True
    include_metadata: bool = True

class QAGenerator:
    def __init__(self, config: Optional[QAGenerationConfig] = None):
        """
        Initialize QA generator with OpenAI integration
        
        Args:
            config: Configuration for QA generation
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or QAGenerationConfig()
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        
        # Validate API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.logger.info(f"Initialized QA Generator with model: {self.config.model}")

    def generate_qa_pairs(self, documents: List[Dict[str, Any]], domain: str, 
                         text_column: str = "text") -> List[Dict[str, Any]]:
        """
        Generate domain-specific QA pairs from documents
        
        Args:
            documents: List of document dictionaries
            domain: Domain for QA generation (e.g., "electric_vehicles", "healthcare")
            text_column: Column name containing the text content
            
        Returns:
            List of QA pair dictionaries
        """
        if not documents:
            self.logger.warning("No documents provided for QA generation")
            return []
        
        self.logger.info(f"Starting QA generation for {len(documents)} documents in {domain} domain")
        
        qa_dataset = []
        total_chunks = 0
        
        for doc in tqdm(documents, desc="Processing documents"):
            try:
                # Extract text content
                text_content = doc.get(text_column, "")
                if not text_content or len(text_content.strip()) < 50:
                    continue
                
                # Split text into chunks
                chunks = self.splitter.split_text(text_content)
                total_chunks += len(chunks)
                
                # Generate QA pairs for each chunk
                for chunk in chunks:
                    if len(chunk.strip()) < 30:
                        continue
                    
                    qa_pairs = self._generate_qa_for_chunk(chunk, domain, doc)
                    qa_dataset.extend(qa_pairs)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Error processing document: {e}")
                continue
        
        self.logger.info(f"Generated {len(qa_dataset)} QA pairs from {total_chunks} chunks")
        return qa_dataset

    def _generate_qa_for_chunk(self, chunk: str, domain: str, source_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate QA pairs for a single text chunk"""
        qa_pairs = []
        
        for attempt in range(self.config.max_retries):
            try:
                # Create system prompt based on domain
                system_prompt = self._create_system_prompt(domain)
                
                # Create user prompt
                user_prompt = self._create_user_prompt(chunk, domain)
                
                # Generate QA pairs
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                # Parse response
                qa_data = json.loads(response.choices[0].message.content)
                
                # Process QA pairs
                if isinstance(qa_data, dict):
                    qa_pairs = self._process_qa_response(qa_data, source_doc)
                elif isinstance(qa_data, list):
                    for qa_item in qa_data:
                        qa_pairs.extend(self._process_qa_response(qa_item, source_doc))
                
                break  # Success, exit retry loop
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    self.logger.error(f"Failed to parse QA response after {self.config.max_retries} attempts")
                else:
                    time.sleep(self.config.retry_delay)
                    
            except Exception as e:
                self.logger.warning(f"QA generation error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    self.logger.error(f"Failed to generate QA pairs after {self.config.max_retries} attempts")
                else:
                    time.sleep(self.config.retry_delay)
        
        return qa_pairs

    def _create_system_prompt(self, domain: str) -> str:
        """Create system prompt for QA generation"""
        domain_prompts = {
            "electric_vehicles": "You are an expert data generator for electric vehicle domain. Generate high-quality, factual question-answer pairs that are informative and accurate.",
            "healthcare": "You are an expert data generator for healthcare domain. Generate high-quality, factual question-answer pairs that are medically accurate and informative.",
            "technology": "You are an expert data generator for technology domain. Generate high-quality, factual question-answer pairs that are technically accurate and informative.",
            "finance": "You are an expert data generator for finance domain. Generate high-quality, factual question-answer pairs that are financially accurate and informative.",
            "education": "You are an expert data generator for education domain. Generate high-quality, factual question-answer pairs that are educationally valuable and informative."
        }
        
        base_prompt = domain_prompts.get(domain, f"You are an expert data generator for {domain} domain. Generate high-quality, factual question-answer pairs.")
        
        return f"{base_prompt} Always respond with valid JSON containing 'question' and 'answer' fields."

    def _create_user_prompt(self, chunk: str, domain: str) -> str:
        """Create user prompt for QA generation"""
        return f"""Generate {self.config.max_qa_per_chunk} high-quality question-answer pair(s) from this text about {domain}.

Text: {chunk}

Requirements:
- Questions should be specific and relevant to the content
- Answers should be accurate and informative
- Use natural, conversational language
- Ensure factual accuracy

Output JSON format:
{{
    "question": "What is the main topic?",
    "answer": "The main topic is..."
}}

Or for multiple QA pairs:
{{
    "qa_pairs": [
        {{"question": "Question 1?", "answer": "Answer 1."}},
        {{"question": "Question 2?", "answer": "Answer 2."}}
    ]
}}"""

    def _process_qa_response(self, qa_data: Dict[str, Any], source_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process QA response and add metadata"""
        qa_pairs = []
        
        # Handle single QA pair
        if "question" in qa_data and "answer" in qa_data:
            qa_pair = {
                "question": qa_data["question"],
                "answer": qa_data["answer"]
            }
            qa_pairs.append(qa_pair)
        
        # Handle multiple QA pairs
        elif "qa_pairs" in qa_data:
            for qa_item in qa_data["qa_pairs"]:
                if "question" in qa_item and "answer" in qa_item:
                    qa_pair = {
                        "question": qa_item["question"],
                        "answer": qa_item["answer"]
                    }
                    qa_pairs.append(qa_pair)
        
        # Add metadata to each QA pair
        for qa_pair in qa_pairs:
            if self.config.include_source:
                qa_pair["source"] = source_doc.get("source", "")
                qa_pair["source_type"] = source_doc.get("type", "")
            
            if self.config.include_metadata:
                # Propagate relevant metadata
                metadata_fields = ["url", "title", "author", "date", "domain"]
                for field in metadata_fields:
                    if field in source_doc:
                        qa_pair[f"source_{field}"] = source_doc[field]
                
                # Add generation metadata
                qa_pair["generated_at"] = pd.Timestamp.now().isoformat()
                qa_pair["model"] = self.config.model
                qa_pair["temperature"] = self.config.temperature
        
        return qa_pairs

    def generate_qa_batch(self, documents: List[Dict[str, Any]], domain: str, 
                         text_column: str = "text") -> List[Dict[str, Any]]:
        """Generate QA pairs in batches for better performance"""
        if len(documents) <= self.config.batch_size:
            return self.generate_qa_pairs(documents, domain, text_column)
        
        self.logger.info(f"Batch QA generation: processing {len(documents)} documents in batches of {self.config.batch_size}")
        
        all_qa_pairs = []
        for i in range(0, len(documents), self.config.batch_size):
            batch = documents[i:i + self.config.batch_size]
            self.logger.info(f"Processing batch {i//self.config.batch_size + 1}/{(len(documents) + self.config.batch_size - 1)//self.config.batch_size}")
            
            batch_qa = self.generate_qa_pairs(batch, domain, text_column)
            all_qa_pairs.extend(batch_qa)
            
            # Rate limiting between batches
            time.sleep(1.0)
        
        return all_qa_pairs

    def validate_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate generated QA pairs"""
        if not qa_pairs:
            return {"valid_count": 0, "invalid_count": 0, "total_count": 0}
        
        valid_count = 0
        invalid_count = 0
        
        for qa in qa_pairs:
            if (isinstance(qa.get("question"), str) and 
                isinstance(qa.get("answer"), str) and
                len(qa["question"].strip()) > 5 and
                len(qa["answer"].strip()) > 10):
                valid_count += 1
            else:
                invalid_count += 1
        
        return {
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "total_count": len(qa_pairs),
            "validity_rate": valid_count / len(qa_pairs) if qa_pairs else 0
        }

    def save_qa_pairs(self, qa_pairs: List[Dict[str, Any]], output_path: str) -> None:
        """Save QA pairs to file"""
        if not qa_pairs:
            self.logger.warning("No QA pairs to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Saved {len(qa_pairs)} QA pairs to {output_path}")

    def load_qa_pairs(self, input_path: str) -> List[Dict[str, Any]]:
        """Load QA pairs from file"""
        qa_pairs = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    qa_pairs.append(json.loads(line))
        
        self.logger.info(f"Loaded {len(qa_pairs)} QA pairs from {input_path}")
        return qa_pairs

    def get_qa_stats(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about QA pairs"""
        if not qa_pairs:
            return {"total_count": 0}
        
        # Calculate statistics
        question_lengths = [len(qa.get("question", "")) for qa in qa_pairs]
        answer_lengths = [len(qa.get("answer", "")) for qa in qa_pairs]
        
        # Source distribution
        sources = {}
        for qa in qa_pairs:
            source = qa.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_count": len(qa_pairs),
            "avg_question_length": sum(question_lengths) / len(question_lengths),
            "avg_answer_length": sum(answer_lengths) / len(answer_lengths),
            "min_question_length": min(question_lengths),
            "max_question_length": max(question_lengths),
            "min_answer_length": min(answer_lengths),
            "max_answer_length": max(answer_lengths),
            "source_distribution": sources,
            "unique_sources": len(sources)
        } 