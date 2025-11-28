"""State extraction from conversation text using pattern matching"""

import re
import logging
import yaml
from typing import List, Tuple, Dict, Any, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class StateExtractor:
    """Extract state changes from text using configurable patterns"""

    def __init__(self, config_path: str = "app/state_config.yaml"):
        self.config_path = config_path
        self.patterns: Dict[str, Dict[str, List[str]]] = {}
        self._load_config()

    def _load_config(self):
        """Load pattern configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"State config not found: {self.config_path}, using empty patterns")
                return

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            self.patterns = config.get('state_types', {})
            logger.info(f"Loaded {len(self.patterns)} state types from {self.config_path}")

        except Exception as e:
            logger.error(f"Error loading state config: {e}")
            self.patterns = {}

    def extract_states(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract states from text using pattern matching.

        Returns:
            List of (state_type, description, inferred_status) tuples
        """
        if not text or not self.patterns:
            return []

        text_lower = text.lower()
        sentences = self._split_sentences(text)
        extracted_states = []
        seen_descriptions: Set[str] = set()  # Simple deduplication

        # Process each sentence
        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Check each state type
            for state_type, pattern_groups in self.patterns.items():
                # Check for completion/invalidation patterns first (higher priority)
                if 'complete' in pattern_groups:
                    for pattern in pattern_groups['complete']:
                        if pattern in sentence_lower:
                            desc = self._extract_description(sentence, pattern, state_type)
                            if desc and desc not in seen_descriptions:
                                extracted_states.append((state_type, desc, 'completed'))
                                seen_descriptions.add(desc)
                                break

                if 'invalidate' in pattern_groups:
                    for pattern in pattern_groups['invalidate']:
                        if pattern in sentence_lower:
                            desc = self._extract_description(sentence, pattern, state_type)
                            if desc and desc not in seen_descriptions:
                                extracted_states.append((state_type, desc, 'invalid'))
                                seen_descriptions.add(desc)
                                break

                # Check for creation patterns
                if 'create' in pattern_groups:
                    for pattern in pattern_groups['create']:
                        if pattern in sentence_lower:
                            desc = self._extract_description(sentence, pattern, state_type)
                            if desc and desc not in seen_descriptions:
                                extracted_states.append((state_type, desc, 'active'))
                                seen_descriptions.add(desc)
                                break

        if extracted_states:
            logger.debug(f"Extracted {len(extracted_states)} states from text")

        return extracted_states

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple approach)"""
        # Simple sentence splitting on common punctuation
        sentences = re.split(r'[.!?]\s+', text)
        # Clean up and filter empty
        return [s.strip() for s in sentences if s.strip()]

    def _extract_description(self, sentence: str, trigger_pattern: str, state_type: str) -> str:
        """
        Extract the meaningful description from a sentence.

        This is a simple heuristic that tries to capture the action/goal/fact
        mentioned after the trigger pattern.
        """
        sentence_lower = sentence.lower()

        # Find where the pattern occurs
        pattern_idx = sentence_lower.find(trigger_pattern)
        if pattern_idx == -1:
            return ""

        # Extract text after the pattern
        after_pattern = sentence[pattern_idx + len(trigger_pattern):].strip()

        # Remove common prefixes/connectors
        after_pattern = re.sub(r'^(to|that|the|a|an)\s+', '', after_pattern, flags=re.IGNORECASE)

        # Take up to the first punctuation or max length
        description = re.split(r'[,;.!?]', after_pattern)[0].strip()

        # Normalize: lowercase, remove extra spaces
        description = ' '.join(description.split()).lower()

        # Length limits: minimum 3 chars, maximum 100 chars
        if len(description) < 3 or len(description) > 100:
            return ""

        # Filter out common false positives
        skip_words = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'we', 'i', 'you'}
        if description in skip_words:
            return ""

        return description

    async def extract_metaphysical_graph(self, text: str, context_domain: str, llm_client: Any = None) -> Dict[str, Any]:
        """
        Extract Entities and Events using LLM with SPO prompting.
        Returns a dict with 'entities' and 'events' lists.
        """
        if not llm_client:
            logger.warning("No LLM client provided for metaphysical extraction")
            return {"entities": [], "events": []}

        system_prompt = """You are a Knowledge Graph Extractor.
        Extract 'Entities' (nouns/objects) and 'Events' (actions/occurrences) from the text.
        
        Output JSON format:
        {
            "entities": [
                {"name": "EntityName", "subtype": "type", "description": "brief desc"}
            ],
            "events": [
                {"name": "EventName", "subtype": "type", "description": "brief desc", "caused_by": ["EntityName"], "next_event": "EventName"}
            ]
        }
        """

        user_prompt = f"""
        Domain: {context_domain}
        Text:
        {text}
        """

        try:
            response = await llm_client.generate(
                context=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            import json
            # Clean up response (remove markdown code blocks if present)
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            data = json.loads(cleaned_response.strip())
            return data
            
        except Exception as e:
            logger.error(f"Error in metaphysical extraction: {e}")
            return {"entities": [], "events": []}

    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()


# Global instance
_extractor_instance = None


def get_extractor(config_path: str = "app/state_config.yaml") -> StateExtractor:
    """Get or create the global state extractor instance"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = StateExtractor(config_path)
    return _extractor_instance
