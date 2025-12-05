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
        Extract Entities and Events using LLM with strict JSON output.
        Returns a dict with 'entities' and 'events' lists.
        """
        if not llm_client:
            logger.warning("No LLM client provided for metaphysical extraction")
            return {"entities": [], "events": []}

        # CRITICAL: Strict JSON-only system prompt
        system_prompt = """You are a JSON extraction engine. You MUST respond with ONLY valid JSON.
Do not include markdown, explanations, code blocks, or any text outside the JSON object.
Do not wrap JSON in ```json or ``` blocks.
Return empty arrays if no entities/events found.

Your ONLY output should be valid JSON matching this exact structure:
{
  "entities": [{"name": "string", "subtype": "string", "description": "string"}],
  "events": [{"name": "string", "subtype": "string", "description": "string", "caused_by": ["string"], "next_event": "string or null"}]
}"""

        # Truncate text to avoid token limits in extraction
        truncated_text = text[:2000] if len(text) > 2000 else text

        user_prompt = f"""Extract entities (nouns/objects) and events (actions/occurrences) from this text.

Domain: {context_domain}

Text:
{truncated_text}

Return ONLY the JSON object (no markdown, no explanation):"""

        try:
            response = await llm_client.generate(
                context=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Lower temperature for consistency
                max_tokens=4096
            )

            logger.info(f"LLM extraction response (first 200 chars): {response[:200]}")

            import json
            # Parse JSON with robust fallbacks
            data = self._parse_json_response(response)

            # Validate and normalize structure
            data = self._validate_extraction(data)

            logger.info(f"Extraction succeeded: {len(data.get('entities', []))} entities, {len(data.get('events', []))} events")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in metaphysical extraction: {e}")
            logger.error(f"Raw LLM response (first 500 chars): {response[:500] if 'response' in locals() else 'N/A'}")
            return {"entities": [], "events": []}
        except Exception as e:
            logger.error(f"Error in metaphysical extraction: {e}")
            if 'response' in locals():
                logger.error(f"Raw LLM response (first 500 chars): {response[:500]}")
            return {"entities": [], "events": []}

    def _parse_json_response(self, response: str) -> Dict:
        """
        Parse JSON from LLM response with multiple fallback strategies.
        """
        import json
        response = response.strip()

        # Strategy 1: Try raw JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Remove markdown code blocks (```json ... ```)
        if "```" in response:
            # Remove opening ```json or ```
            cleaned = re.sub(r'^```(?:json)?\s*', '', response, flags=re.MULTILINE)
            # Remove closing ```
            cleaned = re.sub(r'\s*```\s*$', '', cleaned, flags=re.MULTILINE)
            try:
                return json.loads(cleaned.strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Extract first {...} block
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 4: Check if response is markdown/text (not JSON)
        if response.startswith("#") or response.startswith("**") or "analysis" in response.lower()[:100]:
            logger.warning("LLM returned analysis/markdown instead of JSON, returning empty extraction")
            return {"entities": [], "events": []}

        # Strategy 5: Last resort - log full response and return empty
        logger.error(f"Could not parse JSON from response after all strategies")
        logger.error(f"Full response: {response[:1000]}")
        return {"entities": [], "events": []}

    def _validate_extraction(self, data: Dict) -> Dict:
        """
        Validate and normalize extraction structure.
        Ensures all required fields exist with correct types.
        """
        # Ensure required keys exist
        if "entities" not in data:
            data["entities"] = []
        if "events" not in data:
            data["events"] = []

        # Validate entities
        valid_entities = []
        for entity in data.get("entities", []):
            if isinstance(entity, dict) and "name" in entity:
                entity.setdefault("subtype", "Entity")
                entity.setdefault("description", "")
                # Ensure all values are strings
                entity["name"] = str(entity["name"])
                entity["subtype"] = str(entity["subtype"])
                entity["description"] = str(entity["description"])
                valid_entities.append(entity)
        data["entities"] = valid_entities

        # Validate events
        valid_events = []
        for event in data.get("events", []):
            if isinstance(event, dict) and "name" in event:
                event.setdefault("subtype", "Event")
                event.setdefault("description", "")
                event.setdefault("caused_by", [])
                event.setdefault("next_event", None)
                # Ensure correct types
                event["name"] = str(event["name"])
                event["subtype"] = str(event["subtype"])
                event["description"] = str(event["description"])
                if not isinstance(event["caused_by"], list):
                    event["caused_by"] = []
                valid_events.append(event)
        data["events"] = valid_events

        return data

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
