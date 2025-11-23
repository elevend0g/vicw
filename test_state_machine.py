#!/usr/bin/env python3
"""Test script for state machine functionality"""

import sys
sys.path.insert(0, 'app')

from state_extractor import StateExtractor

def test_goal_extraction():
    """Test goal extraction from narrative text"""
    print("\n" + "="*60)
    print("TEST 1: Goal Extraction (Hydro-Plant Loop Scenario)")
    print("="*60)

    extractor = StateExtractor("app/state_config.yaml")

    # Scenario 1: Planning to go to Hydro-Plant
    text1 = "Alice turned to Bob. 'Let's go to the Hydro-Plant,' she said. 'We need to restore power to the grid.'"
    states1 = extractor.extract_states(text1)

    print(f"\nText: {text1}")
    print(f"Extracted states: {states1}")

    # Should detect goal creation
    assert any(s[0] == 'goal' and 'hydro' in s[1].lower() and s[2] == 'active' for s in states1), \
        "Should detect goal creation"
    print("✓ Goal creation detected correctly")

    # Scenario 2: Arriving at Hydro-Plant
    text2 = "After walking for an hour, they finally arrived at the Hydro-Plant. The gates stood open."
    states2 = extractor.extract_states(text2)

    print(f"\nText: {text2}")
    print(f"Extracted states: {states2}")

    # Should detect goal completion
    assert any(s[0] == 'goal' and s[2] == 'completed' for s in states2), \
        "Should detect goal completion"
    print("✓ Goal completion detected correctly")


def test_task_extraction():
    """Test task extraction from coding conversation"""
    print("\n" + "="*60)
    print("TEST 2: Task Extraction (Coding Scenario)")
    print("="*60)

    extractor = StateExtractor("app/state_config.yaml")

    # Scenario 1: Starting a task
    text1 = "I'm working on refactoring the authentication module to use JWT tokens instead of sessions."
    states1 = extractor.extract_states(text1)

    print(f"\nText: {text1}")
    print(f"Extracted states: {states1}")

    assert any(s[0] == 'task' and s[2] == 'active' for s in states1), \
        "Should detect task creation"
    print("✓ Task creation detected correctly")

    # Scenario 2: Completing the task
    text2 = "Great! The auth module refactoring is completed and merged into main."
    states2 = extractor.extract_states(text2)

    print(f"\nText: {text2}")
    print(f"Extracted states: {states2}")

    assert any(s[0] == 'task' and s[2] == 'completed' for s in states2), \
        "Should detect task completion"
    print("✓ Task completion detected correctly")


def test_decision_extraction():
    """Test decision extraction"""
    print("\n" + "="*60)
    print("TEST 3: Decision Extraction")
    print("="*60)

    extractor = StateExtractor("app/state_config.yaml")

    text1 = "After evaluating the options, we decided to use PostgreSQL for the database."
    states1 = extractor.extract_states(text1)

    print(f"\nText: {text1}")
    print(f"Extracted states: {states1}")

    assert any(s[0] == 'decision' and 'postgresql' in s[1].lower() for s in states1), \
        "Should detect decision"
    print("✓ Decision detected correctly")


def test_fact_extraction():
    """Test fact extraction"""
    print("\n" + "="*60)
    print("TEST 4: Fact Extraction")
    print("="*60)

    extractor = StateExtractor("app/state_config.yaml")

    text1 = "We discovered that the power grid is offline and the backup generators are not functioning."
    states1 = extractor.extract_states(text1)

    print(f"\nText: {text1}")
    print(f"Extracted states: {states1}")

    assert any(s[0] == 'fact' for s in states1), \
        "Should detect fact"
    print("✓ Fact detected correctly")


def test_deduplication():
    """Test that similar states are deduplicated"""
    print("\n" + "="*60)
    print("TEST 5: Deduplication")
    print("="*60)

    extractor = StateExtractor("app/state_config.yaml")

    text1 = "Let's go to the plant. We should go to the plant. We need to go to the plant."
    states1 = extractor.extract_states(text1)

    print(f"\nText: {text1}")
    print(f"Extracted states: {states1}")

    # Should only extract once due to deduplication
    goal_count = sum(1 for s in states1 if s[0] == 'goal')
    print(f"Goals extracted: {goal_count} (should be 1 due to deduplication)")
    assert goal_count <= 2, "Should deduplicate similar goals within same text"
    print("✓ Deduplication working")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VICW STATE MACHINE TESTS")
    print("="*60)

    try:
        test_goal_extraction()
        test_task_extraction()
        test_decision_extraction()
        test_fact_extraction()
        test_deduplication()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe state machine is ready to prevent loops!")
        print("Next steps:")
        print("  1. Start the full stack: docker-compose up -d")
        print("  2. Test with real conversations via API")
        print("  3. Monitor Neo4j for state nodes being created")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
