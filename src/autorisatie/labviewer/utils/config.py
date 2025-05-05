import json
from pathlib import Path

def load_test_config():
    """
    Load test configuration from tests.json
    
    Returns:
        tuple: (included_tests, test_display_names)
            - included_tests: List of test IDs that have include=True
            - test_display_names: Dictionary mapping test IDs to display names with units
    """
    config_path = Path(__file__).parent.parent.parent.parent.parent / 'config' / 'tests.json'
    with open(config_path, 'r') as f:
        test_config = json.load(f)
    
    # Get included tests and their display names
    included_tests = []
    test_display_names = {}
    for test_id, test_info in test_config.items():
        if test_info.get('include', False):
            included_tests.append(test_id)
            test_display_names[test_id] = f"{test_info['name']} ({test_info['unit']})"
    
    return included_tests, test_display_names 