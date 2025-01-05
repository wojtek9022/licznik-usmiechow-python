from typing import Any, Dict, Union, Tuple
from dataclasses import dataclass

@dataclass
class ValidationRule:
    min_value: float
    max_value: float
    type: type
    required: bool = True
    error_message: str = ""

class ConfigValidator:
    def __init__(self):
        self.validation_rules = {
            'FACE_SCALE_FACTOR': ValidationRule(
                min_value=1.01,
                max_value=2.0,
                type=float,
                error_message="Face scale factor must be between 1.01 and 2.0"
            ),
            'FACE_MIN_NEIGHBOURS': ValidationRule(
                min_value=1,
                max_value=20,
                type=int,
                error_message="Face minimum neighbours must be between 1 and 20"
            ),
            'SMILE_SCALE_FACTOR': ValidationRule(
                min_value=1.01,
                max_value=10.0,
                type=float,
                error_message="Smile scale factor must be between 1.01 and 10.0"
            ),
            'SMILE_MIN_NEIGHBOURS': ValidationRule(
                min_value=1,
                max_value=200,
                type=int,
                error_message="Smile minimum neighbours must be between 1 and 200"
            ),
            'TIME_TO_START_COUNTING': ValidationRule(
                min_value=0.1,
                max_value=10.0,
                type=float,
                error_message="Time must be between 0.1 and 10.0 seconds"
            )
        }

    def validate_option(self, option_name: str, value: Any) -> Tuple[bool, str]:
        """Validate single option value"""
        if option_name not in self.validation_rules:
            return True, ""
            
        rule = self.validation_rules[option_name]
        try:
            numeric_value = float(value)
            if numeric_value < rule.min_value or numeric_value > rule.max_value:
                return False, rule.error_message
            return True, ""
        except ValueError:
            return False, f"Invalid value for {option_name}: must be a number"

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
        """Validate entire configuration"""
        errors = {}
        for option_name, value in config.items():
            is_valid, error = self.validate_option(option_name, value)
            if not is_valid:
                errors[option_name] = error
        return len(errors) == 0, errors

    def validate_type(self, value: Any, expected_type: type) -> Tuple[bool, str]:
        """Validate value type"""
        try:
            expected_type(value)
            return True, ""
        except ValueError:
            return False, f"Invalid type: expected {expected_type.__name__}"