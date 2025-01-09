from typing import Any, Dict, Union, Tuple
from dataclasses import dataclass

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

@dataclass
class ValidationRule:
    """
    Data class defining validation rules for configuration options.

    Stores validation parameters for numeric configuration values including
    allowed ranges, type constraints and error messages.

    Attributes:
        min_value (float): Minimum allowed value
        max_value (float): Maximum allowed value
        type (type): Expected value type (int/float)
        required (bool): Whether the option is required
        error_message (str): Custom error message for validation failures
    """
    min_value: float
    max_value: float
    type: type
    required: bool = True
    error_message: str = ""

class OptionsValidator:
    """
    Validates configuration options against defined rules.

    Handles validation of all configuration options according to predefined
    rules including type checking, range validation and required field checking.

    Attributes:
        validation_rules (Dict[str, ValidationRule]): Dictionary mapping option
            names to their validation rules
    """

    def __init__(self):
        # FIXME: Refactor this class and import numbers from config 
        # instead of using magic numbers.
        self.validation_rules = {
            'FACE_SCALE_FACTOR': ValidationRule(
                min_value=1.01,
                max_value=20.0,
                type=float,
                error_message="Face scale factor must be between 1.01 and 20.0"
            ),
            'FACE_MIN_NEIGHBOURS': ValidationRule(
                min_value=1,
                max_value=1000,
                type=int,
                error_message="Face minimum neighbours must be between 1 and 1000"
            ),
            'SMILE_SCALE_FACTOR': ValidationRule(
                min_value=1.01,
                max_value=20.0,
                type=float,
                error_message="Smile scale factor must be between 1.01 and 20.0"
            ),
            'SMILE_MIN_NEIGHBOURS': ValidationRule(
                min_value=1,
                max_value=200,
                type=int,
                error_message="Smile minimum neighbours must be between 1 and 200"
            ),
            'TIME_TO_START_COUNTING': ValidationRule(
                min_value=0.01,
                max_value=10.0,
                type=float,
                error_message="Time must be between 0.01 and 10.0 seconds"
            ),
            'DEBUG_MODE': ValidationRule(
                min_value=0,
                max_value=1,
                type=bool,
                error_message="Debug mode must be True or False"
            ),
            'CAMERA_SOURCE': ValidationRule(
                min_value=0,
                max_value=10,
                type=int,
                error_message="Invalid camera source"
            )
        }

    def validate_value(self, value: Any, option_type: type) -> bool:
        """Validate a value against its expected type."""
        if option_type == bool:
            return isinstance(value, bool)
        try:
            option_type(value)
            return True
        except (ValueError, TypeError):
            return False

    def validate_option(self, option_name: str, value: Any) -> Tuple[bool, str]:
        """Validate a specific option value against its rules."""
        if option_name not in self.validation_rules:
            return True, ""
            
        rule = self.validation_rules[option_name]
        
        if rule.type == bool:
            return True, ""
            
        try:
            numeric_value = rule.type(value)
            if numeric_value < rule.min_value or numeric_value > rule.max_value:
                return False, rule.error_message
            return True, ""
        except ValueError:
            return False, f"Invalid value for {option_name}"

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
        """
        Validate entire configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration dictionary to validate

        Returns:
            Tuple[bool, Dict[str, str]]: Validation result and dictionary of
                error messages keyed by option name
        """
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