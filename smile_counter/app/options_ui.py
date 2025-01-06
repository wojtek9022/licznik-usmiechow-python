import tkinter as tk
import tkinter.ttk as ttk
from typing import Dict, Any, Callable
from .options_types import OptionsConfig
from .options_validator import OptionsValidator, ValidationError
from app.src.utils.camera_utils import get_available_cameras

class OptionsUI:
    """
    Manages the options configuration window interface.

    This class handles creation and management of the options window,
    including input validation, error display, and configuration updates.

    Attributes:
        master (tk.Tk): Main application window
        language (object): Current language module with text strings
        validator (OptionsValidator): Options input validator
        window (Optional[tk.Toplevel]): Options window instance
        entries (Dict[str, tk.Entry]): Option input fields
        labels (Dict[str, tk.Label]): Option labels
        error_labels (Dict[str, tk.Label]): Validation error messages
    """

    def __init__(self, master: tk.Tk, language: object, validator: OptionsValidator):
        self.master = master
        self.language = language
        self.validator = validator
        self.window: tk.Toplevel | None = None
        self.entries: Dict[str, tk.Entry] = {}
        self.labels: Dict[str, tk.Label] = {}
        self.error_labels: Dict[str, tk.Label] = {}
        self.vars: Dict[str, tk.BooleanVar] = {}  # Store BooleanVar objects

    def on_language_change(self, language: object) -> None:
        """Handle language change event."""
        self.language = language
        if self.window and tk.Toplevel.winfo_exists(self.window):
            self.window.title(language.OPTIONS_TITLE_TEXT)
            for option_name, label in self.labels.items():
                label_text = getattr(language, f"{option_name}_TEXT")
                label.config(text=label_text)
            if hasattr(self, "save_button"):
                self.save_button.config(text=language.SAVE_BUTTON_TEXT)

    def create_window(self, config_options: OptionsConfig, 
                     values: Dict[str, Any], 
                     save_callback: Callable) -> None:
        """
        Create and display the options configuration window.

        Creates a new window or destroys existing one if present.
        Sets up input fields for all configuration options and
        adds a save button with the provided callback.

        Args:
            config_options (OptionsConfig): Configuration options schema
            values (Dict[str, Any]): Current option values
            save_callback (Callable): Function to call when saving
        """
        if self.window and tk.Toplevel.winfo_exists(self.window):
            self.window.destroy()
            
        self.window = tk.Toplevel(self.master)
        self.window.title(self.language.OPTIONS_TITLE_TEXT)
        self._create_entries(config_options, values)
        self._create_save_button(save_callback)

    def _create_entries(self, config_options: OptionsConfig, values: Dict[str, Any]) -> None:
        for option_name, options in config_options.items():
            label_text = getattr(self.language, f'{option_name}_TEXT')
            label, entry = self._create_entry(label_text, values[option_name], options['row'], option_name)
            self.labels[option_name] = label
            self.entries[option_name] = entry

    def _create_entry(self, label_text: str, value: Any, row: int, option_name: str) -> tuple:
        label = tk.Label(self.window, text=label_text)
        label.grid(row=row, column=0, padx=10, pady=5)

        if option_name == 'CAMERA_SOURCE':
            cameras = get_available_cameras()
            
            combo = ttk.Combobox(self.window, width=30, state="readonly")
            combo['values'] = [name for _, name in cameras]
            
            # Find current camera index
            current_idx = 0
            for idx, (cam_idx, _) in enumerate(cameras):
                if cam_idx == int(value):
                    current_idx = idx
            combo.current(current_idx)
            
            # Store mapping for retrieving camera index
            combo.camera_indices = {name: idx for idx, name in cameras}
            entry = combo
        elif option_name == 'DEBUG_MODE':
            var = tk.BooleanVar(value=value)  # Convert to bool
            self.vars[option_name] = var  # Store var reference
            entry = tk.Checkbutton(
                self.window,
                variable=var,
                onvalue=True,
                offvalue=False
            )
        else:
            entry = tk.Entry(self.window)
            entry.insert(0, str(value))
        
        entry.grid(row=row, column=1, padx=10, pady=5)
        self.entries[option_name] = entry
        
        error_label = tk.Label(self.window, text="", fg="red")
        error_label.grid(row=row, column=2, padx=10, pady=5)
        self.error_labels[option_name] = error_label
        
        entry.bind('<FocusOut>', lambda e: self._validate_entry(option_name))
        
        return label, entry

    def _create_save_button(self, save_callback: Callable) -> None:
        self.save_button = tk.Button(
            self.window,
            text=self.language.SAVE_BUTTON_TEXT,
            command=save_callback
        )
        self.save_button.grid(row=len(self.entries), columnspan=2, padx=10, pady=10)

    def _validate_entry(self, option_name: str) -> bool:
        try:
            value = self.get_value(option_name)
            self.validator.validate_option(option_name, value)
            self.error_labels[option_name].config(text="")
            
            # Use ttk style for validation
            if isinstance(self.entries[option_name], ttk.Combobox):
                self.entries[option_name].state(['!invalid'])
            else:
                self.entries[option_name].config(bg='white')
                
            return True
            
        except ValidationError as e:
            self.error_labels[option_name].config(text=str(e))
            
            # Use ttk style for validation
            if isinstance(self.entries[option_name], ttk.Combobox):
                self.entries[option_name].state(['invalid'])
            else:
                self.entries[option_name].config(bg='pink')
                
            return False

    def get_value(self, option_name: str) -> Any:
        entry = self.entries[option_name]
        if option_name == 'CAMERA_SOURCE':
            return entry.camera_indices[entry.get()]
        elif option_name == 'DEBUG_MODE':
            return bool(self.vars[option_name].get())
        return entry.get()