from app.src.lang import lang_en, lang_pl
from app.config_manager import ConfigManager

class LanguageManager:
    def __init__(self) -> None:
        self.config_manager = ConfigManager()
        self.language = self._load_language()

    def _load_language(self) -> object:
        try:
            from app.src.config import LANGUAGE
            if LANGUAGE == 'pl':
                return lang_pl
        except ImportError:
            pass
        return lang_en

    def get_current_language_code(self) -> str:
        return 'pl' if self.language == lang_pl else 'en'

    def change_language(self, lang_code: str) -> None:
        self.language = lang_pl if lang_code == "pl" else lang_en
        self.config_manager.update_config({"LANGUAGE": lang_code})