from app.src.lang import lang_en, lang_pl
from app.config_manager import ConfigManager

class LanguageManager:
    def __init__(self) -> None:
        self.config_manager = ConfigManager()
        self.language = self._load_language()

    def _load_language(self) -> object:
        config = self.config_manager.get_config()
        if config.LANGUAGE == 'pl':
            return lang_pl
        return lang_en

    def change_language(self, lang_code: str) -> None:
        """Change language and update config"""
        self.language = lang_pl if lang_code == "pl" else lang_en
        try:
            self.config_manager.update_config({"LANGUAGE": lang_code})
            print(f"Language updated to: {lang_code}")
        except Exception as e:
            print(f"Failed to update language in config: {str(e)}")

    def get_current_language_code(self) -> str:
        return 'pl' if self.language == lang_pl else 'en'
