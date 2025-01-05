from app.src.lang import lang_en, lang_pl
from app.config_handler import ConfigHandler


class LanguageManager:
    def __init__(self) -> None:
        self.config_handler = ConfigHandler()
        self.language = self._load_language()

    def _load_language(self) -> object:
        config = self.config_handler.get_config()
        try:
            if hasattr(config, 'LANGUAGE'):
                language = config.LANGUAGE
            else:
                settings = self.config_handler.config['Settings']
                language = settings.get('LANGUAGE', fallback='en')
            return lang_pl if language == 'pl' else lang_en
        except Exception as e:
            print(f"Error loading language: {e}")
            return lang_en  # Default fallback

    def change_language(self, lang_code: str) -> None:
        """Change language and update config"""
        self.language = lang_pl if lang_code == "pl" else lang_en
        try:
            self.config_handler.config.read(self.config_handler.config_path)
            self.config_handler.config.set('Settings', 'LANGUAGE', lang_code)
            with open(self.config_handler.config_path, 'w') as configfile:
                self.config_handler.config.write(configfile)
            print(f"Language updated to: {lang_code}")
        except Exception as e:
            print(f"Failed to update language in config: {str(e)}")

    def get_current_language_code(self) -> str:
        return 'pl' if self.language == lang_pl else 'en'
