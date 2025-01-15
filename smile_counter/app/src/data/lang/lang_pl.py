# Main menu
TITLE_TEXT = "Licznik Uśmiechów"
VERSION_TEXT = "Wersja: 3.1.0"
START_BUTTON_TEXT = "Start"
OPTIONS_BUTTON_TEXT = "Opcje"
EXIT_BUTTON_TEXT = "Wyjście"
LOGO_NOT_FOUND_TEXT = "[Logo nie znaleziono]"

# Loading messages
OPTIONS_LOADING_TEXT = "Ładowanie opcji, proszę czekać..."
VIDEO_LOADING_TEXT = "Uruchamianie kamery, proszę czekać..."

# Statistics window
STATISTICS_BUTTON_TEXT = "Statystyki"
STATISTICS_TITLE_TEXT = "Statystyki Uśmiechów"
TOTAL_SMILES_TEXT = "Łączna liczba uśmiechów:"
TODAY_SMILES_TEXT = "Uśmiechy dziś:" 
WEEK_SMILES_TEXT = "Uśmiechy w tym tygodniu:"
MONTH_SMILES_TEXT = "Uśmiechy w tym miesiącu:"
YEAR_SMILES_TEXT = "Uśmiechy w tym roku:"

# Options window
OPTIONS_TITLE_TEXT = "Opcje"
FACE_SCALE_FACTOR_TEXT = "Współczynnik skali twarzy:"
FACE_MIN_NEIGHBOURS_TEXT = "Minimalna liczba sąsiadów twarzy:"
SMILE_SCALE_FACTOR_TEXT = "Współczynnik skali uśmiechu:"
SMILE_MIN_NEIGHBOURS_TEXT = "Minimalna liczba sąsiadów uśmiechu:"
COUNTED_SMILE_COOLDOWN_TIME_TEXT = "Czas odnowienia uśmiechu (sekundy):"
TIME_TO_START_COUNTING_TEXT = "Czas do rozpoczęcia liczenia (sekundy):"
SAVE_BUTTON_TEXT = "Zapisz"
OPTIONS_SAVED_TEXT = "Opcje zapisane pomyślnie!"
ERROR_MESSAGE_TEXT = "Błąd zapisywania opcji: {error}"
LANGUAGE_TEXT = "Język:"
DEBUG_MODE_TEXT = "Tryb debugowania"
CAMERA_SOURCE_TEXT = "Źródło kamery:"
APPLY_FACE_EFFECTS_TEXT = "Efekty na twarzy"
AUTO_CONFIG_ADJUSTING_TEXT = "Automatyczne dostosowanie ustawień (eksperymentalne)"

# Tooltips in options window
FACE_SCALE_FACTOR_TOOLTIP = "Współczynnik skalowania dla wykrywania twarzy.\nWiększe wartości wykrywają mniejsze twarze, \
ale zwiększają fałszywe detekcje.\nMa średni wpływ na całościową efektywność detekcji."

FACE_MIN_NEIGHBOURS_TOOLTIP = "Minimalna liczba punktów sąsiednich detekcji wymagana dla wykrycia twarzy.\n\
Wyższe wartości zmniejszają fałszywe detekcje, ale utrudniają rozpoznanie twarzy."

SMILE_SCALE_FACTOR_TOOLTIP = "Współczynnik skalowania dla wykrywania uśmiechu.\nWiększe wartości wykrywają mniejsze uśmiechy, \
ale zwiększają fałszywe detekcje.\nMa duży wpływ na całościową efektywność detekcji."

SMILE_MIN_NEIGHBOURS_TOOLTIP = "Minimalna liczba punktów sąsiednich detekcji wymagana dla wykrycia uśmiechu.\n\
Wyższe wartości zmniejszają fałszywe detekcje, ale znacząco utrudniają wykrycie uśmiechu."

TIME_TO_START_COUNTING_TOOLTIP = "Czas w sekundach przez jaki uśmiech musi być utrzymany zanim zostanie policzony.\n\
Wyższe wartości eliminują sporadyczne fałszywe detekcje w tle, \nale wymagają od użytkownika utrzymania uśmiechu przez dłuższy czas."

COUNTED_SMILE_COOLDOWN_TIME_TOOLTIP = "Czas w sekundach przed możliwością policzenia kolejnego uśmiechu od tej samej osoby.\n\
Zapobiega zliczaniu wielu uśmiechów w krótkim czasie."

CAMERA_SOURCE_TOOLTIP = "Wybierz kamerę do użycia w detekcji.\n\
Domyślna kamera powinna znajdować się w pierwszej dostępnej opcji.\n\
Inne, o ile występują, powinny być dostępne w kolejnych pozycjach w rozwijanej liście."

DEBUG_MODE_TOOLTIP = "Pokaż prostokąty detekcji i dodatkowe informacje debugowania."

APPLY_FACE_EFFECTS_TOOLTIP = "Włącz lub wyłącz zabawne efekty na twarzy jak brody i wąsy."

AUTO_CONFIG_ADJUSTING_TOOLTIP = "Automatycznie dostosuj parametry wykrywania w zależności od wielu czynników."

# Main program
DETECTED_SMILES_TEXT = "Wykryte uśmiechy: {count}"
SMILE_COUNTED_TEXT = "UŚMIECH POLICZONY 😊!"