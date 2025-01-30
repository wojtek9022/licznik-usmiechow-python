# licznik-usmiechow-python
Refaktoryzacja licznika uśmiechów w języku Python przy użyciu OpenCV

# smile-counter-python
Refactoring the smile counter in Python using OpenCV

.
.
.

# INSTRUKCJA URUCHOMIENIA

1. Nalezy zainstalowac Python w wersji 3.12.6 lub nowszej
    Jeżeli istnieje już plik .exe należy go uruchomić (najlepiej z uprawnieniami administratora jeśli to możliwe)

2. W przypadku chęci uruchomienia programu z poziomu dewelopera, w folderze głownym z kodem źródłowym należ otworzyć wiersz polecenia i wykonac
    pip install -r requirements.txt

3. Po zakonczeniu należy wejść w folder smile_counter i przez wiersz polecenia wykonać:
    python smile_counter_app.py

W tym momencie aplikacja powinna się uruchomić, jednocześnie można też edytować kod źrodłowy programu np. w zintegrowanych środowiskach programistycznych (IDE).

# INSTRUKCJA UTWORZENIA PLIKU .EXE

1. Należy wykonać wszystkie poprzednie kroku z instrukcji uruchomienia, jeśli nie było to zrozbione wcześniej.

2. Należy wejść do folderu /smile_counter/app/src/utils/scripts i przez wiersz polecenia wykonać:
    python build_exe.py

Informacja: na czas wykonania skryptu, można wyłączyć ochronę przed detekcją wirusów jeśli wystąpi błąd dostępu. Używany pakiet pyinstaller jest często zaliczany jako biblioteka do tworzenia wirusów i domyślnie blokowana przez programy antywirusowe.


# INSTRUKCJA UTWORZENIA NOWEGO PLIKU REQUIREMENTS.TXT

1. W katalogu głównym programu należy w wierszu poleceń wykonać:
    pip freeze > requirements.txt

#
#
#


# STARTUP INSTRUCTIONS

1. Python version 3.12.6 or later must be installed

If there is already an .exe file, run it (preferably with administrator privileges if possible)

2. If you want to run the program from the developer level, in the main folder with the source code, open the command line and execute
pip install -r requirements.txt

3. After finishing, go to the smile_counter folder and through the command line execute:
python smile_counter_app.py

At this point, the application should start, at the same time you can also edit the program's source code, e.g. in integrated development environments (IDEs).

# INSTRUCTIONS FOR CREATING THE .EXE FILE

1. Perform all the previous steps from the start-up instructions, if this was not understood earlier.

2. Go to the /smile_counter/app/src/utils/scripts folder and execute the following command line command:
python build_exe.py

Note: for the duration of the script execution, you can disable virus detection protection if an access error occurs. The pyinstaller package used is often classified as a virus library and is blocked by default by antivirus programs.

# INSTRUCTIONS FOR CREATING A NEW REQUIREMENTS.TXT FILE

1. In the program's main directory, execute the following command line command:
pip freeze > requirements.txt

---
---