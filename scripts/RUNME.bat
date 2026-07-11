@echo off
REM =====================================================
REM NEURA 300M TOVÁBBFEJLESZTÉS — Gyors üzembe helyezés
REM =====================================================
REM
REM MEGJEGYZÉS: A script-ek futtatásához szükség van a
REM NEURA checkpointra a remote gépen (192.168.0.142).
REM
REM 1. Kapcsold be a remote RTX 3070-et
REM 2. Másold át a scripts mappát a remote-ra
REM 3. Aktiváld a Python környezetet
REM 4. Futtasd a kívánt scriptet
REM
REM =====================================================

REM === 0. OpenSubtitles HU letöltés (egyszer kell) ===
REM curl -L https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/hu.txt.gz -o C:\Users\iga\Desktop\opensubtitles_hu.txt.gz

REM === 1. OpenSubtitles tokenizálás (CPU, ~10 perc) ===
REM python tokenize_opensubs.py

REM === 2. NEURA modell betöltése + 5→2 attention vizsgálat (CPU, ~2 perc) ===
REM python activation_patching.py

REM === 3. LogicAdapter v6 training (GPU, ~7-15 perc) ===
REM python logicadapter_v6.py

REM === 4. ROME edit teszt (CPU, ~1 perc) ===
REM python rome_edit.py --prompt "5 - 2 =" --target "3" --test

REM === 5. ROME edit egyedi hiba javítása (CPU, ~1 perc) ===
REM python rome_edit.py --prompt "5 - 2 =" --target "3" --layer 22

REM === 6. Folytatás a letöltött adaton (ha van checkpoint) ===
REM python logicadapter_v6.py --data combined

echo.
echo ====================================================
echo NEURA 300M fejlesztői scriptek
echo ====================================================
echo.
echo A script-ek elérhetők: C:\Users\iga\Desktop\neuron_modification_book\scripts\
echo.
echo Futtatási sorrend:
echo   1. tokenize_opensubs.py    (OpenSubtitles tokenizálás)
echo   2. activation_patching.py  (Diagnosztika)
echo   3. rome_edit.py --test     (Legjobb paraméterek)
echo   4. logicadapter_v6.py      (Legnagyobb hatás!)
echo.
echo ====================================================
