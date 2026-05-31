@echo off
setlocal

title PIA-CJR - Pipeline Completo

echo.
echo ==========================================
echo    PIA-CJR - PIPELINE COMPLETO
echo ==========================================
echo.

REM CORRER A PARTIR DAQUI, DESTA FOREMA:
REM     (DL_Project_venv) PS D:\partilha\coding\_PIA\PIA-CJR\coding\_CB> .\scripts\_run_full_pipeline.bat

REM --------------------------------------------------
REM FASE 2 - AUGMENT DATASET
REM --------------------------------------------------

echo.
echo ==========================================
echo FASE 2 - AUGMENT DATASET
echo ==========================================
echo.

py .\scripts\augment_roboflow_dataset.py

if errorlevel 1 (
    echo.
    echo [ERRO] Falha na FASE 2.
    pause
    exit /b 1
)

REM --------------------------------------------------
REM FASE 3 - TREINO YOLO
REM --------------------------------------------------

echo.
echo ==========================================
echo FASE 3 - TREINO YOLO
echo ==========================================
echo.

python .\scripts\roboflow\train.py

if errorlevel 1 (
    echo.
    echo [ERRO] Falha na FASE 3.
    pause
    exit /b 1
)

REM --------------------------------------------------
REM FASE 4 - TREINO CNN (ELPV)
REM --------------------------------------------------

echo.
echo ==========================================
echo FASE 4 - TREINO CNN (ELPV)
echo ==========================================
echo.

python .\scripts\train.py

if errorlevel 1 (
    echo.
    echo [ERRO] Falha na FASE 4.
    pause
    exit /b 1
)

REM --------------------------------------------------
REM FASE 5 - GENERATE RECTIFIED PANELS
REM --------------------------------------------------

echo.
echo ==========================================
echo FASE 5 - GENERATE RECTIFIED PANELS
echo ==========================================
echo.

py .\scripts\roboflow\generate_rectified_panels.py

if errorlevel 1 (
    echo.
    echo [ERRO] Falha na FASE 5.
    pause
    exit /b 1
)

REM --------------------------------------------------
REM FASE 6 - EVALUATE
REM --------------------------------------------------

echo.
echo ==========================================
echo FASE 6 - EVALUATE
echo ==========================================
echo.

python .\scripts\evaluate.py

if errorlevel 1 (
    echo.
    echo [ERRO] Falha na FASE 6.
    pause
    exit /b 1
)

REM --------------------------------------------------
REM FASE 7 - PREDICT
REM --------------------------------------------------

echo.
echo ==========================================
echo FASE 7 - PREDICT
echo ==========================================
echo.

python .\scripts\predict.py

if errorlevel 1 (
    echo.
    echo [ERRO] Falha na FASE 7.
    pause
    exit /b 1
)

REM --------------------------------------------------
REM FIM
REM --------------------------------------------------

echo.
echo ==========================================
echo PIPELINE TERMINADO COM SUCESSO
echo ==========================================
echo.

pause