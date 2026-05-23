@echo off
setlocal

:: ─────────────────────────────────────────────
::  CONFIGURAÇÃO — ajusta estes caminhos se necessário
:: ─────────────────────────────────────────────
set INPUT_DIR=D:\_PIA_CJR\dataset\raw\2026_05_08_VALPACOS\visible
set OUTPUT_PANELS=D:\_PIA_CJR\outputs\_CB\_solar_anomaly_detector\panels
set OUTPUT_RESULTS=D:\_PIA_CJR\outputs\_CB\_solar_anomaly_detector\results
set OUTPUT_JSON=D:\partilha\coding\_PIA\PIA-CJR\coding\_CB\scripts\solar_anomaly_detector
set SCRIPT_DIR=%~dp0

:: Epochs de treino (aumenta com mais imagens — ex: 100 para >200 painéis)
set EPOCHS=50

:: ─────────────────────────────────────────────
::  CRIA DIRECTORIAS SE NÃO EXISTIREM
:: ─────────────────────────────────────────────
echo.
echo [1/4] A criar directorias de output...
if not exist "%OUTPUT_PANELS%" mkdir "%OUTPUT_PANELS%"
if not exist "%OUTPUT_RESULTS%" mkdir "%OUTPUT_RESULTS%"
if not exist "%OUTPUT_JSON%" mkdir "%OUTPUT_JSON%"

:: ─────────────────────────────────────────────
::  PASSO 1 — EXTRAÇÃO DE PAINÉIS
:: ─────────────────────────────────────────────
echo.
echo [2/4] A extrair paineis das imagens drone...
echo       Input:  %INPUT_DIR%
echo       Output: %OUTPUT_PANELS%
echo.
python "%SCRIPT_DIR%solar_panel_extractor.py" ^
    --input "%INPUT_DIR%" ^
    --output "%OUTPUT_PANELS%"

if errorlevel 1 (
    echo ERRO na extracao de paineis. Verifica o script e tenta de novo.
    pause
    exit /b 1
)

:: ─────────────────────────────────────────────
::  PASSO 2 — DETEÇÃO DE ANOMALIAS
:: ─────────────────────────────────────────────
echo.
echo [3/4] A treinar modelo e detectar anomalias...
echo       Paineis: %OUTPUT_PANELS%
echo       Output:  %OUTPUT_RESULTS%
echo.
python "%SCRIPT_DIR%solar_anomaly_detector.py" ^
    --panels "%OUTPUT_PANELS%" ^
    --output "%OUTPUT_RESULTS%" ^
    --epochs %EPOCHS%

if errorlevel 1 (
    echo ERRO na detecao de anomalias.
    pause
    exit /b 1
)

:: ─────────────────────────────────────────────
::  PASSO 3 — COPIA anomaly_scores.json
:: ─────────────────────────────────────────────
echo.
echo [4/4] A copiar anomaly_scores.json...
copy /Y "%OUTPUT_RESULTS%\anomaly_scores.json" "%OUTPUT_JSON%\anomaly_scores.json"

if errorlevel 1 (
    echo AVISO: Nao foi possivel copiar o JSON. Verifica o caminho de destino.
) else (
    echo       Copiado para: %OUTPUT_JSON%\anomaly_scores.json
)

:: ─────────────────────────────────────────────
::  SUMÁRIO
:: ─────────────────────────────────────────────
echo.
echo =====================================================
echo  PIPELINE CONCLUIDO
echo =====================================================
echo  Paineis extraidos:  %OUTPUT_PANELS%
echo  Relatorios visuais: %OUTPUT_RESULTS%
echo    - report_top_anomalies.png
echo    - report_overview.png
echo    - training_loss.png
echo  Scores JSON:        %OUTPUT_JSON%\anomaly_scores.json
echo  Modelo treinado:    %OUTPUT_RESULTS%\ae_model.pth
echo =====================================================
echo.
echo Proxima vez que correres novas imagens, podes reutilizar
echo o modelo ja treinado com o flag --no-train:
echo.
echo   python solar_anomaly_detector.py ^
echo     --panels PASTA_NOVOS_PAINEIS ^
echo     --output PASTA_OUTPUT ^
echo     --model "%OUTPUT_RESULTS%\ae_model.pth" ^
echo     --no-train
echo.
pause
