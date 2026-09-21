@echo off
REM SongLights - one-time setup of the stem-separation environment (Demucs via python-audio-separator).
REM Creates Tools\stems\venv next to this script. Needs internet (~3 GB: CUDA PyTorch + model weights on first bake).
REM Requires either `uv` (https://docs.astral.sh/uv/) or a Python 3.10-3.12 on PATH as `py -3.11`.
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if %errorlevel%==0 (
    uv venv --python 3.11 venv || goto :fail
    uv pip install --python venv\Scripts\python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cu124 || goto :fail
    uv pip install --python venv\Scripts\python.exe "audio-separator[gpu]" || goto :fail
) else (
    py -3.11 -m venv venv || py -3.12 -m venv venv || py -3.10 -m venv venv || goto :fail
    venv\Scripts\python.exe -m pip install --upgrade pip || goto :fail
    venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 || goto :fail
    venv\Scripts\python.exe -m pip install "audio-separator[gpu]" || goto :fail
)
venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
echo.
echo SongLights stems environment ready. The Demucs model downloads automatically on the first Analyze And Bake.
pause
exit /b 0
:fail
echo.
echo Setup FAILED. Install uv (https://docs.astral.sh/uv/) or Python 3.11 and re-run.
pause
exit /b 1
