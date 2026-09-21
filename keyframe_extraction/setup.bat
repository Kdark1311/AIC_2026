@echo off
REM Cài môi trường trên Windows (chạy 1 lần): tạo venv .venv\ và cài torch (CUDA) + OmniShotCut.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    python -m venv .venv || goto :err
)
.venv\Scripts\python -m pip install --upgrade pip || goto :err
REM Torch trên PyPI cho Windows là bản CPU, nên lấy bản CUDA từ index của PyTorch.
if "%TORCH_INDEX_URL%"=="" set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install torch torchvision --index-url %TORCH_INDEX_URL% || goto :err
.venv\Scripts\python -m pip install -r requirements.txt || goto :err
.venv\Scripts\python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo.
echo Cai xong. Sua video_dir / output_dir trong config.yaml roi chay: run.bat
exit /b 0
:err
echo Loi khi cai dat.
exit /b 1
