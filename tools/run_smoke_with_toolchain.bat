@echo off
REM Initialize MSVC (pin to 14.39 toolset, which CUDA 11.8 nvcc supports)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39
if errorlevel 1 (echo vcvars64 FAILED & exit /b 1)

REM CUDA 11.8 toolkit
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "PATH=%CUDA_HOME%\bin;%PATH%"

REM conda env Scripts (ninja.exe) on PATH
set "PATH=C:\Users\talgx\miniconda3\envs\roadguard-dl\Scripts;%PATH%"

cd /d "C:\Users\talgx\Desktop\malshinon_master"
"C:\Users\talgx\miniconda3\envs\roadguard-dl\python.exe" tools\smoke_test_env.py
exit /b %errorlevel%
