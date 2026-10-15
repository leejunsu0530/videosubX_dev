# 아래는 가상환경 코드
Write-Host "[Command]: py -3.13 -m venv .venv" -BackgroundColor White -ForegroundColor Blue
py -3.13 -m venv .venv
Write-Host "[Command]: .venv/scripts/activate" -BackgroundColor White -ForegroundColor Blue
.venv/scripts/activate
Write-Host "[Command]: python.exe -m pip install --upgrade pip uv" -BackgroundColor White -ForegroundColor Blue
python.exe -m pip install --upgrade pip uv
# Write-Host "[Command]: python -m uv pip install -U -e .[intel]" -BackgroundColor White -ForegroundColor Blue
# python -m uv pip install -U -e .[intel]
Write-Host "[Command]: python -m uv pip install -U -e .[gui]" -BackgroundColor White -ForegroundColor Blue
python -m uv pip install -U -e .[gui]
# pwcpp 파트
# Write-Host ". C:\Binaries\w_openvino_toolkit_windows_2024.6.0.17404.4c0f47d2335_x86_64\setupvars.ps1" -BackgroundColor White -ForegroundColor Blue
# . C:\Binaries\w_openvino_toolkit_windows_2024.6.0.17404.4c0f47d2335_x86_64\setupvars.ps1
# Write-Host "`$env:WHISPER_OPENVINO=1" -BackgroundColor White -ForegroundColor Blue
# $env:WHISPER_OPENVINO=1
# Write-Host "python -m uv pip install https://github.com/absadiki/pywhispercpp.git --force-reinstall --no-cache" -BackgroundColor White -ForegroundColor Blue
# python -m uv pip install https://github.com/absadiki/pywhispercpp.git --force-reinstall --no-cache