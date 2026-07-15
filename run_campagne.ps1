# run_campagne.ps1
# Esegue in sequenza e senza sorveglianza le due campagne estese:
#   Campagna A -> config_run1_ext.json
#   Campagna B -> config_run2_ext.json
# L'output di entrambe va sulla console (nessun file di log). La B parte
# automaticamente al termine della A. A fine di tutto scrive
# campagne_riepilogo.txt con durate ed esiti.
#
# Uso (dalla cartella del progetto):
#   powershell -ExecutionPolicy Bypass -File .\run_campagne.ps1

$ErrorActionPreference = "Continue"

# --- attiva il virtualenv (se non gia' attivo) ---
if (Test-Path ".\envEnergyBenchmark\Scripts\Activate.ps1") {
    & .\envEnergyBenchmark\Scripts\Activate.ps1
}

# --- impedisci che il PC vada in sospensione durante il run ---
powercfg /change standby-timeout-ac 0   | Out-Null
powercfg /change hibernate-timeout-ac 0 | Out-Null

$start = Get-Date
Write-Host "=== Avvio campagne: $start ===" -ForegroundColor Cyan

# --- Campagna A ---
$aStart = Get-Date
Write-Host "[A] Campagna A avviata: $aStart"
python run_benchmark.py --config config_run1_ext.json
$aCode = $LASTEXITCODE
$aEnd  = Get-Date
$aDur  = [math]::Round(($aEnd - $aStart).TotalHours, 2)
Write-Host "[A] Campagna A finita: $aEnd  (durata $aDur h, exit=$aCode)"

# --- Campagna B (parte comunque, anche se la A ha avuto problemi) ---
$bStart = Get-Date
Write-Host "[B] Campagna B avviata: $bStart"
python run_benchmark.py --config config_run2_ext.json
$bCode = $LASTEXITCODE
$bEnd  = Get-Date
$bDur  = [math]::Round(($bEnd - $bStart).TotalHours, 2)
Write-Host "[B] Campagna B finita: $bEnd  (durata $bDur h, exit=$bCode)"

# --- riepilogo ---
$end = Get-Date
$tot = [math]::Round(($end - $start).TotalHours, 2)
$esitoA = if ($aCode -eq 0) { "OK" } else { "PROBLEMA (rivedi la console)" }
$esitoB = if ($bCode -eq 0) { "OK" } else { "PROBLEMA (rivedi la console)" }

$righe = @(
    "=== Riepilogo campagne ===",
    "Inizio totale: $start",
    "Fine totale:   $end",
    "Durata totale: $tot h",
    "",
    "Campagna A: $esitoA  (durata $aDur h, exit=$aCode)  -> recordings/results_run1_ext.csv",
    "Campagna B: $esitoB  (durata $bDur h, exit=$bCode)  -> recordings/results_run2_ext.csv"
)
$righe | Tee-Object -FilePath campagne_riepilogo.txt
Write-Host ""
Write-Host "=== Fatto. Riepilogo in campagne_riepilogo.txt ===" -ForegroundColor Green
