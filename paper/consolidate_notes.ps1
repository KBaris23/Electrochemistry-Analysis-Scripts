$ErrorActionPreference = 'Stop'
$paperRoot = $PSScriptRoot
$repoRoot = Split-Path -Parent $paperRoot
$planRoot = 'C:\TEMP\BO\analysis_plan'
$notes = @(
    (Join-Path $planRoot 'AI_HANDOFF_PROMPT.md'),
    (Join-Path $planRoot 'STEP_BY_STEP.md'),
    (Join-Path $repoRoot 'audit\bo_paper_data_audit.md'),
    (Join-Path $repoRoot 'audit\storage_audit.md')
)
foreach ($folder in @('archive', 'reference', 'helpers_previous', 'examples_previous')) {
    New-Item -ItemType Directory -Path (Join-Path $paperRoot $folder) -Force | Out-Null
}
$archivePath = Join-Path $paperRoot 'archive\superseded_notes_20261007.zip'
if (Test-Path -LiteralPath $archivePath) { throw 'Archive already exists; refusing to overwrite.' }
foreach ($note in $notes) {
    if (-not (Test-Path -LiteralPath $note -PathType Leaf)) { throw "Missing note: $note" }
}
Compress-Archive -LiteralPath $notes -DestinationPath $archivePath
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    foreach ($note in $notes) {
        $entry = $archive.GetEntry([System.IO.Path]::GetFileName($note))
        if ($null -eq $entry) { throw "Archive entry missing: $note" }
        $entryStream = $entry.Open()
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try { $archivedHash = [BitConverter]::ToString($sha.ComputeHash($entryStream)).Replace('-', '') }
        finally { $entryStream.Dispose(); $sha.Dispose() }
        if ($archivedHash -ne (Get-FileHash -LiteralPath $note -Algorithm SHA256).Hash) {
            throw "Archive hash mismatch: $note"
        }
    }
} finally { $archive.Dispose() }
Copy-Item -LiteralPath (Join-Path $planRoot 'channel_ranking_all_datasets.csv') -Destination (Join-Path $paperRoot 'reference')
foreach ($helper in @('make_titration_only.py', 'render_template_examples.py')) {
    Copy-Item -LiteralPath (Join-Path $planRoot $helper) -Destination (Join-Path $paperRoot 'helpers_previous')
}
Get-ChildItem -LiteralPath (Join-Path $planRoot 'examples') -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $paperRoot 'examples_previous')
}
$sessions = @{
    kana_try = '500um_planar_BO_try_again_20260918_112055\planar_BO_kana_20260918_112056\bo_sessions\bo_113013_8dd581'
    kana_st2 = '500um_planar_kana_20260917_170330\500um_planar_kana_20260917_170349\bo_sessions\bo_171120_dc5dbc'
    amp0 = '100um_amp0_high_conc_20260915_130213\amp0_100um_20260915_130234\bo_sessions\bo_145633_9f5760'
    vanco = 'vanco_first_try_20260826_132944\vanco_try_1_20260826_132945\bo_sessions\bo_142117_84662a'
}
foreach ($dataset in $sessions.Keys) {
    $snapshot = Join-Path (Join-Path 'C:\TEMP\BO' $sessions[$dataset]) 'bo_config_snapshot.json'
    Copy-Item -LiteralPath $snapshot -Destination (Join-Path $paperRoot "reference\${dataset}_bo_config_snapshot.json")
}
# Only these four exact, SHA256-verified archived notes are removed. No recursion.
foreach ($note in $notes) { Remove-Item -LiteralPath $note }
Write-Output "Consolidated four notes; verified recoverable archive: $archivePath"
Write-Output 'Copied small supporting files only. Raw experiment data unchanged.'
