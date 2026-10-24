# Runs the reduction experiments and regenerates the analysis, then commits
# and pushes. Stops at the first failure, so a broken or half-finished run is
# never committed.
#
# Run from the project root:   .\run_all.ps1

function Step($name, $block) {
    Write-Host ""
    Write-Host "=== $name   $(Get-Date -Format 'HH:mm:ss') ==="
    & $block
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "!!! $name failed (exit code $LASTEXITCODE). Nothing was committed."
        exit 1
    }
}

Step "reduction_main.py" { python reduction_main.py }

Step "reduction_analysis.py" {
    python reduction_analysis.py | Out-File -Encoding utf8 reduction_output_analysis.txt
}

Step "plots.py" { python plots.py }

Write-Host ""
Write-Host "=== git   $(Get-Date -Format 'HH:mm:ss') ==="

git add -A

# Nothing staged means nothing changed, and git commit would fail.
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "No files changed, so no commit was made."
    exit 0
}

git commit -m "Re-run the reduction experiments after merging the new reduction code" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
if ($LASTEXITCODE -ne 0) {
    Write-Host "!!! The commit failed. Nothing was pushed."
    exit 1
}

git push
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "The commit is saved locally, but the push was rejected."
    Write-Host "Someone probably pushed while this was running. Fix it with:"
    Write-Host "    git pull --rebase"
    Write-Host "    git push"
    exit 1
}

Write-Host ""
Write-Host "Done at $(Get-Date -Format 'HH:mm:ss'). Committed and pushed."
