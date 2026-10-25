<#
.SYNOPSIS
Quick build without CMake: all engine sources in one translation unit (the same
cross-file inlining LTO gives), or the NNUE kernel microbenchmark.

.DESCRIPTION
Useful for building variants from any source folder and for profiling builds:
with zig (clang) and -Symbols the output gets a .pdb that sample_profile.py and
disasm.py can read. The executable loads its NNUE weights from -Root.

-Src is the Tomahawk source folder: default $env:TOMAHAWK_ROOT, else ..\..\tomahawk
(Tomahawk checked out next to san-jacinto).

-Compiler accepts g++, clang++, or zig (zig.exe runs as "zig c++"; get it with
`pip install ziglang`, then use python -m ziglang or the zig.exe in site-packages).

.EXAMPLE
.\build_unity.ps1 -Out out\current.exe
.\build_unity.ps1 -Out out\current-prof.exe -Compiler zig -Symbols
.\build_unity.ps1 -Out out\verify.exe -Extra '-DTOMAHAWK_VERIFY_NNUE'
.\build_unity.ps1 -Out out\kernelbench.exe -Kernelbench
.\build_unity.ps1 -Out out\other.exe -Src D:\src\tomahawk-experiment
#>
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Src = '',
    [string]$Root = '',
    [string]$Compiler = 'g++',
    [string]$Opt = '-O3 -ffast-math -march=native',
    [string]$Extra = '',
    [switch]$Symbols,
    [switch]$Kernelbench
)
$bench = $PSScriptRoot
if (-not $Src) { $Src = $env:TOMAHAWK_ROOT }
if (-not $Src) { $Src = Join-Path $bench '..\..\tomahawk' }
if (-not (Test-Path (Join-Path $Src 'src'))) {
    throw "Tomahawk sources not found at '$Src': pass -Src <path> or set TOMAHAWK_ROOT"
}
$Src = (Resolve-Path $Src).Path
if (-not $Root) { $Root = $Src }
$Out = [IO.Path]::GetFullPath((Join-Path (Get-Location) $Out))
New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null
$fwd = { param($p) $p -replace '\\', '/' }

$sources = @('board', 'book', 'engine', 'gamestate', 'helpers', 'magics', 'movegen', 'nnue', 'precomp',
             'searcher', 'threads', 'uci', 'affinity', 'main') |
           Where-Object { Test-Path (Join-Path $Src "src\$_.cpp") }
if ($Kernelbench) {
    $main = & $fwd (Join-Path $bench 'kernelbench.cpp')
    $defs = @("-DENGINE_SRC=$(& $fwd $Src)")
} else {
    $main = "$Out.unity.cpp"
    [IO.File]::WriteAllLines($main, [string[]]($sources | ForEach-Object { '#include "' + (& $fwd (Join-Path $Src "src\$_.cpp")) + '"' }))
    $main = & $fwd $main
    $defs = @()
}

$isZig = [IO.Path]::GetFileNameWithoutExtension($Compiler) -eq 'zig'
if (-not (Get-Command $Compiler -ErrorAction SilentlyContinue)) {
    throw "compiler '$Compiler' not found: install MinGW-w64 GCC (or zig) or pass -Compiler <path>"
}
$isGcc = -not $isZig -and ((& $Compiler --version 2>&1 | Out-String) -notmatch 'clang')
$flags = @('-std=c++17') + ($Opt -split ' ' | Where-Object { $_ })
if ($Symbols) { $flags += '-g' }
if ($isGcc -and $env:OS -eq 'Windows_NT') {
    # GCC does not guarantee 32-byte stack alignment for AVX on 64-bit Windows (GCC bug 54412)
    $probe = Join-Path $env:TEMP 'tomahawk_probe.cpp'
    Set-Content $probe 'int main() { return 0; }'
    & $Compiler '-Wa,-muse-unaligned-vector-move' -c $probe -o "$probe.o" 2>$null
    $supported = $LASTEXITCODE -eq 0
    Remove-Item $probe, "$probe.o" -ErrorAction SilentlyContinue
    if ($supported) { $flags += '-Wa,-muse-unaligned-vector-move' }
    else { Write-Warning 'assembler lacks -muse-unaligned-vector-move (binutils < 2.38); AVX builds may crash' }
}
$flags += @('-DENGINE_ID=\"dev\"', ('-DPROJECT_ROOT=\"' + (& $fwd $Root) + '\"')) + $defs
$flags += @('-I' + (& $fwd (Join-Path $Src 'include')), '-I' + (& $fwd (Join-Path $Src 'nnue')))
$flags += ($Extra -split ' ' | Where-Object { $_ })
$flags += @($main, '-o', (& $fwd $Out), '-lole32')

$rsp = "$Out.rsp"
[IO.File]::WriteAllLines($rsp, [string[]]$flags)
$log = "$Out.log"
if ($isZig) { & $Compiler c++ "@$rsp" *> $log } else { & $Compiler "@$rsp" *> $log }
if ($LASTEXITCODE) {
    Get-Content $log | Select-String 'error' | Select-Object -First 15 | ForEach-Object { $_.Line }
    throw "build failed, see $log"
}
Write-Output "built $Out"
