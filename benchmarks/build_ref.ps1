<#
.SYNOPSIS
Build the engine with CMake (your normal toolchain) and copy the executable to
benchmarks\out\<Name>.exe for the benchmark scripts.

.DESCRIPTION
The engine sources come from the Tomahawk repository: -Repo, else $env:TOMAHAWK_ROOT,
else ..\..\tomahawk (Tomahawk checked out next to san-jacinto).

Without -Ref the engine's current working tree is built. With -Ref, the
commit/branch/tag is checked out as a detached git worktree of the Tomahawk repo in
benchmarks\out\src-<Name> and built there. Keep that folder while you use the
executable: the engine loads its NNUE weights from the source tree it was configured
in (PROJECT_ROOT). Remove it later with:
    git -C <tomahawk> worktree remove <san-jacinto>\benchmarks\out\src-<Name>

Note: CMake writes the executable to <source>\engines\prod (or engines\dev), so a
working-tree build also adds bench-<Name>.exe there (tomahawk.exe in Debug builds).
The build reports "id version bench-<Name>" over UCI.

.EXAMPLE
.\build_ref.ps1 -Name current
.\build_ref.ps1 -Name baseline -Ref dd04600
.\build_ref.ps1 -Name verify -CMakeArgs '-DCMAKE_CXX_FLAGS=-DVERIFY_NNUE'
.\build_ref.ps1 -Name current-g -CMakeArgs '-DCMAKE_CXX_FLAGS=-g'   # symbols for sample_profile.py
.\build_ref.ps1 -Name current -Repo D:\src\tomahawk
#>
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Ref = '',
    [string]$Repo = '',
    [string]$BuildType = 'Release',
    [string]$Generator = 'Ninja',
    [string[]]$CMakeArgs = @()
)
$bench = $PSScriptRoot
if (-not $Repo) { $Repo = $env:TOMAHAWK_ROOT }
if (-not $Repo) { $Repo = Join-Path $bench '..\..\tomahawk' }
if (-not (Test-Path (Join-Path $Repo 'CMakeLists.txt'))) {
    throw "Tomahawk sources not found at '$Repo': pass -Repo <path> or set TOMAHAWK_ROOT"
}
$repo = (Resolve-Path $Repo).Path
$out = Join-Path $bench 'out'
New-Item -ItemType Directory -Force -Path $out | Out-Null

if ($Ref) {
    # resolve in the main repository: inside the worktree, HEAD would mean the worktree's own HEAD
    $commit = git -C $repo rev-parse --verify --quiet "$Ref^{commit}"
    if (-not $commit) { throw "unknown git ref: $Ref" }
    $src = Join-Path $out "src-$Name"
    if (Test-Path $src) {
        git -C $src checkout --detach $commit
    } else {
        git -C $repo worktree add --detach $src $commit
    }
    if ($LASTEXITCODE) { throw "could not check out $Ref" }
} else {
    $src = $repo
}

$build = Join-Path $src "build-bench-$Name"
# VERSION names the executable (OUTPUT_NAME, except in Debug builds), so the copy below takes
# exactly this build's output even when engines\<kind> holds other, newer executables.
# It is also the engine's UCI "id version". A -DVERSION in -CMakeArgs comes later and wins.
$version = "bench-$Name"
foreach ($arg in $CMakeArgs) { if ($arg -match '^-DVERSION(:\w+)?=(.+)$') { $version = $Matches[2] } }
cmake -S $src -B $build -G $Generator "-DCMAKE_BUILD_TYPE=$BuildType" "-DVERSION=bench-$Name" @CMakeArgs
if ($LASTEXITCODE) { throw 'cmake configure failed' }
cmake --build $build --parallel
if ($LASTEXITCODE) { throw 'cmake build failed' }

# same (case-sensitive) tests as CMakeLists.txt
$kind = if (@('Release', 'RELEASE', 'Prod', 'PROD') -ccontains $BuildType) { 'prod' } else { 'dev' }
$named = if ($BuildType -ceq 'Debug' -or $version -ceq 'dev') { 'tomahawk.exe' } else { "$version.exe" }
$dir = Join-Path $src "engines\$kind"
$exe = Get-Item (Join-Path $dir $named) -ErrorAction SilentlyContinue
if (-not $exe) {
    $exe = Get-ChildItem $dir -Filter *.exe -ErrorAction SilentlyContinue |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $exe) { throw "no executable in $dir" }
    Write-Warning "$named not found in $dir; copying the newest executable there, $($exe.Name)"
}
$dest = Join-Path $out "$Name.exe"
Copy-Item $exe.FullName $dest -Force
$pdb = [IO.Path]::ChangeExtension($exe.FullName, '.pdb')
if (Test-Path $pdb) { Copy-Item $pdb ([IO.Path]::ChangeExtension($dest, '.pdb')) -Force }
Write-Output "built $dest (sources: $src)"
