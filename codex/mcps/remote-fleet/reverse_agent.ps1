# Reverse-connected development agent for Windows, a PowerShell port of reverse_agent.py.
# Needs only the built-in Windows PowerShell 5.1 (or PowerShell 7); no Python.
#
# One-line start (downloads and runs in the background):
#   & ([scriptblock]::Create((irm https://74-211-111-53.sslip.io:8443/remote-agent/reverse_agent.ps1))) <device-name> -Background
# Foreground:  ... <device-name>
# Stop:        ... <device-name> -Stop
param(
    [Parameter(Position = 0)][string]$Name,
    [Alias('Env')][ValidateSet('merlin', 'volcano', 'null')][string]$AgentEnv = 'null',
    [string[]]$Root = @(),
    [string]$Server = '',
    [string]$StateDir = '',
    [double]$MaxUploadGiB = 0,
    [switch]$Background,
    [switch]$Stop,
    [string]$InstallDir = ''
)

$ErrorActionPreference = 'Stop'
$SelfText = $MyInvocation.MyCommand.ScriptBlock.ToString()

$DefaultServer = 'http://74.211.111.53:18765'
$EnvProxies = @{ merlin = 'http://sys-proxy-rd-relay.byted.org:8118'; volcano = 'http://100.66.18.103:3128' }
$PollWait = 25
$MaxBackoff = 30

# Everything a request handler needs; also loaded into each worker runspace.
$Lib = {
    $COPY_CHUNK = 1MB
    $MAX_DOWNLOAD_CHUNK = 16MB
    $RESULT_RETRIES = 5

    function Write-AgentLog([string]$Message) {
        [Console]::Out.WriteLine((Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $Message)
        [Console]::Out.Flush()
    }

    function ConvertFrom-AgentJson([string]$Text) {
        if ($PSVersionTable.PSVersion.Major -ge 6) { return ($Text | ConvertFrom-Json -AsHashtable -Depth 64) }
        Add-Type -AssemblyName System.Web.Extensions
        $ser = New-Object System.Web.Script.Serialization.JavaScriptSerializer
        $ser.MaxJsonLength = [int]::MaxValue
        return $ser.DeserializeObject($Text)
    }

    function ConvertTo-AgentJson($Value) {
        if ($PSVersionTable.PSVersion.Major -ge 6) { return (ConvertTo-Json -InputObject $Value -Depth 64 -Compress) }
        Add-Type -AssemblyName System.Web.Extensions
        $ser = New-Object System.Web.Script.Serialization.JavaScriptSerializer
        $ser.MaxJsonLength = [int]::MaxValue
        return $ser.Serialize($Value)
    }

    function Get-Field($Map, [string]$Key, $Default = $null) {
        if ($null -ne $Map -and $Map.ContainsKey($Key) -and $null -ne $Map[$Key]) { return $Map[$Key] }
        return $Default
    }

    function Get-UnixTime { return [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0 }

    function Get-Brief([string]$Command, [int]$Limit = 300) {
        $text = ($Command -split '\s+' | Where-Object { $_ }) -join ' '
        if ($text.Length -le $Limit) { return $text }
        return $text.Substring(0, $Limit - 3) + '...'
    }

    function Get-HexSha256([byte[]]$Bytes) {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try { return ([BitConverter]::ToString($sha.ComputeHash($Bytes)) -replace '-', '').ToLowerInvariant() }
        finally { $sha.Dispose() }
    }

    function Get-FileSha256([string]$Path, [long]$Limit = -1) {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $stream = [System.IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
        try {
            $buffer = New-Object byte[] $COPY_CHUNK
            $remaining = $Limit
            while ($remaining -ne 0) {
                $want = if ($remaining -lt 0) { $buffer.Length } else { [Math]::Min([long]$buffer.Length, $remaining) }
                $read = $stream.Read($buffer, 0, $want)
                if ($read -le 0) { break }
                [void]$sha.TransformBlock($buffer, 0, $read, $null, 0)
                if ($remaining -gt 0) { $remaining -= $read }
            }
            if ($remaining -gt 0) { throw [ArgumentException]::new('file ended before requested hash length') }
            [void]$sha.TransformFinalBlock((New-Object byte[] 0), 0, 0)
            return ([BitConverter]::ToString($sha.Hash) -replace '-', '').ToLowerInvariant()
        } finally { $stream.Dispose(); $sha.Dispose() }
    }

    function Lock-Object($Object, [scriptblock]$Body) {
        [System.Threading.Monitor]::Enter($Object)
        try { & $Body } finally { [System.Threading.Monitor]::Exit($Object) }
    }

    function Write-Audit($S, [string]$Action, [hashtable]$Fields = @{}) {
        $record = @{ time = (Get-UnixTime); action = $Action }
        foreach ($k in $Fields.Keys) { $record[$k] = $Fields[$k] }
        $line = (ConvertTo-AgentJson $record) + "`n"
        Lock-Object $S.Lock { [System.IO.File]::AppendAllText($S.AuditPath, $line, [System.Text.UTF8Encoding]::new($false)) }
    }

    function Get-PathLock($S, [string]$Path) {
        $key = $Path.ToLowerInvariant()
        Lock-Object $S.Lock {
            if (-not $S.PathLocks.ContainsKey($key)) { $S.PathLocks[$key] = New-Object object }
        }
        return $S.PathLocks[$key]
    }

    function Resolve-AgentPath($S, [string]$Raw, [switch]$MustExist) {
        if (-not $Raw) { throw [ArgumentException]::new('path is required') }
        if ($Raw -eq '~' -or $Raw.StartsWith('~/') -or $Raw.StartsWith('~\')) {
            $Raw = [Environment]::GetFolderPath('UserProfile') + $Raw.Substring(1)
        }
        if (-not [System.IO.Path]::IsPathRooted($Raw)) { $Raw = [System.IO.Path]::Combine($S.Roots[0], $Raw) }
        $path = [System.IO.Path]::GetFullPath($Raw)
        if ($path.Length -gt 3) { $path = $path.TrimEnd('\', '/') }
        $lower = $path.ToLowerInvariant()
        $allowed = $false
        foreach ($root in $S.Roots) {
            $r = $root.ToLowerInvariant().TrimEnd('\', '/')
            if ($lower.TrimEnd('\', '/') -eq $r -or $lower.StartsWith($r + '\')) { $allowed = $true; break }
        }
        if (-not $allowed) { throw [UnauthorizedAccessException]::new("path is outside allowed roots: $path") }
        if ($MustExist -and -not (Test-Path -LiteralPath $path)) { throw [System.IO.FileNotFoundException]::new("No such file or directory: '$path'") }
        return $path
    }

    function New-JsonResponse([int]$Status, $Payload) {
        return @{
            Status  = $Status
            Body    = [System.Text.Encoding]::UTF8.GetBytes((ConvertTo-AgentJson $Payload))
            Headers = @{ 'Content-Type' = 'application/json; charset=utf-8' }
        }
    }

    function Get-ShellPath {
        foreach ($candidate in 'pwsh.exe', 'powershell.exe') {
            $cmd = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($cmd) { return $cmd.Source }
        }
        throw [System.IO.FileNotFoundException]::new('PowerShell is required (pwsh.exe or powershell.exe)')
    }

    function Get-WrappedScript([string]$Command) {
        return '$utf8 = New-Object System.Text.UTF8Encoding($false); ' +
            '[Console]::InputEncoding = $utf8; [Console]::OutputEncoding = $utf8; ' +
            '$OutputEncoding = $utf8; $ErrorActionPreference = ''Stop''; $ProgressPreference = ''SilentlyContinue''; ' +
            '$global:LASTEXITCODE = 0; ' +
            "& {`n" + $Command + "`n}; " +
            'if (-not $?) { exit 1 }; exit $LASTEXITCODE'
    }

    # Windows PowerShell 5.1 always writes CLIXML to a redirected stderr under
    # -EncodedCommand, so commands run from a script file via -File instead.
    function Write-CommandScript([string]$Path, [string]$Command) {
        [System.IO.File]::WriteAllText($Path, (Get-WrappedScript $Command), [System.Text.UTF8Encoding]::new($true))
    }

    function Get-ShellArguments([string]$ScriptPath) {
        return "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$ScriptPath`""
    }

    function Stop-ProcessTree($Process) {
        $taskkill = Join-Path $env:SystemRoot 'System32\taskkill.exe'
        & $taskkill /PID $Process.Id /T /F 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0 -and -not $Process.HasExited) { throw "taskkill failed for PID $($Process.Id): $LASTEXITCODE" }
    }

    function Get-CommandSpec($S, $Body) {
        $command = Get-Field $Body 'command'
        if (-not ($command -is [string]) -or -not $command.Trim()) { throw [ArgumentException]::new('command must be a non-empty string') }
        $cwd = Resolve-AgentPath $S ([string](Get-Field $Body 'cwd' $S.Roots[0])) -MustExist
        if (-not [System.IO.Directory]::Exists($cwd)) { throw [ArgumentException]::new('cwd must be a directory') }
        $extra = Get-Field $Body 'env' @{}
        if (-not ($extra -is [System.Collections.IDictionary])) { throw [ArgumentException]::new('env must be an object containing string values') }
        foreach ($k in @($extra.Keys)) {
            if (-not ($extra[$k] -is [string])) { throw [ArgumentException]::new('env must be an object containing string values') }
        }
        return @{ Command = $command; Cwd = $cwd; Env = $extra }
    }

    function New-StartInfo([string]$FileName, [string]$Arguments, $Spec) {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $FileName
        $psi.Arguments = $Arguments
        $psi.WorkingDirectory = $Spec.Cwd
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        foreach ($k in @($Spec.Env.Keys)) { $psi.EnvironmentVariables[$k] = [string]$Spec.Env[$k] }
        return $psi
    }

    function Invoke-Exec($S, $Body) {
        $spec = Get-CommandSpec $S $Body
        $timeout = [double](Get-Field $Body 'timeout' 300)
        if (-not ($timeout -gt 0 -and $timeout -le 86400)) { throw [ArgumentException]::new('timeout must be between 0 and 86400 seconds') }
        $tag = 'exec[' + [guid]::NewGuid().ToString('N').Substring(0, 6) + ']'
        Write-AgentLog "$tag `$ $(Get-Brief $spec.Command)  (cwd=$($spec.Cwd))"
        $scriptPath = Join-Path $S.LogsDir ('exec-' + [guid]::NewGuid().ToString('N') + '.ps1')
        Write-CommandScript $scriptPath $spec.Command
        $psi = New-StartInfo (Get-ShellPath) (Get-ShellArguments $scriptPath) $spec
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.StandardOutputEncoding = [System.Text.UTF8Encoding]::new($false)
        $psi.StandardErrorEncoding = [System.Text.UTF8Encoding]::new($false)
        $watch = [System.Diagnostics.Stopwatch]::StartNew()
        $process = [System.Diagnostics.Process]::Start($psi)
        try {
            $process.StandardInput.Close()
            $outTask = $process.StandardOutput.ReadToEndAsync()
            $errTask = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit([int]($timeout * 1000))) {
                Stop-ProcessTree $process
                [void]$process.WaitForExit(15000)
                [void]$outTask.Wait(5000); [void]$errTask.Wait(5000)
                $stdout = if ($outTask.IsCompleted) { $outTask.Result } else { '' }
                $stderr = if ($errTask.IsCompleted) { $errTask.Result } else { '' }
                Write-AgentLog "$tag killed after ${timeout}s timeout"
                Write-Audit $S 'exec_timeout' @{ command = $spec.Command; cwd = $spec.Cwd; timeout = $timeout }
                return New-JsonResponse 408 @{ error = 'command timed out'; stdout = $stdout; stderr = $stderr }
            }
            $process.WaitForExit()
            $code = $process.ExitCode
            $stdout = $outTask.Result
            $stderr = $errTask.Result
        } finally {
            $process.Dispose()
            Remove-Item -LiteralPath $scriptPath -Force -ErrorAction SilentlyContinue
        }
        $elapsed = $watch.Elapsed.TotalSeconds
        Write-AgentLog ("$tag exit=$code in {0:N1}s" -f $elapsed)
        Write-Audit $S 'exec' @{ command = $spec.Command; cwd = $spec.Cwd; exit_code = $code }
        return New-JsonResponse 200 @{ exit_code = $code; stdout = $stdout; stderr = $stderr; duration_seconds = $elapsed }
    }

    function Get-PublicJob($Job) {
        $process = $Job.Process
        $running = -not $process.HasExited
        $code = $null
        if (-not $running) { $code = [int]$process.ExitCode }
        return @{
            id = $Job.Id; command = $Job.Command; cwd = $Job.Cwd; pid = $Job.Pid
            started_at = $Job.StartedAt; running = $running; exit_code = $code
        }
    }

    function Start-AgentJob($S, $Body) {
        $spec = Get-CommandSpec $S $Body
        $jobId = [guid]::NewGuid().ToString('N')
        $logPath = Join-Path $S.LogsDir "$jobId.log"
        $scriptPath = Join-Path $S.LogsDir "$jobId.ps1"
        Write-CommandScript $scriptPath $spec.Command
        $shell = Get-ShellPath
        # cmd.exe merges stdout and stderr into one log file, like the Python agent.
        $arguments = "/d /s /c `"`"$shell`" $(Get-ShellArguments $scriptPath) < NUL > `"$logPath`" 2>&1`""
        $psi = New-StartInfo (Join-Path $env:SystemRoot 'System32\cmd.exe') $arguments $spec
        $process = [System.Diagnostics.Process]::Start($psi)
        $job = [hashtable]::Synchronized(@{
            Id = $jobId; Command = $spec.Command; Cwd = $spec.Cwd; Pid = $process.Id
            Process = $process; LogPath = $logPath; StartedAt = (Get-UnixTime); Reported = $false
        })
        $S.Jobs[$jobId] = $job
        Write-AgentLog "job[$($jobId.Substring(0, 8))] started pid=$($process.Id) `$ $(Get-Brief $spec.Command)  (cwd=$($spec.Cwd))"
        Write-Audit $S 'job_start' @{ job_id = $jobId; command = $spec.Command; cwd = $spec.Cwd; pid = $process.Id }
        return New-JsonResponse 201 (Get-PublicJob $job)
    }

    function Get-AgentJob($S, [string]$JobId) {
        if (-not $S.Jobs.ContainsKey($JobId)) { throw [ArgumentException]::new('unknown job id') }
        return $S.Jobs[$JobId]
    }

    function Get-JobInfo($S, [string]$JobId, [bool]$Logs, $Query) {
        $job = Get-AgentJob $S $JobId
        $public = Get-PublicJob $job
        if (-not $Logs) { return New-JsonResponse 200 $public }
        $offset = [long](Get-Field $Query 'offset' '0')
        $limit = [Math]::Min([long](Get-Field $Query 'limit' ([string]1MB)), [long]4MB)
        if ($offset -lt 0 -or $limit -lt 1) { throw [ArgumentException]::new('offset and limit must be positive') }
        $stream = [System.IO.File]::Open($job.LogPath, 'OpenOrCreate', 'Read', 'ReadWrite')
        try {
            $size = $stream.Length
            [void]$stream.Seek($offset, 'Begin')
            $buffer = New-Object byte[] ([int][Math]::Max(0, [Math]::Min($limit, $size - $offset)))
            $read = 0
            while ($read -lt $buffer.Length) {
                $n = $stream.Read($buffer, $read, $buffer.Length - $read)
                if ($n -le 0) { break }
                $read += $n
            }
        } finally { $stream.Dispose() }
        $public['offset'] = $offset
        $public['next_offset'] = [long]([Math]::Max($offset, $offset + $read))
        $public['size'] = $size
        $public['log'] = [System.Text.Encoding]::UTF8.GetString($buffer, 0, $read)
        return New-JsonResponse 200 $public
    }

    function Stop-AgentJob($S, [string]$JobId) {
        $job = Get-AgentJob $S $JobId
        $process = $job.Process
        if (-not $process.HasExited) {
            Stop-ProcessTree $process
            [void]$process.WaitForExit(10000)
        }
        Write-AgentLog "job[$($JobId.Substring(0, 8))] stop requested"
        $code = $null
        if ($process.HasExited) { $code = [int]$process.ExitCode }
        Write-Audit $S 'job_stop' @{ job_id = $JobId; exit_code = $code }
        return New-JsonResponse 200 (Get-PublicJob $job)
    }

    function Get-FileDownload($S, $Query) {
        $path = Resolve-AgentPath $S ([string](Get-Field $Query 'path' '')) -MustExist
        if (-not [System.IO.File]::Exists($path)) { throw [ArgumentException]::new('path is not a file') }
        $offset = [long](Get-Field $Query 'offset' '0')
        $limit = [long](Get-Field $Query 'limit' ([string]$MAX_DOWNLOAD_CHUNK))
        $stream = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
        try {
            $size = $stream.Length
            if ($offset -lt 0 -or $offset -gt $size -or $limit -lt 0 -or $limit -gt $MAX_DOWNLOAD_CHUNK) {
                throw [ArgumentException]::new("invalid download offset or limit (limit <= $MAX_DOWNLOAD_CHUNK)")
            }
            [void]$stream.Seek($offset, 'Begin')
            $data = New-Object byte[] ([int][Math]::Min($limit, $size - $offset))
            $read = 0
            while ($read -lt $data.Length) {
                $n = $stream.Read($data, $read, $data.Length - $read)
                if ($n -le 0) { break }
                $read += $n
            }
            if ($read -ne $data.Length) { [Array]::Resize([ref]$data, $read) }
        } finally { $stream.Dispose() }
        if ($offset -eq 0) { Write-AgentLog "download $path ($size bytes)" }
        Write-Audit $S 'download' @{ path = $path; offset = $offset; size = $data.Length }
        return @{ Status = 206; Body = $data; Headers = @{ 'Content-Type' = 'application/octet-stream'; 'X-File-Size' = [string]$size } }
    }

    function Get-FileInfoResponse($S, $Query) {
        $path = Resolve-AgentPath $S ([string](Get-Field $Query 'path' ''))
        if (-not (Test-Path -LiteralPath $path)) { return New-JsonResponse 200 @{ exists = $false; size = 0; sha256 = $null } }
        if (-not [System.IO.File]::Exists($path)) { throw [ArgumentException]::new('path is not a file') }
        $size = (New-Object System.IO.FileInfo $path).Length
        $hashLength = -1
        $raw = Get-Field $Query 'hash_length'
        if ($null -ne $raw) {
            $hashLength = [long]$raw
            if ($hashLength -lt 0 -or $hashLength -gt $size) { throw [ArgumentException]::new('invalid hash_length') }
        }
        return New-JsonResponse 200 @{ exists = $true; size = $size; sha256 = (Get-FileSha256 $path $hashLength) }
    }

    function Save-UploadChunk($S, $Request) {
        $path = Resolve-AgentPath $S ([string](Get-Field $Request.Query 'path' ''))
        $offset = [long](Get-Field $Request.Query 'offset' '-1')
        [byte[]]$data = $Request.Body
        if ($offset -lt 0 -or ($S.MaxUpload -gt 0 -and $offset + $data.Length -gt $S.MaxUpload)) {
            throw [ArgumentException]::new('invalid upload offset or chunk size')
        }
        $expected = ([string](Get-Field $Request.Headers 'x-content-sha256' '')).ToLowerInvariant()
        $actual = Get-HexSha256 $data
        if ($expected -and $expected -ne $actual) { throw [ArgumentException]::new('chunk SHA-256 mismatch') }
        [void][System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($path))
        Lock-Object (Get-PathLock $S $path) {
            $current = 0
            if ([System.IO.File]::Exists($path)) { $current = (New-Object System.IO.FileInfo $path).Length }
            if ($current -ne $offset) { throw [ArgumentException]::new("partial size mismatch: $current != $offset") }
            $stream = [System.IO.File]::Open($path, 'OpenOrCreate', 'Write', 'None')
            try {
                [void]$stream.Seek(0, 'End')
                try { $stream.Write($data, 0, $data.Length); $stream.Flush($true) }
                catch { $stream.SetLength($offset); throw }
            } finally { $stream.Dispose() }
        }
        $size = (New-Object System.IO.FileInfo $path).Length
        Write-Audit $S 'upload_chunk' @{ path = $path; offset = $offset; size = $data.Length }
        return New-JsonResponse 201 @{ path = $path; size = $size; sha256 = $actual }
    }

    function Complete-Upload($S, $Body) {
        $partial = Resolve-AgentPath $S ([string](Get-Field $Body 'partial_path' '')) -MustExist
        $destination = Resolve-AgentPath $S ([string](Get-Field $Body 'path' ''))
        $size = [long](Get-Field $Body 'size' -1)
        $expected = ([string](Get-Field $Body 'sha256' '')).ToLowerInvariant()
        $overwrite = [bool](Get-Field $Body 'overwrite' $false)
        if ($partial.ToLowerInvariant() -ne ($destination + '.partial').ToLowerInvariant()) {
            throw [ArgumentException]::new("partial_path must be '<path>.partial'")
        }
        if ($size -lt 0 -or ($S.MaxUpload -gt 0 -and $size -gt $S.MaxUpload)) { throw [ArgumentException]::new('invalid or excessive upload size') }
        if ($expected -notmatch '^[0-9a-f]{64}$') { throw [ArgumentException]::new('invalid SHA-256') }
        $actual = $null
        Lock-Object (Get-PathLock $S $partial) {
            if (-not [System.IO.File]::Exists($partial) -or (New-Object System.IO.FileInfo $partial).Length -ne $size) {
                throw [ArgumentException]::new('upload size mismatch')
            }
            $script:actualHash = Get-FileSha256 $partial
            if ($script:actualHash -ne $expected) { throw [ArgumentException]::new('upload SHA-256 mismatch') }
            Lock-Object (Get-PathLock $S $destination) {
                if ((Test-Path -LiteralPath $destination) -and -not $overwrite) {
                    throw [ArgumentException]::new("destination already exists: $destination")
                }
                [void][System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($destination))
                if ([System.IO.File]::Exists($destination)) { [System.IO.File]::Replace($partial, $destination, $null) }
                else { [System.IO.File]::Move($partial, $destination) }
            }
        }
        $actual = $script:actualHash
        Write-AgentLog "upload $destination ($size bytes)"
        Write-Audit $S 'upload_finalize' @{ path = $destination; size = $size; sha256 = $actual }
        return New-JsonResponse 201 @{ path = $destination; size = $size; sha256 = $actual }
    }

    function Invoke-Route($S, $Request) {
        $method = $Request.Method
        $path = $Request.Path
        $parts = @($path.Trim('/').Split('/'))
        $isJob = ($parts.Count -ge 3 -and $parts[0] -eq 'v1' -and $parts[1] -eq 'jobs')
        if ($method -eq 'GET') {
            if ($path -eq '/v1/health') {
                return New-JsonResponse 200 @{
                    ok = $true; hostname = [Environment]::MachineName; platform = 'nt'; shell = 'powershell'
                    pid = $PID; roots = [string[]]$S.Roots
                }
            }
            if ($path -eq '/v1/files') { return Get-FileDownload $S $Request.Query }
            if ($path -eq '/v1/files/info') { return Get-FileInfoResponse $S $Request.Query }
            if ($isJob -and ($parts.Count -eq 3 -or $parts.Count -eq 4)) {
                if ($parts.Count -eq 4 -and $parts[3] -ne 'logs') { throw [ArgumentException]::new('unknown job endpoint') }
                return Get-JobInfo $S $parts[2] ($parts.Count -eq 4) $Request.Query
            }
        } elseif ($method -eq 'POST') {
            if ($path -eq '/v1/exec') { return Invoke-Exec $S (Get-JsonBody $Request) }
            if ($path -eq '/v1/jobs') { return Start-AgentJob $S (Get-JsonBody $Request) }
            if ($path -eq '/v1/files/finalize') { return Complete-Upload $S (Get-JsonBody $Request) }
            if ($isJob -and $parts.Count -eq 4 -and $parts[3] -eq 'stop') { return Stop-AgentJob $S $parts[2] }
        } elseif ($method -eq 'PUT') {
            if ($path -eq '/v1/files/chunk') { return Save-UploadChunk $S $Request }
        }
        return New-JsonResponse 404 @{ error = 'endpoint not found' }
    }

    function Get-JsonBody($Request) {
        $text = 'null'
        if ($Request.Body.Length -gt 0) { $text = [System.Text.Encoding]::UTF8.GetString($Request.Body) }
        $value = ConvertFrom-AgentJson $text
        if (-not ($value -is [System.Collections.IDictionary])) { throw [ArgumentException]::new('JSON body must be an object') }
        return $value
    }

    function Invoke-Dispatch($S, $Request) {
        try { return Invoke-Route $S $Request }
        catch {
            $exc = $_.Exception
            while ($exc -is [System.Management.Automation.MethodInvocationException] -and $exc.InnerException) { $exc = $exc.InnerException }
            $bad = $exc -is [ArgumentException] -or $exc -is [FormatException] -or $exc -is [InvalidCastException] -or
                $exc -is [UnauthorizedAccessException] -or $exc -is [System.IO.FileNotFoundException] -or
                $exc -is [System.IO.DirectoryNotFoundException] -or $exc -is [System.Management.Automation.PSInvalidCastException]
            if ($bad) {
                Write-AgentLog "rejected $($Request.Method) $($Request.Path): $($exc.Message)"
                return New-JsonResponse 400 @{ error = $exc.Message }
            }
            Write-AgentLog "error in $($Request.Method) $($Request.Path): $($exc.GetType().Name): $($exc.Message)"
            try { Write-Audit $S 'internal_error' @{ method = $Request.Method; path = $Request.Path; error = $exc.ToString() } } catch {}
            return New-JsonResponse 500 @{ error = $exc.Message }
        }
    }

    function ConvertFrom-QueryString([string]$Query) {
        $result = @{}
        foreach ($pair in $Query.Split('&')) {
            if (-not $pair) { continue }
            $kv = $pair.Split('=', 2)
            $key = [Uri]::UnescapeDataString($kv[0].Replace('+', ' '))
            $value = ''
            if ($kv.Count -gt 1) { $value = [Uri]::UnescapeDataString($kv[1].Replace('+', ' ')) }
            if (-not $result.ContainsKey($key)) { $result[$key] = $value }
        }
        return $result
    }

    function Send-AgentPost($Client, [string]$Url, $Payload) {
        $content = New-Object System.Net.Http.StringContent((ConvertTo-AgentJson $Payload), [System.Text.Encoding]::UTF8, 'application/json')
        try {
            $response = $Client.PostAsync($Url, $content).GetAwaiter().GetResult()
            try {
                $text = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
                return @{ Status = [int]$response.StatusCode; Text = $text }
            } finally { $response.Dispose() }
        } finally { $content.Dispose() }
    }

    function Get-ErrorMessage($Reply) {
        try { return [string](Get-Field (ConvertFrom-AgentJson $Reply.Text) 'error' '') } catch { return '' }
    }

    function Test-RetryableStatus([int]$Code) { return ($Code -ge 500 -or $Code -eq 408 -or $Code -eq 429) }

    function Invoke-AgentTask($S, $Task) {
        $rawPath = [string](Get-Field $Task 'path' '')
        $query = ''
        $q = $rawPath.IndexOf('?')
        if ($q -ge 0) { $query = $rawPath.Substring($q + 1); $rawPath = $rawPath.Substring(0, $q) }
        $headers = @{}
        $taskHeaders = Get-Field $Task 'headers' @{}
        foreach ($k in @($taskHeaders.Keys)) { $headers[([string]$k).ToLowerInvariant()] = [string]$taskHeaders[$k] }
        $request = @{
            Method  = ([string](Get-Field $Task 'method' 'GET')).ToUpperInvariant()
            Path    = $rawPath
            Query   = (ConvertFrom-QueryString $query)
            Headers = $headers
            Body    = [Convert]::FromBase64String([string](Get-Field $Task 'body' ''))
        }
        $response = Invoke-Dispatch $S $request
        $payload = @{
            name = $S.Name; instance = $S.Instance; id = (Get-Field $Task 'id')
            status = [int]$response.Status; headers = $response.Headers
            body = [Convert]::ToBase64String([byte[]]$response.Body)
        }
        $delay = 1
        for ($attempt = 0; $attempt -lt $RESULT_RETRIES; $attempt++) {
            try {
                $reply = Send-AgentPost $S.ResultClient ($S.Server + '/v1/agents/result') $payload
                if ($reply.Status -lt 300) { return }
                if (-not (Test-RetryableStatus $reply.Status)) {
                    Write-AgentLog "result for $($request.Method) $($request.Path) rejected: HTTP $($reply.Status)"
                    return
                }
                $reason = "HTTP $($reply.Status): $(Get-ErrorMessage $reply)"
            } catch {
                $reason = $_.Exception.GetBaseException().Message
            }
            if ($attempt -eq $RESULT_RETRIES - 1) {
                Write-AgentLog "dropping result for $($request.Method) $($request.Path): $reason"
                return
            }
            Write-AgentLog "result for $($request.Method) $($request.Path) failed: $reason; retrying in ${delay}s"
            Start-Sleep -Seconds $delay
            $delay = [Math]::Min($delay * 2, 30)
        }
    }
}

function Get-PidFile([string]$Dir, [string]$AgentName) { return Join-Path $Dir "$AgentName.pid" }

function Get-RunningAgent([string]$PidFile) {
    if (-not (Test-Path -LiteralPath $PidFile)) { return $null }
    $raw = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    $agentPid = 0
    if (-not [int]::TryParse($raw, [ref]$agentPid) -or $agentPid -eq $PID) { return $null }
    $process = Get-Process -Id $agentPid -ErrorAction SilentlyContinue
    if ($process -and $process.ProcessName -in 'powershell', 'pwsh') { return $process }
    return $null
}

function Quote-Single([string]$Value) { return "'" + $Value.Replace("'", "''") + "'" }

function New-HttpClient([string]$Proxy, [int]$TimeoutSeconds) {
    Add-Type -AssemblyName System.Net.Http
    $handler = New-Object System.Net.Http.HttpClientHandler
    if ($Proxy) {
        $handler.Proxy = New-Object System.Net.WebProxy($Proxy)
        $handler.UseProxy = $true
    }
    $client = New-Object System.Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSeconds)
    return $client
}

function Start-Background {
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $scriptPath = Join-Path $InstallDir 'reverse_agent.ps1'
    [System.IO.File]::WriteAllText($scriptPath, $SelfText, [System.Text.UTF8Encoding]::new($true))
    $call = "& ([scriptblock]::Create([IO.File]::ReadAllText($(Quote-Single $scriptPath), [Text.Encoding]::UTF8))) " +
        "$(Quote-Single $Name) -AgentEnv $(Quote-Single $AgentEnv) -MaxUploadGiB $MaxUploadGiB"
    if ($Root.Count) { $call += ' -Root ' + (($Root | ForEach-Object { Quote-Single $_ }) -join ',') }
    if ($Server) { $call += ' -Server ' + (Quote-Single $Server) }
    if ($StateDir) { $call += ' -StateDir ' + (Quote-Single $StateDir) }
    $call += ' -InstallDir ' + (Quote-Single $InstallDir)
    $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($call))
    $log = Join-Path $InstallDir "$Name.log"
    $shell = (Get-Process -Id $PID).Path
    $process = Start-Process -FilePath $shell -WindowStyle Hidden -PassThru `
        -ArgumentList '-NoLogo', '-NoProfile', '-NonInteractive', '-EncodedCommand', $encoded `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    Start-Sleep -Seconds 5
    Write-Host "remote-agent '$Name' started in background (pid $($process.Id)); log: $log"
    Get-Content -LiteralPath $log, "$log.err" -ErrorAction SilentlyContinue | Select-Object -Last 10 | ForEach-Object { Write-Host $_ }
}

function Start-Agent {
    . $Lib
    if (-not $Server) { $Server = if ($env:REMOTE_FLEET_SERVER) { $env:REMOTE_FLEET_SERVER } else { $DefaultServer } }
    $Server = $Server.TrimEnd('/')
    if (-not $StateDir) { $StateDir = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.local\state\dev-agent' }
    $roots = @($Root | ForEach-Object { [System.IO.Path]::GetFullPath($_) })
    if (-not $roots.Count) { $roots = @([Environment]::GetFolderPath('UserProfile')) }
    $logsDir = Join-Path $StateDir 'jobs'
    [void][System.IO.Directory]::CreateDirectory($logsDir)

    $proxy = $EnvProxies[$AgentEnv]
    if (-not $proxy -and $AgentEnv -eq 'null') {
        $proxy = if ($Server.StartsWith('https:')) { $env:HTTPS_PROXY, $env:https_proxy } else { $env:HTTP_PROXY, $env:http_proxy }
        $proxy = @($proxy | Where-Object { $_ }) | Select-Object -First 1
    }
    $instance = [guid]::NewGuid().ToString('N')
    $S = [hashtable]::Synchronized(@{
        Name = $Name; Instance = $instance; Server = $Server
        Roots = [string[]]$roots; LogsDir = $logsDir; AuditPath = (Join-Path $StateDir 'audit.jsonl')
        MaxUpload = [long]($MaxUploadGiB * 1GB)
        Jobs = [hashtable]::Synchronized(@{}); PathLocks = [hashtable]::Synchronized(@{}); Lock = (New-Object object)
        ResultClient = (New-HttpClient $proxy 120)
    })
    $pollClient = New-HttpClient $proxy ($PollWait + 30)
    $registration = @{
        name = $Name; instance = $instance; hostname = [Environment]::MachineName; pid = $PID
        roots = [string[]]$roots; env = $(if ($AgentEnv -eq 'null') { $null } else { $AgentEnv }); wait = $PollWait
    }

    $pool = [runspacefactory]::CreateRunspacePool(1, 32)
    $pool.Open()
    $workerScript = "param(`$S, `$Task)`n" + $Lib.ToString() + "`nInvoke-AgentTask `$S `$Task"
    $workers = New-Object System.Collections.ArrayList

    $route = if ($proxy) { "via $proxy" } else { 'system proxy, if any' }
    Write-AgentLog "agent '$Name' (pid $PID) dialing $Server ($route)"
    Write-AgentLog ('allowed roots: ' + ($roots -join ', '))
    $backoff = 1
    $connected = $false
    while ($true) {
        foreach ($w in @($workers)) {
            if ($w.Handle.IsCompleted) {
                try { [void]$w.Shell.EndInvoke($w.Handle) } catch { Write-AgentLog "task failed: $($_.Exception.GetBaseException().Message)" }
                foreach ($e in $w.Shell.Streams.Error) { Write-AgentLog "task error: $e" }
                $w.Shell.Dispose()
                $workers.Remove($w)
            }
        }
        foreach ($job in @($S.Jobs.Values)) {
            if (-not $job.Reported -and $job.Process.HasExited) {
                $job.Reported = $true
                Write-AgentLog "job[$($job.Id.Substring(0, 8))] exited code=$($job.Process.ExitCode)"
            }
        }
        $registration['wait'] = $(if ($connected) { $PollWait } else { 0 })
        $failure = $null
        try {
            $reply = Send-AgentPost $pollClient ($Server + '/v1/agents/poll') $registration
        } catch {
            $failure = $_.Exception.GetBaseException().Message
            $reply = $null
        }
        if ($reply -and $reply.Status -eq 200) {
            if (-not $connected) { Write-AgentLog "connected to $Server as '$Name'"; $connected = $true }
            $backoff = 1
            $value = ConvertFrom-AgentJson $reply.Text
            foreach ($task in @(Get-Field $value 'tasks' @())) {
                $shell = [powershell]::Create()
                $shell.RunspacePool = $pool
                [void]$shell.AddScript($workerScript).AddArgument($S).AddArgument($task)
                [void]$workers.Add(@{ Shell = $shell; Handle = $shell.BeginInvoke() })
            }
            continue
        }
        if ($reply) {
            $connected = $false
            $message = Get-ErrorMessage $reply
            if ($reply.Status -eq 409) {
                Write-AgentLog "registration rejected: $message; retrying in 10s"
                Start-Sleep -Seconds 10
                continue
            }
            if ($reply.Status -ge 400 -and $reply.Status -lt 500) {
                throw "rendezvous rejected agent: HTTP $($reply.Status): $message"
            }
            Write-AgentLog "rendezvous error HTTP $($reply.Status): $message; retrying in ${backoff}s"
        } elseif ($connected) {
            Write-AgentLog "disconnected: $failure"
            $connected = $false
        } else {
            Write-AgentLog "cannot reach ${Server}: $failure; retrying in ${backoff}s"
        }
        Start-Sleep -Seconds $backoff
        $backoff = [Math]::Min($backoff * 2, $MaxBackoff)
    }
}

if (-not $Name) { throw 'usage: reverse_agent.ps1 <device-name> [-AgentEnv merlin|volcano|null] [-Root C:\ ...] [-Background|-Stop]' }
if ($Name -notmatch '^[A-Za-z0-9_.-]+$') { throw "invalid device name: $Name" }
if (-not $InstallDir) { $InstallDir = Join-Path $env:LOCALAPPDATA 'remote-agent' }
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$pidFile = Get-PidFile $InstallDir $Name
$existing = Get-RunningAgent $pidFile

if ($Stop) {
    if ($existing) {
        & (Join-Path $env:SystemRoot 'System32\taskkill.exe') /PID $existing.Id /T /F | Out-Null
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
        Write-Host "stopped remote-agent '$Name' (pid $($existing.Id))"
    } else {
        Write-Host "remote-agent '$Name' is not running"
    }
    return
}
if ($existing) {
    Write-Host "remote-agent '$Name' already running (pid $($existing.Id)); log: $(Join-Path $InstallDir "$Name.log")"
    return
}
if ($Background) { Start-Background; return }

Set-Content -LiteralPath $pidFile -Value $PID
try { Start-Agent }
finally {
    if ((Get-Content -LiteralPath $pidFile -Raw -ErrorAction SilentlyContinue) -match "^$PID\b") {
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    }
}
