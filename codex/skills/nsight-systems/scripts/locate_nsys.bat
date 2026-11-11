@echo off
setlocal disabledelayedexpansion
rem #######################################
rem Print the path of the Nsys CLI executable this machine should use.
rem
rem Checks NSYS_PATH first, then PATH, then the newest runnable installation in
rem known Windows locations.
rem README.md defines the shared contract; errors.txt supplies failure diagnostics.
rem Globals:
rem   Reads NSYS_PATH as the directory containing the Nsys CLI executable.
rem   Reads PATH for executable lookup.
rem   Reads PROCESSOR_ARCHITEW6432 and PROCESSOR_ARCHITECTURE for target selection.
rem   Reads ProgramW6432, ProgramFiles, and SystemDrive for installation discovery.
rem   Keeps discovery and diagnostic variables local with setlocal.
rem Arguments:
rem   None.
rem Outputs:
rem   On success, writes the bare executable path and a newline to standard output.
rem   On failure, writes an ERROR: diagnostic to standard error.
rem   Suppresses output from candidate executables' --version checks.
rem Returns:
rem   Exits the script with status 0 when a runnable executable is found; 1 on resolution failure.
rem Examples:
rem   locate_nsys.bat
rem   Prints C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.1\target-windows-x64\nsys.exe on success.
rem #######################################
rem
rem Editing rules, all three forced by cmd's parser. Delayed expansion stays off, because
rem it would read a "!" in a path as a variable reference; every path is therefore read
rem inside quotes, which is what holds a %var% result together once it is part of the
rem parsed line. The echo that prints a path is the exception -- :emit and :fail_print turn
rem delayed expansion on for that line alone, since a path cannot be quoted there and this
rem is what keeps it out of the parser instead. A subroutine takes a path from a variable
rem the caller sets first, named in its comment, because call re-parses an argument and
rem would consume a "!", a "%" or a "^". A variable written inside a parenthesised block
rem is read past the end of it, through goto, since %var% inside the block expands while
rem the block is parsed.

rem The script's own directory, captured before anything can disturb it.
set "here=%~dp0"

goto :main

rem #######################################
rem Load shared diagnostics once, when the first error is reported.
rem Globals:
rem   Reads here and errors_loaded.
rem   Writes errors_loaded and msg_* diagnostic variables.
rem Arguments:
rem   None.
rem Outputs:
rem   None on success; filesystem errors may reach standard error.
rem Returns:
rem   Status unspecified; a missing errors.txt file is tolerated.
rem #######################################
:load_errors
if defined errors_loaded goto :eof
set "errors_loaded=1"
if not exist "%here%errors.txt" goto :eof
rem Keep complete diagnostic segments after the first equals sign and skip comments.
for /f "usebackq eol=# tokens=1* delims==" %%A in ("%here%errors.txt") do set "msg_%%A=%%B"
goto :eof

rem #######################################
rem Print a shared diagnostic with values between its segments.
rem Globals:
rem   Reads emit_value1 and emit_value2 in insertion order; unused values must be empty.
rem   Reads msg_* diagnostics.
rem   Initializes diagnostics through :load_errors.
rem   Uses fail_seg0, fail_seg1, and fail_seg2 as scratch variables.
rem Arguments:
rem   1: Trusted diagnostic key defined in errors.txt.
rem Outputs:
rem   Writes one diagnostic line to standard error.
rem Returns:
rem   Status unspecified; the caller must exit the script with a failure status.
rem Examples:
rem   Clear emit_value1 and emit_value2, then call :fail err_root.
rem   Set emit_value1 to mips64, clear emit_value2, then call :fail err_arch_unsupported.
rem #######################################
:fail
call :load_errors
rem Values stay in globals so call reparses only the trusted diagnostic key.
call set "fail_seg0=%%msg_%~1%%"
call set "fail_seg1=%%msg_%~1_1%%"
call set "fail_seg2=%%msg_%~1_2%%"
if defined fail_seg0 goto :fail_print
rem The key names the failure whenever errors.txt is unavailable.
set "fail_seg0=ERROR: %~1 "
set "fail_seg1="
set "fail_seg2="
:fail_print
setlocal enabledelayedexpansion
>&2 echo(!fail_seg0!!emit_value1!!fail_seg1!!emit_value2!!fail_seg2!
endlocal
goto :eof

rem #######################################
rem Print the resolved path without interpreting its characters as command text.
rem Globals:
rem   Reads out; changes no global variables.
rem Arguments:
rem   None.
rem Outputs:
rem   Writes out followed by a newline to standard output.
rem Returns:
rem   Status unspecified; the caller controls the script's exit status.
rem #######################################
:emit
rem Delayed expansion inserts the path after parsing and does not rescan its contents.
setlocal enabledelayedexpansion
echo(!out!
endlocal
goto :eof

rem #######################################
rem Confirm that the Nsys CLI executable can run on this machine.
rem Globals:
rem   Reads candidate; changes no global variables.
rem Arguments:
rem   None; the caller sets candidate to the Nsys CLI executable path.
rem Outputs:
rem   None; the version command's output is suppressed.
rem Returns:
rem   0 for a file whose version command succeeds; 1 otherwise.
rem #######################################
:validate_executable
if not exist "%candidate%" exit /b 1
if exist "%candidate%\" exit /b 1
"%candidate%" --version >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

rem #######################################
rem Find a runnable Nsys CLI executable in an installation's target directories.
rem Globals:
rem   Reads vr_dir and platforms in preference order.
rem   Writes found_platform and the candidate global used by :try_platform.
rem Arguments:
rem   None; the caller sets vr_dir to the installation root.
rem Outputs:
rem   None.
rem Returns:
rem   Status unspecified; found_platform is the first runnable target or empty.
rem #######################################
:validate_root
set "found_platform="
for %%P in (%platforms%) do if not defined found_platform call :try_platform %%P
goto :eof

rem #######################################
rem Check one target directory for a runnable Nsys CLI executable.
rem Globals:
rem   Reads vr_dir.
rem   Writes candidate; sets found_platform on success and leaves it unchanged on failure.
rem Arguments:
rem   1: Trusted platform name from platforms.
rem Outputs:
rem   None.
rem Returns:
rem   Status unspecified; found_platform communicates success.
rem #######################################
:try_platform
rem Read the path outside the for loop so loop-variable substitution cannot alter it.
set "candidate=%vr_dir%\target-%1\nsys.exe"
call :validate_executable
if not errorlevel 1 set "found_platform=%1"
goto :eof

rem #######################################
rem Accumulate the newest runnable installation from one parent directory.
rem Globals:
rem   Reads sel_dir, platforms, and best_key.
rem   Updates root, platform, and best_key only when a newer candidate qualifies.
rem   Uses con_dir and con_name as scratch variables; :consider's global effects apply.
rem   Restores the caller's working directory after scanning.
rem Arguments:
rem   None; the caller sets sel_dir and initializes the best-match globals before the first call.
rem Outputs:
rem   None on success; directory-access errors may reach standard error.
rem Returns:
rem   Status unspecified; root and platform contain the best accumulated match.
rem #######################################
:select_root
rem The parent is entered, so the pattern names the version directories alone, for the
rem reason given on :try_platform: a "%%D" inside sel_dir would be read as this loop's
rem variable.
if not exist "%sel_dir%\" goto :eof
pushd "%sel_dir%" || goto :eof
for /d %%D in ("Nsight Systems *") do (
    set "con_dir=%%~fD"
    set "con_name=%%~nxD"
    call :consider
)
popd
goto :eof

rem #######################################
rem Promote a runnable installation when its version outranks the current best.
rem Globals:
rem   Reads con_dir, con_name, platforms, and best_key.
rem   Updates root, platform, and best_key on promotion; keeps them unchanged on a tie.
rem   Uses vk_ver, first, and vr_dir as scratch variables.
rem   Calls :validate_root and :version_key; their global effects apply.
rem Arguments:
rem   None; the caller sets con_dir to the full path and con_name to its directory name.
rem Outputs:
rem   None.
rem Returns:
rem   Status unspecified; the best-match globals communicate the result.
rem #######################################
:consider
set "vk_ver=%con_name:Nsight Systems =%"
set "first=%vk_ver:~0,1%"
if "%first%" LSS "0" goto :eof
if "%first%" GTR "9" goto :eof
set "vr_dir=%con_dir%"
call :validate_root
if not defined found_platform goto :eof
call :version_key
rem Quoted keys compare as strings; cmd's 32-bit numeric comparison would overflow.
if defined best_key if not "%key%" GTR "%best_key%" goto :eof
set "best_key=%key%"
set "root=%con_dir%"
set "platform=%found_platform%"
goto :eof

rem #######################################
rem Extract leading ASCII digits from a version component, preserving any suffix.
rem Globals:
rem   Reads and consumes the leading digits from digits_in.
rem   Writes digits_out; uses digits_c as a scratch variable.
rem Arguments:
rem   None; the caller sets digits_in to the component or an empty value.
rem Outputs:
rem   None.
rem Returns:
rem   Status unspecified; digits_out contains the result and digits_in retains the suffix.
rem Examples:
rem   For digits_in=3rc1, writes digits_out=3 and leaves digits_in=rc1.
rem   For digits_in=12, writes digits_out=12 and leaves digits_in empty.
rem   For an empty digits_in, leaves digits_out empty.
rem #######################################
:digits
set "digits_out="
:digits_loop
if "%digits_in%"=="" goto :eof
set "digits_c=%digits_in:~0,1%"
if "%digits_c%" LSS "0" goto :eof
if "%digits_c%" GTR "9" goto :eof
set "digits_out=%digits_out%%digits_c%"
set "digits_in=%digits_in:~1%"
goto :digits_loop

rem #######################################
rem Build a fixed-width version key that preserves numeric version ordering.
rem Globals:
rem   Reads vk_ver.
rem   Writes key, digits_in, and digits_out.
rem   Uses vk_c1 through vk_c4, k1 through k4, and digits_c as scratch variables.
rem Arguments:
rem   None; the caller sets vk_ver to a version with one to four dot-separated components.
rem Outputs:
rem   None.
rem Returns:
rem   Status unspecified; key contains the result, with missing components padded to zero.
rem Examples:
rem   For vk_ver=2026.9.2, writes key=20260009000200000.
rem   For vk_ver=2026.10.1, writes key=20260010000100000.
rem   For vk_ver=2026.3, writes key=20260003000000000.
rem   For vk_ver=2026.3.1rc2, writes key=20260003000100000.
rem #######################################
:version_key
rem skill_pack_paths.py version_key is the Python equivalent.
rem The version is read with ".0.0.0" behind it, so all four tokens are always there and
rem each of %%a to %%d is a for variable this loop has defined. vk_c1..vk_c4 carry the raw
rem components past the end of the loop, since a variable set inside it reads back its
rem pre-loop value if read there too.
for /f "tokens=1-4 delims=." %%a in ("%vk_ver%.0.0.0") do (
    set "vk_c1=%%a"
    set "vk_c2=%%b"
    set "vk_c3=%%c"
    set "vk_c4=%%d"
)
set "digits_in=%vk_c1%"
call :digits
set "k1=0000%digits_out%"
set "digits_in=%vk_c2%"
call :digits
set "k2=0000%digits_out%"
set "digits_in=%vk_c3%"
call :digits
set "k3=0000%digits_out%"
set "digits_in=%vk_c4%"
call :digits
set "k4=00000%digits_out%"
set "key=%k1:~-4%%k2:~-4%%k3:~-4%%k4:~-5%"
goto :eof

:main
rem 1. NSYS_PATH names the directory containing the Nsys CLI executable. It is set deliberately,
rem    so it answers on its own, once it is confirmed runnable. A caller may
rem    quote it, and dropping those quotes gives the value they meant; one trailing
rem    backslash goes too, which is what a shell completing a directory name leaves behind.
if not defined NSYS_PATH goto :detect
set "NSYS_PATH=%NSYS_PATH:"=%"
if "%NSYS_PATH:~-1%"=="\" set "NSYS_PATH=%NSYS_PATH:~0,-1%"
set "out=%NSYS_PATH%\nsys.exe"
if not exist "%out%" goto :fail_named
if exist "%out%\" goto :fail_named
set "candidate=%out%"
call :validate_executable
if errorlevel 1 goto :fail_not_runnable
call :emit
exit /b 0

rem out is the path NSYS_PATH named. It is the one place this strategy was told about, so
rem it is reported rather than going on to guess at another.
:fail_named
set "emit_value1=%out%"
set "emit_value2="
call :fail err_nsys_missing
exit /b 1

:fail_not_runnable
set "emit_value1=%out%"
set "emit_value2="
call :fail err_nsys_not_runnable
exit /b 1

:detect
rem 2. The Nsys CLI executable already on PATH, which is what the machine is configured to use.
rem
rem    $PATH: performs that lookup inside this process, where `where nsys` would start a
rem    cmd.exe and a where.exe to do it. It expands to the full path of the first match,
rem    but the binary may still fail to load or run. The Nsys CLI executable's filename is
rem    spelled in full, since PATHEXT applies to a typed command.
for %%I in (nsys.exe) do set "onpath=%%~$PATH:I"
if not defined onpath goto :arch
set "out=%onpath%"
set "candidate=%out%"
call :validate_executable
if errorlevel 1 goto :arch
call :emit
exit /b 0

rem The rest builds a path of its own, so from here the target tree has to be named. The
rem running machine names it, and an unknown arch is an error. PROCESSOR_ARCHITEW6432 is
rem read first, as it reports the true machine when this cmd.exe is itself emulated.
:arch
set "arch=%PROCESSOR_ARCHITEW6432%"
if not defined arch set "arch=%PROCESSOR_ARCHITECTURE%"
if defined arch goto :arch_known
set "emit_value1="
set "emit_value2="
call :fail err_arch
exit /b 1
:arch_known
set "platforms="
if /i "%arch%"=="AMD64" set "platforms=windows-x64"
rem Windows-on-Arm runs x64 targets under emulation, so keep that as a fallback
if /i "%arch%"=="ARM64" set "platforms=windows-armv8 windows-x64"
if defined platforms goto :arch_supported
set "emit_value1=%arch%"
set "emit_value2="
call :fail err_arch_unsupported
exit /b 1
:arch_supported

rem setlocal copies the caller's environment, so every variable read before it is first
rem written here is cleared here.
set "root="
set "platform="
set "found_platform="
set "best_key="

rem 3. The newest installation on this machine, folding each known parent into the running
rem    best. %ProgramW6432% and %ProgramFiles% name the same directory except under an
rem    emulated cmd.exe, where only the first reports the 64-bit tree and the second the
rem    32-bit one; whichever is missing from a stripped environment falls back to the other,
rem    and then to %SystemDrive%, which is the drive Windows is actually installed on. The
rem    second is scanned only when it names somewhere else, since listing one directory
rem    twice leaves the same installation winning.
set "prog64=%ProgramW6432%"
if not defined prog64 set "prog64=%ProgramFiles%"
if not defined prog64 set "prog64=%SystemDrive%\Program Files"
set "prog32=%ProgramFiles%"
if not defined prog32 set "prog32=%prog64%"
set "sel_dir=%prog64%\NVIDIA Corporation"
call :select_root
if /i "%prog32%"=="%prog64%" goto :scanned
set "sel_dir=%prog32%\NVIDIA Corporation"
call :select_root
:scanned
if defined root goto :resolved
set "emit_value1="
set "emit_value2="
call :fail err_root
exit /b 1

:resolved
rem :validate_root already proved the Nsys CLI executable runs, so this names it.
set "out=%root%\target-%platform%\nsys.exe"
call :emit
exit /b 0
