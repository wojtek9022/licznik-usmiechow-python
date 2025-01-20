import os
import subprocess
from pathlib import Path

# FIXME: You might need to turn off your antivirus software to run this script
# Windows treats it as a virus because it uses raw pyinstaller command 
# instead of bootstrap version

def cleanup_build_files(output_dir: Path, app_name: str) -> None:
    """Remove all build files except .exe."""
    try:
        # Remove build directory
        if (output_dir / 'build').exists():
            import shutil
            shutil.rmtree(output_dir / 'build')
        
        # Remove spec file
        spec_file = output_dir / f"{app_name}.spec"
        if spec_file.exists():
            spec_file.unlink()
            
        # Remove any .log files
        for log_file in output_dir.glob('*.log'):
            log_file.unlink()
            
        # Remove any .manifest files
        for manifest in output_dir.glob('*.manifest'):
            manifest.unlink()
            
        print("Cleanup completed successfully!")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def build_exe():
    # Get project root directory (5 levels up from scripts folder)
    project_root = Path(__file__).parents[5]
    
    # Define paths and names
    APP_NAME = "Smile Counter 2"
    # Update output path to be directly under project root
    output_dir = project_root / "output"
    icon_path = project_root / "smile_counter" / "app" / "src" / "data" / "img" / "menu" / "icon.ico"
    app_path = project_root / "smile_counter" / "app"
    main_script = project_root / "smile_counter" / "smile_counter_app.py"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Version info
    VERSION = "3.5.1"
    FILE_VERSION = VERSION.replace('.', ',')
    
    # Create version info
    version_info = f"""
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({FILE_VERSION},0),
    prodvers=({FILE_VERSION},0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        u'040904B0',
        [StringStruct(u'FileDescription', u'Smile Counter'),
        StringStruct(u'FileVersion', u'{VERSION}'),
        StringStruct(u'InternalName', u'smile_counter'),
        StringStruct(u'LegalCopyright', u'Milosz Malak, Wojciech Goras 2024-2025'),
        StringStruct(u'OriginalFilename', u'smile_counter.exe'),
        StringStruct(u'ProductName', u'Smile Counter'),
        StringStruct(u'ProductVersion', u'{VERSION}')])
    ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)"""

    # Write version info to file
    version_file = project_root / "file_version_info.txt"
    version_file.write_text(version_info)
    
    # Build command
    command = [
        "pyinstaller",
        "--clean",  # Clean PyInstaller cache
        "--noconfirm",
        "--onefile",
        "--windowed",
        f"--icon={icon_path}",
        f"--version-file={version_file}",
        f"--add-data={app_path};app/",
        f"--distpath={output_dir}",  # Set output directory
        f"--workpath={output_dir / 'build'}",  # Set build directory
        f"--specpath={output_dir}",  # Set spec file directory
        f"--name={APP_NAME}",  # Set executable name
        "--hidden-import=tkinter",
        "--hidden-import=tkinter.ttk",
        "--hidden-import=tkinter.filedialog",
        "--hidden-import=PIL",
        "--hidden-import=cv2",
        "--hidden-import=PIL.Image",
        "--hidden-import=configparser",
        "--hidden-import=win32com",
        "--disable-windowed-traceback",  # Reduce executable size
        "--noupx",  # Disable UPX compression
        str(main_script)
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Build completed successfully! Version: {VERSION}")
        version_file.unlink()  # Clean up version file
        cleanup_build_files(output_dir, APP_NAME)  # Clean up build files
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        if version_file.exists():
            version_file.unlink()

if __name__ == "__main__":
    build_exe()