import cv2
import win32com.client
import pythoncom
import logging

def get_available_cameras():
    """Get list of available cameras from Windows Device Manager."""
    available_cameras = []
    
    try:
        pythoncom.CoInitialize()
        wmi = win32com.client.GetObject("winmgmts:")
        
        # Broader query to catch more camera types
        query = """SELECT * FROM Win32_PnPEntity WHERE 
                  (PNPClass = 'Image' OR 
                   PNPClass = 'Camera' OR 
                   DeviceID LIKE '%USB%VID%PID%' AND 
                   (Caption LIKE '%Camera%' OR Caption LIKE '%Webcam%'))"""
        
        cameras = wmi.ExecQuery(query)
        system_cameras = []
        
        for camera in cameras:
            if hasattr(camera, 'Name'):
                system_cameras.append(camera.Name)
                logging.debug(f"Found camera in WMI: {camera.Name}")
        
        # Test physical camera devices
        for i in range(5):  # Check first 5 indices
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        name = f"Camera {i}"
                        # Try to match with system name if available
                        if i < len(system_cameras):
                            name = system_cameras[i]
                        available_cameras.append((i, name))
                        logging.debug(f"Found working camera: {name} at index {i}")
                cap.release()
            except Exception as camera_error:
                logging.debug(f"Error testing camera {i}: {camera_error}")
                
    except Exception as e:
        logging.error(f"Camera detection error: {e}")
    
    finally:
        pythoncom.CoUninitialize()
    
    # Fallback to default if no cameras found
    if not available_cameras:
        logging.warning("No cameras found, using default camera")
        available_cameras = [(0, "Default Camera")]
    
    return available_cameras