import os
import sys
import tkinter as tk
from PIL import Image, ImageTk
import subprocess

def load_asset(path):
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    assets = os.path.join(base, "assets")
    return os.path.join(assets, path)

def load_image(path):
    img = Image.open(path).convert("RGBA")  # Ensure image is in RGBA mode for transparency
    return ImageTk.PhotoImage(img)

# Initialize the main Tkinter window
window = tk.Tk()
window.geometry("1440x1024")
window.configure(bg="#090808")
window.title("Untitled")

# Create the canvas
canvas = tk.Canvas(
    window,
    bg="#090808",
    width=1440,
    height=1024,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)

canvas.create_rectangle(0, 0, 1440, 163, fill='#000000', outline="")

canvas.create_text(
    600,
    42,
    anchor="nw",
    text="Fit Sensei",
    fill="#28e3da",
    font=("Michroma", 48 * -1)
)

# Load images
image_1 = load_image(load_asset("image_1.png"))
canvas.create_image(525, 100, image=image_1)

button_1_image = load_image(load_asset("/Users/Aditya/Minor Project/Fit-Sensei-GUI-main/gui_ver2/assets/1.png"))
button_2_image = load_image(load_asset("/Users/Aditya/Minor Project/Fit-Sensei-GUI-main/gui_ver2/assets/2.png"))
button_3_image = load_image(load_asset("/Users/Aditya/Minor Project/Fit-Sensei-GUI-main/gui_ver2/assets/3.png"))
button_4_image = load_image(load_asset("/Users/Aditya/Minor Project/Fit-Sensei-GUI-main/gui_ver2/assets/4.png"))
button_5_image = load_image(load_asset("/Users/Aditya/Minor Project/Fit-Sensei-GUI-main/gui_ver2/assets/5.png"))

# Variable to hold the script process
script_process = None

# Function to start the external script
def start_script(script_path):
    global script_process
    if script_process is None:
        script_process = subprocess.Popen(["python3", script_path])

# Function to stop the external script
def stop_script():
    global script_process
    if script_process is not None:
        script_process.terminate()  # Terminate the script process
        script_process = None

# Button click handlers for each exercise
def on_button_1_click():
    start_script("/Users/Aditya/Minor Project/Program Files/plank/PlankPose_Detection.py")  # Replace with your actual script file path

def on_button_2_click():
    start_script("/Users/Aditya/Minor Project/Program Files/pushups/pushup_correction.py")  # Replace with your actual script file path

def on_button_3_click():
    start_script("/Users/Aditya/Minor Project/Program Files/squats/Squat.py")  # Replace with your actual script file path

def on_button_4_click():
    start_script("/Users/Aditya/Minor Project/Program Files/bicep curl/bicep_curl.py")  # Replace with your actual script file path

def on_button_5_click():
    start_script("/Users/Aditya/Minor Project/Program Files/shoulder press/Shoulder Press LeftRight copy.py")  # Replace with your actual script file path

# Create buttons for each exercise
button_1 = tk.Label(
    window,
    image=button_1_image,
    bg="#090808",  # Match canvas background color
    borderwidth=0,
    highlightthickness=0
)
button_1.place(x=448, y=221, width=543, height=104)
button_1.bind("<Button-1>", lambda e: on_button_1_click())

button_2 = tk.Label(
    window,
    image=button_2_image,
    bg="#090808",  # Match canvas background color
    borderwidth=0,
    highlightthickness=0
)
button_2.place(x=448, y=374, width=535, height=104)
button_2.bind("<Button-1>", lambda e: on_button_2_click())

button_3 = tk.Label(
    window,
    image=button_3_image,
    bg="#090808",  # Match canvas background color
    borderwidth=0,
    highlightthickness=0
)
button_3.place(x=448, y=526, width=543, height=104)
button_3.bind("<Button-1>", lambda e: on_button_3_click())

button_4 = tk.Label(
    window,
    image=button_4_image,
    bg="#090808",  # Match canvas background color
    borderwidth=0,
    highlightthickness=0
)
button_4.place(x=448, y=676, width=543, height=104)
button_4.bind("<Button-1>", lambda e: on_button_4_click())

button_5 = tk.Label(
    window,
    image=button_5_image,
    bg="#090808",  # Match canvas background color
    borderwidth=0,
    highlightthickness=0
)
button_5.place(x=450, y=826, width=544, height=102)
button_5.bind("<Button-1>", lambda e: on_button_5_click())

# Create a smaller button to stop any running script, styled as a red box with black text
stop_button = tk.Button(
    window,
    text="Stop Script",
    command=stop_script,
    bg="#FF0000",  # Red background
    fg="#000000",  # Black text
    font=("Arial", 14),
    borderwidth=0,
    highlightthickness=0
)
stop_button.place(x=650, y=950, width=150, height=50)  # Smaller size but aligned horizontally

# Set window properties
window.resizable(False, False)
window.mainloop()
