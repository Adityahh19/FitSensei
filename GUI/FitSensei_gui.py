from pathlib import Path
from tkinter import Tk, Canvas, Button
from PIL import Image, ImageTk
import tkinter.font as tkfont
import subprocess

# Store the process globally if you need to access it later
current_process = None

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"/Users/Aditya/Minor Project/Fit-Sensei-GUI-main")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def run_script(script_path):
    global current_process
    if current_process is not None:
        current_process.terminate()  # Terminate the previous process if needed
        current_process = None
    current_process = subprocess.Popen(["python3", script_path])

def terminate_script():
    global current_process
    if current_process is not None:
        current_process.terminate()  # Terminate the running process
        current_process = None

# Helper function to load images with transparency using Pillow
def load_image_with_transparency(image_path):
    image = Image.open(image_path)
    return ImageTk.PhotoImage(image)

window = Tk()

# Set window size and background color
window.geometry("1440x900")
window.configure(bg="#090808")

canvas = Canvas(
    window,
    bg="#090808",
    height=1024,
    width=1440,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)

# Create a rectangle for the top section
canvas.create_rectangle(
    0.0,
    0.0,
    1440.0,
    163.0,
    fill="#000000",
    outline=""
)

# Load and display the image
image_path = relative_to_assets("image_1.png")
image_image_1 = load_image_with_transparency(image_path)
image_width, image_height = Image.open(image_path).size

# Center the image horizontally at the top with some padding
image_x = 400
image_y = 1  # Adjust the vertical position as needed

canvas.create_image(
    image_x,
    image_y,
    image=image_image_1,
    anchor="nw"
)

# Font settings
font_name = "Michroma"
font_size = 48  # Adjust font size
font = tkfont.Font(family=font_name, size=font_size)

# Text to display
title_text = "Fit Sensei"

# Measure text width and height
text_width = font.measure(title_text)
text_height = font.metrics("linespace")

# Center the text horizontally with the image and vertically aligned
text_x = 550 # Center horizontally
text_y = 10
# Create text beside the image
canvas.create_text(
    text_x,
    text_y,
    anchor="nw",
    text=title_text,
    fill="#28E3DA",
    font=(font_name, font_size)
)

# Load and display buttons
button_image_1 = load_image_with_transparency(relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: run_script("/Users/Aditya/Minor Project/Program Files/plank/PlankPose_Detection.py"),
    relief="flat",
    bg="#090808",
    activebackground="#090808"
)
button_1.place(
    x=448.0,
    y=221.0,
    width=543.0,
    height=104.0
)

button_image_2 = load_image_with_transparency(relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_2 clicked"),
    relief="flat",
    bg="#090808",
    activebackground="#090808"
)
button_2.place(
    x=448.0,
    y=374.0,
    width=535.0,
    height=104.0
)

button_image_3 = load_image_with_transparency(relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: run_script("/Users/Aditya/Minor Project/Program Files/squats/SquatCorrection copy 2.py"),
    relief="flat",
    bg="#090808",
    activebackground="#090808"
)
button_3.place(
    x=448.0,
    y=526.0,
    width=543.0,
    height=104.0
)

button_image_4 = load_image_with_transparency(relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_4 clicked"),
    relief="flat",
    bg="#090808",
    activebackground="#090808"
)
button_4.place(
    x=448.0,
    y=676.0,
    width=543.0,
    height=104.0
)

button_image_5 = load_image_with_transparency(relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: run_script("/Users/Aditya/Minor Project/Program Files/shoulder press/Shoulder Press LeftRight copy.py"),
    relief="flat",
    bg="#090808",
    activebackground="#090808"
)
button_5.place(
    x=450.0,
    y=826.0,
    width=544.0,
    height=102.0
)

window.resizable(False, False)
window.mainloop()


