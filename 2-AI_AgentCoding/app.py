import os
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from groq import Groq

# Hardcoded Groq API Key
GROQ_API_KEY = ""

# Predefined challenges
PRESETS = [
    {
        "title": "🔢 Fibonacci Generator",
        "filename": "fibonacci.py",
        "prompt": "Create a Python script that generates the Fibonacci sequence up to a given number of terms. The script should include a function, input validation, and a command-line interface to test it."
    },
    {
        "title": "🌐 Simple Web Scraper",
        "filename": "web_scraper.py",
        "prompt": "Write a Python script using urllib (no external dependencies if possible, or print instructions if requests/bs4 is needed) to download a webpage, parse all links (<a> tags), and save them to a text file."
    },
    {
        "title": "📊 CSV Data Analyzer",
        "filename": "csv_analyzer.py",
        "prompt": "Build a CLI Python script that reads a CSV file, calculates basic statistics (mean, median, min, max) for numeric columns, lists unique values for categorical columns, and prints a neat summary table."
    },
    {
        "title": "📝 CLI To-Do App",
        "filename": "todo_app.py",
        "prompt": "Develop a CLI Todo list manager that allows users to add tasks, mark them as complete, delete tasks, and list all tasks. Save the tasks to a local JSON file so the data persists between runs."
    },
    {
        "title": "🎨 Tkinter Paint App",
        "filename": "paint_app.py",
        "prompt": "Create a simple GUI painting application using Tkinter. It should allow the user to draw on a canvas using the mouse, change the brush size, clear the canvas, and select different colors."
    }
]

class CodingAgentApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure Window
        self.title("AI Coding Agent & File Builder")
        self.geometry("1100x700")
        self.minsize(900, 600)

        # Set appearance and theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")  # Using default blue theme but overriding some colors for custom premium look

        # Custom Palette
        self.bg_color = "#12121e"
        self.sidebar_color = "#1a192c"
        self.card_color = "#242338"
        self.accent_purple = "#8A2BE2"
        self.accent_purple_hover = "#9932CC"
        self.text_color_bright = "#EAEAEA"
        self.text_color_muted = "#A0A0B0"

        # Apply Window Background
        self.configure(fg_color=self.bg_color)

        # Output Folder state
        self.output_dir = tk.StringVar(value=os.getcwd())
        self.generated_filepath = None

        # Setup GUI Grid Layout
        self.grid_columnconfigure(0, weight=1)  # Sidebar
        self.grid_columnconfigure(1, weight=3)  # Main panel
        self.grid_rowconfigure(0, weight=1)

        self.create_sidebar()
        self.create_main_panel()

    def create_sidebar(self):
        # Sidebar Frame
        self.sidebar = ctk.CTkFrame(self, fg_color=self.sidebar_color, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(7, weight=1)  # Spacer pushes content up

        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar, 
            text="⚙️ CODER AGENT", 
            font=ctk.CTkFont(family="Helvetica Neue", size=20, weight="bold"),
            text_color=self.text_color_bright
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10), sticky="w")

        self.logo_desc = ctk.CTkLabel(
            self.sidebar,
            text="Llama 3 Powered Agent",
            font=ctk.CTkFont(family="Helvetica Neue", size=12, weight="normal"),
            text_color=self.text_color_muted
        )
        self.logo_desc.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        # Divider
        self.divider = ctk.CTkFrame(self.sidebar, height=2, fg_color=self.card_color)
        self.divider.grid(row=2, column=0, sticky="ew", padx=20, pady=5)

        # Preset Challenges Header
        self.preset_label = ctk.CTkLabel(
            self.sidebar, 
            text="CHALLENGE PRESETS", 
            font=ctk.CTkFont(family="Helvetica Neue", size=11, weight="bold"),
            text_color=self.text_color_muted
        )
        self.preset_label.grid(row=3, column=0, padx=20, pady=(15, 10), sticky="w")

        # Add buttons for presets
        self.preset_buttons = []
        for i, preset in enumerate(PRESETS):
            btn = ctk.CTkButton(
                self.sidebar,
                text=preset["title"],
                anchor="w",
                font=ctk.CTkFont(family="Helvetica Neue", size=13),
                fg_color="transparent",
                text_color=self.text_color_bright,
                hover_color=self.card_color,
                height=38,
                command=lambda p=preset: self.load_preset(p)
            )
            btn.grid(row=4 + i, column=0, padx=15, pady=4, sticky="ew")
            self.preset_buttons.append(btn)

        # Theme toggle (Light/Dark mode)
        self.appearance_mode_label = ctk.CTkLabel(
            self.sidebar, 
            text="Theme Mode:", 
            font=ctk.CTkFont(family="Helvetica Neue", size=12),
            text_color=self.text_color_muted
        )
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(
            self.sidebar, 
            values=["Dark", "Light"],
            command=self.change_appearance_mode,
            fg_color=self.card_color,
            button_color=self.card_color,
            button_hover_color=self.accent_purple,
            dropdown_fg_color=self.card_color,
            dropdown_hover_color=self.accent_purple
        )
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(5, 20), sticky="ew")

    def create_main_panel(self):
        # Main Scrollable Frame or Frame with nested layout
        self.main_container = ctk.CTkFrame(self, fg_color=self.bg_color, corner_radius=0)
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=25, pady=25)
        
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(3, weight=1) # Code preview area gets space

        # App Header
        self.header_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="✨ Code Builder Expert AI",
            font=ctk.CTkFont(family="Helvetica Neue", size=24, weight="bold"),
            text_color=self.text_color_bright
        )
        self.title_label.grid(row=0, column=0, sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self.header_frame,
            text="Input a problem, and the agent will synthesize Python code and write it to your filesystem.",
            font=ctk.CTkFont(family="Helvetica Neue", size=14),
            text_color=self.text_color_muted
        )
        self.subtitle_label.grid(row=1, column=0, sticky="w", pady=(2, 0))

        # Inputs section (Filename & Target Directory)
        self.input_config_frame = ctk.CTkFrame(self.main_container, fg_color=self.sidebar_color, corner_radius=10, border_color=self.card_color, border_width=1)
        self.input_config_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15), padx=2)
        self.input_config_frame.grid_columnconfigure(0, weight=1)
        self.input_config_frame.grid_columnconfigure(1, weight=1)

        # Filename Entry
        self.fn_frame = ctk.CTkFrame(self.input_config_frame, fg_color="transparent")
        self.fn_frame.grid(row=0, column=0, padx=15, pady=10, sticky="ew")
        
        self.fn_label = ctk.CTkLabel(self.fn_frame, text="Target Filename (e.g. script.py):", text_color=self.text_color_muted, font=ctk.CTkFont(size=12))
        self.fn_label.pack(anchor="w", pady=(0, 4))
        self.filename_entry = ctk.CTkEntry(
            self.fn_frame, 
            placeholder_text="auto_generated.py", 
            fg_color=self.bg_color,
            border_color=self.card_color,
            text_color=self.text_color_bright
        )
        self.filename_entry.pack(fill="x", ipady=3)

        # Directory selector
        self.dir_frame = ctk.CTkFrame(self.input_config_frame, fg_color="transparent")
        self.dir_frame.grid(row=0, column=1, padx=15, pady=10, sticky="ew")

        self.dir_label = ctk.CTkLabel(self.dir_frame, text="Output Directory:", text_color=self.text_color_muted, font=ctk.CTkFont(size=12))
        self.dir_label.pack(anchor="w", pady=(0, 4))
        
        self.dir_inner = ctk.CTkFrame(self.dir_frame, fg_color="transparent")
        self.dir_inner.pack(fill="x")
        
        self.dir_entry = ctk.CTkEntry(
            self.dir_inner, 
            textvariable=self.output_dir, 
            fg_color=self.bg_color,
            border_color=self.card_color,
            text_color=self.text_color_bright
        )
        self.dir_entry.pack(side="left", fill="x", expand=True, ipady=3, padx=(0, 5))
        
        self.dir_btn = ctk.CTkButton(
            self.dir_inner, 
            text="Browse", 
            width=70, 
            fg_color=self.card_color, 
            hover_color=self.accent_purple,
            command=self.browse_directory
        )
        self.dir_btn.pack(side="right")

        # Challenge Prompt Text area
        self.prompt_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.prompt_frame.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        
        self.prompt_label = ctk.CTkLabel(
            self.prompt_frame, 
            text="Coding Challenge / Problem Description:", 
            text_color=self.text_color_bright,
            font=ctk.CTkFont(weight="bold")
        )
        self.prompt_label.pack(anchor="w", pady=(0, 5))
        
        self.prompt_textbox = ctk.CTkTextbox(
            self.prompt_frame, 
            height=120, 
            fg_color=self.sidebar_color,
            border_color=self.card_color,
            border_width=1,
            text_color=self.text_color_bright,
            font=ctk.CTkFont(family="Consolas", size=13)
        )
        self.prompt_textbox.pack(fill="x")

        # Action Buttons frame
        self.actions_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.actions_frame.grid(row=3, column=0, sticky="ew", pady=(0, 15))

        self.btn_generate = ctk.CTkButton(
            self.actions_frame,
            text="🚀 Generate & Create File",
            fg_color=self.accent_purple,
            hover_color=self.accent_purple_hover,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45,
            command=self.start_generation
        )
        self.btn_generate.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_run_code = ctk.CTkButton(
            self.actions_frame,
            text="▶️ Run Generated File",
            fg_color="#2EB67D",
            hover_color="#228B5E",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45,
            width=200,
            state="disabled",
            command=self.run_generated_file
        )
        self.btn_run_code.pack(side="right")

        # Status & Progress Panel
        self.status_frame = ctk.CTkFrame(self.main_container, fg_color=self.sidebar_color, corner_radius=8, border_color=self.card_color, border_width=1)
        self.status_frame.grid(row=4, column=0, sticky="ew", pady=(0, 15))
        
        self.status_label = ctk.CTkLabel(
            self.status_frame, 
            text="Status: Ready", 
            font=ctk.CTkFont(weight="bold"), 
            text_color=self.text_color_muted
        )
        self.status_label.pack(side="left", padx=15, pady=10)
        
        self.progress_bar = ctk.CTkProgressBar(self.status_frame, width=200, progress_color=self.accent_purple)
        self.progress_bar.pack(side="right", padx=15, pady=10)
        self.progress_bar.set(0)

        # Output/Code Preview Panel
        self.output_panel_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.output_panel_frame.grid(row=5, column=0, sticky="nsew")
        self.main_container.grid_rowconfigure(5, weight=1) # Allow code preview to expand

        self.output_lbl = ctk.CTkLabel(
            self.output_panel_frame, 
            text="Agent Thoughts & Generated Code Preview:", 
            text_color=self.text_color_bright,
            font=ctk.CTkFont(weight="bold")
        )
        self.output_lbl.pack(anchor="w", pady=(0, 5))

        self.output_textbox = ctk.CTkTextbox(
            self.output_panel_frame, 
            fg_color=self.sidebar_color, 
            border_color=self.card_color, 
            border_width=1,
            text_color="#F8F8F2", # Monokai text style
            font=ctk.CTkFont(family="Consolas", size=13)
        )
        self.output_textbox.pack(fill="both", expand=True)

    def change_appearance_mode(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode.lower())

    def load_preset(self, preset):
        self.filename_entry.delete(0, tk.END)
        self.filename_entry.insert(0, preset["filename"])
        self.prompt_textbox.delete("1.0", tk.END)
        self.prompt_textbox.insert("1.0", preset["prompt"])
        self.update_status(f"Preset loaded: {preset['title']}", self.text_color_muted)

    def browse_directory(self):
        selected = filedialog.askdirectory(initialdir=self.output_dir.get())
        if selected:
            self.output_dir.set(selected)

    def update_status(self, text, color=None):
        self.status_label.configure(text=f"Status: {text}")
        if color:
            self.status_label.configure(text_color=color)

    def start_generation(self):
        prompt = self.prompt_textbox.get("1.0", tk.END).strip()
        filename = self.filename_entry.get().strip()
        out_dir = self.output_dir.get().strip()

        if not prompt:
            messagebox.showwarning("Empty Challenge", "Please write a problem description or select a preset.")
            return

        if not out_dir or not os.path.exists(out_dir):
            messagebox.showwarning("Invalid Directory", "The specified output directory does not exist.")
            return

        # Prepare file name
        if not filename:
            filename = "solution.py"
        elif not filename.endswith(".py"):
            filename += ".py"

        # Update UI state
        self.update_status("Querying Groq Agent...", self.accent_purple)
        self.progress_bar.start()
        self.btn_generate.configure(state="disabled")
        self.btn_run_code.configure(state="disabled")
        self.output_textbox.delete("1.0", tk.END)
        self.generated_filepath = None

        # Run model API call in background thread to avoid freezing GUI
        thread = threading.Thread(target=self.run_agent_task, args=(prompt, filename, out_dir), daemon=True)
        thread.start()

    def run_agent_task(self, prompt, filename, out_dir):
        try:
            client = Groq(api_key=GROQ_API_KEY)
            
            system_instruction = (
                "You are an expert python problem-solving coding agent. "
                "Your objective is to provide a complete, clean, working Python script for the user's challenge. "
                "Format your response so it is easy to read. "
                "CRITICAL: You MUST include the full executable Python code inside a standard markdown code block: ```python\n<your code here>\n```."
            )

            # Query the Groq Client
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Challenge: {prompt}\n\nPlease generate a Python script named '{filename}'."}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
            )

            response_content = chat_completion.choices[0].message.content
            
            # Extract Python code
            code_pattern = r"```python\s*(.*?)\s*```"
            code_match = re.search(code_pattern, response_content, re.DOTALL)
            
            extracted_code = ""
            if code_match:
                extracted_code = code_match.group(1).strip()
            else:
                # Try generic code block fallback
                generic_pattern = r"```\s*(.*?)\s*```"
                generic_match = re.search(generic_pattern, response_content, re.DOTALL)
                if generic_match:
                    extracted_code = generic_match.group(1).strip()
                else:
                    # Look for lines with code if no block exists
                    extracted_code = response_content

            # Write code to file
            full_path = os.path.join(out_dir, filename)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(extracted_code)

            self.generated_filepath = full_path

            # Update UI on Main Thread
            self.after(0, self.on_generation_success, response_content, full_path)

        except Exception as e:
            self.after(0, self.on_generation_error, str(e))

    def on_generation_success(self, response_text, filepath):
        self.progress_bar.stop()
        self.progress_bar.set(1.0)
        self.btn_generate.configure(state="normal")
        self.btn_run_code.configure(state="normal")
        self.update_status(f"File created successfully: {os.path.basename(filepath)}", "#2EB67D")
        
        # Display response
        self.output_textbox.insert("1.0", response_text)
        
        messagebox.showinfo("Success", f"File created successfully at:\n{filepath}")

    def on_generation_error(self, error_message):
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.btn_generate.configure(state="normal")
        self.update_status("Generation Failed!", "#E01E5A")
        
        self.output_textbox.insert("1.0", f"Error occurred during generation:\n\n{error_message}")
        messagebox.showerror("Error", f"Failed to generate code:\n{error_message}")

    def run_generated_file(self):
        if not self.generated_filepath or not os.path.exists(self.generated_filepath):
            messagebox.showerror("File Error", "Generated file not found on disk.")
            return

        import subprocess
        try:
            # Run the python script in a separate console window/process
            if os.name == 'nt':
                # Windows: run in a new cmd window
                subprocess.Popen(f'start cmd /k "python \\"{self.generated_filepath}\\""', shell=True)
            else:
                # Unix/Mac
                subprocess.Popen(['python', self.generated_filepath])
            
            self.update_status(f"Running file: {os.path.basename(self.generated_filepath)}", self.text_color_bright)
        except Exception as e:
            messagebox.showerror("Run Error", f"Could not execute file:\n{str(e)}")


if __name__ == "__main__":
    app = CodingAgentApp()
    app.mainloop()
