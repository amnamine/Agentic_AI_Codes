import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import pandas as pd

# Import the AI agent
from agent import DatasetSearchAgent

# Set default CustomTkinter appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

class AIDatasetApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure Window
        self.title("📊 AI CSV Dataset Search Agent")
        self.geometry("1100x700")
        self.minsize(950, 600)
        
        # Define Preset Datasets
        self.presets = {
            "Iris": {"query": "iris csv dataset", "filename": "iris.csv", "desc": "Classic flower classification dataset"},
            "Diabetes": {"query": "diabetes csv dataset raw", "filename": "diabetes.csv", "desc": "Predict diabetes onset using medical measurements"},
            "Heart": {"query": "heart disease csv dataset", "filename": "heart.csv", "desc": "Predict heart disease from patient parameters"},
            "Titanic": {"query": "titanic survivor csv dataset", "filename": "titanic.csv", "desc": "Famous passenger survival classification dataset"},
            "Fraud": {"query": "credit card fraud detection csv dataset", "filename": "fraud.csv", "desc": "Anonymized credit card transaction fraud logs"}
        }

        self.agent = DatasetSearchAgent(log_callback=self.append_log)
        
        # Configure layout grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1, minsize=350)
        self.grid_columnconfigure(1, weight=2, minsize=550)
        
        self.create_header()
        self.create_left_panel()
        self.create_right_panel()
        
    def create_header(self):
        # Header Frame
        header_frame = ctk.CTkFrame(self, height=80, corner_radius=0, fg_color="#1E293B")
        header_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=0, pady=(0, 10))
        header_frame.grid_rowconfigure(0, weight=1)
        header_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame, 
            text="📊 AI DATASET SEARCH AGENT", 
            font=ctk.CTkFont(family="Inter", size=24, weight="bold"),
            text_color="#F8FAFC"
        )
        title_label.grid(row=0, column=0, sticky="w", padx=25, pady=(15, 2))
        
        subtitle_label = ctk.CTkLabel(
            header_frame, 
            text="Sourcing, downloading, and verifying datasets using Groq Llama 3 70B & DuckDuckGo Search", 
            font=ctk.CTkFont(family="Inter", size=13),
            text_color="#94A3B8"
        )
        subtitle_label.grid(row=0, column=0, sticky="w", padx=25, pady=(38, 15))

    def create_left_panel(self):
        # Left Panel (Controls & Log Console)
        left_frame = ctk.CTkFrame(self, fg_color="transparent")
        left_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=10)
        
        left_frame.grid_rowconfigure(2, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        
        # Presets Section
        presets_container = ctk.CTkFrame(left_frame, corner_radius=12, border_width=1, border_color="#334155")
        presets_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=(5, 15))
        
        presets_title = ctk.CTkLabel(
            presets_container, 
            text="PRESET DATASETS", 
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"), 
            text_color="#38BDF8"
        )
        presets_title.pack(anchor="w", padx=15, pady=(10, 5))
        
        # Button Grid for Presets
        btn_frame = ctk.CTkFrame(presets_container, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        for idx, (name, details) in enumerate(self.presets.items()):
            btn = ctk.CTkButton(
                btn_frame, 
                text=name, 
                command=lambda n=name, d=details: self.start_search_thread(d["query"], d["filename"]),
                font=ctk.CTkFont(family="Inter", size=12, weight="bold"),
                fg_color="#1E293B",
                hover_color="#334155",
                border_width=1,
                border_color="#475569",
                height=32
            )
            btn.pack(side="left", expand=True, padx=4, fill="x")

        # Custom Search Section
        search_container = ctk.CTkFrame(left_frame, corner_radius=12, border_width=1, border_color="#334155")
        search_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 15))
        
        search_title = ctk.CTkLabel(
            search_container, 
            text="CUSTOM DATASET SEARCH", 
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"), 
            text_color="#38BDF8"
        )
        search_title.pack(anchor="w", padx=15, pady=(10, 5))
        
        search_sub_frame = ctk.CTkFrame(search_container, fg_color="transparent")
        search_sub_frame.pack(fill="x", padx=12, pady=(5, 12))
        
        self.search_entry = ctk.CTkEntry(
            search_sub_frame, 
            placeholder_text="Enter dataset name (e.g. california housing csv)...",
            font=ctk.CTkFont(family="Inter", size=12),
            height=36,
            fg_color="#0F172A",
            border_color="#475569"
        )
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        
        self.search_btn = ctk.CTkButton(
            search_sub_frame,
            text="🔍 Find & Download",
            command=self.on_custom_search,
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"),
            fg_color="#0EA5E9",
            hover_color="#0284C7",
            height=36,
            width=130
        )
        self.search_btn.pack(side="right")

        # Agent Console Logs Section
        log_container = ctk.CTkFrame(left_frame, corner_radius=12, border_width=1, border_color="#334155")
        log_container.grid(row=2, column=0, sticky="nsew", padx=5, pady=0)
        
        log_title = ctk.CTkLabel(
            log_container, 
            text="AGENT ACTIVITY LOG", 
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"), 
            text_color="#38BDF8"
        )
        log_title.pack(anchor="w", padx=15, pady=(10, 5))
        
        self.log_textbox = ctk.CTkTextbox(
            log_container, 
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#020617",
            text_color="#10B981",
            activate_scrollbars=True
        )
        self.log_textbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.log_textbox.configure(state="disabled")

    def create_right_panel(self):
        # Right Panel (Dataset Preview & Details)
        right_frame = ctk.CTkFrame(self, fg_color="transparent")
        right_frame.grid(row=1, column=1, sticky="nsew", padx=15, pady=10)
        
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Dataset Info Card
        self.info_card = ctk.CTkFrame(right_frame, corner_radius=12, border_width=1, border_color="#334155")
        self.info_card.grid(row=0, column=0, sticky="nsew", padx=5, pady=(5, 15))
        
        info_title = ctk.CTkLabel(
            self.info_card, 
            text="DATASET METADATA & STATUS", 
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"), 
            text_color="#38BDF8"
        )
        info_title.pack(anchor="w", padx=15, pady=(10, 5))
        
        # Grid inside Info Card
        info_grid = ctk.CTkFrame(self.info_card, fg_color="transparent")
        info_grid.pack(fill="x", padx=15, pady=(0, 12))
        info_grid.grid_columnconfigure(1, weight=1)
        
        # Labels for path, shape, columns
        ctk.CTkLabel(info_grid, text="Status:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", pady=2)
        self.lbl_status = ctk.CTkLabel(info_grid, text="Idle", text_color="#94A3B8")
        self.lbl_status.grid(row=0, column=1, sticky="w", padx=10, pady=2)
        
        ctk.CTkLabel(info_grid, text="Saved File:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, sticky="w", pady=2)
        self.lbl_file = ctk.CTkLabel(info_grid, text="N/A", text_color="#94A3B8")
        self.lbl_file.grid(row=1, column=1, sticky="w", padx=10, pady=2)
        
        ctk.CTkLabel(info_grid, text="Dimensions:", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, sticky="w", pady=2)
        self.lbl_shape = ctk.CTkLabel(info_grid, text="N/A", text_color="#94A3B8")
        self.lbl_shape.grid(row=2, column=1, sticky="w", padx=10, pady=2)
        
        ctk.CTkLabel(info_grid, text="Columns:", font=ctk.CTkFont(weight="bold")).grid(row=3, column=0, sticky="nw", pady=2)
        self.lbl_columns = ctk.CTkLabel(info_grid, text="N/A", text_color="#94A3B8", wraplength=500, justify="left")
        self.lbl_columns.grid(row=3, column=1, sticky="w", padx=10, pady=2)
        
        # Preview Table Container
        preview_container = ctk.CTkFrame(right_frame, corner_radius=12, border_width=1, border_color="#334155")
        preview_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=0)
        
        preview_title = ctk.CTkLabel(
            preview_container, 
            text="DATASET PREVIEW (FIRST 5 ROWS)", 
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"), 
            text_color="#38BDF8"
        )
        preview_title.pack(anchor="w", padx=15, pady=(10, 5))
        
        # Modern Styled Treeview for displaying CSV Data
        style = ttk.Style()
        style.theme_use("clam")
        
        # Style treeview to blend into the Dark Theme
        style.configure("Treeview", 
                        background="#0F172A", 
                        fieldbackground="#0F172A", 
                        foreground="#E2E8F0",
                        font=("Inter", 9),
                        rowheight=28)
        style.configure("Treeview.Heading", 
                        background="#1E293B", 
                        foreground="#94A3B8", 
                        font=("Inter", 9, "bold"))
        style.map("Treeview.Heading", background=[('active', '#334155')])
        
        table_frame = tk.Frame(preview_container, bg="#0F172A")
        table_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        
        # Scrollbars
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal")
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical")
        
        self.tree = ttk.Treeview(table_frame, show="headings", xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        
        x_scroll.config(command=self.tree.xview)
        y_scroll.config(command=self.tree.yview)
        
        y_scroll.pack(side="right", fill="y")
        x_scroll.pack(side="bottom", fill="x")
        self.tree.pack(side="left", fill="both", expand=True)

    def append_log(self, text):
        """Append log output to the live activity console."""
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", text + "\n")
        self.log_textbox.configure(state="disabled")
        self.log_textbox.see("end")

    def clear_log(self):
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

    def on_custom_search(self):
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a dataset name to search.")
            return
        
        # Make a safe filename out of the query
        safe_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', query)
        safe_name = safe_name.replace(" ", "_").lower()
        if not safe_name.endswith(".csv"):
            safe_name += ".csv"
            
        self.start_search_thread(query, safe_name)

    def start_search_thread(self, query, filename):
        """Runs the search agent in a background thread to keep UI interactive."""
        self.clear_log()
        self.lbl_status.configure(text="🔍 Sourcing...", text_color="#F59E0B")
        self.lbl_file.configure(text="N/A")
        self.lbl_shape.configure(text="N/A")
        self.lbl_columns.configure(text="N/A")
        
        # Clear Table
        self.tree.delete(*self.tree.get_children())
        
        # Disable buttons during search
        self.search_btn.configure(state="disabled")
        
        thread = threading.Thread(target=self.run_agent, args=(query, filename), daemon=True)
        thread.start()

    def run_agent(self, query, filename):
        try:
            result = self.agent.execute_agent(query, filename)
            
            if result:
                self.after(0, lambda: self.update_ui_success(result))
            else:
                self.after(0, lambda: self.update_ui_failure())
        except Exception as e:
            self.after(0, lambda: self.update_ui_error(str(e)))

    def update_ui_success(self, info):
        self.lbl_status.configure(text="✅ Success", text_color="#10B981")
        self.lbl_file.configure(text=os.path.basename(info["filepath"]))
        self.lbl_shape.configure(text=f"{info['shape'][0]} rows x {info['shape'][1]} columns")
        
        col_list = ", ".join(info["columns"])
        if len(col_list) > 120:
            col_list = col_list[:120] + "..."
        self.lbl_columns.configure(text=col_list)
        
        # Populating treeview preview
        cols = info["columns"]
        self.tree.configure(columns=cols)
        
        for col in cols:
            self.tree.heading(col, text=col)
            # Estimate width
            self.tree.column(col, width=max(100, int(len(col)*10)))
            
        for row in info["head"]:
            row_values = [row.get(col, "") for col in cols]
            self.tree.insert("", "end", values=row_values)
            
        self.search_btn.configure(state="normal")
        messagebox.showinfo("Success", f"Dataset downloaded successfully:\n{os.path.basename(info['filepath'])}")

    def update_ui_failure(self):
        self.lbl_status.configure(text="❌ Failed to source dataset", text_color="#EF4444")
        self.search_btn.configure(state="normal")
        messagebox.showerror("Failed", "AI agent was unable to find and verify a valid CSV dataset.")

    def update_ui_error(self, err_msg):
        self.lbl_status.configure(text="❌ Error", text_color="#EF4444")
        self.search_btn.configure(state="normal")
        messagebox.showerror("Error", f"An error occurred: {err_msg}")

if __name__ == "__main__":
    app = AIDatasetApp()
    app.mainloop()
