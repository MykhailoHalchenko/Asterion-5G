"""CustomTkinter desktop interface for the integrated Asterion pipeline."""

from __future__ import annotations

from pathlib import Path
import threading
import traceback
import tkinter.filedialog as filedialog

import customtkinter as ctk

from integration.bridge import simulate_and_export


class AsterionApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Asterion 5G + ASTERIX")
        self.geometry("760x560")
        self.minsize(640, 460)

        self.output_path = Path(__file__).resolve().parent / "output-logs" / "5g_asterix.parquet"
        self.data_dir = Path(__file__).resolve().parent / "Datasets"
        self._build_layout()

    def _build_layout(self) -> None:
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(
            self,
            text="Asterion 5G + ASTERIX",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=0, padx=24, pady=(24, 8), sticky="w")
        ctk.CTkLabel(
            self,
            text="RF track simulation, CAT 062 encoding, and Parquet model training",
        ).grid(row=1, column=0, padx=24, pady=(0, 18), sticky="w")

        controls = ctk.CTkFrame(self)
        controls.grid(row=2, column=0, padx=24, pady=8, sticky="ew")
        controls.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(controls, text="Number of RF samples:").grid(
            row=0, column=0, padx=12, pady=12
        )
        self.samples = ctk.CTkEntry(controls, width=100)
        self.samples.insert(0, "20")
        self.samples.grid(row=0, column=1, padx=12, pady=12, sticky="w")
        ctk.CTkButton(
            controls, text="Simulate + ASTERIX", command=self.start_simulation
        ).grid(row=0, column=2, padx=12, pady=12)
        ctk.CTkButton(
            controls, text="Train model", command=self.start_training
        ).grid(row=0, column=3, padx=12, pady=12)

        paths = ctk.CTkFrame(self)
        paths.grid(row=3, column=0, padx=24, pady=8, sticky="ew")
        paths.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(paths, text="Datasets:").grid(row=0, column=0, padx=12, pady=10)
        self.data_label = ctk.CTkLabel(paths, text=str(self.data_dir), anchor="w")
        self.data_label.grid(row=0, column=1, padx=12, pady=10, sticky="ew")
        ctk.CTkButton(paths, text="Choose folder", command=self.choose_data_dir).grid(
            row=0, column=2, padx=12, pady=10
        )

        self.log = ctk.CTkTextbox(self, wrap="word")
        self.log.grid(row=4, column=0, padx=24, pady=(8, 24), sticky="nsew")
        self._write("Ready. Choose an action.")

    def _write(self, message: str) -> None:
        self.after(0, self._append_log, message)

    def _append_log(self, message: str) -> None:
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")

    def choose_data_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.data_dir)
        if selected:
            self.data_dir = Path(selected)
            self.data_label.configure(text=str(self.data_dir))

    def start_simulation(self) -> None:
        try:
            samples = int(self.samples.get())
        except ValueError:
            self._write("Error: the sample count must be an integer.")
            return
        threading.Thread(
            target=self._run_simulation, args=(samples,), daemon=True
        ).start()

    def _run_simulation(self, samples: int) -> None:
        try:
            result = simulate_and_export(samples, self.output_path)
            self._write(
                f"Done: saved {len(result)} CAT 062 records to {self.output_path}"
            )
        except Exception:
            self._write(traceback.format_exc())

    def start_training(self) -> None:
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self) -> None:
        try:
            from Model.train_model import train

            parquet_count = len(list(self.data_dir.glob("*.parquet")))
            if parquet_count == 0:
                raise FileNotFoundError(f"No .parquet files found in {self.data_dir}")
            self._write(f"Training on {parquet_count} Parquet files...")
            metrics = train(data_dir=self.data_dir)
            self._write(f"Training completed: {metrics}")
        except Exception:
            self._write(traceback.format_exc())


if __name__ == "__main__":
    app = AsterionApp()
    app.mainloop()
