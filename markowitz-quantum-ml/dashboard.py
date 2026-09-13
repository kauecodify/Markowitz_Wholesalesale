# -*- coding: utf-8 -*-
"""

Dashboard Markowitz

Arquitetura Modular: DataManager | MLForecaster | QuantumOptimizer

"pip install qiskit qiskit-optimization qiskit-"

"python -m pip install customtkinter"

"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import threading
import time
import json
import os
import yfinance as yf
from datetime import datetime

try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠️ Aviso: Qiskit não encontrado. A otimização quântica será desativada.")

# ==========================================
# CONFIGURAÇÃO DE TEMA
# ==========================================

ctk.set_appearance_mode("Dark")

COLORS = {
    "primary": "#00CED1",    
    "secondary": "#20B2AA",   
    "dark": "#008B8B",         
    "light": "#40E0D0",      
    "bg_main": "#0a1929",     
    "bg_surface": "#132f4c",   
    "text": "#e6f1ff",    
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444"
}

class DataManager:
    def __init__(self):
        self.offline_data = None
        self.online_data = {}
        self.stacked_docs = []
        self.cache_file = "cloud_cache_stack.json"
        self.load_cache()

    def load_cache(self):
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, 'w') as f:
                json.dump({"stacked_docs": [], "last_sync": ""}, f)
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                self.stacked_docs = data.get("stacked_docs", [])
        except:
            self.stacked_docs = []

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump({"stacked_docs": self.stacked_docs, "last_sync": datetime.now().isoformat()}, f)

    def load_offline_files(self, file_paths):
        dfs = []
        for f in file_paths:
            try:
                if f.endswith('.csv'):
                    df = pd.read_csv(f)
                else:
                    df = pd.read_excel(f)
                dfs.append(df)
                self.stacked_docs.append({"name": os.path.basename(f), "date": datetime.now().isoformat()})
            except Exception as e:
                print(f"Erro ao ler {f}: {e}")
        
        if dfs:
            self.save_cache()
            self.offline_data = pd.concat(dfs, ignore_index=True)
            return self.offline_data
        return None

    def get_numeric_columns(self):
        if self.offline_data is None:
            return []
        return self.offline_data.select_dtypes(include=[np.number]).columns.tolist()

    def get_live_price(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            # Tenta fast_info primeiro (mais rápido), fallback para history
            try:
                return stock.fast_info['lastPrice']
            except:
                hist = stock.history(period='1d')
                if not hist.empty:
                    return hist['Close'].iloc[-1]
            return None
        except Exception:
            return None

class MLForecaster:
    def __init__(self):
        self.models = {
            "Linear": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        }

    def forecast(self, data, columns, model_type, horizon):
        subset = data[columns].dropna()
        if subset.empty:
            raise ValueError("Dados vazios após limpeza")
            
        X_train = np.arange(len(subset)).reshape(-1, 1)
        X_future = np.arange(len(subset), len(subset) + horizon).reshape(-1, 1)
        predictions = {}
        
        for col in columns:
            y_train = subset[col].values
            # Clona o modelo para não reusar estado antigo
            if model_type == "Linear":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                
            model.fit(X_train, y_train)
            y_future = model.predict(X_future)
            predictions[col] = {
                "historical": y_train.tolist(),
                "forecast": y_future.tolist(),
                "last_real": y_train[-1],
                "last_pred": y_future[-1]
            }
        return predictions

class QuantumOptimizer:
    def __init__(self):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit não está instalado.")
        self.sampler = Sampler()
        self.optimizer = COBYLA(maxiter=100)

    def optimize(self, data, columns, q_risco, k_ativos):
        returns = data[columns].pct_change().dropna()
        if len(returns) < 5:
            raise ValueError("Dados insuficientes para cálculo de covariância")
            
        mu = returns.mean().values
        sigma = returns.cov().values
        n = len(columns)

        qp = QuadraticProgram()
        for i in range(n):
            qp.binary_var(name=f"x{i}")

        # Função Objetivo: Minimizar (-Retorno + Risco)
        linear_obj = {f"x{i}": -mu[i] * 100 for i in range(n)} # Escala retorno
        
        quadratic_obj = {}
        for i in range(n):
            for j in range(n):
                quadratic_obj[(f"x{i}", f"x{j}")] = q_risco * sigma[i, j] * 50 # Escala risco
        
        qp.minimize(linear=linear_obj, quadratic=quadratic_obj)

        # Restrição: Escolher exatamente K ativos
        constraint_linear = {f"x{i}": 1.0 for i in range(n)}
        qp.linear_constraint(linear=constraint_linear, sense="==", rhs=k_ativos, name="budget")

        qaoa = QAOA(sampler=self.sampler, optimizer=self.optimizer, reps=2)
        meo = MinimumEigenOptimizer(qaoa)
        resultado = meo.solve(qp)

        alocacao = {columns[i]: int(resultado.x[i]) for i in range(n)}
        
        # Cálculos pós-otimização
        selected_indices = [i for i in range(n) if resultado.x[i] == 1]
        total_return = sum(mu[i] for i in selected_indices)
        
        # Risco da carteira selecionada
        total_risk = 0
        for i in selected_indices:
            for j in selected_indices:
                total_risk += sigma[i, j]

        return {
            "alocacao": alocacao,
            "score": resultado.fval,
            "retorno_esperado": total_return,
            "risco": total_risk,
            "status": resultado.status.name
        }

class CloudSyncThread(threading.Thread):
    def __init__(self, data_manager, update_callback):
        super().__init__(daemon=True)
        self.data_manager = data_manager
        self.update_callback = update_callback
        self.running = True
        self.tickers = ["BTC-USD", "ETH-USD", "PETR4.SA", "VALE3.SA"]

    def run(self):
        while self.running:
            for ticker in self.tickers:
                price = self.data_manager.get_live_price(ticker)
                if price:
                    msg = f"{datetime.now().strftime('%H:%M:%S')} | {ticker}: ${price:.2f}"
                    try:
                        self.update_callback(msg)
                    except:
                        pass # Ignora erro de GUI fechada
            time.sleep(10) # Reduzido para 10s para não sobrecarregar API

    def stop(self):
        self.running = False

class MarkowitzDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Markowitz Quantum & ML")
        self.geometry("1200x800")
        self.configure(fg_color=COLORS["bg_main"])
        
        self.data_manager = DataManager()
        self.forecaster = MLForecaster()
        
        # Inicializa Optimizer apenas se Qiskit estiver disponível
        if QISKIT_AVAILABLE:
            self.optimizer = QuantumOptimizer()
        else:
            self.optimizer = None
            
        self.cloud_thread = CloudSyncThread(self.data_manager, self.update_live_feed)
        self.cloud_thread.start()
        
        self.check_vars = {}
        self.last_predictions = None
        self.last_qaoa_result = None
        self.last_selected_cols = []
        self.last_horizon = 0
        
        self.setup_ui()

    def setup_ui(self):
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0, fg_color=COLORS["bg_surface"])
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Main Area
        self.main_area = ctk.CTkFrame(self, corner_radius=0, fg_color=COLORS["bg_main"])
        self.main_area.pack(side="right", fill="both", expand=True)

        # --- Sidebar Widgets ---
        ctk.CTkLabel(self.sidebar, text="MARKOWITZ\nQUANTUM", 
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=COLORS["primary"]).pack(pady=20)

        self.btn_load = ctk.CTkButton(self.sidebar, text="📂 Carregar Dados", command=self.load_files,
                                      fg_color=COLORS["primary"], hover_color=COLORS["dark"], text_color="white")
        self.btn_load.pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="Ativos para Análise:", text_color=COLORS["light"]).pack(pady=(20, 5), padx=20, anchor="w")
        self.frame_checks = ctk.CTkScrollableFrame(self.sidebar, width=260, height=150, fg_color=COLORS["bg_main"])
        self.frame_checks.pack(padx=20, fill="x")

        ctk.CTkLabel(self.sidebar, text="Modelo de Previsão:", text_color=COLORS["light"]).pack(pady=(20, 5), padx=20, anchor="w")
        self.model_var = ctk.StringVar(value="Random Forest")
        ctk.CTkRadioButton(self.sidebar, text="Regressão Linear", variable=self.model_var, value="Linear",
                           fg_color=COLORS["primary"]).pack(padx=20, anchor="w")
        ctk.CTkRadioButton(self.sidebar, text="Random Forest", variable=self.model_var, value="Random Forest",
                           fg_color=COLORS["primary"]).pack(padx=20, anchor="w")

        ctk.CTkLabel(self.sidebar, text="Horizonte (Dias):", text_color=COLORS["light"]).pack(pady=(20, 5), padx=20, anchor="w")
        self.slider_horizon = ctk.CTkSlider(self.sidebar, from_=1, to=30, number_of_steps=29,
                                            button_color=COLORS["primary"], progress_color=COLORS["primary"])
        self.slider_horizon.pack(padx=20, fill="x")
        self.slider_horizon.set(10)

        if self.optimizer:
            ctk.CTkLabel(self.sidebar, text="Ativos na Carteira (K):", text_color=COLORS["light"]).pack(pady=(20, 5), padx=20, anchor="w")
            self.slider_k = ctk.CTkSlider(self.sidebar, from_=1, to=6, number_of_steps=5,
                                          button_color=COLORS["primary"], progress_color=COLORS["primary"])
            self.slider_k.pack(padx=20, fill="x")
            self.slider_k.set(2)

            ctk.CTkLabel(self.sidebar, text="Fator de Risco (q):", text_color=COLORS["light"]).pack(pady=(20, 5), padx=20, anchor="w")
            self.slider_q = ctk.CTkSlider(self.sidebar, from_=0.1, to=2.0, number_of_steps=19,
                                          button_color=COLORS["primary"], progress_color=COLORS["primary"])
            self.slider_q.pack(padx=20, fill="x")
            self.slider_q.set(0.5)

        btn_color = COLORS["primary"] if self.optimizer else COLORS["warning"]
        btn_text = "Executar Pipeline" if self.optimizer else "⚠️ Apenas ML (Sem Qiskit)"
        
        self.btn_run = ctk.CTkButton(self.sidebar, text=btn_text, command=self.run_pipeline, 
                                     fg_color=btn_color, hover_color=COLORS["dark"], height=40, text_color="white")
        self.btn_run.pack(pady=10, padx=20, fill="x")

        self.btn_save = ctk.CTkButton(self.sidebar, text="💾 Salvar Resultados", command=self.save_results, 
                                      fg_color="#1f6feb", hover_color="#1158c7", height=40, text_color="white")
        self.btn_save.pack(pady=10, padx=20, fill="x")

        self.cloud_status = ctk.CTkLabel(self.sidebar, text="● Cloud Sync: Ativo", text_color=COLORS["success"])
        self.cloud_status.pack(side="bottom", pady=20)

        # --- Main Tabs ---
        self.tabs = ctk.CTkTabview(self.main_area, width=850,
                                   segmented_button_selected_color=COLORS["primary"])
        self.tabs.pack(fill="both", expand=True, padx=20, pady=20)

        self.tab_forecast = self.tabs.add("Previsão ML")
        self.tab_quantum = self.tabs.add("Otimização Quântica")
        self.tab_cloud = self.tabs.add("Feed Nuvem & Docs")

        self.log_text = ctk.CTkTextbox(self.tab_cloud, font=ctk.CTkFont(family="Consolas", size=11),
                                       fg_color=COLORS["bg_surface"], text_color=COLORS["light"])
        self.log_text.pack(fill="both", expand=True)
        self.update_live_feed("[SISTEMA] Dashboard iniciado. Tema Verde Aqua ativo.")

    def load_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Dados", "*.csv *.xlsx")])
        if not files:
            return
        result = self.data_manager.load_offline_files(files)
        if result is not None:
            self.update_checkboxes()
            self.update_live_feed(f"Documentos carregados: {len(files)} arquivos.")

    def update_checkboxes(self):
        for widget in self.frame_checks.winfo_children():
            widget.destroy()
        self.check_vars = {}
        numeric_cols = self.data_manager.get_numeric_columns()
        for col in numeric_cols:
            var = ctk.BooleanVar(value=True)
            chk = ctk.CTkCheckBox(self.frame_checks, text=col, variable=var,
                                  fg_color=COLORS["primary"], border_color=COLORS["primary"], text_color=COLORS["text"])
            chk.pack(anchor="w", pady=2)
            self.check_vars[col] = var

    def update_live_feed(self, message):
        try:
            self.log_text.insert("0.0", f"{message}\n")
        except:
            pass

    def run_pipeline(self):
        selected = [col for col, var in self.check_vars.items() if var.get()]
        if len(selected) < 2 or self.data_manager.offline_data is None:
            messagebox.showwarning("Atenção", "Selecione pelo menos 2 ativos e carregue dados.")
            return
        
        self.update_live_feed("Iniciando processamento...")
        
        # Thread ML
        threading.Thread(target=self.execute_forecast, args=(selected,), daemon=True).start()
        
        # Thread Quantum (se disponível)
        if self.optimizer:
            threading.Thread(target=self.execute_quantum, args=(selected,), daemon=True).start()
        else:
            self.update_live_feed("⚠️ Qiskit não instalado. Pulando otimização quântica.")

    def execute_forecast(self, selected_cols):
        try:
            model_type = self.model_var.get()
            horizon = int(self.slider_horizon.get())
            self.last_horizon = horizon
            self.last_selected_cols = selected_cols
            
            predictions = self.forecaster.forecast(self.data_manager.offline_data, selected_cols, model_type, horizon)
            self.last_predictions = predictions

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(COLORS["bg_surface"])
            ax.set_facecolor(COLORS["bg_main"])
            
            # Estilização Aqua
            ax.tick_params(colors=COLORS["light"])
            ax.title.set_color(COLORS["primary"])
            ax.spines['bottom'].set_color(COLORS["primary"])
            ax.spines['top'].set_color(COLORS["primary"])
            ax.spines['left'].set_color(COLORS["primary"])
            ax.spines['right'].set_color(COLORS["primary"])
            ax.grid(True, alpha=0.2, color=COLORS["dark"])

            colors_plot = [COLORS["primary"], COLORS["light"], COLORS["success"], COLORS["warning"]]

            for i, col in enumerate(selected_cols[:5]): # Limita a 5 para clareza
                hist = predictions[col]["historical"]
                forecast = predictions[col]["forecast"]
                x_hist = np.arange(len(hist))
                x_forecast = np.arange(len(hist), len(hist) + horizon)
                
                color = colors_plot[i % len(colors_plot)]
                
                ax.plot(x_hist, hist, label=f'{col} (Real)', linestyle='--', alpha=0.6, color=color)
                ax.plot(x_forecast, forecast, label=f'{col} (Prev)', linewidth=2, color=color)

            ax.set_title(f"Previsão ({model_type}) - Horizonte: {horizon} dias", fontsize=14)
            ax.legend(facecolor=COLORS["bg_surface"], labelcolor=COLORS["text"])

            self.after(0, lambda: self.render_plot(fig, self.tab_forecast))
            self.after(0, lambda: self.update_live_feed("✅ Previsão ML concluída."))
            
        except Exception as e:
            self.after(0, lambda: self.update_live_feed(f"❌ Erro ML: {str(e)}"))

    def execute_quantum(self, selected_cols):
        if not self.optimizer: return
        try:
            q_risco = float(self.slider_q.get())
            k_ativos = int(self.slider_k.get())
            k_ativos = min(k_ativos, len(selected_cols))
            
            resultado = self.optimizer.optimize(self.data_manager.offline_data, selected_cols, q_risco, k_ativos)
            self.last_qaoa_result = resultado

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(COLORS["bg_surface"])
            ax.set_facecolor(COLORS["bg_main"])
            
            ax.tick_params(colors=COLORS["light"])
            ax.title.set_color(COLORS["primary"])
            ax.spines['bottom'].set_color(COLORS["primary"])
            ax.spines['top'].set_color(COLORS["primary"])
            ax.spines['left'].set_color(COLORS["primary"])
            ax.spines['right'].set_color(COLORS["primary"])

            alocados = [k for k, v in resultado["alocacao"].items() if v == 1]
            if not alocados: alocados = ["Nenhum"]
            
            bars = ax.bar(alocados, [1] * len(alocados), color=COLORS["success"], edgecolor=COLORS["light"])
            
            ax.set_title(f"Alocação Quântica QAOA (Score: {resultado['score']:.4f})")
            ax.set_ylabel("Selecionado (1)", color=COLORS["light"])

            self.after(0, lambda: self.render_plot_with_info(fig, self.tab_quantum, resultado))
            self.after(0, lambda: self.update_live_feed(f" QAOA Concluído. Score: {resultado['score']:.4f}"))

        except Exception as e:
            self.after(0, lambda: self.update_live_feed(f"❌ Erro Quântico: {str(e)}"))

    def render_plot(self, fig, tab):
        for widget in tab.winfo_children(): widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def render_plot_with_info(self, fig, tab, resultado):
        for widget in tab.winfo_children(): widget.destroy()
        
        # Frame superior para gráfico
        frame_top = ctk.CTkFrame(tab, fg_color=COLORS["bg_main"])
        frame_top.pack(fill="both", expand=True)
        
        canvas = FigureCanvasTkAgg(fig, master=frame_top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Frame inferior para texto
        frame_bottom = ctk.CTkFrame(tab, fg_color=COLORS["bg_surface"])
        frame_bottom.pack(fill="x", pady=10)
        
        info_text = (
            f" Retorno Esperado: {resultado['retorno_esperado']:.4f}\n"
            f"⚠️ Risco Total: {resultado['risco']:.4f}\n"
            f" Score Final: {resultado['score']:.4f}\n"
            f" Status: {resultado['status']}"
        )
        
        label = ctk.CTkLabel(frame_bottom, text=info_text, font=ctk.CTkFont(size=13), 
                             justify="left", text_color=COLORS["light"])
        label.pack(pady=10, padx=20, anchor="w")

    def save_results(self):
        if self.data_manager.offline_data is None:
            messagebox.showwarning("Atenção", "Nenhum dado carregado.")
            return
        if self.last_predictions is None:
            messagebox.showwarning("Atenção", "Execute a previsão antes de salvar.")
            return

        df_export = self.data_manager.offline_data.copy()
        
        # Adiciona previsões
        future_rows = []
        for i in range(self.last_horizon):
            row = {col: np.nan for col in df_export.columns}
            for col in self.last_selected_cols:
                if col in self.last_predictions:
                    row[col] = self.last_predictions[col]["forecast"][i]
            future_rows.append(row)
            
        df_future = pd.DataFrame(future_rows)
        df_export = pd.concat([df_export, df_future], ignore_index=True)
        
        # Adiciona dados QAOA se existirem
        if self.last_qaoa_result:
            df_export["QAOA_Score"] = self.last_qaoa_result["score"]
            df_export["QAOA_Return"] = self.last_qaoa_result["retorno_esperado"]

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            title="Salvar Resultados"
        )

        if file_path:
            try:
                if file_path.endswith('.csv'):
                    df_export.to_csv(file_path, index=False)
                else:
                    df_export.to_excel(file_path, index=False)
                self.update_live_feed(f"💾 Resultados salvos em: {os.path.basename(file_path)}")
                messagebox.showinfo("Sucesso", "Resultados salvos com sucesso.")
            except Exception as e:
                messagebox.showerror("Erro", str(e))

if __name__ == "__main__":
    try:
        app = MarkowitzDashboard()
        app.mainloop()
    except Exception as e:
        print(f"Erro fatal: {e}")
