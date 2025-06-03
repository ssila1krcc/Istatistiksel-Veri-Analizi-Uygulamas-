

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from io import StringIO
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder


df = pd.DataFrame()
aktif_figur = None


def csv_yukle():
    global df
    dosya_yolu = filedialog.askopenfilename(filetypes=[("CSV veya Excel dosyaları", "*.csv *.xlsx")])
    if dosya_yolu:
        try:
            if dosya_yolu.endswith(".csv"):
                df = pd.read_csv(dosya_yolu)
            elif dosya_yolu.endswith(".xlsx"):
                df = pd.read_excel(dosya_yolu)
            temizle_ve_yukle_bilgi()
        except Exception as e:
            messagebox.showerror("Hata", f"Dosya yüklenemedi:\n{str(e)}")
def genel_bilgi():
    output.delete(1.0, tk.END)
    if df.empty:
        output.insert(tk.END, "Veri seti yüklenmedi.")
        return
    output.insert(tk.END, "🔹 İlk 20 Gözlem:\n")
    ilk20 = df.head(20).copy()
    ilk20.index += 1
    output.insert(tk.END, ilk20.to_string())
    output.insert(tk.END, "\n\n🔹 Veri Bilgisi:\n")
    buffer = StringIO()
    df.info(buf=buffer)
    output.insert(tk.END, buffer.getvalue())
    output.insert(tk.END, "\n🔹 Eksik Değerler:\n")
    output.insert(tk.END, str(df.isnull().sum()))

def varyans_analizi():
    output.delete(1.0, tk.END)
    if df.empty:
        return
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) < 1:
        output.insert(tk.END, "Sayısal sütun bulunamadı.")
        return
    col = num_cols[0]
    data = df[col].dropna()
    result = np.var(data, ddof=1)  # Örnek varyansı
    output.insert(tk.END, f"🔹 Varyans Analizi:\n{col} sütunu için varyans: {result:.4f}")

def ortalama_analizi():
    output.delete(1.0, tk.END)
    if df.empty:
        return
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) < 1:
        output.insert(tk.END, "Sayısal sütun bulunamadı.")
        return
    col = num_cols[0]
    data = df[col].dropna()
    result = np.mean(data)
    output.insert(tk.END, f"🔹 Ortalama Analizi:\n{col} sütunu için ortalama: {result:.4f}")

def stdsapma_analizi():
    output.delete(1.0, tk.END)
    if df.empty:
        return
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) < 1:
        output.insert(tk.END, "Sayısal sütun bulunamadı.")
        return
    col = num_cols[0]
    data = df[col].dropna()
    result = np.std(data, ddof=1)  # Örnek standart sapma
    output.insert(tk.END, f"🔹 Standart Sapma Analizi:\n{col} sütunu için std sapma: {result:.4f}")




def temizle_ve_yukle_bilgi():
    output.delete(1.0, tk.END)
    if df.empty:
        output.insert(tk.END, "Veri seti yüklenmedi.")
    else:
        output.insert(tk.END, f"✅ Veri seti yüklendi. Toplam Satır: {len(df)}, Sütun: {len(df.columns)}\n\n")
        ilk20 = df.head(20).copy()
        ilk20.index += 1
        output.insert(tk.END, "🔹 İlk 20 Gözlem:\n")
        output.insert(tk.END, ilk20.to_string())
        output.insert(tk.END, "\n")

# --- GRAFİK 
def grafik_goster(fig, baslik="Grafik"):
    global aktif_figur
    for tab in graph_notebook.tabs():
        if graph_notebook.tab(tab, "text") == baslik:
            graph_notebook.forget(tab)
            break
    frame = ttk.Frame(graph_notebook)
    graph_notebook.add(frame, text=baslik)
    graph_notebook.select(frame)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
    aktif_figur = fig
    plt.close(fig)

# ANALİZ FONKSİYONLARI ---
def sayisal_istatistikler():
    if df.empty:
        messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
        return
    output.delete(1.0, tk.END)
    output.insert(tk.END, "🔹 Sayısal İstatistikler:\n")
    output.insert(tk.END, str(df.describe()))

def kategorik_ozet():
    output.delete(1.0, tk.END)
    if df.empty:
        output.insert(tk.END, "Veri seti yüklenmedi.")
        return
    output.insert(tk.END, "🔹 Kategorik Değişkenler:\n")
    kategorik_sutunlar = df.select_dtypes(include=["object", "category"]).columns
    if len(kategorik_sutunlar) == 0:
        output.insert(tk.END, "Kategorik değişken yok.\n")
        return
    for sutun in kategorik_sutunlar:
        output.insert(tk.END, f"\n🔸 {sutun}:\n")
        output.insert(tk.END, str(df[sutun].value_counts()) + "\n")

def eksik_veri_gorsel():
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Eksik Veri Haritası")
    grafik_goster(fig, "Eksik Veri")

def eksik_doldur_ortalama():
    global df
    if df.empty:
        return
    df.fillna(df.mean(numeric_only=True), inplace=True)
    output.insert(tk.END, "\n✅ Sayısal eksik veriler ortalama ile dolduruldu.\n")

def histogram_gorsel():
    if df.empty:
        return
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) < 1:
        return
    fig, axs = plt.subplots(nrows=1, ncols=len(numeric_cols), figsize=(6 * len(numeric_cols), 5))
    if len(numeric_cols) == 1:
        axs = [axs]
    for ax, col in zip(axs, numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"{col} Histogram")
    fig.tight_layout()
    grafik_goster(fig, "Histogram")

def pasta_dilimi_gorsel():
    if df.empty:
        return
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) < 1:
        return
    fig, axs = plt.subplots(nrows=1, ncols=len(cat_cols), figsize=(6 * len(cat_cols), 5))
    if len(cat_cols) == 1:
        axs = [axs]
    for ax, col in zip(axs, cat_cols):
        df[col].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        ax.set_title(f"{col} Dağılımı")
    fig.tight_layout()
    grafik_goster(fig, "Pasta Grafiği")

def korelasyon_haritasi():
    if df.empty:
        return
    df_corr = df.select_dtypes(include=["number"])
    if df_corr.shape[1] < 2:
        return
    corr = df_corr.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Korelasyon Matrisi")
    grafik_goster(fig, "Korelasyon")

def kume_grafigi():
    if df.empty:
        messagebox.showwarning("Uyarı", "Önce bir veri seti yükleyin.")
        return

    # Sayısal sütunları al
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        messagebox.showinfo("Bilgi", "Sayısal sütun bulunamadı.")
        return

    # Sadece ilk sayısal sütunu kullan (isteğe bağlı değiştirilebilir)
    data = df[numeric_cols[0]].dropna()
    if len(data) < 10:
        messagebox.showwarning("Uyarı", "Yeterli veri yok.")
        return

    mu = data.mean()
    sigma = data.std()
    n_bins = 25

    fig = plt.figure(figsize=(9, 4), layout="constrained")
    axs = fig.subplots(1, 2, sharex=True, sharey=True)

    # CDF (Cumulative)
    n, bins, patches = axs[0].hist(data, n_bins, density=True, histtype="step",
    cumulative=True, label="Kümülatif Histogram")
    x = np.linspace(data.min(), data.max(), 300)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (x - mu))**2))
    y = y.cumsum()
    y /= y[-1]
    axs[0].plot(x, y, "k--", linewidth=1.5, label="Teorik CDF")
    axs[0].set_title("Kümülatif Dağılım (CDF)")

    # CCDF (Complementary)
    axs[1].hist(data, bins=bins, density=True, histtype="step", cumulative=-1,
                label="Ters Kümülatif Histogram")
    axs[1].plot(x, 1 - y, "k--", linewidth=1.5, label="Teorik CCDF")
    axs[1].set_title("Ters Kümülatif Dağılım (CCDF)")

    for ax in axs:
        ax.grid(True)
        ax.legend()
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel("Olasılık")
        ax.label_outer()

    fig.suptitle(f"Kümülatif Dağılımlar - {numeric_cols[0]}")
    grafik_goster(fig, "CDF & CCDF")


def kategorik_boxplot():
    if df.empty:
        return
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(cat_cols) == 0 or len(num_cols) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x=cat_cols[0], y=num_cols[0], ax=ax)
    ax.set_title(f"{cat_cols[0]} - {num_cols[0]} Boxplot")
    fig.tight_layout()
    grafik_goster(fig, "Boxplot")

def normalite_testi():
    output.delete(1.0, tk.END)
    if df.empty:
        return
    output.insert(tk.END, "🔹 Normal Dağılım Testi (Shapiro-Wilk):\n")
    for col in df.select_dtypes(include="number"):
        data = df[col].dropna()
        if 3 < len(data) < 5000:
            stat, p = stats.shapiro(data)
            sonuc = "✅ Normale uygun" if p > 0.05 else "❌ Normale uymuyor"
            output.insert(tk.END, f"{col}: p={p:.4f} → {sonuc}\n")

def t_test():
    output.delete(1.0, tk.END)
    if df.empty:
        return
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) < 2:
        return
    stat, p = stats.ttest_ind(df[num_cols[0]].dropna(), df[num_cols[1]].dropna())
    output.insert(tk.END, f"🔹 T-Test ({num_cols[0]} vs {num_cols[1]}):\nTest istatistiği={stat:.4f}, p={p:.4f}\n")

def anova_analizi():
    output.delete(1.0, tk.END)
    if df.empty:
        return
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(cat_cols) < 1 or len(num_cols) < 1:
        return
    model = ols(f"{num_cols[0]} ~ C({cat_cols[0]})", data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    output.insert(tk.END, f"🔹 ANOVA: {num_cols[0]} ~ {cat_cols[0]}\n")
    output.insert(tk.END, str(table))

def regresyon_analizi():
    output.delete(1.0, tk.END)
    if df.empty:
        messagebox.showwarning("Uyarı", "Önce bir veri yükleyin.")
        return

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        messagebox.showinfo("Bilgi", "En az 2 sayısal sütun gerekli.")
        return

    # Yeni pencere aç
    reg_win = tk.Toplevel(pencere)
    reg_win.title("Regresyon Değişken Seçimi")
    reg_win.geometry("300x200")

    tk.Label(reg_win, text="Bağımlı Değişken:").pack()
    dependent_var = tk.StringVar(reg_win)
    dependent_menu = ttk.Combobox(reg_win, textvariable=dependent_var, values=num_cols, state="readonly")
    dependent_menu.pack(pady=5)

    tk.Label(reg_win, text="Bağımsız Değişkenler (Ctrl ile çoklu seçim):").pack()
    independent_vars_listbox = tk.Listbox(reg_win, selectmode=tk.MULTIPLE, exportselection=0, height=6)
    for col in num_cols:
        independent_vars_listbox.insert(tk.END, col)
    independent_vars_listbox.pack(pady=5, fill=tk.X)


    def analiz_basla():
        y = dependent_var.get()
        x_indices = independent_vars_listbox.curselection()
        if not y or not x_indices:
            messagebox.showwarning("Uyarı", "Lütfen tüm seçimleri yapın.")
            return
        x = [num_cols[i] for i in x_indices if num_cols[i] != y]
        if not x:
            messagebox.showwarning("Uyarı", "Bağımlı ve bağımsız değişkenler aynı olamaz.")
            return
        formül = f"{y} ~ {' + '.join(x)}"
        try:
            model = ols(formül, data=df).fit()
            output.delete(1.0, tk.END)
            output.insert(tk.END, f"📊 Regresyon Formülü: {formül}\n\n")
            output.insert(tk.END, model.summary().as_text())
            reg_win.destroy()
        except Exception as e:
            messagebox.showerror("Hata", f"Regresyon modeli oluşturulamadı:\n{str(e)}")

    ttk.Button(reg_win, text="Analizi Başlat", command=analiz_basla).pack(pady=10)



def grafik_kaydet():
    if aktif_figur is None:
        return
    dosya_yolu = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Dosyası", "*.png")])
    if dosya_yolu:
        aktif_figur.savefig(dosya_yolu)
        messagebox.showinfo("Bilgi", "Grafik kaydedildi.")


df = pd.DataFrame()
aktif_figur = None


pencere = tk.Tk()
pencere.title("İstatistiksel Veri Analizi Uygulaması")
pencere.geometry("1200x700")


ttk.Style().theme_use("clam")
ttk.Style().configure("TButton", font=("Segoe UI", 10), padding=6)

menu = tk.Menu(pencere)
pencere.config(menu=menu)
dosya_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Dosya", menu=dosya_menu)
dosya_menu.add_command(label="Veri Yükle (CSV/Excel)", command=csv_yukle)
dosya_menu.add_separator()
dosya_menu.add_command(label="Çıkış", command=pencere.destroy)

main_frame = ttk.Frame(pencere)
main_frame.pack(fill=tk.BOTH, expand=1)

left_frame = ttk.Frame(main_frame, width=350)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

veri_yukle_btn = ttk.Button(left_frame, text="📂 Veri Yükle (CSV/Excel)", command=csv_yukle)
veri_yukle_btn.pack(fill=tk.X, pady=(0, 5))

button_frame = ttk.Frame(left_frame)
button_frame.pack(fill=tk.X, pady=5)

butonlar = [
    ("Genel Bilgi", genel_bilgi),
    ("Varyans Analizi", varyans_analizi),
    ("Ortalama Analizi", ortalama_analizi),
    ("Standart Sapma Analizi", stdsapma_analizi),
    ("Sayısal İstatistikler", sayisal_istatistikler),
    ("Kategorik Özet", kategorik_ozet),
    ("Eksik Veri Görseli", eksik_veri_gorsel),
    ("Eksik Veri Doldur (Ortalama)", eksik_doldur_ortalama),
    ("Histogram", histogram_gorsel),
    ("Pasta Grafiği", pasta_dilimi_gorsel),
    ("Korelasyon Haritası", korelasyon_haritasi),
    ("Boxplot", kategorik_boxplot),
    ("Normalite Testi", normalite_testi),
    ("T-Testi", t_test),
    ("ANOVA Analizi", anova_analizi),
    ("Regresyon Analizi", regresyon_analizi),
    ("Küme Grafiği", kume_grafigi)
]


for i, (yazi, fonk) in enumerate(butonlar):
    row = i // 2
    col = i % 2
    btn = ttk.Button(button_frame, text=yazi, command=fonk)
    btn.grid(row=row, column=col, padx=6, pady=4, sticky="ew")

for c in range(2):
    button_frame.columnconfigure(c, weight=1)

ttk.Label(left_frame, text="Çıktı / Log").pack()
output = scrolledtext.ScrolledText(left_frame, height=25, width=45, wrap=tk.WORD)
output.pack(fill=tk.BOTH, expand=1, pady=5)


graph_frame = ttk.Frame(main_frame)
graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, padx=5, pady=5)

graph_notebook = ttk.Notebook(graph_frame)
graph_notebook.pack(fill=tk.BOTH, expand=1)

grafik_kaydet_buton = ttk.Button(graph_frame, text="Grafiği PNG Kaydet", command=grafik_kaydet)
grafik_kaydet_buton.pack(side=tk.BOTTOM, pady=5)

pencere.mainloop()
