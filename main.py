import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from scipy.stats import norm


class GeneExpressionAnalyzer:
    """Главный класс приложения для анализа экспрессии генов.

    Позволяет загружать CSV-файлы с данными экспрессии генов,
    выбирать конкретные гены и визуализировать их распределение
    в нормальной и опухолевой тканях.
    """

    def __init__(self, root):
        """Инициализация главного окна приложения.

        Args:
            root (tk.Tk): Корневое окно Tkinter
        """
        self.root = root
        self.root.title("Gene Expression Analyzer")
        self.root.geometry("500x300")

        self.df = None  # DataFrame для хранения загруженных данных

        self.create_widgets()

    def create_widgets(self):
        """Создает и размещает все элементы интерфейса в главном окне."""
        # Кнопка для загрузки файла
        ttk.Label(self.root, text="1. Сначала загрузите CSV файл с данными").pack(pady=5)
        self.load_button = ttk.Button(self.root, text="Загрузить CSV файл", command=self.load_csv)
        self.load_button.pack(pady=5)

        # Выбор гена
        ttk.Label(self.root, text="2. Выберите ген для анализа:").pack(pady=5)
        self.gene_combobox = ttk.Combobox(self.root, state="readonly")
        self.gene_combobox.pack(pady=5)

        # Фрейм для кнопок графиков
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        # Кнопки для отображения графиков
        self.combined_button = ttk.Button(
            button_frame,
            text="Совмещенный график",
            command=lambda: self.show_gene_plot("combined"),
            state=tk.DISABLED
        )
        self.combined_button.pack(side=tk.LEFT, padx=5)

        self.tumor_button = ttk.Button(
            button_frame,
            text="Только опухоль",
            command=lambda: self.show_gene_plot("tumor"),
            state=tk.DISABLED
        )
        self.tumor_button.pack(side=tk.LEFT, padx=5)

        self.normal_button = ttk.Button(
            button_frame,
            text="Только норма",
            command=lambda: self.show_gene_plot("normal"),
            state=tk.DISABLED
        )
        self.normal_button.pack(side=tk.LEFT, padx=5)

        # Статус загрузки
        self.status_label = ttk.Label(self.root, text="Файл не загружен", foreground="red")
        self.status_label.pack(pady=5)

    def load_csv(self):
        """Загружает CSV-файл с данными экспрессии генов.

        Открывает диалоговое окно для выбора файла, загружает данные в DataFrame
        и активирует элементы интерфейса для работы с данными.
        """
        file_path = filedialog.askopenfilename(
            title="Выберите CSV файл",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )

        if file_path:
            try:
                # Загрузка данных из CSV
                self.df = pd.read_csv(file_path)
                # Заполнение выпадающего списка gene_id
                self.gene_combobox['values'] = self.df['gene_id'].tolist()
                # Активация кнопок
                self.combined_button['state'] = tk.NORMAL
                self.tumor_button['state'] = tk.NORMAL
                self.normal_button['state'] = tk.NORMAL
                # Обновление статуса
                self.status_label.config(text=f"Загружен файл: {file_path.split('/')[-1]}", foreground="green")
            except Exception as e:
                # Обработка ошибок загрузки
                self.status_label.config(text=f"Ошибка загрузки файла: {str(e)}", foreground="red")

    def generate_normal_distribution(self, median, size=1000):
        """Генерирует нормальное распределение на основе медианного значения.

        Args:
            median (float): Медианное значение для распределения
            size (int): Количество генерируемых точек (по умолчанию 1000)

        Returns:
            np.ndarray: Массив сгенерированных значений
        """
        std = median * 0.1 if median != 0 else 0.1  # Стандартное отклонение как 10% от медианы
        return np.random.normal(loc=median, scale=std, size=size)

    def plot_gene_distribution(self, gene_id, ax, plot_type="combined"):
        """Строит график распределения экспрессии гена.

        Args:
            gene_id (str): Идентификатор гена
            ax (matplotlib.axes.Axes): Ось для построения графика
            plot_type (str): Тип графика ("combined", "tumor" или "normal")
        """
        # Получение данных для выбранного гена
        gene_data = self.df[self.df['gene_id'] == gene_id].iloc[0]

        # Генерация данных распределений
        tumor_data = self.generate_normal_distribution(gene_data['median_tum'])
        norm_data = self.generate_normal_distribution(gene_data['median_norm'])

        # Очистка предыдущего графика
        ax.clear()

        # Построение гистограммы для опухолевых данных (если нужно)
        if plot_type in ["combined", "tumor"]:
            ax.hist(tumor_data, bins=30, alpha=0.5 if plot_type == "combined" else 0.7,
                    color='red', label='Tumor', density=True)
            tumor_kde = norm(np.median(tumor_data), tumor_data.std())

        # Построение гистограммы для нормальных данных (если нужно)
        if plot_type in ["combined", "normal"]:
            ax.hist(norm_data, bins=30, alpha=0.5 if plot_type == "combined" else 0.7,
                    color='blue', label='Normal', density=True)
            norm_kde = norm(np.median(norm_data), norm_data.std())

        # Определение диапазона значений для оси X
        x_min = min(np.min(tumor_data), np.min(norm_data)) if plot_type == "combined" else (
            np.min(tumor_data) if plot_type == "tumor" else np.min(norm_data))
        x_max = max(np.max(tumor_data), np.max(norm_data)) if plot_type == "combined" else (
            np.max(tumor_data) if plot_type == "tumor" else np.max(norm_data))

        x = np.linspace(x_min - 1, x_max + 1, 1000)

        # Построение KDE для опухолевых данных (если нужно)
        if plot_type in ["combined", "tumor"]:
            ax.plot(x, tumor_kde.pdf(x), color='red', linewidth=2)

        # Построение KDE для нормальных данных (если нужно)
        if plot_type in ["combined", "normal"]:
            ax.plot(x, norm_kde.pdf(x), color='blue', linewidth=2)

        # Формирование заголовка
        title = f'Gene: {gene_id}\n'
        if plot_type in ["combined", "tumor"]:
            title += f'Tumor median: {gene_data["median_tum"]:.4f}'
        if plot_type == "combined":
            title += ', '
        if plot_type in ["combined", "normal"]:
            title += f'Normal median: {gene_data["median_norm"]:.4f}'

        # Настройка внешнего вида графика
        ax.set_title(title)
        ax.set_xlabel('Expression Level')
        ax.set_ylabel('Density')
        if plot_type == "combined":
            ax.legend()
        ax.grid(True)

    def save_plot(self, fig, gene_id, plot_type):
        """Сохраняет график в файл.

        Args:
            fig (matplotlib.figure.Figure): Фигура для сохранения
            gene_id (str): Идентификатор гена
            plot_type (str): Тип графика ("combined", "tumor" или "normal")
        """
        # Определение предлагаемого имени файла
        file_types = [
            ('PNG Image', '*.png'),
            ('PDF Document', '*.pdf'),
            ('SVG Vector', '*.svg'),
            ('All Files', '*.*')
        ]

        # Создание диалогового окна сохранения
        file_path = filedialog.asksaveasfilename(
            title="Сохранить график как",
            defaultextension=".png",
            filetypes=file_types,
            initialfile=f"{gene_id}_{plot_type}"
        )

        if file_path:
            try:
                fig.savefig(file_path, bbox_inches='tight', dpi=300)
                tk.messagebox.showinfo("Успех", f"График успешно сохранен в:\n{file_path}")
            except Exception as e:
                tk.messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")

    def show_gene_plot(self, plot_type):
        """Отображает окно с графиком распределения экспрессии гена.

        Args:
            plot_type (str): Тип графика ("combined", "tumor" или "normal")
        """
        selected_gene = self.gene_combobox.get()
        if not selected_gene or self.df is None:
            return

        # Заголовки для разных типов графиков
        titles = {
            "combined": "Совмещенное распределение",
            "tumor": "Распределение опухоли",
            "normal": "Нормальное распределение"
        }

        # Создание нового окна
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"{titles[plot_type]} - {selected_gene}")
        plot_window.geometry("800x600")

        # Создание фигуры matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        self.plot_gene_distribution(selected_gene, ax, plot_type)

        # Встраивание графика в Tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Фрейм для кнопок
        button_frame = ttk.Frame(plot_window)
        button_frame.pack(pady=10)

        # Кнопка сохранения графика
        save_button = ttk.Button(
            button_frame,
            text="Сохранить график",
            command=lambda: self.save_plot(fig, selected_gene, plot_type)
        )
        save_button.pack(side=tk.LEFT, padx=5)

        # Кнопка закрытия окна
        close_button = ttk.Button(
            button_frame,
            text="Закрыть",
            command=plot_window.destroy
        )
        close_button.pack(side=tk.LEFT, padx=5)


if __name__ == "__main__":
    """Точка входа в приложение."""
    root = tk.Tk()  # Создание главного окна
    app = GeneExpressionAnalyzer(root)  # Создание экземпляра приложения
    root.mainloop()  # Запуск главного цикла обработки событий