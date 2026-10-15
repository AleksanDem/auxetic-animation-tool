"""
IPMech Auxetic Animation Tool | PHD_0826
Интерактивный параметрический генератор анимаций деформирования ауксетиков и неауксетиков.
Лаборатория механики перспективных материалов и технологий ИПМех РАН.
Автор: соискатель Демин Александр Иванович.
"""

import os
import sys
import io
import time
from typing import Tuple, Dict, Any, List
import numpy as np
import streamlit as st
from PIL import Image


# Добавляем путь к локальным модулям
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from core.styles import PALETTES
from core.engine_uniaxial import UniaxialEngine
from core.engine_bending import BendingEngine
from core.engine_indentation import IndentationEngine
from core.exporter import export_to_gif, export_to_mp4, check_ffmpeg_available

# 1. Настройка страницы Streamlit
st.set_page_config(
    page_title="Auxetic Animation Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Фирменный стиль IPMech RAS (темная академическая тема)
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #0e1117 !important;
        color: #e0e6ed !important;
    }
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 1.0rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
    }
    header[data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stTabs"] {
        margin-top: 0px !important;
    }
    button[data-baseweb="tab"] {
        cursor: pointer !important;
    }
    .sidebar-brand-title {
        font-size: 1.30rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
        line-height: 1.25;
    }
    .sidebar-brand-subtitle {
        font-size: 0.82rem;
        color: #8fa0b5;
        margin-bottom: 0.6rem;
        line-height: 1.35;
    }
    .sidebar-divider {
        border-bottom: 1px solid #282c37;
        margin-top: 4px;
        margin-bottom: 12px;
    }
    .section-header {
        margin-top: 10px !important;
        margin-bottom: 6px !important;
        font-size: 0.90rem !important;
        font-weight: bold;
        color: #5c88be;
        border-bottom: 1px solid #464b5d;
        padding-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stDownloadButton>button {
        width: 100%;
        background-color: #1E4D8C;
        color: white !important;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.45rem 1rem;
        border: 1px solid #3d4455;
    }
    .stDownloadButton>button:hover {
        background-color: #2563EB;
        color: white !important;
    }
    .column-footer {
        text-align: center;
        color: #6f7380;
        padding-top: 10px;
        padding-bottom: 6px;
        font-size: 0.78rem;
        border-top: 1px solid #282c37;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)


def parse_material_preset(prefix: str, preset_str: str, custom_val: float) -> Tuple[float, str]:
    """Формирует точный коэффициент Пуассона и компактный двухстрочный заголовок: Префикс\n(Материал, nu = ...)"""
    if "0.00" in preset_str:
        return 0.00, f"{prefix}\n(Пробка, ν = 0.00)"
    elif "0.33" in preset_str:
        return 0.33, f"{prefix}\n(PLA / металл, ν = +0.33)"
    elif "0.50" in preset_str and "-" not in preset_str:
        return 0.50, f"{prefix}\n(Резина / эластомер, ν = +0.50)"
    elif "-0.50" in preset_str:
        return -0.50, f"{prefix}\n(Умеренный метаматериал, ν = -0.50)"
    elif "-1.00" in preset_str:
        return -1.00, f"{prefix}\n(Изотропный предел, ν = -1.00)"
    elif "-2.50" in preset_str:
        return -2.50, f"{prefix}\n(Анизотропный метаматериал, ν = -2.50)"
    else:
        sign_str = "+" if custom_val >= 0 else ""
        return custom_val, f"{prefix}\n(Свой выбор, ν = {sign_str}{custom_val:.2f})"


def main():
    # =========================================================================
    # БОКОВАЯ ПАНЕЛЬ УПРАВЛЕНИЯ (СТРОГО ВСЕ ПАРАМЕТРЫ В ST.SIDEBAR)
    # =========================================================================
    st.sidebar.markdown('<div class="sidebar-brand-title">🧬 Auxetic Animation Tool</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-brand-subtitle">Инструмент параметрической генерации анимаций деформирования ауксетиков (GIF / MP4)</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="section-header">⚙️ 1. Физический сценарий</div>', unsafe_allow_html=True)
    scenario = st.sidebar.radio(
        "Выберите фундаментальный сценарий деформирования:",
        [
            "1. Одноосное растяжение / сжатие",
            "2. 3D-изгиб пластины (Купол vs Седло)",
            "3. Контактное индентирование (Самоуплотнение)"
        ],
        index=0,
        label_visibility="collapsed"
    )

    # 1. Соотношение сторон отдельной рамки и расстояние между рамками
    st.sidebar.markdown('<div class="section-header">📐 2. Геометрия отдельной рамки</div>', unsafe_allow_html=True)
    frame_aspect_choice = st.sidebar.radio(
        "Соотношение сторон отдельной рамки (W / H):",
        [
            "1:1 (Квадратная рамка)",
            "3:4 (Вертикальная рамка 3:4)",
            "4:3 (Классическая рамка 4:3)",
            "16:9 (Широкоформатная 16:9)",
            "Кастомное (W / H)"
        ],
        index=0
    )

    if "1:1" in frame_aspect_choice:
        frame_aspect = 1.0
    elif "3:4" in frame_aspect_choice:
        frame_aspect = 3.0 / 4.0
    elif "4:3" in frame_aspect_choice:
        frame_aspect = 4.0 / 3.0
    elif "16:9" in frame_aspect_choice:
        frame_aspect = 16.0 / 9.0
    else:
        col_w, col_h = st.sidebar.columns(2)
        with col_w:
            w_frame = st.number_input("Ширина рамки:", min_value=1.0, value=3.0, step=0.5)
        with col_h:
            h_frame = st.number_input("Высота рамки:", min_value=1.0, value=3.0, step=0.5)
        frame_aspect = w_frame / h_frame

    # Расстояние между рамками (зазор)
    col_gap, col_dpi = st.sidebar.columns(2)
    with col_gap:
        frame_gap = st.slider("Зазор между рамками:", min_value=0.01, max_value=0.40, value=0.06, step=0.01,
                              help="Регулирует расстояние между карточками в полотне")
    with col_dpi:
        dpi = st.number_input("DPI рендера:", min_value=50, max_value=200, value=85, step=5)

    # 3. Компоновка панелей
    st.sidebar.markdown('<div class="section-header">🪟 3. Компоновка панелей</div>', unsafe_allow_html=True)
    layout_name = st.sidebar.radio(
        "Вариант размещения панелей:",
        [
            "Горизонтальная (Side-by-Side 1x2)",
            "Вертикальная (Top/Bottom 2x1)",
            "Сетка 2 на 2 (2x2)"
        ],
        index=0,
        label_visibility="collapsed"
    )
    is_grid_2x2 = ("2 на 2" in layout_name or "2x2" in layout_name)

    # 5. Кружок на выбор внешних рамок (по умолчанию - без рамок)
    st.sidebar.markdown('<div class="section-header">🔘 4. Стиль внешней рамки окон</div>', unsafe_allow_html=True)
    border_style = st.sidebar.radio(
        "Внешняя окантовка карточек:",
        [
            "Без внешних рамок (по умолчанию)",
            "Тонкая нейтральная рамка",
            "Ультратонкая нейтральная рамка",
            "Цветная рамка (в цвет материала)",
            "Пунктирная рамка"
        ],
        index=0,
        label_visibility="collapsed"
    )

    # 2 & 4. Параметры материалов
    st.sidebar.markdown('<div class="section-header">🧪 5. Параметры материалов</div>', unsafe_allow_html=True)

    # Материал 1 (по умолчанию неауксетик)
    st.sidebar.markdown("**Материал 1:**")
    m1_preset = st.sidebar.radio(
        "Пресет Материала 1:",
        [
            "Неауксетик (PLA / металл, ν = +0.33)",
            "Неауксетик (Пробка, ν = 0.00)",
            "Неауксетик (Резина / эластомер, ν = +0.50)",
            "Свой выбор ν₁ (≥ 0)"
        ],
        index=0,
        key="rad_m1",
        label_visibility="collapsed"
    )
    custom_nu1 = 0.33
    if "Свой выбор" in m1_preset:
        custom_nu1 = st.sidebar.number_input("Значение ν₁ (≥ 0):", min_value=0.00, max_value=1.50, value=0.33, step=0.01)
    nu1, title1_auto = parse_material_preset("Неауксетик", m1_preset, custom_nu1)

    # Материал 2 (по умолчанию ауксетик)
    st.sidebar.markdown("**Материал 2:**")
    m2_preset = st.sidebar.radio(
        "Пресет Материала 2:",
        [
            "Ауксетик (Умеренный метаматериал, ν = -0.50)",
            "Ауксетик (Изотропный предел, ν = -1.00)",
            "Ауксетик (Анизотропный метаматериал, ν = -2.50)",
            "Свой выбор ν₂ (< 0)"
        ],
        index=0,
        key="rad_m2",
        label_visibility="collapsed"
    )
    custom_nu2 = -0.50
    if "Свой выбор" in m2_preset:
        custom_nu2 = st.sidebar.number_input("Значение ν₂ (< 0):", min_value=-5.00, max_value=-0.01, value=-0.50, step=0.01)
    nu2, title2_auto = parse_material_preset("Ауксетик", m2_preset, custom_nu2)

    # Материалы 3 и 4 при сетке 2х2
    nu3, title3_auto = 0.0, "Неауксетик\n(Пробка, ν = 0.00)"
    nu4, title4_auto = -1.0, "Ауксетик\n(Изотропный предел, ν = -1.00)"

    if is_grid_2x2:
        st.sidebar.markdown("**Материал 3 (Сетка 2x2):**")
        m3_preset = st.sidebar.radio(
            "Пресет Материала 3:",
            [
                "Неауксетик (Пробка, ν = 0.00)",
                "Неауксетик (PLA / металл, ν = +0.33)",
                "Неауксетик (Резина / эластомер, ν = +0.50)",
                "Ауксетик (Умеренный метаматериал, ν = -0.50)",
                "Свой выбор ν₃"
            ],
            index=0,
            key="rad_m3",
            label_visibility="collapsed"
        )
        custom_nu3 = 0.0
        if "Свой выбор" in m3_preset:
            custom_nu3 = st.sidebar.number_input("Значение ν₃:", value=0.00, step=0.05)
        prefix3 = "Ауксетик" if (custom_nu3 < 0 or "Ауксетик" in m3_preset) else "Неауксетик"
        nu3, title3_auto = parse_material_preset(prefix3, m3_preset, custom_nu3)

        st.sidebar.markdown("**Материал 4 (Сетка 2x2):**")
        m4_preset = st.sidebar.radio(
            "Пресет Материала 4:",
            [
                "Ауксетик (Изотропный предел, ν = -1.00)",
                "Ауксетик (Анизотропный метаматериал, ν = -2.50)",
                "Ауксетик (Умеренный метаматериал, ν = -0.50)",
                "Неауксетик (PLA / металл, ν = +0.33)",
                "Свой выбор ν₄"
            ],
            index=0,
            key="rad_m4",
            label_visibility="collapsed"
        )
        custom_nu4 = -1.0
        if "Свой выбор" in m4_preset:
            custom_nu4 = st.sidebar.number_input("Значение ν₄:", value=-1.00, step=0.05)
        prefix4 = "Ауксетик" if (custom_nu4 < 0 or "Ауксетик" in m4_preset) else "Неауксетик"
        nu4, title4_auto = parse_material_preset(prefix4, m4_preset, custom_nu4)

    # Масштабный множитель деформаций
    st.sidebar.markdown("**Масштабный множитель деформаций:**")
    k_disp = st.sidebar.number_input(
        "Коэффициент k_disp (произвольное число):",
        min_value=0.01,
        value=1.00,
        step=0.10,
        format="%.2f",
        help="Позволяет акцентировать отклик на слайде"
    )

    # Параметры выбранного сценария
    scenario_cfg = {}

    if "1. Одноосное" in scenario:
        max_strain = st.sidebar.number_input("Макс. деформация ε_max:", min_value=0.02, max_value=0.80, value=0.20, step=0.02)
        scenario_cfg["max_strain"] = max_strain

    elif "2. 3D-изгиб" in scenario:
        col_elev, col_azim = st.sidebar.columns(2)
        with col_elev:
            elev = st.number_input("Угол места (Elev):", min_value=5, max_value=85, value=28, step=2)
        with col_azim:
            azim = st.number_input("Азимут (Azim):", min_value=-180, max_value=180, value=-55, step=5)
        scenario_cfg["camera_elev"] = float(elev)
        scenario_cfg["camera_azim"] = float(azim)
        max_curv = st.sidebar.number_input("Макс. кривизна κ_max:", min_value=0.05, max_value=2.00, value=0.55, step=0.05)
        scenario_cfg["max_curvature"] = max_curv

    elif "3. Контактное" in scenario:
        st.sidebar.markdown('<div class="section-header">🔨 Форма индентора</div>', unsafe_allow_html=True)
        ind_choice = st.sidebar.radio(
            "Геометрия наконечника индентора:",
            [
                "Сферическая (полусфера, по умолчанию)",
                "Коническая (клин / конус)"
            ],
            index=0,
            label_visibility="collapsed"
        )
        scenario_cfg["indentor_shape"] = "Сферический (Spherical)" if "Сферическая" in ind_choice else "Конический (Conical/Wedge)"

        indentor_transparent = st.sidebar.checkbox(
            "Прозрачный индентор (пунктирный контур)",
            value=False,
            help="Позволяет четко видеть границу полупространства непосредственно под индентором"
        )
        scenario_cfg["indentor_transparent"] = indentor_transparent

        show_quiver = st.sidebar.checkbox("Стрелки смещений узлов (Quiver)", value=True)
        scenario_cfg["show_quiver"] = show_quiver
        max_indent = st.sidebar.number_input("Глубина внедрения h_max:", min_value=0.10, max_value=2.50, value=0.95, step=0.05)
        scenario_cfg["max_indent"] = max_indent

    # Размеры шрифтов и положение надписей
    st.sidebar.markdown('<div class="section-header">📝 6. Шрифты и положение плашек</div>', unsafe_allow_html=True)
    col_fs1, col_fs2 = st.sidebar.columns(2)
    with col_fs1:
        font_size_title = st.number_input("Шрифт заголовка (pt):", min_value=8.0, max_value=24.0, value=12.5, step=0.5)
    with col_fs2:
        font_size_box = st.number_input("Шрифт плашек (pt):", min_value=6.0, max_value=20.0, value=10.0, step=0.5)

    font_size_status = st.sidebar.number_input("Шрифт статуса (pt):", min_value=6.0, max_value=18.0, value=9.5, step=0.5)

    box_pos_choice = st.sidebar.radio(
        "Положение поясняющего текста:",
        [
            "Скрыть пояснения (по умолчанию)",
            "Внизу карточки",
            "Вверху карточки (под заголовком)"
        ],
        index=0
    )
    box_position_str = "Скрыть" if "Скрыть" in box_pos_choice else ("Вверху" if "Вверху" in box_pos_choice else "Внизу")

    box_style_choice = st.sidebar.radio(
        "Оформление поясняющего текста:",
        [
            "Чистый текст (без плашки)",
            "Классическая плашка с фоном"
        ],
        index=0,
        help="Минималистичный текст без прямоугольного фона по умолчанию для статей и слайдов"
    )

    col_off1, col_off2 = st.sidebar.columns(2)
    with col_off1:
        box_y_offset = st.slider("Сдвиг текста Y:", min_value=-0.30, max_value=0.30, value=0.00, step=0.02)
    with col_off2:
        title_y_offset = st.slider("Сдвиг заголовка Y:", min_value=-0.20, max_value=0.20, value=0.00, step=0.02)

    # 7. Аналитическая формула на анимации
    st.sidebar.markdown('<div class="section-header">📐 7. Аналитическая формула на анимации</div>', unsafe_allow_html=True)
    formula_pos_choice = st.sidebar.radio(
        "Положение формулы на анимации:",
        [
            "Не отображать",
            "1. Между рамками",
            "2. Под рамками",
            "3. Справа от рамок"
        ],
        index=0,
        help="Позволяет разместить теоретические соотношения прямо на экспортируемой анимации"
    )

    # Дефолтные формулы для каждого сценария
    if "1. Одноосное" in scenario:
        default_formula = r"$\varepsilon_x = -\nu \cdot \varepsilon_y$" + "\n" + r"$\theta = \frac{\Delta V}{V_0} = (1 - 2\nu)\,\varepsilon_y$"
    elif "2. 3D-изгиб" in scenario:
        default_formula = r"$w(x, y) = \frac{1}{2}\kappa (x^2 - \nu y^2)$" + "\n" + r"$K_G = \kappa_x \kappa_y = -\nu \kappa^2$"
    else:
        default_formula = r"$u_r \propto \nu \frac{1+\nu}{E}\frac{P}{r}$" + "\n" + r"$\nabla \cdot \mathbf{u} = \frac{1-2\nu}{2G}\sigma_{kk}$"

    if formula_pos_choice != "Не отображать":
        formula_text_input = st.sidebar.text_area(
            "Текст формулы (LaTeX / Mathtext $...$):",
            value=default_formula,
            height=75,
            help="Используйте стандартный синтаксис формул в долларах $...$"
        )
        
        formula_bg_choice = st.sidebar.radio(
            "Фон блока формулы:",
            [
                "Прозрачный фон (по умолчанию)",
                "Белая плашка",
                "Темная плашка"
            ],
            index=0,
            help="Минималистичный вывод чистым шрифтом непосредственно на полотне"
        )

        col_ffs, col_fbox = st.sidebar.columns([1, 1])
        with col_ffs:
            formula_fs = st.number_input("Размер шрифта формулы (pt):", min_value=8.0, max_value=36.0, value=20.0, step=1.0)

        col_fx, col_fy = st.sidebar.columns(2)
        with col_fx:
            formula_x_off = st.slider("Сдвиг формулы X:", min_value=-0.35, max_value=0.35, value=0.00, step=0.01)
        with col_fy:
            formula_y_off = st.slider("Сдвиг формулы Y:", min_value=-0.35, max_value=0.35, value=0.00, step=0.01)
    else:
        formula_text_input = ""
        formula_bg_choice = "Прозрачный фон (по умолчанию)"
        formula_fs = 20.0
        formula_x_off = 0.0
        formula_y_off = 0.0

    # Кинематика и тайминги
    st.sidebar.markdown('<div class="section-header">⏱️ 8. Кинематика и тайминги</div>', unsafe_allow_html=True)
    motion_type_label = st.sidebar.selectbox(
        "Закон движения:",
        [
            "Однонаправленный с фазой удержания пика",
            "Гармонический знакопеременный (Растяжение -> Сжатие)",
            "Чистый синус (без пауз)"
        ],
        index=0
    )
    motion_type_map = {
        "Однонаправленный с фазой удержания пика": "unidirectional_hold",
        "Гармонический знакопеременный (Растяжение -> Сжатие)": "symmetric_cycle",
        "Чистый синус (без пауз)": "pure_sine"
    }
    motion_type = motion_type_map[motion_type_label]

    col_fps, col_frames = st.sidebar.columns(2)
    with col_fps:
        fps = st.number_input("Частота (FPS):", min_value=5, max_value=30, value=15, step=1)
    with col_frames:
        n_frames = st.number_input("Всего кадров:", min_value=8, max_value=60, value=24, step=2)

    hold_frames = st.sidebar.slider("Кадры удержания в пике:", 0, 10, 4, 1)

    # Академическая палитра и флаги
    st.sidebar.markdown('<div class="section-header">🎨 9. Палитра и элементы</div>', unsafe_allow_html=True)
    palette_choice = st.sidebar.selectbox("Цветовая палитра:", list(PALETTES.keys()), index=0)

    col_c1, col_c2 = st.sidebar.columns(2)
    with col_c1:
        show_arrows = st.checkbox("Стрелки сил / реакции", value=True)
        show_labels = st.checkbox("Заголовки панелей", value=True)
    with col_c2:
        show_boxes = st.checkbox("Поясняющие плашки", value=("Скрыть" not in box_pos_choice))
        show_ghost = st.checkbox("Недеформированный контур", value=True)

    show_grid = st.sidebar.checkbox("Сетка деформации (КЭ)", value=True)

    # Экспандер для кастомных формулировок
    with st.sidebar.expander("📝 Настроить тексты и подписи"):
        custom_title1 = st.text_input("Заголовок 1:", value=title1_auto)
        custom_title2 = st.text_input("Заголовок 2:", value=title2_auto)
        custom_title3 = st.text_input("Заголовок 3:", value=title3_auto) if is_grid_2x2 else title3_auto
        custom_title4 = st.text_input("Заголовок 4:", value=title4_auto) if is_grid_2x2 else title4_auto

        if "1. Одноосное" in scenario:
            default_sub1 = "Поперечное обужение при растяжении\n(Классический закон Пуассона)"
            default_sub2 = "Поперечное расширение при растяжении\n(Контринтуитивный ауксетический отклик)"
            default_sub3 = "Поперечный отклик среды (ν = 0)\n(Нулевая поперечная деформация)"
            default_sub4 = "Предельный ауксетический отклик\n(Синхронное объемное расширение)"
        elif "2. 3D-изгиб" in scenario:
            default_sub1 = "Антикластический изгиб → СЕДЛО\nКривизны разных знаков (склонность к короблению)"
            default_sub2 = "Синкластический изгиб → КУПОЛ\nКривизны одного знака (сопротивление потере устойчивости)"
            default_sub3 = "Цилиндрический изгиб (ν = 0)\nКривизна строго в одной плоскости"
            default_sub4 = "Сферический купол (ν = -1.0)\nИдеальная изотропная чаша"
        else:
            default_sub1 = "Поперечное вытеснение материала в стороны\n→ локальное утонение → механизм среза пробки"
            default_sub2 = "Поперечное стягивание массы к оси удара\n→ локальное самоуплотнение → динамическое запирание"
            default_sub3 = "Одноосная деформация под штампом\n(Отсутствие поперечных потоков массы)"
            default_sub4 = "Максимальное поперечное стягивание\n(Формирование плотного клина жесткости)"

        custom_sub1 = st.text_area("Плашка 1:", value=default_sub1, height=65)
        custom_sub2 = st.text_area("Плашка 2:", value=default_sub2, height=65)
        custom_sub3 = st.text_area("Плашка 3:", value=default_sub3, height=65) if is_grid_2x2 else default_sub3
        custom_sub4 = st.text_area("Плашка 4:", value=default_sub4, height=65) if is_grid_2x2 else default_sub4

    # ------------------ СБОРКА КОНФИГУРАЦИИ ДВИЖКА ------------------
    config = {
        "palette": palette_choice,
        "frame_aspect": frame_aspect,
        "frame_gap": frame_gap,
        "border_style": border_style,
        "dpi": dpi,
        "layout": layout_name,
        "font_size_title": font_size_title,
        "font_size_box": font_size_box,
        "font_size_status": font_size_status,
        "box_position": box_position_str,
        "box_style": box_style_choice,
        "box_y_offset": box_y_offset,
        "title_y_offset": title_y_offset,
        "formula_position": formula_pos_choice,
        "formula_text": formula_text_input,
        "formula_font_size": formula_fs,
        "formula_x_offset": formula_x_off,
        "formula_y_offset": formula_y_off,
        "formula_bg_style": formula_bg_choice,
        "show_arrows": show_arrows,
        "show_labels": show_labels,
        "show_boxes": show_boxes,
        "show_grid": show_grid,
        "show_ghost": show_ghost,
        "k_disp": k_disp,
        "nu1": nu1,
        "nu2": nu2,
        "nu3": nu3,
        "nu4": nu4,
        "title1": custom_title1,
        "title2": custom_title2,
        "title3": custom_title3,
        "title4": custom_title4,
        "sub1": custom_sub1,
        "sub2": custom_sub2,
        "sub3": custom_sub3,
        "sub4": custom_sub4,
        "motion_type": motion_type,
        "hold_frames": hold_frames,
        **scenario_cfg
    }

    # Инициализация соответствующего движка
    if "1. Одноосное" in scenario:
        engine = UniaxialEngine(config)
        scenario_short = "uniaxial"
    elif "2. 3D-изгиб" in scenario:
        engine = BendingEngine(config)
        scenario_short = "bending"
    else:
        engine = IndentationEngine(config)
        scenario_short = "indentation"

    # =========================================================================
    # ГЛАВНАЯ ОБЛАСТЬ (MAIN AREA) — СТИЛЬ IPMECH RAS (МАКСИМАЛЬНО КОМПАКТНО)
    # =========================================================================
    tab_preview, tab_generate, tab_theory = st.tabs([
        "👁️ Интерактивный предпросмотр (Стоп-кадр)",
        "🎬 Генерация и Экспорт (GIF / MP4)",
        "📖 Физическая теория и формулы"
    ])

    # ------------------ ВКЛАДКА 1: ПРЕДПРОСМОТР ------------------
    with tab_preview:
        col_prev_ctrl, col_prev_info = st.columns([3, 1])
        with col_prev_ctrl:
            t_slider = st.slider(
                "Фаза деформации для предпросмотра (t):",
                min_value=-1.0 if motion_type == "symmetric_cycle" else 0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                help="Позволяет моментально проверить геометрию, положение шрифтов и границы перед запуском рендера"
            )

        with col_prev_info:
            ffmpeg_ok = check_ffmpeg_available()
            if ffmpeg_ok:
                st.markdown('<div style="margin-top:28px; padding:6px 8px; background:#1e3a29; border:1px solid #2e7d32; border-radius:6px; font-size:0.78rem; color:#a5d6a7; text-align:center; font-weight:600;">✅ FFmpeg активен</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="margin-top:28px; padding:6px 8px; background:#3e2723; border:1px solid #d84315; border-radius:6px; font-size:0.78rem; color:#ffab91; text-align:center; font-weight:600;">⚠️ Режим Pillow</div>', unsafe_allow_html=True)

        # Генерация и отображение стоп-кадра
        with st.spinner("Рендеринг кадра предпросмотра..."):
            preview_img = engine.generate_single_frame(t_slider)
            st.image(preview_img, use_container_width=True, caption=f"Предпросмотр: {scenario} при t = {t_slider:+.2f}")

        # Кнопка мгновенного скачивания текущего стоп-кадра в PNG
        buf_png = io.BytesIO()
        preview_img.save(buf_png, format="PNG")
        st.download_button(
            label="📸 Скачать этот стоп-кадр в высоком разрешении (PNG)",
            data=buf_png.getvalue(),
            file_name=f"auxetic_{scenario_short}_frame_t_{t_slider:+.2f}.png",
            mime="image/png"
        )

    # ------------------ ВКЛАДКА 2: ГЕНЕРАЦИЯ И ЭКСПОРТ ------------------
    with tab_generate:
        st.markdown("### 🚀 Рендеринг полной анимации")
        st.info(f"Параметры видеоряда: **{n_frames} кадров** при **{fps} FPS** (длительность: **{n_frames/fps:.1f} с**). Разрешение: **{preview_img.size[0]} × {preview_img.size[1]} px**.")

        col_btn_gif, col_btn_mp4, col_btn_all = st.columns(3)
        render_gif = col_btn_gif.button("🎞️ Сгенерировать GIF", use_container_width=True)
        render_mp4 = col_btn_mp4.button("🎥 Сгенерировать MP4 (H.264)", use_container_width=True)
        render_both = col_btn_all.button("⚡ Сгенерировать оба формата", use_container_width=True)

        # Безопасный выбор выходной папки (локальный режим капсулы vs облако Streamlit Cloud)
        dissertation_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
        if os.path.basename(dissertation_root) == "PHD_0826_Dissertation" and os.path.exists(os.path.join(dissertation_root, "04_Artifacts")):
            output_dir = os.path.join(dissertation_root, "04_Artifacts", "01_Drafts", "Generated_Animations")
        else:
            output_dir = os.path.join(CURRENT_DIR, "output_animations")

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            import tempfile
            output_dir = os.path.join(tempfile.gettempdir(), "auxetic_animations")
            os.makedirs(output_dir, exist_ok=True)

        if render_gif or render_mp4 or render_both:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            status_text.text("Рендеринг кадров анимации в памяти...")

            def progress_callback(cur, total):
                prog = cur / total
                progress_bar.progress(prog * 0.7)
                status_text.text(f"Отрендерено кадров: {cur}/{total} ({prog*100:.0f}%)")

            frames = engine.generate_all_frames(n_frames=n_frames, progress_callback=progress_callback)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")

            # Экспорт в GIF
            if render_gif or render_both:
                status_text.text("Компиляция и сжатие GIF-файла...")
                progress_bar.progress(0.85)
                gif_name = f"auxetic_{scenario_short}_{timestamp_str}.gif"
                gif_path = os.path.join(output_dir, gif_name)
                export_to_gif(frames, gif_path, fps=fps, optimize_palette=True)

                st.session_state["last_gif"] = gif_path
                st.session_state["gif_name"] = gif_name

            # Экспорт в MP4
            if render_mp4 or render_both:
                status_text.text("Кодирование видеопотока MP4 (H.264 / yuv420p)...")
                progress_bar.progress(0.95)
                mp4_name = f"auxetic_{scenario_short}_{timestamp_str}.mp4"
                mp4_path = os.path.join(output_dir, mp4_name)
                export_to_mp4(frames, mp4_path, fps=fps)

                st.session_state["last_mp4"] = mp4_path
                st.session_state["mp4_name"] = mp4_name

            progress_bar.empty()
            status_text.empty()

        # Отображение готовых результатов и кнопок скачивания
        if "last_gif" in st.session_state and os.path.exists(st.session_state["last_gif"]):
            st.markdown("---")
            st.markdown("#### 🎞️ Зацикленная анимация GIF (Для слайдов PowerPoint)")
            col_v1, col_d1 = st.columns([3, 1])
            with col_v1:
                st.image(st.session_state["last_gif"], use_container_width=True)
            with col_d1:
                gif_bytes = open(st.session_state["last_gif"], "rb").read()
                st.markdown(f"**Размер файла:** {len(gif_bytes)/1024:.1f} КБ")
                st.download_button(
                    label="📥 Скачать файл GIF",
                    data=gif_bytes,
                    file_name=st.session_state["gif_name"],
                    mime="image/gif",
                    key="dl_gif"
                )

        if "last_mp4" in st.session_state and os.path.exists(st.session_state["last_mp4"]):
            st.markdown("---")
            st.markdown("#### 🎥 Видеофайл MP4 (Full HD / PowerPoint Compatible)")
            col_v2, col_d2 = st.columns([3, 1])
            with col_v2:
                st.video(st.session_state["last_mp4"])
            with col_d2:
                mp4_bytes = open(st.session_state["last_mp4"], "rb").read()
                st.markdown(f"**Размер файла:** {len(mp4_bytes)/1024:.1f} КБ")
                st.download_button(
                    label="📥 Скачать файл MP4",
                    data=mp4_bytes,
                    file_name=st.session_state["mp4_name"],
                    mime="video/mp4",
                    key="dl_mp4"
                )

    # ------------------ ВКЛАДКА 3: ТЕОРИЯ И ФОРМУЛЫ ------------------
    with tab_theory:
        st.markdown("### 📚 Теоретические основы механики ауксетиков")
        st.markdown("""
        **Ауксетики (Auxetics)** — метаматериалы и структуры с **отрицательным коэффициентом Пуассона** ($\nu < 0$).
        В отличие от традиционных материалов (неауксетиков), при одноосном растяжении ауксетики расширяются в поперечном направлении, 
        а при сжатии — стягиваются.
        """)

        st.markdown("#### 1. Одноосное деформированное состояние")
        st.latex(r"\varepsilon_x = -\nu \cdot \varepsilon_y, \quad \theta = \frac{\Delta V}{V_0} = (1 - 2\nu)\varepsilon_y")

        st.markdown("#### 2. Изгиб пластин (Купол vs Седло)")
        st.latex(r"w(x, y) = \frac{1}{2}\kappa \left(x^2 - \nu y^2\right)")

        st.markdown("#### 3. Контактное внедрение и локальное самоуплотнение")
        st.latex(r"u_y(x, y) = -\delta e^{-\frac{x^2}{2a^2}} e^{\frac{y}{b}}, \quad u_x(x, y) = \nu \frac{\delta}{b} \sqrt{\frac{\pi}{2}} a \cdot \text{erf}\left(\frac{x}{\sqrt{2}a}\right) e^{\frac{y}{b}}")

    # Авторский футер ИПМех РАН
    st.markdown(
        '<div class="column-footer">'
        '© 2026 Demin A.I. — Laboratory of Mechanics of Novel Materials and Technologies IPMech RAS'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
