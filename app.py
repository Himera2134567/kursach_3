# -*- coding: utf-8 -*-
"""
Аниме-помощница — быстрые точные ответы (фикс времени по городам РФ, чистый TTS, меньше «спама» партиалов)
— Тёмный UI (QSS)
— Стриминг LLM (OpenRouter, SSE), «⚡ Быстрый режим»
— Поиск (ddgs) с кэшем и таймаутом
— Погода (Open-Meteo, без ключей)
— Время по городам РФ (включая Саранск, Саратов и др.) — локально, БЕЗ поиска
— STT: Vosk (автокопия в ASCII-путь), приглушённые partial-подсказки
— TTS: pyttsx3 (офлайн), Санитайзер текста (никаких «жирных знаков», HTML и мусора)
— 3D-персонаж (Qt3D, вращающийся куб), фоллбэк на 2D-аватар
"""

import os
import sys
import json
import time
import shutil
import queue
import ctypes
import tempfile
import threading
import functools
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from html import escape as html_escape
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import requests
import sounddevice as sd

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QAction, QTextCursor, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTextEdit, QLineEdit, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QComboBox, QCheckBox, QSplitter, QMessageBox
)

# ---- Qt3D опционально ----
QT3D_OK = False
try:
    from PySide6 import Qt3DCore, Qt3DExtras, Qt3DRender
    QT3D_OK = True
except Exception:
    QT3D_OK = False

# ---------- окружение ----------
APP_TITLE = "Аниме-помощница"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat")  # быстрый default
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_HEADERS_BASE = {"HTTP-Referer": "https://local.assistant.app", "X-Title": APP_TITLE}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOSK_MODEL_DIR_PRIMARY = os.path.join(BASE_DIR, "models", "vosk-ru")
VOSK_MODEL_DIR_ALT = os.path.join(BASE_DIR, "vosk-model-small-ru-0.22")

# ---------- словари городов ----------
# Координаты (Open-Meteo)
CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "москва": (55.7558, 37.6173), "мск": (55.7558, 37.6173),
    "санкт-петербург": (59.9386, 30.3141), "питер": (59.9386, 30.3141), "спб": (59.9386, 30.3141),
    "новосибирск": (55.0084, 82.9357),
    "екатеринбург": (56.8389, 60.6057),
    "казань": (55.7903, 49.1125),
    "нижний новгород": (56.3269, 44.0075),
    "самара": (53.1959, 50.1008),
    "саратов": (51.5331, 46.0342),
    "саранск": (54.1806, 45.1860),
    "омск": (54.9893, 73.3682),
    "челябинск": (55.1644, 61.4368),
    "ростов-на-дону": (47.2357, 39.7015),
    "уфа": (54.7388, 55.9721),
    "красноярск": (56.0153, 92.8932),
    "пермь": (58.0105, 56.2294),
    "воронеж": (51.6608, 39.2003),
    "волгоград": (48.7080, 44.5133),
    "краснодар": (45.0355, 38.9753),
    "тюмень": (57.1530, 65.5343),
    "тольятти": (53.5088, 49.4196),
    "ижевск": (56.8526, 53.2045),
    "барнаул": (53.3481, 83.7798),
    "уляновск": (54.3142, 48.4031),
    "иркутск": (52.2869, 104.3050),
    "хабаровск": (48.4802, 135.0710),
    "владивосток": (43.1155, 131.8855),
    "ярославль": (57.6266, 39.8938),
    "томск": (56.4847, 84.9482),
    "калининград": (54.7104, 20.4522),
    "москов": (55.7558, 37.6173), "мск.": (55.7558, 37.6173),
}

# Таймзоны (локальное точное время)
CITY_TZ: Dict[str, str] = {
    "москва": "Europe/Moscow", "мск": "Europe/Moscow", "москов": "Europe/Moscow",
    "санкт-петербург": "Europe/Moscow", "питер": "Europe/Moscow", "спб": "Europe/Moscow",
    "саратов": "Europe/Saratov",   # UTC+4
    "саранск": "Europe/Moscow",    # Республика Мордовия — МСК
    "самара": "Europe/Samara",
    "казань": "Europe/Moscow",
    "екатеринбург": "Asia/Yekaterinburg",
    "новосибирск": "Asia/Novosibirsk",
    "красноярск": "Asia/Krasnoyarsk",
    "омск": "Asia/Omsk",
    "владивосток": "Asia/Vladivostok",
    "калининград": "Europe/Kaliningrad",
}

# ---------- поиск ----------
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except Exception:
    DDGS_AVAILABLE = False

@functools.lru_cache(maxsize=256)
def ddg_search_cached(query: str, max_results: int = 8, timeout_sec: float = 4.0) -> List[Tuple[str, str]]:
    if not DDGS_AVAILABLE:
        return [("Установи пакет ddgs: pip install ddgs", "")]
    out: List[Tuple[str, str]] = []
    done = threading.Event()

    def worker():
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results, safesearch="moderate"):
                    title = r.get("title") or r.get("body") or ""
                    href = r.get("href") or r.get("link") or ""
                    if title and href:
                        out.append((title, href))
        except Exception as e:
            out.append((f"[Поиск: ошибка] {e}", ""))
        finally:
            done.set()

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    done.wait(timeout=timeout_sec)
    return out

# ---------- утилиты ----------
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_EMOJI_RE = re.compile(
    "["                     # убрать эмодзи/символы, чтобы TTS не «уренг 256»
    "\U0001F300-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U00002B00-\U00002BFF"
    "]+", flags=re.UNICODE
)

def strip_openrouter_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()

def sanitize_tts(text: str) -> str:
    # удаляем HTML, эмодзи, сжимаем пробелы/пунктацию
    t = _TAG_RE.sub("", text or "")
    t = _EMOJI_RE.sub("", t)
    t = re.sub(r"\s+", " ", t)
    # удаляем URL, артефакты типа [DONE], data:, и т.п.
    t = re.sub(r"https?://\S+", "", t, flags=re.I)
    t = t.replace("[DONE]", "").replace("data:", "")
    return t.strip()

def is_search_intent(user_text: str) -> bool:
    t = (user_text or "").lower().strip()
    if not t:
        return False
    starts = ["найди", "поиск", "поищи", "открой"]
    qwords = ["что","кто","где","когда","как","сколько","какая","какой","какие"]
    keywords = ["погода","курс","новости","определение","инструкция","время","температура"]
    return any(t.startswith(s) for s in starts) or any(t.startswith(w+" ") for w in qwords) or any(k in t for k in keywords)

_TIME_PAT = re.compile(r"(сколько\s+сейчас\s+времени|который\s+час|время)\s*(в|на)?\s*(?P<city>[а-яё\- ]+)?", re.I)

def detect_time_city(user_text: str) -> Optional[str]:
    t = (user_text or "").lower()
    m = _TIME_PAT.search(t)
    if m:
        city = (m.group("city") or "").strip(" .,:;!?")
        if city:
            # нормализуем падежи: «в саратове/саранске/москве»
            city = city.replace("  ", " ")
            # самое простое — убрать окончания -е/-у/-е/-и где очевидно
            forms = [city]
            if city.endswith("е"): forms.append(city[:-1])
            if city.endswith("у"): forms.append(city[:-1])
            if city.endswith("и"): forms.append(city[:-1])
            if city.endswith("ии"): forms.append(city[:-2])
            if city.endswith("ы"): forms.append(city[:-1])
            # спец случаи
            repl = {
                "москов": "москва",
                "питере": "питер",
                "санкт петербурге": "санкт-петербург",
                "санкт-петербурге": "санкт-петербург",
            }
            for k, v in repl.items():
                if city == k: forms.append(v)
            # проверяем в словаре
            for f in forms:
                f = f.strip()
                if f in CITY_TZ: return CITY_TZ[f]
    # если просто «сколько времени» без города — ответим по Мск
    if "сколько" in t and "времен" in t:
        return "Europe/Moscow"
    return None

def detect_time_tz(user_text: str) -> Optional[str]:
    # прямой поиск по CITY_TZ
    t = (user_text or "").lower()
    for city, tz in CITY_TZ.items():
        if city in t:
            return tz
    # fallback — регэксп
    return detect_time_city(user_text)

def get_time_for_tz(tz_name: str) -> datetime:
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        # запасной: Москва
        return datetime.utcnow() + timedelta(hours=3)

def find_city_coords(text: str) -> Optional[Tuple[float, float, str]]:
    t = (text or "").lower()
    for name, (lat, lon) in CITY_COORDS.items():
        if name in t:
            return lat, lon, name
    if "погод" in t or "температур" in t:
        return CITY_COORDS["москва"][0], CITY_COORDS["москва"][1], "москва"
    return None

# ---------- погода (Open-Meteo) ----------
WEATHER_CODE_MAP = {
    0: "Ясно", 1: "Преимущественно ясно", 2: "Переменная облачность", 3: "Пасмурно",
    45: "Туман", 48: "Инейный туман",
    51: "Лёгкая морось", 53: "Умеренная морось", 55: "Сильная морось",
    61: "Лёгкий дождь", 63: "Дождь", 65: "Ливень",
    71: "Небольшой снег", 73: "Снег", 75: "Сильный снег",
    80: "Кратковременные дожди", 81: "Ливни", 82: "Сильные ливни",
    95: "Гроза", 96: "Гроза с градом", 99: "Сильная гроза с градом",
}

@functools.lru_cache(maxsize=256)
def get_weather_now(lat: float, lon: float) -> Optional[str]:
    try:
        url = ("https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code,wind_speed_10m")
        r = requests.get(url, timeout=4)
        r.raise_for_status()
        data = r.json()
        cur = data.get("current", {})
        t = cur.get("temperature_2m")
        w = cur.get("wind_speed_10m", 0)
        code = cur.get("weather_code")
        desc = WEATHER_CODE_MAP.get(code, "на улице нормально")
        if t is None: return None
        return f"{desc}, {t:.0f}°C, ветер {w:.0f} м/с"
    except Exception:
        return None

# ---------- TTS: pyttsx3 ----------
import pyttsx3
class SpeakerPyttsx3:
    def __init__(self, info_cb=None):
        self._q: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._info_cb = info_cb
        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()

    def say(self, text: str):
        txt = sanitize_tts(text)
        if not txt: return
        try: self._q.put_nowait(txt)
        except Exception: pass

    def _select_ru_voice(self, engine: pyttsx3.Engine):
        chosen = None
        for v in engine.getProperty("voices"):
            name = (v.name or "").lower()
            lang = ",".join(getattr(v, "languages", []) or []).lower()
            if "ru" in name or "rus" in name or "ru" in lang:
                chosen = v.id; break
        if chosen:
            engine.setProperty("voice", chosen)

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)   # чуть медленнее и разборчивее
        engine.setProperty("volume", 1.0)
        self._select_ru_voice(engine)
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not text: continue
            try:
                engine.say(text); engine.runAndWait()
            except Exception as e:
                if self._info_cb: self._info_cb(f"❗ TTS: {e}")

    def shutdown(self):
        self._stop.set()
        try: self._q.put_nowait("")
        except Exception: pass

# ---------- OpenRouter: stream ----------
@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.22
    max_tokens: int = 256
    stream: bool = True

def openrouter_stream(messages: List[dict], cfg: LLMConfig):
    if not OPENROUTER_KEY:
        yield "❗ Не задан ключ OPENROUTER_API_KEY."
        return
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        **OPENROUTER_HEADERS_BASE,
    }
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": True
    }
    try:
        with requests.post(url, headers=headers, data=json.dumps(payload), stream=True, timeout=120) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("data: "):
                    data = raw[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        js = json.loads(data)
                        delta = js["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
    except Exception as e:
        yield f"\n\n❗ Ошибка OpenRouter: {e}"

# ---------- Vosk ----------
def is_ascii_path(path: str) -> bool:
    try: path.encode("ascii")
    except UnicodeEncodeError: return False
    return " " not in path

def prepare_vosk_path(original_dir: str) -> Optional[str]:
    if not os.path.isdir(original_dir): return None
    conf = os.path.join(original_dir, "conf", "model.conf")
    if not os.path.isfile(conf): return None
    if is_ascii_path(original_dir): return original_dir
    safe_target = os.path.join(tempfile.gettempdir(), "vosk-ru")
    if os.path.isdir(safe_target) and os.path.isfile(os.path.join(safe_target, "conf", "model.conf")):
        return safe_target
    try:
        if os.path.isdir(safe_target): shutil.rmtree(safe_target, ignore_errors=True)
        shutil.copytree(original_dir, safe_target)
        return safe_target
    except Exception as e:
        print("[VOSK] copy fail:", e); return None

class VoskRecognizerThread(QtCore.QThread):
    partial_update = QtCore.Signal(str)
    final_result = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, model_dir: str, parent=None, samplerate: int = 16000, blocksize: int = 4096):
        super().__init__(parent)
        self._running = False
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._q: "queue.Queue[bytes]" = queue.Queue()
        self._model_dir = model_dir
        self._last_partial = ""
        self._last_emit = 0.0

    def run(self):
        try:
            from vosk import Model, KaldiRecognizer
            model = Model(self._model_dir)
            rec = KaldiRecognizer(model, self.samplerate)
            rec.SetWords(True)
        except Exception as e:
            self.error.emit(f"Ошибка инициализации Vosk: {e}")
            return

        def _callback(indata, frames, time_info, status):
            data = indata[:, 0].copy()
            data_bytes = (data * 32767).astype(np.int16).tobytes()
            self._q.put(data_bytes)

        try:
            self._running = True
            with sd.InputStream(samplerate=self.samplerate, channels=1, dtype="float32",
                                blocksize=self.blocksize, callback=_callback):
                while self._running:
                    try:
                        data = self._q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result()); text = (res.get("text") or "").strip()
                        if text: self.final_result.emit(text)
                    else:
                        part = json.loads(rec.PartialResult()).get("partial", "").strip()
                        now = time.time()
                        if part and part != self._last_partial and (now - self._last_emit) > 0.25:
                            self._last_partial = part
                            self._last_emit = now
                            self.partial_update.emit(part)
                res = json.loads(rec.FinalResult()); text = (res.get("text") or "").strip()
                if text: self.final_result.emit(text)
        except Exception as e:
            self.error.emit(f"Ошибка аудиопотока: {e}")

    def stop(self):
        self._running = False

# ---------- Аватары ----------
class Avatar3DWidget(QWidget):
    """Qt3D: вращающийся «чиби-куб»."""
    def __init__(self, parent=None):
        super().__init__(parent)
        if not QT3D_OK:
            raise RuntimeError("Qt3D недоступен")
        self.view = Qt3DExtras.Qt3DWindow()
        self.container = QWidget.createWindowContainer(self.view, self)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.addWidget(self.container)

        root = Qt3DCore.QEntity()
        cam = self.view.camera()
        cam.lens().setPerspectiveProjection(45.0, 1.6, 0.1, 1000.0)
        cam.setPosition(QtGui.QVector3D(0, 2, 8))
        cam.setViewCenter(QtGui.QVector3D(0, 0, 0))

        lightEntity = Qt3DCore.QEntity(root)
        light = Qt3DRender.QPointLight(lightEntity)
        light.setColor(Qt.white); light.setIntensity(1.3)
        lightEntity.addComponent(light)
        lt = Qt3DCore.QTransform(); lt.setTranslation(QtGui.QVector3D(10, 10, 10))
        lightEntity.addComponent(lt)

        self.ent = Qt3DCore.QEntity(root)
        mesh = Qt3DExtras.QCuboidMesh()
        mat = Qt3DExtras.QPhongMaterial(self.ent); mat.setDiffuse(QtGui.QColor("#ff99cc"))
        tr = Qt3DCore.QTransform(); tr.setScale3D(QtGui.QVector3D(1.8, 1.8, 1.8))
        self.ent.addComponent(mesh); self.ent.addComponent(mat); self.ent.addComponent(tr)

        self.anim = QtCore.QPropertyAnimation(tr, b"rotation")
        self.anim.setDuration(8000)
        self.anim.setStartValue(QtGui.QQuaternion.fromAxisAndAngle(QtGui.QVector3D(0,1,0), 0))
        self.anim.setEndValue(QtGui.QQuaternion.fromAxisAndAngle(QtGui.QVector3D(0,1,0), 360))
        self.anim.setLoopCount(-1); self.anim.start()

        self.view.setRootEntity(root)

class Avatar2DWidget(QPushButton):
    """Круглый 2D-аватар (фоллбэк)."""
    def __init__(self, size=220, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(size, size)
        self.setStyleSheet("border:none;")
        self.setIconSize(QtCore.QSize(size, size))
        self.setIcon(QIcon(self._make_pixmap(size)))

    def _make_pixmap(self, size: int) -> QPixmap:
        w = h = size
        pm = QPixmap(w, h); pm.fill(Qt.transparent)
        p = QtGui.QPainter(pm); p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        grad = QtGui.QRadialGradient(QtCore.QPointF(w*0.5, h*0.45), w*0.55)
        grad.setColorAt(0, QtGui.QColor(255, 220, 240)); grad.setColorAt(1, QtGui.QColor(230, 200, 255))
        p.setBrush(QtGui.QBrush(grad)); p.setPen(Qt.NoPen); p.drawEllipse(0, 0, w, h)
        face = QtGui.QColor(255, 245, 230); p.setBrush(face); p.drawEllipse(w*0.2, h*0.2, w*0.6, h*0.6)
        p.setBrush(Qt.white); p.drawEllipse(w*0.32, h*0.42, w*0.12, h*0.12); p.drawEllipse(w*0.56, h*0.42, w*0.12, h*0.12)
        p.setBrush(Qt.black); p.drawEllipse(w*0.355, h*0.46, w*0.05, h*0.05); p.drawEllipse(w*0.595, h*0.46, w*0.05, h*0.05)
        pen = QtGui.QPen(QtGui.QColor(200, 80, 120), 3); p.setPen(pen); p.setBrush(Qt.NoBrush)
        p.drawArc(int(w*0.35), int(h*0.58), int(h*0.3), int(h*0.15), 0, -1440); p.end()
        return pm

# ---------- UI ----------
DARK_QSS = """
QMainWindow { background: #0d1117; }
QTextEdit { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; }
QLineEdit { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 6px; }
QLabel { color: #8b949e; }
QPushButton { background: #238636; color: white; border: 1px solid #2ea043; border-radius: 6px; padding: 6px 10px; }
QPushButton:hover { background: #2ea043; }
QPushButton#micBtn { background: #30363d; border: 1px solid #3b434b; }
QProgressBar { background: #0d1117; border: 1px solid #30363d; border-radius: 3px; height: 6px; }
QProgressBar::chunk { background: #8957e5; }
QComboBox, QCheckBox { color: #c9d1d9; }
QSplitter::handle { background: #161b22; }
QMenuBar { background: #0d1117; color: #c9d1d9; }
QMenu { background: #161b22; color: #c9d1d9; }
"""

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1180, 760)
        self.setStyleSheet(DARK_QSS)

        try:
            if sys.platform.startswith("win"):
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass

        self.listening = False
        self.rec_thread: Optional[VoskRecognizerThread] = None
        self.messages = [{"role": "system", "content": "Ты дружелюбная виртуальная помощница-анимешница. Отвечай по-русски и по делу. Если это поиск — верни ссылки."}]
        self.llm_model = DEFAULT_MODEL

        self.speaker = SpeakerPyttsx3(info_cb=self._append_info)

        self.chat = QTextEdit(); self.chat.setReadOnly(True)
        self.input = QLineEdit(); self.input.setPlaceholderText("Скажи или напиши запрос…")
        self.input.returnPressed.connect(self.on_send)

        self.progress = QProgressBar(); self.progress.setTextVisible(False); self.progress.setRange(0,1)

        self.mic_btn = QPushButton("🎤"); self.mic_btn.setObjectName("micBtn"); self.mic_btn.setFixedWidth(44)
        self.mic_btn.clicked.connect(self.toggle_listen)

        self.speak_cb = QCheckBox("🔈 Озвучивать"); self.speak_cb.setChecked(True)
        self.fast_cb = QCheckBox("⚡ Быстрый режим"); self.fast_cb.setChecked(True)

        self.model_box = QComboBox()
        self.model_box.addItems([
            "deepseek/deepseek-chat",
            "mistralai/mistral-small-latest",
            "qwen/qwen-2.5-7b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "anthropic/claude-3-haiku"
        ])
        idx = self.model_box.findText(self.llm_model)
        if idx >= 0: self.model_box.setCurrentIndex(idx)
        self.model_box.currentTextChanged.connect(self.on_model_change)

        self.send_btn = QPushButton("Ответить"); self.send_btn.clicked.connect(self.on_send)

        self.partial_label = QLabel(""); self.partial_label.setMaximumHeight(18)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.input, 1)
        top_bar.addWidget(self.mic_btn)
        top_bar.addWidget(self.speak_cb)
        top_bar.addWidget(self.fast_cb)
        top_bar.addWidget(self.model_box)
        top_bar.addWidget(self.send_btn)

        left = QVBoxLayout()
        left.addLayout(top_bar)
        left.addWidget(self.progress)
        left.addWidget(self.partial_label)
        left.addWidget(self.chat)

        left_wrap = QWidget(); left_wrap.setLayout(left)

        if QT3D_OK:
            try:
                self.avatar = Avatar3DWidget()
            except Exception:
                self.avatar = Avatar2DWidget()
        else:
            self.avatar = Avatar2DWidget()

        splitter = QSplitter()
        splitter.addWidget(left_wrap)
        splitter.addWidget(self.avatar)
        splitter.setSizes([820, 360])
        self.setCentralWidget(splitter)

        m = self.menuBar().addMenu("Справка")
        act = QAction("О программе", self); act.triggered.connect(self.show_about); m.addAction(act)

        raw_model_dir = ""
        if self._has_vosk_model(VOSK_MODEL_DIR_PRIMARY): raw_model_dir = VOSK_MODEL_DIR_PRIMARY
        elif self._has_vosk_model(VOSK_MODEL_DIR_ALT):   raw_model_dir = VOSK_MODEL_DIR_ALT

        if raw_model_dir:
            prepared = prepare_vosk_path(raw_model_dir)
            if prepared and os.path.isfile(os.path.join(prepared, "conf", "model.conf")):
                self.vosk_model_dir = prepared; self.vosk_ready = True
                text = "✅ Модель распознавания речи найдена: " if prepared==raw_model_dir else "✅ Модель найдена и скопирована в безопасный путь: "
                self._append_info(text + self._shorten(prepared))
            else:
                self.vosk_model_dir = ""; self.vosk_ready = False
                self._append_info("⚠️ Модель нашёл, но не смог подготовить (проверь conf/model.conf).")
        else:
            self.vosk_model_dir = ""; self.vosk_ready = False
            self._append_info("⚠️ Положи Vosk в models\\vosk-ru или vosk-model-small-ru-0.22")

        if not OPENROUTER_KEY:
            self._append_info("⚠️ OPENROUTER_API_KEY не задан — LLM не ответит.")
        else:
            self._append_info(f"🧠 Модель LLM: {html_escape(self.llm_model)}")

    # ------- сервис
    def _has_vosk_model(self, path: str) -> bool:
        return os.path.isdir(path) and os.path.isfile(os.path.join(path, "conf", "model.conf"))
    def _shorten(self, path: str, max_len: int = 80) -> str:
        return path if len(path) <= max_len else "..." + path[-max_len:]
    def _progress_on(self): QTimer.singleShot(0, lambda: self.progress.setRange(0,0))
    def _progress_off(self): QTimer.singleShot(0, lambda: self.progress.setRange(0,1))
    def _append_line(self, who: str, text: str, color: str):
        self.chat.moveCursor(QTextCursor.End)
        self.chat.insertHtml(f'<p><b style="color:{color}">{html_escape(who)}:</b> {text if text.strip().startswith("<") else html_escape(text)}</p>')
        self.chat.moveCursor(QTextCursor.End)
    def _append_info(self, text: str): self._append_line("Система", text, "#8b949e")
    def _append_user(self, text: str): self._append_line("Вы", text, "#58a6ff")

    # ------- действия
    def show_about(self):
        QMessageBox.information(self, "О программе",
            "Стриминг LLM, быстрый поиск/погода, офлайн озвучка, STT и 3D-персонаж. "
            "«⚡ Быстрый режим» ограничивает многословие и ускоряет ответ."
        )

    def on_model_change(self, m: str):
        self.llm_model = m
        self._append_info(f"🧠 Модель LLM: {html_escape(self.llm_model)}")

    def toggle_listen(self):
        if self.listening: self.stop_listen()
        else: self.start_listen()

    def start_listen(self):
        if not getattr(self, "vosk_ready", False):
            self._append_info("⚠️ Голосовой ввод отключён: модель не найдена/не подготовлена."); return
        if self.rec_thread is not None and self.rec_thread.isRunning():
            self._append_info("🎙️ Уже слушаю."); return
        self.listening = True; self.mic_btn.setText("⏹")
        self.partial_label.setText("🎙️ Слушаю…")
        self.rec_thread = VoskRecognizerThread(self.vosk_model_dir)
        self.rec_thread.partial_update.connect(self.on_stt_partial)
        self.rec_thread.final_result.connect(self.on_stt_final)
        self.rec_thread.error.connect(lambda msg: (self._append_info("❗ STT: " + msg), self.stop_listen()))
        self.rec_thread.start()

    def stop_listen(self):
        if not self.listening: return
        self.listening = False; self.mic_btn.setText("🎤")
        self.partial_label.setText("")
        if self.rec_thread:
            try: self.rec_thread.stop(); self.rec_thread.wait(1500)
            except Exception: pass
            self.rec_thread = None
        self._append_info("⏹ Запись остановлена.")

    @QtCore.Slot(str)
    def on_stt_partial(self, text: str):
        if text:
            self.partial_label.setText("⏳ " + text)

    @QtCore.Slot(str)
    def on_stt_final(self, text: str):
        self.partial_label.setText("")
        if not text: return
        self.input.setText(text)
        self._append_user(f"(голос) {text}")
        self._handle_and_dispatch(text)

    def on_send(self):
        q = self.input.text().strip()
        if not q: return
        self._append_user(q)
        self._handle_and_dispatch(q)

    # ---- маршрутизация: время → погода → поиск/LLM
    def _handle_and_dispatch(self, text: str):
        low = text.lower()

        # 1) Время (включая «сколько сейчас времени в саранске»)
        tz = detect_time_tz(low)
        if tz:
            dt = get_time_for_tz(tz)
            self._post("Помощница", dt.strftime("Сейчас %H:%M, %d.%m.%Y"), speak=True)
            return

        # 2) Погода
        if "погод" in low or "температур" in low:
            coords = find_city_coords(low)
            if coords:
                lat, lon, name = coords
                ans = get_weather_now(lat, lon)
                if ans:
                    self._post("Помощница", f"Погода ({name}): {ans}", speak=True)
                    return

        # 3) Поиск / LLM
        if is_search_intent(low):
            self._do_search(text)
        else:
            self._do_llm_stream(text)

    # ---- поиск
    def _do_search(self, query: str):
        self._progress_on()
        low = query.lower()
        for head in ["найди","поиск","поищи","открой"]:
            if low.startswith(head): query = query[len(head):].strip(" :,-"); break
        self._append_info("🔎 Ищу в интернете…")

        def run():
            results = ddg_search_cached(query, max_results=8, timeout_sec=4.0)
            parts = []
            for title, url in results:
                if url: parts.append(f"• <a href='{html_escape(url)}'>{html_escape(title)}</a>")
                else:  parts.append(f"• {html_escape(title)}")
            html_ans = "<br>".join(parts) if parts else "Ничего не нашла, попробуй иначе сформулировать."
            self._post("Поиск", html_ans, speak=True, speak_override=f"Показала ссылки по запросу: {query}")
            self._progress_off()

        threading.Thread(target=run, daemon=True).start()

        def watch():
            self._append_info("⏱️ Поиск дольше обычного… сеть может тормозить.")
        QTimer.singleShot(6000, watch)

    # ---- LLM: стриминг
    def _do_llm_stream(self, text: str):
        self._progress_on()
        fast = self.fast_cb.isChecked()
        cfg = LLMConfig(model=self.llm_model, temperature=0.2 if fast else 0.33,
                        max_tokens=192 if fast else 512, stream=True)
        self.messages.append({"role": "user", "content": text})

        self.chat.moveCursor(QTextCursor.End)
        self.chat.insertHtml(f'<p><b style="color:#3fb950">Помощница:</b> ')
        self.chat.moveCursor(QTextCursor.End)

        def run():
            acc = []
            for chunk in openrouter_stream(self.messages, cfg):
                chunk = strip_openrouter_think(chunk)
                if not chunk: continue
                acc.append(chunk)
                def append_chunk(c=chunk):
                    self.chat.insertPlainText(c)
                    self.chat.moveCursor(QTextCursor.End)
                QTimer.singleShot(0, append_chunk)

            def close_p():
                self.chat.insertHtml("</p>")
                self.chat.moveCursor(QTextCursor.End)
            QTimer.singleShot(0, close_p)

            final_text = sanitize_tts("".join(acc).strip() or "…")
            self.messages.append({"role": "assistant", "content": final_text})
            if self.speak_cb.isChecked():
                self.speaker.say(final_text)
            self._progress_off()

        threading.Thread(target=run, daemon=True).start()

    # ---- пост без стриминга
    def _post(self, who: str, text: str, speak: bool = False, speak_override: Optional[str] = None):
        safe_text = text if text.strip().startswith("<") else html_escape(text)
        def gui():
            self.chat.moveCursor(QTextCursor.End)
            self.chat.insertHtml(f'<p><b style="color:#3fb950">{html_escape(who)}:</b> {safe_text}</p>')
            self.chat.moveCursor(QTextCursor.End)
            if speak and self.speak_cb.isChecked():
                self.speaker.say(sanitize_tts(speak_override or text))
        QTimer.singleShot(0, gui)

    # ---- lifecycle
    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            if self.rec_thread: self.rec_thread.stop(); self.rec_thread.wait(1500)
        except Exception: pass
        try: self.speaker.shutdown()
        except Exception: pass
        super().closeEvent(e)

# ---------- main ----------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    win = ChatWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
