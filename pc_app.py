import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import QThread, pyqtSignal
from faster_whisper import WhisperModel

class TranscribeThread(QThread):
    """Фоновый поток, чтобы интерфейс программы не зависал во время работы ИИ"""
    finished = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path

    def run(self):
        try:
            self.status.emit("🤖 Загрузка локальной модели ИИ...")
            # Инициализируем быструю модель tiny на CPU в режиме оптимизации int8
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            
            self.status.emit("⏳ Расшифровка аудио...")
            segments, info = model.transcribe(self.audio_path, beam_size=5, language="ru")
            
            text_segments = [segment.text for segment in segments]
            full_text = " ".join(text_segments).strip()
            
            if not full_text:
                full_text = "🎙 Речь не обнаружена или файл пуст."
                
            self.finished.emit(full_text)
        except Exception as e:
            self.finished.emit(f"❌ Ошибка: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SilentSTT — Бесплатный ИИ Транскрибатор")
        self.setMinimumSize(500, 400)

        # Элементы интерфейса
        self.label = QLabel("Загрузите аудиофайл для перевода в текст", self)
        self.text_edit = QTextEdit(self)
        self.text_edit.setPlaceholderText("Здесь появится ваш текст...")
        
        self.btn_select = QPushButton("Выбрать файл и распознать", self)
        self.btn_select.clicked.connect(self.start_transcription)

        # Макет окна
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.btn_select)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_transcription(self):
        # Для теста: программа ищет файл test.mp3 в своей папке
        test_file = "test.mp3"
        if not os.path.exists(test_file):
            self.label.setText("❌ Положите файл 'test.mp3' в папку с программой для теста!")
            return

        self.btn_select.setEnabled(False)
        self.text_edit.clear()
        
        # Запуск ИИ в отдельном потоке
        self.thread = TranscribeThread(test_file)
        self.thread.status.connect(lambda text: self.label.setText(text))
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def on_finished(self, text):
        self.text_edit.setPlainText(text)
        self.label.setText("✅ Успешно завершено!")
        self.btn_select.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
