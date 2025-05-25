import multiprocessing

import sys
import time
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSizePolicy,
    QScrollArea,
    QCheckBox,
)
from PyQt6.QtCore import QThreadPool, QRunnable, pyqtSlot, QMetaObject, Qt
from src.manager import SoundManager
from src.ai_manager import AiManager
from pygments import highlight
from pygments.lexers.python import PythonLexer
from pygments.formatters import HtmlFormatter
import logging

multiprocessing.set_start_method("spawn", force=True)

logger = logging.getLogger("AudioPopup")
logger.setLevel(logging.DEBUG)

# Add this block to enable console logging
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] - %(levelname)s [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class RecordWorker(QRunnable):
    def __init__(self, sound_manager, callback):
        super().__init__()
        self.sound_manager = sound_manager
        self.callback = callback

    @pyqtSlot()
    def run(self):
        self.sound_manager.record()
        if self.callback:
            self.callback()


class RecognitionWorker(QRunnable):
    def __init__(self, sound_manager, callback):
        super().__init__()
        self.sound_manager = sound_manager
        self.callback = callback

    @pyqtSlot()
    def run(self):
        result = self.sound_manager.recognition()
        if self.callback:
            self.callback(result)


class AudioPopup(QDialog):
    """
    A popup dialog for audio control and interaction with AI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Control Popup")
        self.setFixedSize(600, 700)  # Fixed size for the popup
        self.sound_manager = SoundManager()
        self.ai_manager = AiManager()
        self.result = None
        self.threadpool = QThreadPool()
        # Main layout (vertical)
        main_layout = QVBoxLayout()

        # Label for instructions or status
        self.status_label = QLabel("Press a button to start.")
        main_layout.addWidget(self.status_label)

        # Nested button layout (vertical)
        self.button_layout = QVBoxLayout()

        # Buttons
        self.start_button = QPushButton("Start Voice Recording")
        # Create a horizontal layout for Recognize button and Auto checkbox

        self.recognize_layout = QHBoxLayout()
        self.recognize_button = QPushButton("Recognize")
        self.auto_recognize = QCheckBox("Auto")
        self.auto_recognize.setChecked(True)
        self.recognize_layout.addWidget(self.recognize_button)
        self.recognize_layout.addWidget(self.auto_recognize)

        self.explain_layout = QHBoxLayout()
        self.explain_button = QPushButton("Explain")
        self.auto_explain = QCheckBox("Auto")
        self.auto_explain.setChecked(True)
        self.use_chain = QCheckBox("Use Chain")
        self.explain_layout.addWidget(self.explain_button)
        self.explain_layout.addWidget(self.auto_explain)
        self.explain_layout.addWidget(self.use_chain)

        # Add buttons to the layout
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addLayout(self.recognize_layout)  # Add the horizontal layout
        self.button_layout.addLayout(self.explain_layout)

        # Add question layout to the main layout
        self.question_layout = QVBoxLayout()
        self.question_label = QLabel("Question: ")
        self.execution_label = QLabel("Recognition time: ")
        self.answer_time_label = QLabel("Explain time: ")
        self.question_label.setWordWrap(True)  # Enable word wrapping
        self.question_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )  # Allow the label to expand horizontally
        self.question_layout.addWidget(self.question_label)
        self.question_layout.addWidget(self.execution_label)
        self.question_layout.addWidget(self.answer_time_label)

        # Nest the button layout into the main layout
        main_layout.addLayout(self.button_layout)
        # Add the answer label to the main layout
        main_layout.addLayout(self.question_layout)
        # main_layout.addWidget(self.question_label)
        # main_layout.addWidget(self.execution_label)

        self.answer_layout = QHBoxLayout()
        # Add answer label
        self.answer_label = QLabel("Answer: ")
        self.answer_label.setWordWrap(True)
        self.answer_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )  # Allow the label to expand horizontally
        self.answer_layout.addWidget(self.answer_label)
        # main_layout.addWidget(self.answer_label)

        # Create a scroll area for the answer label
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow resizing
        scroll_area.setWidget(self.answer_label)  # Set the answer label as the scrollable widget
        main_layout.addWidget(scroll_area)

        # Set the main layout for the dialog

        self.setLayout(main_layout)

        # Connect buttons to functions
        self.start_button.clicked.connect(self.start_recording)
        self.recognize_button.clicked.connect(self.recognize_audio)
        self.explain_button.clicked.connect(self.explain_question)

        # State to track recording
        self.is_recording = False

    # audio recording part
    def start_recording(self):
        """
        Start recording audio and update the UI accordingly.
        """
        logger.info("🎤 Starting recording...")
        self.status_label.setText("🎤 Recording started...")
        self.start_button.setText("✅ Recording")
        QApplication.processEvents()

        self.is_recording = True
        print("🎤 Recording started...")

        worker = RecordWorker(self.sound_manager, self.update_ui_after_recording)
        self.threadpool.start(worker)

    @pyqtSlot()
    def update_ui_after_recording(self):
        # self.stop_recording
        if self.auto_recognize.isChecked():
            self.recognize_audio()
        if self.result and self.auto_explain.isChecked():
            self.explain_question()
        self.status_label.setText("✅ Recording ends. Click 'Recognize' to process.")
        self.start_button.setText("Start Voice Recording")
        logger.info("🎤 Recording finished. UI updated")

    # voice
    @pyqtSlot(object)
    def update_ui_after_recognition(self, result):
        try:
            self.result = result
            self.question_label.setText(f"Question: {self.result['result']}")
            self.execution_label.setText(f"Recognition time: {self.result['execution_time']:.2f} seconds")
            self.status_label.setText("✅ Recognition complete.")
            logger.info("✅ Recognition complete. UI updated")
            logger.debug("Recognition result: %s", self.result["result"])
            # Auto-explain if checkbox is checked
            if self.auto_explain.isChecked():
                self.explain_question()
        except Exception as e:
            logger.error(f"Error in update_ui_after_recognition: {e}")
            self.status_label.setText(f"Error during recognition: {str(e)}")

    # def on_recognition_finished(self):
    #     # This runs in the worker thread, so use signals or QMetaObject.invokeMethod to update UI safely
    #     QMetaObject.invokeMethod(self, "update_ui_after_recognition", Qt.ConnectionType.QueuedConnection)

    def recognize_audio(self):
        print("🎤 Recoginze started...")
        self.status_label.setText("✅ Recognizing audio...")
        # QApplication.processEvents()
        worker = RecognitionWorker(self.sound_manager, self.update_ui_after_recognition)
        self.threadpool.start(worker)

    def explain_question(self):
        self.status_label.setText("✅ Start explain....")
        print("✅  Explaining function...")
        if not self.result:
            self.status_label.setText("❌ No question to explain. Please record and recognize first.")
            return
        if self.use_chain.isChecked():
            answer = self.ai_manager.ask_ollama_memory(self.result["result"], translate=True)
        else:
            answer = self.ai_manager.ask_ollama(self.result["result"], translate=True)
        # Highlight Python syntax
        # formatter = HtmlFormatter(style="colorful", full=False, noclasses=True)
        # highlighted_code = highlight(answer, PythonLexer(), formatter)

        self.answer_label.setText(answer["result"])
        self.status_label.setText("✅ Explain complete.")
        self.answer_time_label.setText(f"Explaining took: {answer['execution_time']:.2f} seconds")

    def closeEvent(self, event):
        """Handle cleanup when window is closed"""
        print("Cleaning up resources...")
        self.threadpool.waitForDone(1000)  # Wait up to 1 second
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    popup = AudioPopup()
    popup.show()
    exit_code = app.exec()  # Get the exit code
    popup.threadpool.waitForDone(3000)  # Wait up to 3 seconds for threads to finish
    sys.exit(exit_code)  # Exit with the saved code
