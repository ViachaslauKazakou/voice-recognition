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
    QCheckBox
)
from src.manager import SoundManager, AiManager
from pygments import highlight
from pygments.lexers.python import PythonLexer
from pygments.formatters import HtmlFormatter


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
        self.recognize_layout.addWidget(self.recognize_button)
        self.recognize_layout.addWidget(self.auto_recognize)

        self.explain_layout = QHBoxLayout()
        self.explain_button = QPushButton("Explain")
        self.auto_explain = QCheckBox("Auto")
        self.explain_layout.addWidget(self.explain_button)
        self.explain_layout.addWidget(self.auto_explain)

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
        self.explain_button.clicked.connect(self.explain_function)

        # State to track recording
        self.is_recording = False

    def toggle_recording(self):
        if not self.is_recording:
            self.status_label.setText("🎤 Recording started...")
            time.sleep(1)  # Simulate a short delay for UI update
            self.start_recording()
            self.start_button.setText("✅ Stop Record")
            self.is_recording = True
        else:
            self.stop_recording()

    def start_recording(self):
        """
        Start recording audio and update the UI accordingly."""
        
        self.status_label.setText("🎤 Recording started...")
        self.start_button.setText("✅ Recording")
        QApplication.processEvents()

        self.is_recording = True
        print("🎤 Recording started...")
        self.sound_manager.record()
        self.status_label.setText("✅ Recording ends.")
        self.start_button.setText("✅ Start Voice Recording")
        if self.auto_recognize.isChecked():
            self.recognize_audio()
        if self.result and self.auto_explain.isChecked():
            self.explain_function()
        self.status_label.setText("✅ Recording ends.")
        # Add your recording logic here (e.g., self.sound_manager.record())

    def stop_recording(self):
        self.status_label.setText("✅ Recording stopped.")
        self.start_button.setText("✅ Start Record")
        self.is_recording = False
        # Add your stop recording logic here (e.g., self.sound_manager.stop())

    def recognize_audio(self):
        print("🎤 Recoginze started...")
        self.status_label.setText("✅ Recognizing audio...")
        QApplication.processEvents()
        self.result = self.sound_manager.recognition()
        self.question_label.setText(f"Question: {self.result['result']}")
        self.execution_label.setText(
            f"Recognition time: {self.result['execution_time']:.2f} seconds"
        )
        self.status_label.setText("✅ Recognition complete.")
        # Add your recognition logic here

    def explain_function(self):
        self.status_label.setText("✅ Start explain....")
        print("✅  Explaining function...")
        answer = self.ai_manager.ask_ollama(self.result['result'], translate=True)
        # Highlight Python syntax
        # formatter = HtmlFormatter(style="colorful", full=False, noclasses=True)
        # highlighted_code = highlight(answer, PythonLexer(), formatter)

        self.answer_label.setText(answer["result"])
        self.status_label.setText("✅ Explain complete.")
        self.answer_time_label.setText(
            f"Explaining took: {answer['execution_time']:.2f} seconds"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    popup = AudioPopup()
    popup.show()  # Show the popup as a modal dialog
    sys.exit(app.exec())
