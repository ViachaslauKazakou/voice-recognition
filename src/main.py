import sys
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSizePolicy,
    QScrollArea   
)
from manager import SoundManager, AiManager
from pygments import highlight
from pygments.lexers.python import PythonLexer
from pygments.formatters import HtmlFormatter


class AudioPopup(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Control Popup")
        self.setFixedSize(600, 600)  # Fixed size for the popup
        self.sound_manager = SoundManager()
        self.ai_manager = AiManager()
        self.result = None
        # Main layout (vertical)
        main_layout = QVBoxLayout()

        # Label for instructions or status
        self.status_label = QLabel("Press a button to start.")
        main_layout.addWidget(self.status_label)

        # Nested button layout (vertical)
        button_layout = QVBoxLayout()

        # Buttons
        self.start_button = QPushButton("Start Record")
        self.recognize_button = QPushButton("Recognize")
        self.explain_button = QPushButton("Explain")

        # Add buttons to the layout
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.recognize_button)
        button_layout.addWidget(self.explain_button)
        
        # Add answer layout to the main layout
        self.question_layout = QHBoxLayout()
        self.question_label = QLabel("Question: ")
        self.question_label.setWordWrap(True)  # Enable word wrapping
        self.question_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)  # Allow the label to expand horizontally
        self.question_layout.addWidget(self.question_label)

        # Nest the button layout into the main layout
        main_layout.addLayout(button_layout)
        # Add the answer label to the main layout
        main_layout.addWidget(self.question_label)
        
        self.answer_layout = QHBoxLayout()
        # Add answer label
        self.answer_label = QLabel("Answer: ")
        self.answer_label.setWordWrap(True)
        self.answer_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)  # Allow the label to expand horizontally
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
        self.start_button.clicked.connect(self.toggle_recording)
        self.recognize_button.clicked.connect(self.recognize_audio)
        self.explain_button.clicked.connect(self.explain_function)

        # State to track recording
        self.is_recording = False

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.status_label.setText("🎤 Recording started...")
        self.start_button.setText("✅ Stop Record")
        self.is_recording = True
        self.sound_manager.record()
        self.status_label.setText("✅ Recording ends.")
        self.recognize_audio()
        # Add your recording logic here (e.g., self.sound_manager.record())

    def stop_recording(self):
        self.status_label.setText("✅ Recording stopped.")
        self.start_button.setText("✅ Start Record")
        self.is_recording = False
        # Add your stop recording logic here (e.g., self.sound_manager.stop())

    def recognize_audio(self):
        self.status_label.setText("✅ Recognizing audio...")
        self.result = self.sound_manager.recognition()
        self.question_label.setText(f"Question: {self.result}")
        self.status_label.setText("✅ Recognition complete.")
        # Add your recognition logic here

    def explain_function(self):
        self.status_label.setText("✅ Start explain....")
        print("✅  Explaining function...")
        answer = self.ai_manager.ask_ollama(self.result)
        # Highlight Python syntax
        # formatter = HtmlFormatter(style="colorful", full=False, noclasses=True)
        # highlighted_code = highlight(answer, PythonLexer(), formatter)

        self.answer_label.setText(answer)
        self.status_label.setText("✅ Explain complete.")
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    popup = AudioPopup()
    popup.show()  # Show the popup as a modal dialog
    sys.exit(app.exec())
