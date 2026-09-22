


import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap, QMovie, QFont
from PyQt5.QtCore import Qt, QTimer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# === Load your trained model ===
# Adjust path and input dimensions accordingly
MODEL_PATH = "FER_model.h5"  # <- change this to your model's filename
IMG_HEIGHT = 48   # replace with your model's input height
IMG_WIDTH = 48    # replace with your model's input width

model = load_model(MODEL_PATH)

# Provide class labels (in the same order as during training)
class_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# ^ adjust these according to your training dataset


class FERApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Facial Expression Recognition")
        self.setFixedSize(500, 600)

        # === Main container ===
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setAlignment(Qt.AlignCenter)

        # === Background animation ===
        self.bg_label = QLabel(self)
        self.bg_movie = QMovie("background.gif")  # your background gif
        self.bg_label.setMovie(self.bg_movie)
        self.bg_label.setScaledContents(True)
        self.bg_label.setGeometry(0, 0, 500, 600)
        self.bg_movie.start()

        # === Foreground container ===
        self.fg_widget = QWidget(self)
        self.fg_widget.setGeometry(0, 0, 500, 600)
        self.fg_layout = QVBoxLayout(self.fg_widget)
        self.fg_layout.setAlignment(Qt.AlignCenter)
        self.fg_widget.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.7);
        """)

        # === Heading ===
        self.heading = QLabel("Facial Expression Recognition")
        self.heading.setAlignment(Qt.AlignCenter)
        self.heading.setFont(QFont("Arial", 22, QFont.Bold))
        self.heading.setStyleSheet("color: #00FFCC; margin-bottom: 20px;")
        self.fg_layout.addWidget(self.heading)

        # === Image display ===
        self.imageLabel = QLabel("No image")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setFixedSize(450, 300)
        self.imageLabel.setStyleSheet("""
            border: 3px dashed #00FFCC;
            background: rgba(255, 255, 255, 0.05);
            color: #ddd;
        """)
        self.fg_layout.addWidget(self.imageLabel)

        # === Browse Button ===
        self.browseButton = QPushButton("📂 Browse Image")
        self.browseButton.clicked.connect(self.browse_image)
        self.browseButton.setStyleSheet(self.button_style())
        self.fg_layout.addWidget(self.browseButton)

        # === Predict Button ===
        self.predictButton = QPushButton("🤖 Predict Expression")
        self.predictButton.clicked.connect(self.predict_expression)
        self.predictButton.setStyleSheet(self.button_style())
        self.fg_layout.addWidget(self.predictButton)

        # === Result Label ===
        self.resultLabel = QLabel("Result will appear here.")
        self.resultLabel.setAlignment(Qt.AlignCenter)
        self.resultLabel.setFont(QFont("Arial", 14))
        self.resultLabel.setStyleSheet("color: #EEE; margin-top: 10px;")
        self.fg_layout.addWidget(self.resultLabel)

        # === Loading spinner ===
        self.loadingLabel = QLabel()
        self.loadingLabel.setAlignment(Qt.AlignCenter)
        self.movie = QMovie("loading.gif")  # your spinner gif
        self.loadingLabel.setMovie(self.movie)
        self.loadingLabel.setVisible(False)
        self.fg_layout.addWidget(self.loadingLabel)

        self.image_path = None

    def button_style(self):
        return """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #00FFC6, stop:1 #005454);
                color: #000;
                font-weight: bold;
                font-size: 16px;
                padding: 12px 24px;
                border: 2px solid #00FFC6;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #00FFF7, stop:1 #007070);
                border: 2px solid #00FFF7;
            }
        """

    def browse_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            pixmap = QPixmap(file_name)
            self.imageLabel.setPixmap(pixmap.scaled(
                self.imageLabel.width(), self.imageLabel.height(), Qt.KeepAspectRatio))
            self.image_path = file_name
            self.resultLabel.setText("✅ Image loaded.")

    def predict_expression(self):
        if not self.image_path:
            self.resultLabel.setText("❌ Please upload an image first.")
            return

        # Show loading spinner
        self.loadingLabel.setVisible(True)
        self.movie.start()
        self.resultLabel.setText("⏳ Predicting...")

        # Predict after a short delay to show spinner
        QTimer.singleShot(1000, self.run_prediction)

    def run_prediction(self):
        try:
            # === Preprocess image ===
            img = image.load_img(self.image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # === Predict ===
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            result = f"😄 Predicted Expression: {class_labels[predicted_class]}"

        except Exception as e:
            result = f"⚠️ Error: {str(e)}"

        self.movie.stop()
        self.loadingLabel.setVisible(False)
        self.resultLabel.setText(result)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FERApp()
    window.show()
    sys.exit(app.exec_())
