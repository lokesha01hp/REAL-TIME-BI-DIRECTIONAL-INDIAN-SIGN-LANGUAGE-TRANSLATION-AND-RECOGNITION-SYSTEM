# Text / Audio to Indian Sign Language (ISL) Video Conversion

This module converts **text or spoken audio input into animated Indian Sign Language (ISL) videos**.  
It is designed to improve accessibility by allowing non-signers to communicate effectively with the deaf and hard-of-hearing community using visual sign language representations.

---

## ✨ Features

- 🔤 Text to ISL video translation
- 🎤 Audio input with speech-to-text support
- 🧠 Intelligent word-level and character-level gesture mapping
- 🎬 Automatic ISL video concatenation
- ⏩ Adjustable playback speed
- ⬇️ Downloadable ISL output video
- 🌐 Web-based interface using Flask

---

## 🛠️ Technologies Used

- **Python 3.10**
- **Flask** – backend web framework
- **OpenCV** – video processing
- **PyAV** – MP4 video encoding (H.264)
- **SpeechRecognition (Web API)** – audio to text
- **HTML, CSS, JavaScript** – frontend interface

---

## 📁 Directory Structure

Text_Audio_to_ISL/
├── app.py # Main Flask application
├── convertor.py # Human video → 2D avatar conversion
├── isl_index.json # ISL gesture index (words, alphabets, numbers)
├── static/
│ ├── outputs/ # Generated ISL videos
│ └── css/
│ └── style.css
├── templates/
│ ├── cover.html # Home page
│ ├── text_to_isl.html # Text/Audio to ISL interface
│ └── about.html
└── Animated_ISl_Videos/ # Pre-generated animated ISL gesture videos

---

## ⚙️ How It Works

1. **Text or Audio Input**

   - Users can either type text or speak using a microphone.
   - Audio input is converted into text using speech recognition.

2. **Text Processing**

   - Input text is tokenized and normalized.
   - The system first attempts **word-level ISL mapping**.
   - If a word is unavailable, it falls back to **alphabet-level gesture mapping**.

3. **Gesture Mapping**

   - Each recognized token is mapped to a corresponding ISL animation video using `isl_index.json`.

4. **Video Generation**

   - Mapped ISL videos are resized and concatenated in sequence.
   - Playback speed is adjusted based on user preference.
   - Final output is encoded as an MP4 (H.264) video.

5. **Output**
   - The generated ISL animation is displayed in the browser.
   - Users can replay or download the video.

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

````bash
pip install flask opencv-python av SpeechRecognition

2️⃣ Run the Application
```bash
python app.py

3️⃣ Open in Browser
```bash
http://localhost:5000
````
