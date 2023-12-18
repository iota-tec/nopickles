# Chatopotamus: Where Your Voice is the Special Ingredient

Chatopotamus is an innovative solution designed to streamline order management and customer interaction in restaurant chains. Utilizing cutting-edge technologies like Automatic Speech Recognition (ASR), Natural Language Processing (NLP), and facial recognition, it offers a seamless, human-like interaction experience for customers while automating order-taking and inventory management.

## Features
- **Voice-activated Ordering**: Processes customer orders through voice, using ASR and NLP technologies.
- **Facial Recognition**: Recognizes returning customers and personalizes the interaction.
- **Real-time Inventory Management**: Keeps track of inventory in real-time, updating availability based on orders.
- **Manager Interaction**: Allows managers to interact with the system to update inventory or communicate with customers.
- **Local and Cloud Data Storage**: Stores data locally and on the cloud ensuring robust data management and analytics.

## Getting Started

### Prerequisites
- Python 3.9
- MySQL 8.x 
- TensorFlow, Keras, DeepSpeech, Pyttsx, and other necessary libraries (see ``requirements.txt``).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ryuukkk/chatopotamus.git
   cd chatopotamus
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   **OR**
   create a conda environment using the following command before beginning:
   ```bash
   conda env create -f environment.yml
   ```

3. Set up your MySQL databases as per the instructions in `db_setup.md`.

### Usage
1. Run the main script to start the application:
   ```bash
   python src/main.py
   ```
2. Follow the on-screen instructions to interact with Chatopotamus.

### Development
Detailed documentation for development, including the setup of various modules (audio management, facial recognition, and NLP) can be found in the `docs` folder.
/src directory has the complete library.
main.ipynb is a demonstration of the implemented parts.
