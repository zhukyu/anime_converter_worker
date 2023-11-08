
# Anime Converter Worker

This guide will walk you through the setup process for running the Flask application contained in this repository.

## Prerequisites

Before you begin, make sure you have Python installed on your system. You can download Python from [python.org](https://www.python.org/downloads/).

## Setup

Follow these steps to set up the Flask app:

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```
git clone https://github.com/zhukyu/anime_converter_worker.git
cd anime_converter_worker
```

### 2. Create a Virtual Environment

Create a virtual environment in the project directory:

```
python -m venv ./venv
```

This will create a new `venv` directory within your project folder.

### 3. Activate the Virtual Environment

Activate the virtual environment using the command below:

#### For Windows:

```
venv\Scripts\activate
```

#### For macOS and Linux:

```
source venv/bin/activate
```

### 4. Install Dependencies

Install the required dependencies with pip:

```
pip install -r requirements.txt
```

### 5. Run the Flask Application

Now you can run the Flask application using:

```
flask --app ./Converter.py run
```

This will start the Flask server, and you should be able to access the application on your browser at `http://127.0.0.1:5000/`.

## Additional Information

- Ensure that you are always within the activated virtual environment while running the app or installing additional dependencies.
- To deactivate the virtual environment, you can use the `deactivate` command.

Thank you for setting up the Flask application!
