# 🩻 Image Mask Backend API

This project is the backend API for the Plus Also application. It provides the necessary endpoints and functionality to support the application's features.

## 📦 Prerequisites

Before setting up the project, ensure you have the following installed on your system:

- **Python 3.8 or higher**: Required to run the backend application.
- **Git**: For cloning the repository and version control.
- **Google API Key**: API key for accessing Google AI services (e.g., Generative AI APIs). Instructions to getting Google Gemini API key can be found [here](https://image-mask-backend-delta.vercel.app). 

## 🚀 Setup Instructions

### 1. Create and activate a virtual environment:
  
  ```bash
  python3 -m venv name_of_virtual_env
  source name_of_virtual_env/bin/activate
  ```

  You deactivate the virtual environment when you're done with the command
  ```bash 
  deactivate
  ``` 
### 2. Clone the repository 

  ```bash
    git clone https://github.com/hkam0006/image_mask_backend.git
    cd image_mask_backend
  ```

### 3. Install dependencies

  Make sure the virtual environment is activated before installing modules.
  ```bash
  pip install -r requirements.txt
  ```

### 4. Configure Environment Variables

To run the project locally, you need to set up environment variables.

1. Duplicate the `.env.example` file and rename it to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Update the values in .env with your local configuration.

    The .env.example file includes all required variables to run this project locally.
    | Variable          | Description                                                                        |
    | ----------------- | ---------------------------------------------------------------------------------- |
    | `GOOGLE_API_KEY` | Your API key for accessing Google AI services (e.g., Generative AI APIs). |
    | `FRONTEND_URL`     | The base URL of your frontend (e.g., `http://localhost:3000`).              |
    | `AUTH_TOKEN`      | A secret token used for authentication between the frontend and backend. You can set this to any value, but make sure it matches the token configured in your frontend .env file as well.             |

    🔑 Important: Ensure the AUTH_TOKEN value is the same in both your frontend and backend .env files to avoid authentication errors.
    🔑 Important: See Prerequsites section above for information about obtaining Google Gemini API Key.

### 5. Start the development server:

  ```bash
  python3 main.py
  ```

## 🛠️ Tools and Technologies

- **Python**: The primary programming language used for the backend.
- **FastAPI**: A lightweight WSGI web application framework for building APIs.
- **dotenv**: A module for loading environment variables from a `.env` file.
- **Google AI APIs**: Used for integrating generative AI features into the application.
- **Vercel**: Hosting platform for deploying the live production site.

## 🌐 Live Production Site

The live production site is hosted at: [https://image-mask-backend-delta.vercel.app](https://image-mask-backend-delta.vercel.app)
