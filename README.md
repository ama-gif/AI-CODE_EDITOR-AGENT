# AI Code Editor with GitHub Integration
An AI-powered code documentation assistant that analyzes GitHub repositories and provides intelligent code analysis and chat capabilities.

## Features
- **Repository Analysis**: Load and analyze GitHub repositories
- **AI Chat**: Interactive chat with AI about your codebase
- **GitHub Issue Analysis**: Analyze and get insights on GitHub issues
- **Code Documentation**: Generate documentation for your code
- **Multi-language Support**: Supports various programming languages

## Setup
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   GITHUB_TOKEN=your_github_token
   ```
4. Run the app: `streamlit run app.py`

## Deployment
This app is configured for deployment on Streamlit Cloud.

## Technologies Used
- Streamlit
- LangChain
- Google Gemini AI
- HuggingFace Embeddings
- ChromaDB
- GitHub API