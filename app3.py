import streamlit as st
import os

from github import Github
import tempfile
import shutil
from typing import List, Dict

from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# Environment variables are now handled by Streamlit secrets


# ==================== HELPER FUNCTIONS ====================

def load_and_index_repo(repo_url: str, branch: str, file_extensions: List[str]) -> bool:
    """Load repository and create vector store"""
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Define file filter
        def file_filter(file_path: str) -> bool:
            return any(file_path.endswith(ext) for ext in file_extensions)
        
        # Load repository
        loader = GitLoader(
            clone_url=repo_url,
            repo_path=temp_dir,
            branch=branch,
            file_filter=file_filter
        )
        documents = loader.load()
        
        if not documents:
            st.error("No files found matching the selected extensions")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
        
        st.info(f"📚 Loaded {len(documents)} files from repository")
        
        # Determine primary language based on file extensions
        language_map = {
            ".py": Language.PYTHON,
            ".js": Language.JS,
            ".jsx": Language.JS,
            ".ts": Language.TS,
            ".tsx": Language.TS,
            ".java": Language.JAVA,
            ".cpp": Language.CPP,
            ".c": Language.CPP,
            ".go": Language.GO,
            ".rb": Language.RUBY
        }
        
        language = Language.PYTHON  # Default
        for ext in file_extensions:
            if ext in language_map:
                language = language_map[ext]
                break
        
        # Split documents using language-aware splitter
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        st.info(f"✂️ Split into {len(splits)} chunks")
        
        # Create FREE local embeddings (no API quota limits!)
        try:
            st.info("🔄 Loading HuggingFace embeddings model (first time may take a minute)...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            st.success("✅ Using free local embeddings (unlimited usage)")
        except Exception as e:
            st.error(f"Error initializing embeddings: {str(e)}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
        
        # Create vector store
        st.info("🔄 Creating vector database...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create LLM
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0,
                convert_system_message_to_human=True
            )
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
        
        # Create prompt template with chat history support
        system_prompt = """You are an AI assistant that helps developers understand code.
Use the following pieces of context from the codebase to answer the question.
If you don't know the answer, just say that you don't know, don't make up an answer.
Provide code examples when relevant and be specific about file locations.

Context from codebase:
{context}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Store in session state
        st.session_state.vectorstore = vectorstore
        st.session_state.retrieval_chain = retrieval_chain
        st.session_state.chat_memory = ChatMessageHistory()
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        st.error(f"Error loading repository: {str(e)}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return False


def get_response(question: str) -> Dict:
    """Get response from retrieval chain with chat history"""
    if st.session_state.retrieval_chain is None:
        return {"answer": "Please load a repository first", "context": []}
    
    try:
        # Get chat history
        chat_history = st.session_state.chat_memory.messages
        
        # Invoke chain
        response = st.session_state.retrieval_chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        
        # Update chat history
        st.session_state.chat_memory.add_user_message(question)
        st.session_state.chat_memory.add_ai_message(response["answer"])
        
        return response
    except Exception as e:
        return {"answer": f"Error generating response: {str(e)}", "context": []}


def analyze_github_issue(repo_name: str, issue_number: int) -> Dict:
    """Fetch and analyze a specific GitHub issue"""
    try:
        g = Github(st.session_state.github_token)
        repo = g.get_repo(repo_name)
        issue = repo.get_issue(issue_number)
        
        # Get issue details
        issue_data = {
            "title": issue.title,
            "number": issue.number,
            "state": issue.state,
            "created_at": issue.created_at,
            "updated_at": issue.updated_at,
            "body": issue.body or "No description provided",
            "labels": [label.name for label in issue.labels],
            "comments_count": issue.comments,
            "url": issue.html_url,
            "user": issue.user.login
        }
        
        # Get comments
        comments = []
        for comment in issue.get_comments()[:5]:
            comments.append({
                "user": comment.user.login,
                "body": comment.body,
                "created_at": comment.created_at
            })
        issue_data["comments"] = comments
        
        # Use LLM to analyze the issue
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                convert_system_message_to_human=True
            )
        except Exception as e:
            st.error(f"Error initializing analysis LLM: {str(e)}")
            return None
        
        analysis_prompt = f"""Analyze this GitHub issue and provide:
1. A brief summary of the problem
2. Possible root causes
3. Suggested solutions or next steps
4. Priority level (High/Medium/Low)

Issue Title: {issue_data['title']}
Issue Body: {issue_data['body'][:1000]}
Labels: {', '.join(issue_data['labels']) if issue_data['labels'] else 'None'}
Comments: {len(comments)} comments
State: {issue_data['state']}
"""
        
        analysis = llm.invoke([HumanMessage(content=analysis_prompt)]).content
        issue_data["ai_analysis"] = analysis
        
        return issue_data
        
    except Exception as e:
        st.error(f"Error analyzing issue: {str(e)}")
        return None


def search_github_issues(repo_name: str, keywords: str, labels: str, state: str, max_results: int) -> List[Dict]:
    """Search GitHub issues based on criteria"""
    try:
        g = Github(st.session_state.github_token)
        repo = g.get_repo(repo_name)
        
        issues_list = []
        label_list = [l.strip() for l in labels.split(",")] if labels else []
        
        for issue in repo.get_issues(state=state, labels=label_list):
            if len(issues_list) >= max_results:
                break
                
            # Filter by keywords if provided
            if keywords:
                keyword_list = keywords.lower().split()
                issue_text = f"{issue.title} {issue.body or ''}".lower()
                if not any(kw in issue_text for kw in keyword_list):
                    continue
            
            issues_list.append({
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "labels": [label.name for label in issue.labels],
                "created_at": issue.created_at,
                "comments": issue.comments,
                "url": issue.html_url
            })
        
        return issues_list
        
    except Exception as e:
        st.error(f"Error searching issues: {str(e)}")
        return []


def display_issue_analysis(issue_data: Dict):
    """Display analyzed issue in a formatted way"""
    st.markdown(f"### Issue #{issue_data['number']}: {issue_data['title']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        state_color = "🟢" if issue_data['state'] == "open" else "🔴"
        st.metric("State", f"{state_color} {issue_data['state'].upper()}")
    with col2:
        st.metric("Comments", issue_data['comments_count'])
    with col3:
        st.metric("Labels", len(issue_data['labels']))
    
    st.markdown(f"**Created by:** {issue_data['user']} | **Created:** {issue_data['created_at'].strftime('%Y-%m-%d')}")
    st.markdown(f"**URL:** [{issue_data['url']}]({issue_data['url']})")
    
    if issue_data['labels']:
        st.markdown("**Labels:** " + ", ".join([f"`{label}`" for label in issue_data['labels']]))
    
    st.markdown("#### Issue Description")
    with st.container():
        st.write(issue_data['body'][:500] + "..." if len(issue_data['body']) > 500 else issue_data['body'])
    
    st.markdown("#### 🤖 AI Analysis")
    st.info(issue_data['ai_analysis'])
    
    if issue_data['comments']:
        with st.expander(f"💬 View Comments ({len(issue_data['comments'])})"):
            for i, comment in enumerate(issue_data['comments'], 1):
                st.markdown(f"**Comment {i}** by {comment['user']}:")
                st.write(comment['body'][:300] + "..." if len(comment['body']) > 300 else comment['body'])
                st.divider()


def display_search_results(issues: List[Dict]):
    """Display search results"""
    st.markdown(f"### Found {len(issues)} issues")
    
    if not issues:
        st.warning("No issues found matching your criteria")
        return
    
    for issue in issues:
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"**[#{issue['number']}]({issue['url']})** {issue['title']}")
                if issue['labels']:
                    labels_html = " ".join([f'<span style="background-color:#d4d4d4;padding:2px 8px;border-radius:10px;font-size:12px;">{label}</span>' for label in issue['labels']])
                    st.markdown(labels_html, unsafe_allow_html=True)
            
            with col2:
                state_emoji = "🟢" if issue['state'] == "open" else "🔴"
                st.markdown(f"**State:** {state_emoji} {issue['state']}")
                st.markdown(f"💬 {issue['comments']}")
            
            st.caption(f"Created: {issue['created_at'].strftime('%Y-%m-%d %H:%M')}")
            st.divider()


# ==================== STREAMLIT UI ====================

# Page configuration
st.set_page_config(
    page_title="AI Code Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = ChatMessageHistory()
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'github_token' not in st.session_state:
    st.session_state.github_token = st.secrets.get("GITHUB_TOKEN", "")
if 'google_key' not in st.session_state:
    st.session_state.google_key = st.secrets.get("GOOGLE_API_KEY", "")

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Keys
    st.subheader("API Keys")
    google_key = st.text_input("Google API Key", value=st.session_state.google_key, type="password")
    github_token = st.text_input("GitHub Token (Optional)", value=st.session_state.github_token, type="password")
    
    if google_key:
        st.session_state.google_key = google_key
        os.environ["GOOGLE_API_KEY"] = google_key
    
    if github_token:
        st.session_state.github_token = github_token
    
    st.divider()
    
    # Repository input
    st.subheader("Repository Settings")
    repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/username/repo")
    branch = st.text_input("Branch", value="main")
    
    # File filter options
    file_extensions = st.multiselect(
        "File Types to Index",
        [".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".c", ".go", ".rb", ".md"],
        default=[".py"]
    )
    
    if st.button("🔄 Load Repository", use_container_width=True):
        if not google_key:
            st.error("Please provide Google API Key")
        elif not repo_url:
            st.error("Please provide a repository URL")
        else:
            with st.spinner("Loading repository and creating vector store..."):
                success = load_and_index_repo(repo_url, branch, file_extensions)
                if success:
                    st.success("✅ Repository indexed successfully!")
                else:
                    st.error("❌ Failed to load repository")
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.chat_memory = ChatMessageHistory()
        st.success("Chat history cleared!")

# Main title
st.title("🤖 AI-Powered Code Documentation Assistant")
st.markdown("**Chat with your codebase, analyze GitHub issues, and find solutions faster**")

# Tabs for different features
tab1, tab2, tab3 = st.tabs(["💬 Chat with Code", "🐛 Analyze Issues", "🔍 Find Issues"])

# Tab 1: Chat with Codebase
with tab1:
    st.subheader("Ask questions about your codebase")
    
    if st.session_state.vectorstore is None:
        st.info("👈 Please load a repository from the sidebar to start chatting")
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        if user_question := st.chat_input("Ask about the code..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_response(user_question)
                    st.markdown(response['answer'])
                    
                    # Show source documents
                    if response.get('context'):
                        with st.expander("📄 Source Code References"):
                            for i, doc in enumerate(response['context'][:3], 1):
                                st.markdown(f"**Reference {i}:**")
                                st.code(doc.page_content[:500], language="python")
                                st.caption(f"📁 {doc.metadata.get('source', 'Unknown')}")
                                if i < len(response['context'][:3]):
                                    st.divider()
            
            st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})

# Tab 2: Analyze GitHub Issues
with tab2:
    st.subheader("Analyze and Explain GitHub Issues")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        issue_repo = st.text_input("Repository (format: owner/repo)", placeholder="langchain-ai/langchain")
    with col2:
        issue_number = st.number_input("Issue Number", min_value=1, value=1)
    
    if st.button("🔍 Analyze Issue", use_container_width=True):
        if not st.session_state.github_token:
            st.error("Please provide GitHub token in sidebar")
        elif not issue_repo:
            st.error("Please provide repository name")
        else:
            with st.spinner("Fetching and analyzing issue..."):
                issue_analysis = analyze_github_issue(issue_repo, issue_number)
                if issue_analysis:
                    display_issue_analysis(issue_analysis)

# Tab 3: Find Issues
with tab3:
    st.subheader("Search Issues by Keywords or Labels")
    
    search_repo = st.text_input("Repository (format: owner/repo)", placeholder="facebook/react", key="search_repo")
    
    col1, col2 = st.columns(2)
    with col1:
        search_keywords = st.text_input("Search Keywords", placeholder="bug crash error")
    with col2:
        state_filter = st.selectbox("Issue State", ["open", "closed", "all"])
    
    labels_input = st.text_input("Labels (comma-separated)", placeholder="bug, high-priority")
    max_results = st.slider("Max Results", 5, 50, 10)
    
    if st.button("🔎 Search Issues", use_container_width=True):
        if not st.session_state.github_token:
            st.error("Please provide GitHub token in sidebar")
        elif not search_repo:
            st.error("Please provide repository name")
        else:
            with st.spinner("Searching issues..."):
                issues = search_github_issues(search_repo, search_keywords, labels_input, state_filter, max_results)
                if issues:
                    display_search_results(issues)

# Footer
st.sidebar.divider()
st.sidebar.markdown("### 📚 How to Use")
st.sidebar.markdown("""
1. Add your Google API key
2. Enter GitHub repository URL
3. Select file types to index
4. Click 'Load Repository'
5. Start chatting or analyzing issues!
""")

st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Uses **free local embeddings** (no quota limits!)
- Ask specific questions for better answers
- Check source references for context
- Analyze issues to understand problems faster
""")

st.sidebar.markdown("### ⚡ Features")
st.sidebar.markdown("""
- ✅ Unlimited embedding usage (local model)
- ✅ Conversation memory
- ✅ Code-aware text splitting
- ✅ GitHub issue analysis
""")
