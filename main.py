import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QLabel,
    QRadioButton,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from dotenv import load_dotenv
from langchain.utils.math import cosine_similarity
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from prompt import (
    regulation_chat_qa_prompt_template,
    legal_chat_qa_prompt_template,
    greeting_chat_qa_prompt_template,
    internet_search_prompt_template,
)
from tool import rag_general_chat, rag_legal_chat, condense_question, search_internet
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import settings

# Load environment variables
load_dotenv()

# Initialize embeddings and templates
llm = ChatOpenAI(model_name=settings.LLM_MODEL_NAME, temperature=0.3, max_tokens=3000)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
prompt_templates = [
    regulation_chat_qa_prompt_template,
    legal_chat_qa_prompt_template,
    greeting_chat_qa_prompt_template,
    internet_search_prompt_template,
]
classifications = [
    "Statues and Regulation",
    "legal cases(case laws or judicial decisions and precedents) and court decisions",
    "greetings",
    "current events",
]
prompt_embeddings = embeddings.embed_documents(prompt_templates)
classification_embeddings = embeddings.embed_documents(prompt_templates)


def greeting(question: str):
    greeting_prompt = PromptTemplate.from_template(greeting_chat_qa_prompt_template)

    greeting_chain = LLMChain(llm=llm, prompt=greeting_prompt, verbose=True)

    response = greeting_chain.invoke({"question": question})

    print("answer:", response["text"])

    answer = response["text"]
    return answer


def agent(
    question: str,
    session_id: str = "2c1fb0f9-c88c-4137-8a43-cbe5e8c1b935",
    selected_template: str = None,
):
    standalone_question = condense_question(question=question, session_id=session_id)
    query_embedding = embeddings.embed_query(standalone_question)
    classification_embedding = embeddings.embed_query(standalone_question)

    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    similarity_classification = cosine_similarity(
        [classification_embedding], classification_embeddings
    )
    print(similarity)
    print(similarity_classification)
    most_similar = prompt_templates[similarity.argmax()]

    if most_similar == legal_chat_qa_prompt_template:
        template_name = "Legal Chat"
    elif most_similar == regulation_chat_qa_prompt_template:
        template_name = "Regulation Chat"
    elif most_similar == greeting_chat_qa_prompt_template:
        template_name = "Greeting Chat"
    elif most_similar == internet_search_prompt_template:
        template_name = "Internet Search"

    prompt = PromptTemplate.from_template(most_similar)
    print("Intent Router:", template_name)

    if most_similar == legal_chat_qa_prompt_template:
        response = rag_legal_chat(question=standalone_question)
    elif most_similar == regulation_chat_qa_prompt_template:
        response = rag_general_chat(question=standalone_question)
    elif most_similar == greeting_chat_qa_prompt_template:
        response = greeting(question=standalone_question)
    elif most_similar == internet_search_prompt_template:
        response = search_internet(query=standalone_question)

    return template_name, response


class Worker(QThread):
    finished = pyqtSignal(tuple)

    def __init__(self, query, selected_template):
        super().__init__()
        self.query = query
        self.selected_template = selected_template

    def run(self):
        result = agent(self.query, selected_template=self.selected_template)
        self.finished.emit(result)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Adaletgpt Intent Router")
        self.setGeometry(100, 100, 800, 600)

        self.input_label = QLabel("Question:", self)
        self.input_label.setStyleSheet("font-size: 18px;")
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Enter your question here...")
        self.input_field.setStyleSheet(
            "font-size: 14px; padding: 10px; border: 2px solid #ccc; border-radius: 5px;"
        )

        self.output_label = QLabel("Answer:", self)
        self.output_label.setStyleSheet("font-size: 18px;")
        self.output_field = QTextEdit(self)
        self.output_field.setReadOnly(True)
        self.output_field.setStyleSheet(
            "font-size: 14px; padding: 10px; border: 2px solid #ccc; border-radius: 5px;"
        )

        self.loading_label = QLabel("Loading...", self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.hide()

        self.submit_button = QPushButton("Submit", self)
        self.submit_button.clicked.connect(self.on_submit)
        self.submit_button.setStyleSheet(
            """
            QPushButton {
                font-size: 16px;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            """
        )

        self.radio_general = QRadioButton("Regulation Chat", self)
        self.radio_legal = QRadioButton("Legal Chat", self)
        self.radio_greeting = QRadioButton("Greeting Chat", self)
        self.radio_search = QRadioButton("Internet Search", self)

        # Colored radio buttons using stylesheets
        self.radio_general.setStyleSheet(
            "font-size: 14px; color: black; background-color: #e6e6e6; padding: 10px; border: 2px solid #ccc; border-radius: 5px;"
        )
        self.radio_legal.setStyleSheet(
            "font-size: 14px; color: black; background-color: #e6e6e6; padding: 10px; border: 2px solid #ccc; border-radius: 5px;"
        )
        self.radio_greeting.setStyleSheet(
            "font-size: 14px; color: black; background-color: #e6e6e6; padding: 10px; border: 2px solid #ccc; border-radius: 5px;"
        )
        self.radio_search.setStyleSheet(
            "font-size: 14px; color: black; background-color: #e6e6e6; padding: 10px; border: 2px solid #ccc; border-radius: 5px;"
        )

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_general)
        radio_layout.addWidget(self.radio_legal)
        radio_layout.addWidget(self.radio_greeting)
        radio_layout.addWidget(self.radio_search)

        layout = QVBoxLayout()
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_field)
        layout.addLayout(radio_layout)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.loading_label)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_field)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

    def on_submit(self):
        query = self.input_field.text()
        if query:
            self.loading_label.show()
            self.submit_button.setDisabled(True)
            self.output_field.clear()

            selected_template = None
            if self.radio_general.isChecked():
                selected_template = "Regulation Chat"
            elif self.radio_legal.isChecked():
                selected_template = "Legal Chat"
            elif self.radio_greeting.isChecked():
                selected_template = "Greeting Chat"
            elif self.radio_search.isChecked():
                selected_template = "Internet Search"

            self.worker = Worker(query, selected_template)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()

    def on_finished(self, result):
        template_name, answer = result
        self.output_field.setText(answer)
        self.loading_label.hide()
        self.submit_button.setDisabled(False)

        # Automatically select the appropriate radio button
        if template_name == "Legal Chat":
            self.radio_legal.setChecked(True)
        elif template_name == "Greeting Chat":
            self.radio_greeting.setChecked(True)
        elif template_name == "Regulation Chat":
            self.radio_general.setChecked(True)
        else:
            self.radio_search.setChecked(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
