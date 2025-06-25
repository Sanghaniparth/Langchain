import random
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import json

# Step 1: Scrape Q&A
# url = "https://www.interviewbit.com/spring-boot-interview-questions/"
# soup = BeautifulSoup(requests.get(url).text, "html.parser")

# questions = [q.get_text(strip=True) for q in soup.select("h2")[:20]]
# answers = [a.get_text(strip=True) for a in soup.select("div.single_answer")[:20]]
# pairs = list(zip(questions, answers))
# selected = random.sample(pairs, 10)
qa_pairs = [
    ("What is Spring Boot?", "Spring Boot is a project built on top of the Spring Framework. It helps create stand-alone, production-grade Spring-based applications easily."),
    ("How does Spring Boot make development easier?", "Spring Boot simplifies development by providing default configurations, embedded servers, and minimal Spring configuration."),
    ("What are the features of Spring Boot?", "Key features include auto-configuration, starter dependencies, embedded servers, Actuator, and CLI support."),
    ("What are the advantages of Spring Boot?", "It reduces development time, increases productivity, and simplifies deployment with embedded servers."),
    ("What is @SpringBootApplication annotation?", "It is a combination of @Configuration, @EnableAutoConfiguration, and @ComponentScan annotations."),
    ("What is the use of application.properties?", "It's used to configure application-level settings like port number, DB connection, logging, etc."),
    ("What is Spring Boot Starter?", "Starters are dependency descriptors that simplify adding common dependencies like web, data-jpa, security."),
    ("What is the role of Spring Boot CLI?", "It allows you to quickly run and test Spring Boot applications using Groovy scripts."),
    ("How is Spring Boot different from Spring?", "Spring Boot is opinionated and convention-based, while Spring requires more setup and configuration."),
    ("What is Spring Boot DevTools?", "DevTools helps in auto-reloading and live reloading of changes during development.")
]
# Step 2: Build Documents
# docs = [Document(page_content=f"Q: {q}\nA: {a}") for q, a in selected]
docs = [Document(page_content=f"Q: {q}\nA: {a}") for q, a in qa_pairs]

# Step 3: Vector DB and Retriever
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
vectordb = FAISS.from_documents(docs, embedding_model)
retriever = vectordb.as_retriever()

# Step 4: Prompt + Chain
prompt = ChatPromptTemplate.from_template(
    """You are a Spring Boot expert. Evaluate the candidate's answer using the following reference context.

<context>
{context}
</context>

Question/Answer: {input}

Give a JSON with fields:
- score (0-10)
- feedback (1-line reason)
"""
)
llm = Ollama(model="llama3", temperature=0)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Step 5: Ask and Evaluate
import json

# === Step 1: Ask 10 questions and store user answers ===
candidate_responses = {}

for i, (q, _) in enumerate(qa_pairs[:10], 1):  # Ensure only 10 questions
    print(f"\nQ{i}: {q}")
    user_ans = input("Your Answer: ")
    candidate_responses[q] = user_ans  # Store in dict: {q: a}

# === Step 2: Evaluate answers using retrieval chain ===
total_score = 0
feedback_list = []
evaluations = {}

for i, (question, user_ans) in enumerate(candidate_responses.items(), 1):
    combined_input = f"Question: {question}\nAnswer: {user_ans}"
    response = retrieval_chain.invoke({"input": combined_input})
    
    try:
        result = json.loads(response["answer"])
        score = int(result.get("score", 0))
        feedback = result.get("feedback", "✅ Answer evaluated.")
    except Exception:
        score = 0
        feedback = "⚠️ Could not evaluate."

    # Save evaluation details
    evaluations[question] = {
        "answer": user_ans,
        "score": score,
        "feedback": feedback
    }

    total_score += score
    feedback_list.append((question, score, feedback))

# === Step 3: Final evaluation result ===
avg = total_score / len(candidate_responses)
print(f"\n📊 Final Score: {avg}/10")
if avg >= 6:
    print("✅ Candidate QUALIFIED for the technical round.")
else:
    print("❌ Candidate NOT qualified.")

# === Step 4: Show detailed feedback ===
print("\n📝 Feedback Summary:")
for q, s, f in feedback_list:
    print(f"Q: {q}\nScore: {s}, Feedback: {f}\n")

# === Step 5: Optionally save full report ===
final_report = {
    "score": avg,
    "qualified": avg >= 6,
    "responses": evaluations
}

with open("tech_eval_result.json", "w") as f:
    json.dump(final_report, f, indent=2)
