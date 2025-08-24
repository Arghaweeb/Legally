# ⚖️ Legal Assistant – Multi-Agent CrewAI System  

## 📌 Introduction  
The **Legal Assistant** is a multi-agent system built with **CrewAI** that helps automate repetitive and time-consuming legal tasks.  
It is designed to assist **lawyers, students, and citizens** by structuring case information, identifying applicable IPC sections, drafting legal documents, and retrieving relevant precedents.  

The system leverages:  
- **Specialized AI Agents** for modular responsibilities.  
- **Custom Search Tools** for IPC sections and legal precedents.  
- **Vector Databases (ChromaDB)** for semantic search and retrieval.  

By combining these components, the assistant reduces research time, increases accuracy, and makes legal knowledge more accessible.  

---

## 🤔 Why is this necessary?  
Legal research and drafting often involve navigating thousands of IPC sections and judgments, which is both **time-intensive** and **prone to human error**.  

This system provides a **structured and automated workflow** that enables:  
- Faster case preparation.  
- More accurate identification of applicable laws.  
- Simplified access to precedent-based reasoning.  

Ultimately, it **democratizes legal knowledge** and improves efficiency for professionals and students alike.  

---

## 🔗 Key Agents  
1. **Case Intake Agent**  
   - Structures unorganized user input into a standardized case brief.  

2. **IPC Section Agent**  
   - Finds relevant IPC sections using ChromaDB vector search.  

3. **Legal Drafter Agent**  
   - Generates drafts for petitions, FIRs, and notices.  

4. **Legal Precedent Agent**  
   - Retrieves and summarizes precedents supporting the case.  

---

## ⚙️ Custom Tools  
- **IPC Research Search Tool** → Vectorized `ipc.json` stored in ChromaDB for semantic retrieval.  
- **Legal Precedent Search Tool** → Retrieves judgments and summarizes their relevance.  

---

## 📚 Example Use Cases  
### 👨‍⚖️ For Lawyers  
- Automate IPC section lookup.  
- Draft legal documents (petitions, FIRs, notices).  
- Retrieve and summarize supporting precedents.  

### 🎓 For Law Students  
- Input hypothetical scenarios and instantly see applicable IPC sections.  
- Generate structured case briefs for study/moot court.  
- Learn from precedent summaries without sifting through volumes of judgments.  

### 🧑‍💼 For Citizens  
- Describe a real-life issue and get simplified explanations of applicable IPC laws.  
- Receive a draft of an FIR or complaint.  
- Understand past judgments to know their rights and protections.  

---

## 🚀 Workflow Overview  
1. **User provides a case description.**  
2. **Case Intake Agent** → Converts it into a structured case brief.  
3. **IPC Section Agent** → Queries IPC sections from ChromaDB.  
4. **Legal Drafter Agent** → Prepares draft petitions/complaints.  
5. **Legal Precedent Agent** → Fetches relevant precedents.  
6. **Final consolidated output** → Presented to the user.  

---

## 🛠️ Tech Stack  
- **CrewAI** – Multi-agent orchestration.  
- **ChromaDB** – Vector database for semantic IPC search.  
- **LangChain / LLMs** – Legal text processing and reasoning.  
- **Python** – Core implementation language.  

---

## 📂 Documentation Roadmap  
The repository will also contain a `docs/` folder with detailed documentation:  
- `docs/overview.md` → Vision + Use Cases  
- `docs/agents.md` → Agents & Task Descriptions  
- `docs/tools.md` → Custom Search Tools + ChromaDB Setup  
- `docs/workflow.md` → System Workflow & Architecture  

---

## 📜 Disclaimer  
This project is intended for **educational and research purposes** only.  
It should not be used as a substitute for professional legal advice.  

---
