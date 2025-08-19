# generate_roadmap.py
import argparse
from rag_utils2 import build_vectorstore, ask_with_llm

def main():
    parser = argparse.ArgumentParser(description="Generate a role-specific roadmap using RAG + LLM")
    parser.add_argument("--docs", type=str, required=True, help="Path to folder containing onboarding PDFs/TXT docs")
    parser.add_argument("--role", type=str, required=True, help="Role to generate roadmap (e.g., 'Frontend Developer Intern')")
    parser.add_argument("--days", type=int, default=30, help="Duration in days (default: 30)")
    args = parser.parse_args()

    # Step 1: Build/load vectorstore from docs
    coll, embeddings = build_vectorstore(args.docs)

    # Step 2: Generate roadmap using role + context
    query = f"""
You are an onboarding assistant responsible for creating structured and practical roadmaps for new hires. 
Your task is to generate a **{args.days}-day roadmap** for onboarding a new {args.role} using the provided company documents as the primary reference. 

### Instructions:
1. Use the uploaded documents as the main source of truth. If the documents do not contain enough information, make reasonable assumptions based on standard industry practices. 
2. The roadmap must cover **{args.days} days**, divided into weekly phases (Week 1, Week 2, Week 3, Week 4).
3. Each week should contain:
   - **Focus Areas** (main learning/activities for the week)
   - **Daily Breakdown** (Day 1 to Day 7 with tasks and objectives)
   - **Deliverables/Checkpoints** (what should be achieved by end of the week)
4. The plan should balance **learning**, **hands-on practice**, **team collaboration**, and **evaluation**.
5. Make sure the roadmap feels **customized to the role** ({args.role}) and aligns with company processes found in the documents.

### Output Format:
- Title: “{args.days}-Day Roadmap for {args.role}”
- Week-by-week sections
- Clear bullet points for daily tasks
- End-of-week summary with deliverables

Now, generate the **{args.days}-day roadmap for onboarding a {args.role}**.
"""

    answer = ask_with_llm(coll, embeddings, query)

    # Step 3: Print roadmap
    print("\n=== Generated Roadmap ===\n")
    print(answer)
    print("\n=========================\n")

if __name__ == "__main__":
    main()
