from dotenv import load_dotenv
import numpy as np
import openai
import os
import faiss 
import pandas as pd 
import torch 


class RAG:
    def __init__(self, index_path:str, transaction_embeddings_path:str, autoencoder, openai_api_key:str):
        # Load FAISS index & embeddings
        self.index = faiss.read_index(index_path)
        self.autoencoder = autoencoder
        self.transaction_embeddings = np.load(transaction_embeddings_path).astype("float32")
        # Set up OpenAI API key
        openai.api_key = openai_api_key
        # load_dotenv()
        # openai.api_key = os.getenv("OPENAI_API_KEY")


    # Function to find similar transactions
    def find_similar_transactions(self, query_embedding, top_k=5):
        query_embedding = np.array([query_embedding], dtype="float32")  # Reshape for FAISS
        distances, indices = self.index.search(query_embedding, top_k)  # Retrieve top-k similar transactions
        return distances, indices
    

    def generate_audit_explanation_rag(self, query_id, df_original):
        """
        Generate an audit-focused explanation for an anomalous transaction using RAG.
        """
        # Retrieve query transaction embedding
        query_embedding = self.transaction_embeddings[query_id]

        # Find similar transactions
        distances, indices = self.find_similar_transactions(query_embedding, top_k=5)

        # Extract transaction metadata for similar cases
        retrieved_transactions = df_original.iloc[indices[0]].to_dict(orient="records")

        # Format retrieved data into structured knowledge
        retrieved_info = "\n".join([
            f"- **Transaction ID**: {t['BELNR']}, **Amount**: {t['DMBTR']}, **Company Code**: {t['BUKRS']}, **Posting Key**: {t['BSCHL']}"
            for t in retrieved_transactions
        ])

        # Create a structured prompt for the LLM
        prompt = f"""
        You are an expert financial auditor analyzing an anomalous transaction (ID: {query_id}).
        Below is the retrieved knowledge about similar past anomalies:

        {retrieved_info}

        Using this information, explain **why transaction {query_id} is anomalous** based on **audit risks**, including:
        - **Fraud Indicators** (e.g., duplicate payments, unusual amounts, suspicious vendor activity)
        - **Regulatory Violations** (e.g., missing approvals, misclassified expenses)
        - **Operational Risks** (e.g., frequent transactions to unknown accounts)

        Provide a **detailed financial audit explanation**, and suggest **next steps for the auditor**.
        """
        # Generate explanation from LLM
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert financial auditor."},
                    {"role": "user", "content": prompt}]
        )

        explanation = response["choices"][0]["message"]["content"]
        return explanation
    

    def generate_rag_audit_report(self, transaction_row, df_original):
        """
        Generate an audit-focused RAG report for a given transaction row.
        transaction row is passed as Pandas Series object.
        """
        # Encode the transaction using the trained autoencoder
        transaction_data = transaction_row[self.feature_columns].values.reshape(1, -1)
        transaction_embedding = self.autoencoder.encoder(torch.tensor(transaction_data, dtype=torch.float32)).detach().numpy()

        # Find similar transactions
        distances, indices = self.find_similar_transactions(transaction_embedding, top_k=5)

        # Extract metadata for similar transactions
        retrieved_transactions = df_original.iloc[indices[0]].to_dict(orient="records")

        # Derive potential regulatory concerns
        red_flags = []
        if transaction_row["DMBTR"] > df_original["DMBTR"].quantile(0.95):  # High-value approval needed
            red_flags.append(f"🚨 **High-Value Transaction:** Amount ({transaction_row['DMBTR']}) is in the top 5% of all transactions, requiring higher approvals.")

        if df_original[(df_original["DMBTR"] == transaction_row["DMBTR"]) & 
                    (df_original["BUKRS"] == transaction_row["BUKRS"])].shape[0] > 1:
            red_flags.append(f"🚨 **Potential Duplicate Payment:** A transaction with the same amount and company code already exists.")

        if transaction_row["HKONT"] not in df_original["HKONT"].value_counts().head(20).index:
            red_flags.append(f"🚨 **Uncommon Account Used:** The posting key ({transaction_row['HKONT']}) is rarely used in similar transactions.")

        # Format retrieved transaction data
        retrieved_info = "\n".join([
            f"- **Transaction ID**: {t['BELNR']}, **Amount**: {t['DMBTR']}, **Company Code**: {t['BUKRS']}, **Posting Key**: {t['BSCHL']}"
            for t in retrieved_transactions
        ])
        # Precompute the formatted red flags string
        red_flags_text = "\n".join(red_flags) if red_flags else "No immediate compliance concerns detected, but further review recommended."


        # Create a structured prompt for LLM
        prompt = f"""
        ### 📊 **Audit Report for Transaction ID: {transaction_row['BELNR']}**
        
        **🔎 Transaction Overview:**
        - **Amount:** {transaction_row['DMBTR']}
        - **Company Code:** {transaction_row['BUKRS']}
        - **Profit Center:** {transaction_row['PRCTR']}
        - **Posting Key:** {transaction_row['BSCHL']}
        - **Ledger Account:** {transaction_row['HKONT']}
        
        **📌 Retrieved Similar Anomalies:**
        {retrieved_info}
        
        **🚨 Potential Regulatory Concerns:**
        {red_flags_text}
        **🛠️ Audit Assessment:**
        Provide a **detailed financial audit explanation** for why this transaction is considered anomalous. Focus on:
        - **Fraud Risks** (e.g., duplicate payments, high-value transfers, suspicious vendor activity)
        - **Regulatory Violations** (e.g., missing approvals, incorrect classifications)
        - **Operational Errors** (e.g., unusual timing, unauthorized accounts)

        Provide clear **recommendations for auditors** on how to investigate further.
        """

        # Generate explanation from LLM
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert financial auditor."},
                    {"role": "user", "content": prompt}]
        )

        explanation = response["choices"][0]["message"]["content"]
        
        # Precompute the red flags text
        red_flags_text = "\n".join(red_flags) if red_flags else "✅ No immediate compliance concerns detected."

        # Return a structured audit report
        report = f"""
        # 📊 **Audit Report: Transaction {transaction_row['BELNR']}**
        
        ## 🔎 **Transaction Details**
        - **Amount:** {transaction_row['DMBTR']}
        - **Company Code:** {transaction_row['BUKRS']}
        - **Profit Center:** {transaction_row['PRCTR']}
        - **Posting Key:** {transaction_row['BSCHL']}
        - **Ledger Account:** {transaction_row['HKONT']}
        
        ## 🔍 **Retrieved Similar Transactions**
        {retrieved_info}
        
        ## 🚨 **Potential Compliance Risks**
        {red_flags_text}

        ## 🛠️ **AI-Generated Audit Explanation**
        {explanation}
        
        ## 📌 **Next Steps for Auditors**
        - Review supporting documentation (invoices, approval records)
        - Validate vendor legitimacy and past transaction history
        - Cross-check similar flagged transactions
        """

        return report




if __name__=="__main__":
    # Load metadata first
    checkpoint = torch.load("weights/refined_autoencoder_with_metadata.pth")
    metadata = checkpoint["metadata"]
    from model import RefinedTransactionAutoencoder

    # Ensure correct architecture
    autoencoder = RefinedTransactionAutoencoder(input_dim=metadata["input_dim"], latent_dim=metadata["latent_dim"])

    # Load model weights
    autoencoder.load_state_dict(checkpoint["model_state"])
    autoencoder.eval()

    print("✅ Model and metadata loaded successfully.")

    # Load your original transaction dataset (same one used to train embeddings)
    df_original = pd.read_csv("datathon_data.csv")  # Replace with your actual file path

    # Select an anomaly transaction ID for analysis
    query_id = 506926  # Change this to any anomaly ID you want to analyze
    rag = RAG(index_path="index/refined_transaction_faiss.index", transaction_embeddings_path="weights/refined_transaction_embeddings.npy", autoencoder=autoencoder)
    # Generate audit explanation using RAG
    explanation = rag.generate_audit_explanation_rag(query_id, df_original)

    # Print the generated explanation
    print("\n🔍 Audit Explanation for Transaction", query_id)
    print(explanation)
