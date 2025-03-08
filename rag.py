from dotenv import load_dotenv
import numpy as np
import openai
import os
import faiss 
import pandas as pd 
import torch 


class RAG:
    def __init__(self, index_path:str, transaction_embeddings_path:str, autoencoder=None, key=None):
        # Load FAISS index & embeddings
        self.index = faiss.read_index(index_path)
        self.autoencoder = autoencoder
        self.transaction_embeddings = np.load(transaction_embeddings_path).astype("float32")
        # Set up OpenAI API key
        if key is not None:
            openai.api_key = key
        else:
            load_dotenv()
            openai.api_key = os.getenv("OPENAI_API_KEY")


    # Function to find similar transactions
    def find_similar_transactions(self, query_embedding, top_k=5):
        query_embedding = np.asarray(query_embedding, dtype="float32").reshape(1, -1)  # Ensure correct shape (1, d)
        distances, indices = self.index.search(query_embedding, top_k)  # Retrieve top-k similar transactions
        return distances, indices
    

    def generate_rag_from_id(self, query_id, df_original):
        """
        Generate an audit-focused RAG report for a given transaction ID.
        Uses FAISS for similarity search instead of autoencoder.
        
        - query_id: The index of the transaction in df_original.
        - df_original: The full DataFrame containing transaction data.
        """

        # Ensure query_id exists
        if query_id not in df_original.index:
            return f"❌ Error: Transaction at index {query_id} not found in the dataset."

        # Retrieve the transaction row based on query_id
        transaction_row = df_original.loc[query_id]

        # Get the FAISS embedding for the transaction
        transaction_embedding = np.asarray(self.transaction_embeddings[query_id], dtype="float32").reshape(1, -1)  # Ensure correct shape

        # Find similar transactions using FAISS
        distances, indices = self.find_similar_transactions(transaction_embedding, top_k=5)

        # Extract metadata for similar transactions
        retrieved_transactions = df_original.iloc[indices[0]].to_dict(orient="records")

        # Derive potential regulatory concerns
        red_flags = []
        if transaction_row["DMBTR"] > df_original["DMBTR"].quantile(0.95):
            red_flags.append(f"🚨 **High-Value Transaction:** Amount ({transaction_row['DMBTR']}) is in the top 5% of all transactions, requiring higher approvals.")

        if df_original[(df_original["DMBTR"] == transaction_row["DMBTR"]) & 
                       (df_original["BUKRS"] == transaction_row["BUKRS"])].shape[0] > 1:
            red_flags.append(f"🚨 **Potential Duplicate Payment:** A transaction with the same amount and company code already exists.")

        if transaction_row["HKONT"] not in df_original["HKONT"].value_counts().head(20).index:
            red_flags.append(f"🚨 **Uncommon Account Used:** The ledger account ({transaction_row['HKONT']}) is rarely used in similar transactions.")

        # Applying additional rule-based anomaly checks
        if df_original.duplicated(subset=["BUKRS", "WRBTR", "DMBTR"], keep=False).any():
            red_flags.append("🚨 **Unique Value Combination Detected:** This transaction has a unique combination of company, currency, and amount, which may require further review.")

        if transaction_row["WAERS"] not in df_original[df_original["BUKRS"] == transaction_row["BUKRS"]]["WAERS"].unique():
            red_flags.append(f"🚨 **Unusual Currency for Company:** The currency ({transaction_row['WAERS']}) is uncommon for company ({transaction_row['BUKRS']}).")

        if 54423 <= transaction_row["WRBTR"] <= 54478 and 910631 <= transaction_row["DMBTR"] <= 910677:
            red_flags.append("🚨 **Suspicious Amount Range:** Both WRBTR and DMBTR are within specific flagged ranges, indicating possible irregularities.")

        if transaction_row["PRCTR"] == "C20" and transaction_row["HKONT"] == "B1":
            red_flags.append("🚨 **Unusual Profit Center & Ledger Combination:** PRCTR is 'C20' and HKONT is 'B1', which is rarely observed in normal transactions.")

        if transaction_row["BUKRS"] == "C11" and (54423 <= transaction_row["WRBTR"] <= 54478 or 910631 <= transaction_row["DMBTR"] <= 910677):
            red_flags.append("🚨 **Company-Specific Anomaly:** Transactions from company 'C11' with these WRBTR or DMBTR values are flagged for unusual activity.")

        # Format retrieved transaction data
        retrieved_info = "\n".join([
            f"- **Amount**: {t['DMBTR']}, **Company Code**: {t['BUKRS']}, **Posting Key**: {t['BSCHL']}, **Ledger Account**: {t['HKONT']}"
            for t in retrieved_transactions
        ])

        # Precompute the formatted red flags string
        red_flags_text = "\n".join(red_flags) if red_flags else "✅ No immediate compliance concerns detected."

        # Create a structured prompt for LLM
        prompt = f"""
        ### 📊 **Audit Report for a Transaction**
        
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
            temperature=0,  # Make responses deterministic
            messages=[{"role": "system", "content": "You are an expert financial auditor."},
                    {"role": "user", "content": prompt}]
        )

        explanation = response["choices"][0]["message"]["content"]

        # Return a structured audit report
        report = f"""
        # 📊 **Audit Report**
        
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
    # checkpoint = torch.load("weights/refined_autoencoder_with_metadata.pth")
    # metadata = checkpoint["metadata"]
    # from model import RefinedTransactionAutoencoder

    # # Ensure correct architecture
    # autoencoder = RefinedTransactionAutoencoder(input_dim=metadata["input_dim"], latent_dim=metadata["latent_dim"])

    # # Load model weights
    # autoencoder.load_state_dict(checkpoint["model_state"])
    # autoencoder.eval()


    # Load your original transaction dataset (same one used to train embeddings)
    df_original = pd.read_csv("data/datathon_data.csv")  # Replace with your actual file path

    # Select an anomaly transaction ID for analysis
    query_id = 506926  # Change this to any anomaly ID you want to analyze
    rag = RAG(index_path="index/refined_transaction_faiss.index", transaction_embeddings_path="weights/refined_transaction_embeddings.npy")
    # Generate audit explanation using RAG
    explanation = rag.generate_rag_from_id(query_id, df_original)

    # Print the generated explanation
    print("\n🔍 Audit Explanation for Transaction", query_id)
    print(explanation)
