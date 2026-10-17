import os
import json
import polars as pl
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

# NEW: Import the modern GenAI SDK
from google import genai
from google.genai import types

# Load the hidden .env file safely
load_dotenv()

# 1. Setup Gemini Client 
# The modern client automatically picks up the GEMINI_API_KEY from your environment
client = genai.Client()

# 2. Define the Strict CRM Target Schema (Pydantic)
class CRMLead(BaseModel):
    first_name: str = Field(description="The lead's first name. Capitalized.")
    last_name: str = Field(description="The lead's last name. Capitalized.")
    company_name: str = Field(description="The company name. Infer from email domain if missing.")
    email: str = Field(description="The anonymized email address (e.g. xxxxx@acme.com).")
    job_title: Optional[str] = Field(default="Unknown", description="The job title. Default to 'Unknown' if not present.")
    phone_number: Optional[str] = Field(default=None, description="The hashed phone number.")

# 3. The Core Processing Logic
def process_csv(input_filepath: str, output_filepath: str, deadletter_filepath: str):
    print(f"Loading {input_filepath} with Polars...")
    
    df = pl.read_csv(input_filepath, infer_schema_length=0)
    
    clean_leads = []
    dead_letters = []

    print(f"Processing {len(df)} rows through Gemini 2.5 Flash...")

    for index, row_dict in enumerate(df.iter_rows(named=True)):
        raw_json = json.dumps(row_dict)
        
        prompt = f"""
        You are a strict data hygiene ETL worker. 
        Take the following broken JSON extracted from a failed scraper log.
        Extract the information to match the required schema.
        If the company name is missing, intelligently infer it from the email domain (e.g., '@acme.com' -> 'Acme').
        Fix all capitalization, unescaped quotes, and mojibake (weird character encodings).
        
        Raw Data:
        {raw_json}
        """

      # Initialize response to None to prevent 'unbound variable' errors in the except block
        response = None 
        
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt,
                config=types.GenerateContentConfig( # type: ignore
                    response_mime_type="application/json",
                    response_schema=CRMLead,
                    temperature=0.1
                )
            )
            
            # Use 'or ""' to guarantee raw_text is always a string, fixing the NoneType error
            raw_text = response.text or "" 
            
            # Clean rogue markdown
            if raw_text.strip().startswith("```"):
                raw_text = raw_text.strip("` \n").replace("json", "").strip()
                
            validated_lead = CRMLead.model_validate_json(raw_text)
            clean_leads.append(validated_lead.model_dump())
            print(f"Row {index + 1}: Success -> {validated_lead.email}")

        except Exception as e:
            print(f"Row {index + 1}: FAILED -> {e}")
            # Safely extract debug text only if response was successfully created
            debug_text = response.text if response and hasattr(response, 'text') else "No response generated. Possible network/API failure."
            print(f"Debug Output: {debug_text}") 
            row_dict['error_reason'] = str(e)
            dead_letters.append(row_dict)
    # 4. Save Outputs
    if clean_leads:
        pl.DataFrame(clean_leads).write_csv(output_filepath)
        print(f"\n✅ Saved {len(clean_leads)} clean leads to {output_filepath}")
        
    if dead_letters:
        pl.DataFrame(dead_letters).write_csv(deadletter_filepath)
        print(f"⚠️ Saved {len(dead_letters)} unrecoverable leads to {deadletter_filepath}")

if __name__ == "__main__":
    INPUT_FILE = "anonymized_audit_data.csv"
    CLEAN_OUTPUT = "recovered_leads.csv"
    DEAD_OUTPUT = "dead_leads.csv"
    
    if os.path.exists(INPUT_FILE):
        process_csv(INPUT_FILE, CLEAN_OUTPUT, DEAD_OUTPUT)
    else:
        print(f"Waiting for {INPUT_FILE}... Drop a file on your GitHub Pages site to generate it.")