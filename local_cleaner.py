import os
import json
import polars as pl
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

# 1. Setup Gemini API
# You must set your API key in your terminal before running: export GEMINI_API_KEY="your_key"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# We use Gemini 2.5 Flash for speed and cost-efficiency
model = genai.GenerativeModel('gemini-2.5-flash')

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
    
    # Ingest using Polars. Strictly cast everything to strings to prevent schema crashes on weird data.
    df = pl.read_csv(input_filepath, infer_schema_length=0)
    
    clean_leads = []
    dead_letters = []

    print(f"Processing {len(df)} rows through Gemini 2.5 Flash...")

    # For the local MVP, we iterate synchronously. 
    # (In the final GCP version, we use the Batch API for massive concurrency)
    for index, row_dict in enumerate(df.iter_rows(named=True)):
        # Convert the raw row into a JSON string for the LLM prompt
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

        try:
            # Enforce structured output via Pydantic
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=CRMLead,
                    temperature=0.1 # Keep it deterministic
                )
            )
            
            # Load the JSON string back into a Python dictionary, then validate with Pydantic
            validated_lead = CRMLead.model_validate_json(response.text)
            clean_leads.append(validated_lead.model_dump())
            print(f"Row {index + 1}: Success -> {validated_lead.email}")

        except Exception as e:
            # If Pydantic fails validation (e.g., missing a primary key like email), route to dead letter
            print(f"Row {index + 1}: FAILED -> {e}")
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
    # Point this to the file you download from the web dropzone
    INPUT_FILE = "anonymized_audit_data.csv"
    CLEAN_OUTPUT = "recovered_leads.csv"
    DEAD_OUTPUT = "dead_leads.csv"
    
    if os.path.exists(INPUT_FILE):
        process_csv(INPUT_FILE, CLEAN_OUTPUT, DEAD_OUTPUT)
    else:
        print(f"Waiting for {INPUT_FILE}... Drop a file on your GitHub Pages site to generate it.")