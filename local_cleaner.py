import os
import json
import asyncio
import polars as pl
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load the hidden .env file safely
load_dotenv()
client = genai.Client()

# 1. Define the Strict CRM Target Schema
class CRMLead(BaseModel):
    first_name: str = Field(description="The lead's first name. Capitalized.")
    last_name: str = Field(description="The lead's last name. Capitalized.")
    company_name: str = Field(description="The company name. Infer from email domain if missing.")
    email: str = Field(description="The anonymized email address.")
    job_title: Optional[str] = Field(default="Unknown")
    phone_number: Optional[str] = Field(default=None)

# 2. Asynchronous Worker Function
async def process_single_row(index, row_dict, semaphore, clean_leads, dead_letters):
    # The semaphore acts as a bouncer, only letting X requests hit the API at the exact same time
    async with semaphore:
        raw_json = json.dumps(row_dict)
        
        prompt = f"""
        You are a strict data hygiene ETL worker. 
        Take the following broken JSON extracted from a failed scraper log.
        Extract the information to match the required schema.
        If the company name is missing, intelligently infer it from the email domain.
        Fix all capitalization, unescaped quotes, and mojibake.
        
        Raw Data:
        {raw_json}
        """

        try:
            # NEW: Use the .aio asynchronous client
            response = await client.aio.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt,
                config=types.GenerateContentConfig( # type: ignore
                    response_mime_type="application/json",
                    response_schema=CRMLead,
                    temperature=0.1
                )
            )
            
            raw_text = response.text or ""
            if raw_text.strip().startswith("```"):
                raw_text = raw_text.strip("` \n").replace("json", "").strip()
                
            validated_lead = CRMLead.model_validate_json(raw_text)
            updated_row = {**row_dict, **validated_lead.model_dump()}
            clean_leads.append(updated_row)
            print(f"Row {index + 1}: Success -> {validated_lead.email}")

        except Exception as e:
            print(f"Row {index + 1}: FAILED -> {e}")
            row_dict['error_reason'] = str(e)
            dead_letters.append(row_dict)

# 3. Main Asynchronous Controller
async def process_csv_async(input_filepath: str, output_filepath: str, deadletter_filepath: str):
    print(f"Loading {input_filepath} with Polars...")
    df = pl.read_csv(input_filepath, infer_schema_length=0)
    
    clean_leads = []
    dead_letters = []

    # CONCURRENCY LIMITER: Set to 15 for Free Tier. 
    # When you add a billing card to Google AI Studio, change this to 100+ for massive speed.
    semaphore = asyncio.Semaphore(15)

    print(f"Processing {len(df)} rows asynchronously...")

    # Create a massive batch of concurrent tasks
    tasks = []
    for index, row_dict in enumerate(df.iter_rows(named=True)):
        task = asyncio.create_task(process_single_row(index, row_dict, semaphore, clean_leads, dead_letters))
        tasks.append(task)

    # Execute all tasks in parallel and wait for them to finish
    await asyncio.gather(*tasks)

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
        # NEW: Run the async loop
        asyncio.run(process_csv_async(INPUT_FILE, CLEAN_OUTPUT, DEAD_OUTPUT))
    else:
        print(f"Waiting for {INPUT_FILE}... Drop a file on your GitHub Pages site to generate it.")