import subprocess
import json
import re

def load_ocr_output():
    with open("donut_output.txt", "r", encoding="utf-8") as f:
        return f.read()

def call_llama(prompt):
    result = subprocess.run(
        ['ollama', 'run', 'llama3'],
        input=prompt.encode('utf-8'),
        capture_output=True
    )
    return result.stdout.decode()

def extract_json_from_response(text):
    match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def create_prompt(raw_text):
    return f"""
You are a professional medical NLP engine. Convert the following OCR-extracted lab report into structured JSON.

Only output the final JSON inside <json></json> tags. DO NOT explain anything.

Structure:
<json>
{{
  "hospital_info": {{
    "hospital_name": "",
    "address": "",
    "phone": "",
    "website": "",
    "accreditation": "",
    "panel": [],
    "certificate_number": "",
    "accreditation_date": ""
  }},
  "patient_info": {{
    "lab_id": "",
    "name": "",
    "age": "",
    "gender": "",
    "client_code": ""
  }},
  "doctor_info": {{
    "referred_by": "",
    "consultant": null,
    "pathologist": "",
    "reporting_location": ""
  }},
  "report_info": {{
    "report_type": "Pathology Report",
    "collection_date": "",
    "collection_time": "",
    "received_date": "",
    "received_time": "",
    "report_date": "",
    "report_time": "",
    "sample_type": "Serum"
  }},
  "test_results": [
    {{
      "test_name": "",
      "result_value": "",
      "unit": "",
      "reference_range": "",
      "status": "",
      "method": ""
    }}
  ]
}}
</json>

Report text:
\"\"\"
{raw_text}
\"\"\"
"""

if __name__ == "__main__":
    raw_text = load_ocr_output()
    prompt = create_prompt(raw_text)
    response = call_llama(prompt)
    extracted_json = extract_json_from_response(response)

    if extracted_json:
        try:
            parsed = json.loads(extracted_json)
            with open("final_output.json", "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            print("✅ JSON parsed and saved to final_output.json")
        except json.JSONDecodeError as e:
            with open("final_output.json", "w", encoding="utf-8") as f:
                f.write(response)
            print("⚠️ JSON tags found, but structure invalid. Raw output saved.")
    else:
        with open("final_output.json", "w", encoding="utf-8") as f:
            f.write(response)
        print("⚠️ No <json> tag found. Raw output saved instead.")
