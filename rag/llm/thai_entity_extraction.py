"""
Thai-specific entity extraction module for GraphRAG
"""
import json
from typing import List, Dict, Any, Tuple
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

THAI_ENTITY_EXTRACTION_PROMPT = """
คุณเป็นผู้เชี่ยวชาญในการสกัดเอนทิตีและความสัมพันธ์จากเอกสารภาษาไทย
กรุณาวิเคราะห์ข้อความต่อไปนี้และสกัด:
1. เอนทิตีที่สำคัญทั้งหมด (คน, องค์กร, สถานที่, แนวคิด ฯลฯ)
2. ความสัมพันธ์ระหว่างเอนทิตีเหล่านี้

โปรดตอบกลับในรูปแบบ JSON เท่านั้น ไม่ต้องมีคำอธิบายเพิ่มเติม:
{{
  "entities": [
    {{"id": "person_1", "name": "ชื่อบุคคล", "type": "person"}},
    {{"id": "org_1", "name": "ชื่อองค์กร", "type": "organization"}}
  ],
  "relations": [
    {{"source": "person_1", "target": "org_1", "type": "works_for"}}
  ]
}}

ข้อควรระวัง:
- สร้าง ID ที่เป็นเอกลักษณ์และสั้นแต่เข้าใจง่าย (เช่น "pm_prayut", "mot", "bangkok")
- สกัดเฉพาะความสัมพันธ์ที่มีการระบุไว้อย่างชัดเจน ไม่คาดเดา
- ระบุประเภทความสัมพันธ์ให้เฉพาะเจาะจง (เช่น "ทำงานให้", "ตั้งอยู่ใน", "เป็นผู้กำกับดูแล")
- หากเป็นเอกสารนโยบาย ระบุความสัมพันธ์ระหว่างหน่วยงานและนโยบาย

เนื้อหาที่ต้องวิเคราะห์:
{text}

กรุณาตอบกลับเป็น JSON:
"""

def extract_thai_entities_relations(llm, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract entities and relations from Thai text using LLM
    
    Args:
        llm: LLM model
        text: Text to analyze
        
    Returns:
        Tuple of (entities, relations)
    """
    # Create a direct prompt for JSON generation
    prompt = PromptTemplate(
        template=THAI_ENTITY_EXTRACTION_PROMPT,
        input_variables=["text"]
    )
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run chain
    response = chain.invoke({"text": text})
    result_text = response["text"]
    
    # Extract JSON from the response
    try:
        # Try to find JSON between curly braces
        if "{" in result_text and "}" in result_text:
            start_idx = result_text.find("{")
            end_idx = result_text.rfind("}") + 1
            json_str = result_text[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Extract entities and relations
            entities = result.get("entities", [])
            relations = result.get("relations", [])
            
            # Add default IDs if missing
            for i, entity in enumerate(entities):
                if "id" not in entity and "name" in entity:
                    # Create a simple ID from the name
                    name = entity["name"].lower()
                    # Replace Thai characters with romanized versions where possible
                    name = ''.join(c if c.isalnum() else '_' for c in name)
                    entity["id"] = f"{entity.get('type', 'entity')}_{name}_{i}"
            
            # Verify source and target exist
            valid_relations = []
            entity_ids = {entity.get("id") for entity in entities}
            
            for relation in relations:
                source = relation.get("source")
                target = relation.get("target")
                
                if source in entity_ids and target in entity_ids:
                    valid_relations.append(relation)
            
            return entities, valid_relations
        else:
            print("No JSON object found in the response")
            return [], []
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing entity extraction response: {e}")
        print(f"Response: {result_text[:100]}...")  # Print just the first 100 chars
        return [], []
