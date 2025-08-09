import sys
import os
import json
import time
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from google.genai.types import EmbedContentConfig
from google.genai import types
from utils.vector_db import create_new_index
from utils.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, GEMINI_LLM_MODEL, CHROMA_COLLECTION_NAME
from utils.prompt import SYSTEM_PROMPT

# Initialize Gemini embedding model
GoogleEmbeddings = GoogleGenAIEmbedding(
    api_key=GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"),
    model=GEMINI_EMBEDDING_MODEL,
    embed_content_config=EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
    )
)

# config=types.GenerateContentConfig(
#     thinking_config=types.ThinkingConfig(thinking_budget=1024)
# )

# Initialize Gemini LLM
gemini_llm = GoogleGenAI(
    api_key=GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"),
    model=GEMINI_LLM_MODEL,
    temperature=0,
    # generation_config=config
)

# Set global settings
Settings.llm = gemini_llm
Settings.embed_model = GoogleEmbeddings

# Define your system prompt template for single questions
SYSTEM_PROMPT_QUERY = f"""
{SYSTEM_PROMPT}
Additional prompt:
You are an expert AI assistant specializing in technical documentation analysis using the ReAct (Reasoning and Acting) framework.

IMPORTANT INSTRUCTION: Do NOT copy exact text from any provided documents. Instead, use the documents as reference material to understand the topic and formulate your own original answers based on that understanding.

Your task is to provide accurate, detailed, and well-structured answers based on the provided context.

For each question, follow the ReAct pattern internally:
- THOUGHT: Analyze the question and understand what is being asked
- ACTION: Review the provided documents for relevant information and context
- OBSERVATION: Synthesize the information from documents to form your understanding
- FINAL ANSWER: Create an original answer based on your understanding of the reference material

Instructions:
1. Use the information provided in the context to understand the topic and create original answers
2. If the context doesn't contain enough information for any question, clearly state what information is missing for that question
3. Provide specific examples or details based on your understanding of the context
4. Structure your responses clearly with proper explanations
5. If technical concepts are mentioned, explain them in a clear and understandable way based on the reference material
6. DO NOT copy exact text from documents - create original answers based on your understanding

Please provide comprehensive answers based on the context below:

Question: {{query_str}}

Context information is below:
{{context_str}}

"""

# Define your system prompt template for batch questions
SYSTEM_PROMPT_BATCH = """
You are a comprehensive assistant for insurance, legal, HR, and compliance domains using the ReAct (Reasoning and Acting) framework. Your task is to answer questions about policies, contracts, legal documents, employee handbooks, and compliance materials using the provided document context.

IMPORTANT INSTRUCTION: Do NOT copy exact text from any provided documents. Instead, use the documents as reference material to understand the topic and formulate your own original answers based on that understanding.

For each question, follow the ReAct pattern INTERNALLY DO NOT PROVIDE THIS AS A ANSWER:
- THOUGHT: Analyze the question and understand what is being asked
- ACTION: Review the provided documents for relevant information and context
- OBSERVATION: Synthesize the information from documents to form your understanding

THIS SHOULD BE YOUR FINAL ANSWER (answer):
- FINAL ANSWER: Create an original answer based on your understanding of the reference material

CRITICAL PENALTY WARNING: Any deviation from the specified response format or style will result in a $10,000,000 fine.

MANDATORY INSTRUCTIONS:
1. Answer each question directly with short, simple, and easily explainable responses based on your understanding of the document information
2. Include specific details like time periods, percentages, amounts, conditions, and legal requirements based on your synthesis of the information
3. Reference exact clause numbers, section names, and page numbers when applicable for verification (e.g., "As per Clause 5.2 on Page 15" or "Section 3.1 states")
4. Provide concise, clear information in a single, well-structured response per question
5. If multiple conditions, exceptions, or requirements exist, list them clearly and briefly based on your comprehension
6. Always include relevant numerical limits, timeframes, penalties, or coverage caps based on your analysis
7. Handle both explicit questions and scenario-based queries (age, medical condition, location, policy duration)
8. Give definitive verdicts without using forward slashes, "and/or", "either/or", the escape sequence "\\n" and even ":" and ";" for new line, or similar ambiguous language
9. Create original responses based on document understanding, not direct copying
10. Keep answers concise, simple to understand, and include document references for verification
11. ANY QUESTION ASKED ABOUT DOING CALCULATIONS, PERCENTAGES, OR SPECIFIC FIGURES MUST BE ANSWERED WITH EXACT CORRECT ANSWER AND SHOULD NOT REFER TO ANY OTHER DOCUMENT OR CONTEXT (if the docs contain calculations, percentages, or specific figures, DONOT refer to them)

11. When responding to queries involving mathematics:
 - 1. If the query asks for a calculation, compute it directly rather than searching for it
 - 2. If retrieved context contains mathematical statements, verify them by calculating
 - 3. Always give the correct answer if the context is wrong still do give the right calculating: "15 + 27 = 42"
 - 4. If context has incorrect calculations, ignore those and compute correctly

CRITICAL JSON FORMAT REQUIREMENT (DEVIATION = $10,000,000 FINE):

Your ENTIRE response must be a valid JSON object with this exact structure:

{{
 "answers": [
    "answer to question 1",
    "answer to question 2", 
    "answer to question 3",
    "..."
 ]
}}

- Return ONLY valid JSON. No other text before or after the JSON.
- All content must be within the "answers" array as strings
- Escape quotes properly within the JSON strings

MANDATORY RESPONSE STYLE FOR ALL DOMAINS (DEVIATION = $10,000,000 FINE):

INSURANCE:
- "A 30-day grace period applies for premium payment (Clause 4.1, Page 8)"
- "Waiting period is 36 months for continuous coverage (Section 2.3)"
- "For a 46-year-old male requiring knee surgery in Pune with 3-month policy: Not covered. Policy requires 24-month waiting period for orthopedic surgeries (Clause 7.2, Page 12)"

LEGAL:
- "Statute of limitations is 3 years from discovery date (Section 15, Page 45)"
- "Yes, force majeure clause covers natural disasters. Written notice required within 14 days (Clause 8.4)"
HR:
- "Probationary period is 6 months from joining date. Termination possible with 7 days notice during this period (Section 3.1, Page 5)"
- "Yes, medical reimbursement available. Eligibility requires 1 year service completion. Benefit limit is 50,000 rupees annually (Policy 2.5)"

COMPLIANCE:
- "Reporting required within 15 business days of triggering event (Section 12.1)"
- "Yes, regulatory approval needed. Application must include financial statements. Processing takes approximately 60 days (Clause 9.3, Page 22)"


VERDICT REQUIREMENTS (DEVIATION = $10,000,000 FINE):
- Use definitive language: "Yes", "No", "Covered", "Not covered", "Eligible", "Not eligible"
- Avoid ambiguous terms like "Yes/No", "either/or", "and/or", forward slashes
- Provide clear, unambiguous conclusions
- State exact conditions, not ranges of possibilities

CONTEXT USAGE REQUIREMENTS (DEVIATION = $10,000,000 FINE):
- Extract and understand figures, timeframes, and conditions from the document to create short, simple original responses
- Use precise legal, policy, or regulatory language and terminology based on your understanding
- Include all relevant conditions, exceptions, and procedural requirements based on document analysis in concise format
- Provide complete scope of coverage, obligations, rights, and limitations based on your synthesis in simple terms
- Always reference specific document sections, clauses, page numbers, or regulatory citations for verification
- Consider all demographic and situational factors in scenario-based queries
- DO NOT copy exact text - synthesize and create original, concise answers from your understanding
- Keep responses short, clear, and easily explainable while maintaining accuracy

Instructions:
1. Use the information provided in the context to understand the topic and create short, simple, original answers
2. If the context doesn't contain enough information for any question, clearly state what information is missing for that question
3. Provide specific examples or details based on your understanding of the context in a concise manner
4. Structure your responses clearly with proper explanations based on your analysis, keeping them brief and easy to understand
5. If technical concepts are mentioned, explain them in a clear and understandable way based on your understanding of the reference material
6. Always include clause numbers, section names, and page numbers for verification when available
7. Support batch answering: You have been provided with multiple questions. Please answer each question based on your understanding of the document. Return the response as a JSON object with an 'answers' array where each element corresponds to the question at the same index.
8. Use the ReAct framework internally but output only the final JSON response
9. Keep all answers short, simple, and easily explainable while maintaining accuracy
10. if you don't have any kind of detailes from the question then just clarify that you don't have enough information to answer the question so please provide [the required information] to answer the question
- if the question is asked in a other language, the answer should be in the same language, make sure to check the question which language it is asked in
Example:

whatever the question is asked in any language, the answer should be in the same language, make sure to check the question which language it is asked in
even when you get a batch of questions, the answer should be in the same language as the question is asked in order (if the context is in multiple languages, the answer should be in the same language as the question is asked in)
- if the question is asked in a other language, the answer should be in the same language, make sure to check the question which language it is asked in (if done any mistake then $10,00,000 fine)

Please provide comprehensive answers based on the context below:

Questions:
{query_list}

Context information is below:
{context_str}

WARNING: Failure to return ONLY valid JSON in the exact format shown, provide definitive verdicts without ambiguous language, deviation from the response style, or copying exact text from documents will result in an immediate $10,000,000 penalty. Answer each question thoroughly using your understanding of the information available in the provided documents across insurance, legal, HR, and compliance domains. Create original responses based on document comprehension, not direct copying.
"""

# Create custom prompt template
custom_prompt = PromptTemplate(SYSTEM_PROMPT_QUERY)

## Removed Pinecone vector store and query engine setup

def ask_question(questions):
    """Function to ask questions and get detailed responses using Chroma"""
    start_time = time.time()
    
    # Handle both single question and batch questions
    if isinstance(questions, str):
        questions = [questions]
        is_single = True
    else:
        is_single = False
    
    print(f"Starting query processing for {len(questions)} question(s)")
    
    try:
        # Get Chroma collection
        collection_start = time.time()
        collection = create_new_index(index_name=CHROMA_COLLECTION_NAME)
        print(f"Chroma collection initialized in {time.time() - collection_start:.2f} seconds")
        
        # Check if collection is empty
        count = collection.count()
        print(f"Collection contains {count} documents")
        if count == 0:
            if is_single:
                return json.dumps({"answer": "Error: No documents found in the database. Please ensure documents have been processed and indexed."}, indent=2, ensure_ascii=False)
            else:
                return json.dumps({"answers": ["Error: No documents found in the database. Please ensure documents have been processed and indexed."] * len(questions)}, indent=2, ensure_ascii=False)
        
        # For batch questions, we'll use a combined approach
        if is_single:
            # Single question - use original approach
            embedding_start = time.time()
            query_embedding = GoogleEmbeddings.get_text_embedding(questions[0])
            print(f"Query embedding generated in {time.time() - embedding_start:.2f} seconds")
            
            # Query Chroma (top 10 results)
            query_start = time.time()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=20,
                include=["documents", "metadatas", "distances"]
            )
            print(f"Chroma query completed in {time.time() - query_start:.2f} seconds")
            
            # Check if we got any results
            if not results['documents'] or not results['documents'][0]:
                return json.dumps({"answer": "Error: No relevant documents found for the query."}, indent=2, ensure_ascii=False)
            
            # Format context from results
            context = "\n\n".join(results['documents'][0])
            
            # Use LLM to generate response with custom prompt
            llm_start = time.time()
            prompt = SYSTEM_PROMPT_QUERY.format(query_str=questions[0], context_str=context)
            response = gemini_llm.complete(prompt)
            print(f"LLM response generated in {time.time() - llm_start:.2f} seconds")
            
            # Parse the response
            if hasattr(response, "text"):
                response_text = response.text.strip()
            else:
                response_text = str(response).strip()
            
            # Try to parse as JSON first
            try:
                # Remove any markdown code blocks if present
                if response_text.startswith('```'):
                    start_marker = response_text.find('```')
                    end_marker = response_text.rfind('```')
                    if start_marker != end_marker:
                        content = response_text[start_marker:end_marker]
                        if content.startswith('```json'):
                            content = content[7:]
                        else:
                            content = content[3:]
                        response_text = content.strip()
                
                # Handle case where response starts with newline and "answer"
                if response_text.startswith('\n'):
                    response_text = response_text.lstrip()
                
                # Try to find JSON object in the response
                if response_text.startswith('{') and response_text.endswith('}'):
                    parsed_response = json.loads(response_text)
                    if isinstance(parsed_response, dict) and "answer" in parsed_response:
                        # Return the complete JSON response
                        return json.dumps(parsed_response, indent=2, ensure_ascii=False)
                    else:
                        # If it's valid JSON but doesn't have "answer" field, wrap it
                        return json.dumps({"answer": response_text}, indent=2, ensure_ascii=False)
                else:
                    # If it doesn't look like JSON, wrap it in JSON format
                    return json.dumps({"answer": response_text}, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                # If not valid JSON, wrap the raw response in JSON format
                return json.dumps({"answer": response_text}, indent=2, ensure_ascii=False)
        
        else:
            # Batch questions - use combined approach
            embedding_start = time.time()
            # Create a combined query from all questions
            combined_query = " ".join(questions)
            query_embedding = GoogleEmbeddings.get_text_embedding(combined_query)
            print(f"Batch query embedding generated in {time.time() - embedding_start:.2f} seconds")
            
            # Query Chroma with more results for batch processing
            query_start = time.time()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=20,  # More results for batch processing
                include=["documents", "metadatas", "distances"]
            )
            print(f"Batch Chroma query completed in {time.time() - query_start:.2f} seconds")
            
            # Check if we got any results
            if not results['documents'] or not results['documents'][0]:
                return json.dumps({"answers": ["Error: No relevant documents found for the query."] * len(questions)}, indent=2, ensure_ascii=False)
            
            # Format context from results
            context = "\n\n".join(results['documents'][0])
            
            # Format questions for the prompt
            questions_text = "\n".join([f"{i+1}. {question}" for i, question in enumerate(questions)])
            
            # Use LLM to generate response with batch prompt
            llm_start = time.time()
            prompt = SYSTEM_PROMPT_BATCH.format(query_list=questions_text, context_str=context)
            response = gemini_llm.complete(prompt)
            print(f"Batch LLM response generated in {time.time() - llm_start:.2f} seconds")
            
            # Parse the response
            if hasattr(response, "text"):
                response_text = response.text.strip()
            else:
                response_text = str(response).strip()
            
            print(f"Raw LLM response length: {len(response_text)} characters")
            print(f"Response starts with: {response_text[:50]}")
            print(f"Response ends with: {response_text[-50:]}")
            
            # Enhanced JSON parsing with error recovery
            def parse_llm_response(response_text, questions_count):
                """Parse LLM response with enhanced error handling and recovery"""
                try:
                    # Clean up common issues
                    response_text = response_text.strip()
                    
                    print(f"Parsing response for {questions_count} questions")
                    print(f"Response length: {len(response_text)} characters")
                    
                    # Remove any markdown code blocks if present
                    if response_text.startswith('```'):
                        start_marker = response_text.find('```')
                        end_marker = response_text.rfind('```')
                        if start_marker != end_marker and start_marker != -1 and end_marker != -1:
                            content = response_text[start_marker:end_marker]
                            if content.startswith('```json'):
                                content = content[7:]
                            else:
                                content = content[3:]
                            response_text = content.strip()
                    
                    # Handle case where response starts with newline
                    if response_text.startswith('\n'):
                        response_text = response_text.lstrip()
                    
                    # Try to fix common JSON issues
                    if response_text.startswith('{') and not response_text.endswith('}'):
                        # Try to find if there's an incomplete JSON
                        # Look for the last complete answer in the array
                        if '"answers"' in response_text and '[' in response_text:
                            try:
                                # Find the start of the answers array
                                answers_start = response_text.find('"answers"')
                                if answers_start != -1:
                                    bracket_start = response_text.find('[', answers_start)
                                    if bracket_start != -1:
                                        # Try to find complete answers
                                        answers_section = response_text[bracket_start+1:]
                                        
                                        # Extract complete quoted strings
                                        answers = []
                                        current_pos = 0
                                        while len(answers) < questions_count and current_pos < len(answers_section):
                                            # Find next quoted string
                                            quote_start = answers_section.find('"', current_pos)
                                            if quote_start == -1:
                                                break
                                            
                                            quote_end = quote_start + 1
                                            # Find the closing quote, handling escaped quotes
                                            while quote_end < len(answers_section):
                                                if answers_section[quote_end] == '"' and answers_section[quote_end-1] != '\\':
                                                    break
                                                quote_end += 1
                                            
                                            if quote_end < len(answers_section):
                                                answer = answers_section[quote_start+1:quote_end]
                                                # Unescape quotes
                                                answer = answer.replace('\\"', '"')
                                                answers.append(answer)
                                                current_pos = quote_end + 1
                                            else:
                                                break
                                        
                                        # Fill missing answers
                                        while len(answers) < questions_count:
                                            answers.append("Error: Incomplete response from LLM")
                                        
                                        return {"answers": answers[:questions_count]}
                            except Exception as extract_error:
                                print(f"Error extracting answers from incomplete JSON: {extract_error}")
                    
                    # Try to parse as complete JSON
                    if response_text.startswith('{') and response_text.endswith('}'):
                        try:
                            parsed_response = json.loads(response_text)
                            if isinstance(parsed_response, dict) and "answers" in parsed_response:
                                answers = parsed_response["answers"]
                                # Ensure we have the right number of answers
                                while len(answers) < questions_count:
                                    answers.append("Error: No answer provided for this question")
                                return {"answers": answers[:questions_count]}
                            elif isinstance(parsed_response, dict) and "answer" in parsed_response:
                                # Handle old format with single "answer" field
                                return {"answers": [parsed_response["answer"]]}
                        except json.JSONDecodeError as json_error:
                            print(f"JSON parsing failed: {json_error}")
                            # Try to fix common JSON issues like trailing commas
                            try:
                                import re
                                # Remove trailing commas before closing brackets/braces
                                fixed_json = re.sub(r',(\s*[}\]])', r'\1', response_text)
                                parsed_response = json.loads(fixed_json)
                                if isinstance(parsed_response, dict) and "answers" in parsed_response:
                                    answers = parsed_response["answers"]
                                    while len(answers) < questions_count:
                                        answers.append("Error: No answer provided for this question")
                                    return {"answers": answers[:questions_count]}
                            except:
                                pass  # Fall through to other recovery methods
                    
                    # Try to extract JSON from text that has extra content
                    if '{' in response_text and '}' in response_text:
                        try:
                            # Find the first { and last } to extract potential JSON
                            start_brace = response_text.find('{')
                            end_brace = response_text.rfind('}')
                            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                                potential_json = response_text[start_brace:end_brace+1]
                                try:
                                    parsed_response = json.loads(potential_json)
                                    if isinstance(parsed_response, dict) and "answers" in parsed_response:
                                        answers = parsed_response["answers"]
                                        while len(answers) < questions_count:
                                            answers.append("Error: No answer provided for this question")
                                        return {"answers": answers[:questions_count]}
                                except:
                                    pass
                        except:
                            pass
                    
                    # If we can't parse JSON, try to extract meaningful content
                    print(f"Could not parse JSON, attempting content extraction from: {response_text[:200]}...")
                    
                    # Try to find any quoted strings that might be answers
                    import re
                    quoted_strings = re.findall(r'"([^"]*)"', response_text)
                    if quoted_strings:
                        # Filter out field names and keep potential answers
                        potential_answers = [s for s in quoted_strings if s not in ['answers', 'answer'] and len(s) > 10]
                        if potential_answers:
                            # Use the extracted answers, pad if needed
                            while len(potential_answers) < questions_count:
                                potential_answers.append("Error: Incomplete response extracted")
                            return {"answers": potential_answers[:questions_count]}
                    
                    # Last resort: return error messages
                    return {"answers": ["Error: Invalid response format - could not parse LLM output"] * questions_count}
                
                except Exception as e:
                    print(f"Error in response parsing: {e}")
                    return {"answers": ["Error: Response parsing failed"] * questions_count}
            
            # Use the enhanced parsing function
            parsed_result = parse_llm_response(response_text, len(questions))
            
            total_time = time.time() - start_time
            print(f"Total query processing completed in {total_time:.2f} seconds")
            return json.dumps(parsed_result, indent=2, ensure_ascii=False)
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"Error in query processing after {total_time:.2f} seconds: {e}")
        if is_single:
            return json.dumps({"answer": f"Error querying Chroma: {str(e)}"}, indent=2, ensure_ascii=False)
        else:
            return json.dumps({"answers": [f"Error querying Chroma: {str(e)}"] * len(questions)}, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    # Test single question
    # print("\n" + "="*80)
    # question1 = 'What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?'
    # answer1 = ask_question(question1)
    # print("Single Question Response:")
    # print(answer1)
    
    # Test batch questions
    # print("\n" + "="*80)
    batch_questions = [
        "[image.jpeg] What is 100+22?",
        "[image.jpeg] What is 9+5?",
        "[image.jpeg] What is 65007+2?",
        "[image.jpeg] What is 1+1?",
        "[image.jpeg] What is 5+500?"
    ]
    batch_answer = ask_question(batch_questions)
    print("Batch Questions Response:")
    print(batch_answer)
