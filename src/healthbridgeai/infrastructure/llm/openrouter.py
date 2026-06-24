# modules/llm_handler.py: Multilingual LLM handler for Nigerian languages
import logging
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

from modules.language_utils import (
    detect_language, is_pidgin, get_language_name
)

from config import OPENROUTER_API_KEY, LLM_MODEL, MIN_SCORE

logger = logging.getLogger(__name__)


# ------------------- LLM Initialization -------------------

def get_llm():
    """Initialize the LLM with optimized settings for multilingual use."""
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set")
        raise ValueError("OPENROUTER_API_KEY environment variable is missing")

    return ChatOpenAI(
        model=LLM_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.2,
        max_tokens=600,
        timeout=35,
    )


# ------------------- Translation -------------------

def translate_to_english(text: str, source_lang: str) -> str:
    """Translate Nigerian languages to English for KB search."""
    if source_lang == "en":
        return text

    try:
        llm = get_llm()
        translation_prompts = {
            "yo": f"Translate this Yoruba health question to clear English. Preserve the meaning exactly.\n\nYoruba: {text}\n\nEnglish translation:",
            "ig": f"Translate this Igbo health question to clear English. Preserve the meaning exactly.\n\nIgbo: {text}\n\nEnglish translation:",
            "ha": f"Translate this Hausa health question to clear English. Preserve the meaning exactly.\n\nHausa: {text}\n\nEnglish translation:",
            "pidgin": f"Translate this Nigerian Pidgin health question to standard English. Preserve the meaning exactly.\n\nPidgin: {text}\n\nEnglish translation:"
        }
        prompt = translation_prompts.get(source_lang, f"Translate to English: {text}")
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Translation from {source_lang} failed: {e}")
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate English responses back to Nigerian languages with proper standardized language."""
    if target_lang == "en":
        return text

    try:
        llm = get_llm()
        
        # Enhanced prompts for proper, standardized language output
        response_prompts = {
            "yo": f"""Translate this English health information to STANDARD Yoruba language.
Use proper Yoruba diacritics (à, á, è, é, ì, í, ò, ó, ù, ú, ẹ, ọ, ṣ).
Make sure the Yoruba is grammatically correct and easily understandable.
Use common, everyday Yoruba words that ordinary people use.

English: {text}

Standard Yoruba translation:""",
            
            "ig": f"""Translate this English health information to STANDARD Igbo language.
Use proper Igbo orthography and diacritics where needed.
Make sure the Igbo is grammatically correct and easily understandable.
Use common, everyday Igbo words that ordinary people use.

English: {text}

Standard Igbo translation:""",
            
            "ha": f"""Translate this English health information to STANDARD Hausa language.
Use proper Hausa spelling and orthography.
Make sure the Hausa is grammatically correct and easily understandable.
Use common, everyday Hausa words that ordinary people use.

English: {text}

Standard Hausa translation:""",
            
            "pidgin": f"""Convert this English health information to STANDARD Nigerian Pidgin.
Make it conversational but clear and accurate.
Use proper Nigerian Pidgin that is widely understood across Nigeria.
Keep medical terms simple and explain them in Pidgin.

English: {text}

Nigerian Pidgin translation:"""
        }
        
        prompt = response_prompts.get(target_lang, f"Translate to {get_language_name(target_lang)}: {text}")
        response = llm.invoke([HumanMessage(content=prompt)])
        translated = response.content.strip()
        
        # Log the translation for debugging
        logger.info(f"Translation to {target_lang}: {translated[:100]}...")
        
        return translated
    except Exception as e:
        logger.error(f"Translation to {target_lang} failed: {e}")
        return text


# ------------------- KB Search -------------------

def create_multilingual_search_variations(query: str, detected_lang: str) -> List[str]:
    """Create search variations for better KB retrieval across Nigerian languages."""
    variations = [query]

    if detected_lang != "en":
        english_query = translate_to_english(query, detected_lang)
        if english_query and english_query != query:
            variations.append(english_query)

    if "tb" not in query.lower() and "tuberculosis" not in query.lower():
        variations.append(f"{query} tuberculosis")

    return list(dict.fromkeys(variations))  # deduplicate, keep order

def get_qa_chain(vector_store):
    """Create RetrievalQA chain for multilingual queries."""
    llm = get_llm()

    qa_template = """You are an expert tuberculosis (TB) health assistant.
Use the provided context to give accurate, clear, and medically correct answers.
Provide specific, actionable information when possible.

If the context does not contain enough information, say so clearly.

Context: {context}
Question: {question}

Detailed Answer:"""

    QA_PROMPT = PromptTemplate(
        template=qa_template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )


def calculate_result_quality_score(result_text: str, sources: List, query: str) -> float:
    """Score KB search results for relevance and completeness.
    
    STRICT SCORING: Higher threshold ensures we only use KB when we have good answers.
    This forces fallback to web search when KB results are weak.
    """
    score = 0.0

    if not result_text or len(result_text.strip()) < 10:
        return 0.0

    # CRITICAL: Strong negative indicators mean KB doesn't have the answer
    strong_negatives = [
        "don't know", "not sure", "cannot answer", "insufficient information",
        "not enough information", "context does not contain", "i don't have",
        "unable to answer", "no information", "not available in"
    ]
    
    result_lower = result_text.lower()
    if any(ind in result_lower for ind in strong_negatives):
        logger.info(f"KB result contains negative indicator - forcing web search fallback")
        return 0.0  # Force web search by returning score below threshold

    # Length scoring (good answers have substance)
    text_length = len(result_text.strip())
    if 100 <= text_length <= 500:
        score += 4.0  # Increased for substantial answers
    elif 50 <= text_length < 100:
        score += 2.0
    elif text_length > 500:
        score += 3.0
    elif text_length < 50:
        score += 0.5  # Very short answers are suspect

    # Source documents (KB should have sources)
    if sources and len(sources) >= 2:
        score += 3.0
    elif sources and len(sources) == 1:
        score += 1.5
    else:
        score -= 2.0  # No sources is bad

    # Positive content indicators
    score += 2.0  # Base positive score

    # TB relevance (must be TB-related)
    tb_keywords = ["tuberculosis", "tb", "infection", "treatment", "symptoms", 
                   "prevention", "lungs", "bacteria", "disease", "diagnosis"]
    tb_matches = sum(1 for kw in tb_keywords if kw.lower() in result_lower)
    
    if tb_matches >= 3:
        score += 3.0  # Strong TB relevance
    elif tb_matches >= 1:
        score += 1.5
    else:
        score -= 2.0  # Not TB-related is suspicious

    # Query overlap (answer should address the query)
    query_words = set(w.lower() for w in query.split() if len(w) > 3)
    result_words = set(w.lower() for w in result_text.split())
    overlap = query_words.intersection(result_words)
    
    if query_words:
        overlap_ratio = len(overlap) / len(query_words)
        if overlap_ratio >= 0.5:
            score += 3.0
        elif overlap_ratio >= 0.3:
            score += 1.5
        else:
            score -= 1.0

    logger.info(f"Detailed score breakdown: length={text_length}, sources={len(sources) if sources else 0}, tb_matches={tb_matches}, final_score={score:.2f}")
    
    return score


def search_kb_with_multiple_strategies(query_variations: List[str], vector_store) -> Dict[str, Any]:
    """Try multiple KB search strategies and return the best scored result.
    
    STRICT MODE: Only returns results that meet quality threshold.
    """
    best_result = None
    best_score = 0

    for variation in query_variations:
        try:
            qa_chain = get_qa_chain(vector_store)
            response = qa_chain.invoke({"query": variation})
            result_text = response.get("result", "")
            sources = response.get("source_documents", [])

            score = calculate_result_quality_score(result_text, sources, variation)
            logger.info(f"KB search score for '{variation[:50]}...': {score:.2f}")

            if score > best_score:
                best_score = score
                best_result = {
                    "result": result_text,
                    "source_documents": sources,
                    "query_used": variation,
                    "score": score
                }
        except Exception as e:
            logger.warning(f"KB search failed for '{variation}': {e}")
            continue

    return best_result


# ------------------- Web Search Fallback -------------------

def web_search_fallback(query: str, target_lang: str) -> str:
    """IMPROVED: Real-time web search fallback using DuckDuckGo for TB info.
    
    Uses updated API wrapper for reliable real-time search results.
    """
    try:
        logger.info(f"Starting web search for query: '{query[:50]}...'")
        
        # Use the API wrapper for more reliable results
        wrapper = DuckDuckGoSearchAPIWrapper(
            max_results=5,
            region="wt-wt",  # Worldwide
            safesearch="moderate",
            time="y"  # Past year for recent information
        )
        
        # Enhanced search query for TB-specific medical sources
        search_terms = f"tuberculosis {query} site:who.int OR site:cdc.gov OR site:nhs.uk OR site:mayoclinic.org"
        
        logger.info(f"Search terms: {search_terms}")
        
        # Perform the search
        results = wrapper.run(search_terms)
        
        if not results or len(results.strip()) < 20:
            logger.warning("Web search returned insufficient results, trying broader search")
            # Try broader search without site restrictions
            search_terms = f"tuberculosis {query} health medical"
            results = wrapper.run(search_terms)
        
        if not results or len(results.strip()) < 20:
            logger.error("Web search failed to return usable results")
            return "I couldn't find current information online. Please consult a healthcare professional for the most accurate information."

        logger.info(f"Web search returned {len(results)} characters of results")

        # Summarize the search results using LLM
        llm = get_llm()
        summary_prompt = f"""You are a tuberculosis (TB) health expert. Based on the following search results, 
provide a clear, accurate, and helpful answer to the user's question.

User's Question: {query}

Search Results:
{results}

Instructions:
- Provide factual, medically accurate information
- Be specific and actionable when possible
- Keep the response between 100-300 words
- Focus on tuberculosis-related information
- Do not invent information not in the search results
- If the results don't contain relevant TB information, say so

Answer:"""
        
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary = response.content.strip()
        
        logger.info(f"Generated summary: {summary[:100]}...")

        # Translate if needed
        if target_lang != "en":
            logger.info(f"Translating web search result to {get_language_name(target_lang)}")
            summary = translate_from_english(summary, target_lang)

        return summary
        
    except Exception as e:
        logger.error(f"Web search fallback failed: {e}", exc_info=True)
        return "Unable to fetch current information at this time. Please try again or consult a healthcare professional."


# ------------------- Main Response -------------------

def get_response(query: str, vector_store, lang) -> Dict[str, Any]:
    """Main function to get multilingual responses.
    
    STRICT CONTROL FLOW:
    1. ALWAYS try vector store FIRST
    2. ONLY use web search if vector store score < MIN_SCORE
    3. NEVER skip vector store
    """
    try:
        # Auto-detect language if lang="auto"
        if lang == "auto":
            detected_lang = detect_language(query)
            logger.info(f"Auto-detected language: {detected_lang} from query: '{query[:50]}...'")
        else:
            detected_lang = lang
        
        # Check for Pidgin specifically (but don't override English detection)
        if detected_lang == "en" and is_pidgin(query):
            detected_lang = "pidgin"
        
        # Ensure we have a valid language code
        valid_langs = ["en", "yo", "ig", "ha", "pidgin"]
        if detected_lang not in valid_langs:
            detected_lang = "en"  # Default to English if unknown
        
        # The target language for response should match the query language
        target_lang = detected_lang
        
        logger.info(f"=" * 70)
        logger.info(f"NEW QUERY PROCESSING")
        logger.info(f"Query: '{query[:100]}...'")
        logger.info(f"Detected language: {detected_lang} ({get_language_name(detected_lang)})")
        logger.info(f"Target response language: {target_lang}")
        logger.info(f"=" * 70)

        # STEP 1: ALWAYS search knowledge base FIRST
        logger.info("STEP 1: Searching vector store (knowledge base)...")
        search_variations = create_multilingual_search_variations(query, lang)
        logger.info(f"Search variations: {search_variations}")
        
        kb_result = search_kb_with_multiple_strategies(search_variations, vector_store)

        # STEP 2: Evaluate KB result quality
        if kb_result:
            kb_score = kb_result["score"]
            logger.info(f"STEP 2: KB search completed. Score: {kb_score:.2f} | Threshold: {MIN_SCORE:.2f}")
        else:
            kb_score = 0.0
            logger.info(f"STEP 2: KB search returned no results")

        # STEP 3: Decision point - KB or Web Search
        if kb_result and kb_score >= MIN_SCORE:
            # USE KNOWLEDGE BASE RESULT
            logger.info(f"✓ DECISION: Using KB result (score {kb_score:.2f} >= threshold {MIN_SCORE:.2f})")
            answer = kb_result["result"].strip()
            
            # Translate the answer to the target language if needed
            if target_lang != "en":
                logger.info(f"Translating KB response from English to {get_language_name(target_lang)}")
                answer = translate_from_english(answer, target_lang)

            sources = kb_result.get("source_documents", [])
            filenames = []
            for doc in sources[:3]:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    filenames.append(doc.metadata["source"].split("/")[-1])
            source_info = "knowledge_base"
            if filenames:
                source_info += f" ({', '.join(set(filenames))})"

            logger.info(f"Final answer source: {source_info}")
            logger.info(f"=" * 70)

            return {
                "source": source_info,
                "answer": answer,
                "lang": target_lang,
                "detected_lang": lang
            }
        
        # USE WEB SEARCH FALLBACK
        logger.info(f"✗ DECISION: KB insufficient (score {kb_score:.2f} < threshold {MIN_SCORE:.2f})")
        logger.info("STEP 3: Activating web search fallback...")
        
        fallback_answer = web_search_fallback(query, target_lang)
        
        logger.info(f"Web search completed. Answer length: {len(fallback_answer)} chars")
        logger.info(f"Final answer source: internet_search")
        logger.info(f"=" * 70)

        return {
            "source": "internet_search",
            "answer": fallback_answer,
            "lang": target_lang,
            "detected_lang": lang
        }

    except Exception as e:
        logger.error(f"CRITICAL ERROR in get_response: {e}", exc_info=True)
        return {
            "source": "error",
            "answer": "An error occurred while processing your question. Please try again.",
            "lang": "en",
            "detected_lang": "en"
        }