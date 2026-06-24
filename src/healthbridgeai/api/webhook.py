import json
import flask
import asyncio
import logging
import requests
from threading import Thread
from flask import Flask, render_template, request
from modules.llm_handler import get_response
from modules.utills import parse_json, get_media_url, download_media, upload_media_to_whatsapp, get_text_message_input, send_message, send_audio_message, send_text_message, send_language_selection_menu, send_language_switch_confirmation, is_language_switch_request
from modules.audio_transcriber import get_transcriber
from modules.audio_synthesizer import get_synthesizer
from modules.vector_store_manager import get_vector_store
from modules.user_preferences import get_preference_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ======== PRE-LOAD MODELS AT STARTUP ========
# This ensures WhatsApp webhook can respond quickly with 200
# Models are loaded once and cached by singleton pattern

logger.info("🚀 Initializing audio models at startup...")
try:
    # Pre-load transcriber (N-ATLAS, Whisper will lazy-load on first use)
    transcriber = get_transcriber()
    logger.info("✓ Audio transcriber initialized")
    
    # Pre-load synthesizer
    synthesizer = get_synthesizer()
    logger.info("✓ Audio synthesizer initialized")
    
    # Pre-load vector store
    vector_store = get_vector_store()
    logger.info("✓ Vector store initialized")
    
    # Pre-load preference manager
    preference_manager = get_preference_manager()
    logger.info("✓ User preference manager initialized")
    
    logger.info("🎉 All models loaded successfully! Webhook ready for fast responses.")
except Exception as e:
    logger.error(f"⚠️  Warning: Model initialization failed: {e}")
    logger.error("App will still start, but first request may be slow")
    transcriber = None
    synthesizer = None
    vector_store = None
    preference_manager = None


@app.route('/')
def index():
    return "<h1>Welcome to the Flask App</h1>"

with open('config.json') as f:
    config = json.load(f)

print(config) 
app.config.update(config)

 

# ======== BACKGROUND MESSAGE PROCESSING ========
def process_audio_message(data: dict):
    """
    Process audio message in background thread.
    Sends audio response first (matching input format), then text.
    This keeps the webhook fast and prevents WhatsApp retries.
    """
    try:
        logger.info("🎤 Processing audio message in background...")
        
        # Get sender's WhatsApp ID (phone number)
        sender_phone = data.get('wa_id', 'unknown')
        logger.info(f"Message from: {sender_phone}")
        
        # Check if first-time user
        if preference_manager.is_first_time_user(sender_phone):
            logger.info(f"🆕 First-time user detected: {sender_phone}")
            send_language_selection_menu(sender_phone)
            logger.info("✓ Language selection menu sent to first-time user")
            return
        
        # Get user's preferred language
        user_lang = preference_manager.get_user_preference(sender_phone)
        if not user_lang:
            user_lang = "en"  # Default to English if preference not found
            logger.warning(f"⚠️  No language preference found for {sender_phone}, defaulting to English")

        logger.info(f"Using user's preferred language: {user_lang}")
        
        # Download audio
        media_id = data['audio']['id']
        json_url = get_media_url(media_id)
        file_path = download_media(json_url)
        logger.info(f"✓ Audio downloaded: {file_path}")
        
        # Transcribe using pre-loaded transcriber with user's language
        transcribed_text, detected_lang = transcriber.transcribe(file_path, user_lang)
        logger.info(f"✓ Transcription: '{transcribed_text[:100]}...' (lang={detected_lang})")
        
        # Get LLM response using user's language
        result = get_response(transcribed_text, vector_store, user_lang)
        answer = result.get("answer", "Sorry, I encountered an error processing your request.")
        logger.info(f"✓ LLM response generated: '{answer[:100]}...'")
        
        # Generate audio response first using pre-loaded synthesizer with user's language
        audio_file_path = synthesizer.synthesize(answer, user_lang)
        logger.info(f"✓ Audio response generated: {audio_file_path}")
        
        # Upload and send audio first (matches user's input format)
        upload_response = upload_media_to_whatsapp(audio_file_path)
        
        if 'id' in upload_response:
            media_object_id = upload_response['id']
            logger.info(f"✓ Media uploaded with ID: {media_object_id}")
            
            # Send audio message back to user using wa_id
            send_audio_message(media_object_id, sender_phone)
            logger.info(f"✓ Audio message sent to {sender_phone}")
            
            # Then send text version (for reference/accessibility)
            send_text_message(answer, sender_phone)
            logger.info(f"✓ Text message sent to {sender_phone}")
            
            logger.info("✅ Audio message processed - sent audio first, then text!")
        else:
            logger.error(f"❌ Failed to upload media: {upload_response.get('error')}")
            # If audio upload fails, at least send text
            send_text_message(answer, sender_phone)
            logger.info("✅ Audio message processed - text sent (audio upload failed)")
        
        # Cleanup temporary audio file
        try:
            synthesizer.cleanup_audio_file(audio_file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup audio file: {e}")
        
    except Exception as e:
        logger.error(f"❌ Error processing audio message: {e}", exc_info=True)


def process_text_message(data: dict):
    """
    Process text message in background thread.
    Sends both text AND audio response.
    This keeps the webhook fast and prevents WhatsApp retries.
    """
    try:
        logger.info("💬 Processing text message in background...")
        
        # Get sender's WhatsApp ID (phone number)
        sender_phone = data.get('wa_id', 'unknown')
        text = data['text']
        logger.info(f"Message from {sender_phone}: '{text}'")
        
        # Check if user wants to change language
        if is_language_switch_request(text):
            logger.info(f"🔄 Language switch request detected from {sender_phone}")
            send_language_selection_menu(sender_phone)
            logger.info("✓ Language selection menu sent")
            return
        
        # Check if first-time user
        if preference_manager.is_first_time_user(sender_phone):
            logger.info(f"🆕 First-time user detected: {sender_phone}")
            send_language_selection_menu(sender_phone)
            logger.info("✓ Language selection menu sent to first-time user")
            return
        
        # Get user's preferred language
        user_lang = preference_manager.get_user_preference(sender_phone)
        if not user_lang:
            user_lang = "en"  # Default to English if preference not found
            logger.warning(f"⚠️  No language preference found for {sender_phone}, defaulting to English")
        
        logger.info(f"Using user's preferred language: {user_lang}")
        
        # Get LLM response using user's language
        result = get_response(text, vector_store, user_lang)
        answer = result.get("answer", "Sorry, I encountered an error processing your request.")
        logger.info(f"✓ LLM response generated: '{answer[:100]}...'")
        
        # Send text response first (instant delivery)
        send_text_message(answer, sender_phone)
        logger.info(f"✓ Text message sent to {sender_phone}")
        
        # Generate audio response using user's language
        try:
            audio_file_path = synthesizer.synthesize(answer, user_lang)
            logger.info(f"✓ Audio response generated: {audio_file_path}")
            
            # Upload audio to WhatsApp
            upload_response = upload_media_to_whatsapp(audio_file_path)
            
            if 'id' in upload_response:
                media_object_id = upload_response['id']
                logger.info(f"✓ Media uploaded with ID: {media_object_id}")
                
                # Send audio message back to user
                send_audio_message(media_object_id, sender_phone)
                logger.info(f"✓ Audio message sent to {sender_phone}")
                
                logger.info("✅ Text message processed - sent both text and audio!")
            else:
                logger.error(f"❌ Failed to upload media: {upload_response.get('error')}")
                logger.info("✅ Text message processed - text sent (audio upload failed)")
            
            # Cleanup temporary audio file
            try:
                synthesizer.cleanup_audio_file(audio_file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup audio file: {e}")
                
        except Exception as audio_error:
            logger.warning(f"⚠️ Audio generation failed, text was sent: {audio_error}")
            logger.info("✅ Text message processed - text sent (audio generation failed)")
        
    except Exception as e:
        logger.error(f"❌ Error processing text message: {e}", exc_info=True)


def process_interactive_button(data: dict):
    """
    Process interactive button clicks in background thread.
    Saves user's language preference and sends confirmation.
    """
    try:
        logger.info("🔘 Processing interactive button in background...")
        
        # Get sender's WhatsApp ID (phone number)
        sender_phone = data.get('wa_id', 'unknown')
        button_id = data.get('button_id')
        button_title = data.get('button_title')
        
        logger.info(f"Button click from {sender_phone}: {button_id} ({button_title})")
        
        # Extract language code from button_id (format: "lang_xx")
        if button_id and button_id.startswith("lang_"):
            lang_code = button_id.replace("lang_", "")
            
            # Validate language code
            valid_langs = ["ha", "yo", "ig", "en"]
            if lang_code in valid_langs:
                # Save preference
                success = preference_manager.set_user_preference(sender_phone, lang_code)
                
                if success:
                    logger.info(f"✓ Saved language preference: {sender_phone} -> {lang_code}")
                    
                    # Send confirmation in selected language
                    send_language_switch_confirmation(sender_phone, button_id)
                    logger.info(f"✓ Confirmation sent to {sender_phone}")
                else:
                    logger.error(f"❌ Failed to save preference for {sender_phone}")
                    send_text_message("Sorry, there was an error saving your preference. Please try again.", sender_phone)
            else:
                logger.warning(f"⚠️  Invalid language code: {lang_code}")
                send_text_message("Invalid language selection. Please try again.", sender_phone)
        else:
            logger.warning(f"⚠️  Invalid button_id format: {button_id}")
            send_text_message("Invalid button response. Please try again.", sender_phone)
        
        logger.info("✅ Interactive button processed!")
        
    except Exception as e:
        logger.error(f"❌ Error processing interactive button: {e}", exc_info=True)


# ======== WEBHOOK PROCESS ========
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    """
    WhatsApp webhook endpoint.
    Returns 200 immediately to prevent retries, processes messages in background.
    """
    if request.method == 'GET':
        # Verification (used by Meta)
        verify_token = app.config['VERIFY_TOKEN']
        if request.args.get('hub.verify_token') == verify_token:
            return request.args.get('hub.challenge')
        return 'Verification failed', 403

    if request.method == 'POST':
        logger.info("📨 Webhook received POST request")
        
        try:
            data = parse_json(request.get_json())
            
            if not data:
                logger.warning("⚠️  Received empty or invalid data")
                return 'Event received', 200
            
            # Process message in background thread to return 200 immediately
            if data.get('type') == 'audio':
                logger.info("🎤 Audio message detected, processing in background...")
                thread = Thread(target=process_audio_message, args=(data,))
                thread.daemon = True
                thread.start()
                
            elif data.get('type') == 'text':
                logger.info("💬 Text message detected, processing in background...")
                thread = Thread(target=process_text_message, args=(data,))
                thread.daemon = True
                thread.start()
                
            elif data.get('type') == 'interactive':
                logger.info("🔘 Interactive button clicked, processing in background...")
                thread = Thread(target=process_interactive_button, args=(data,))
                thread.daemon = True
                thread.start()
                
            else:
                logger.info(f"ℹ️  Unsupported message type: {data.get('type')}")
            
        except Exception as e:
            logger.error(f"❌ Error in webhook: {e}", exc_info=True)
        
        # CRITICAL: Always return 200 immediately to prevent WhatsApp retries
        return 'Event received', 200
