import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
from openai import OpenAI
import tempfile
import asyncio
from pydub import AudioSegment
from rapidfuzz import fuzz
from typing import Dict, Optional
from collections import deque
import re
import unicodedata
import json
from datetime import datetime

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ENROLLMENT_FILE = 'enrollments.json'
COURSES_FILE = 'courses.json'


# Language codes and names mapping
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ru': 'Russian'
}

class UserSettings:
    def __init__(self):
        self.target_phrase: str = ""
        self.language: str = "en"
        self.original_phrase: str = ""  # Store original phrase
        self.translated_phrase: str = ""  
        self.last_score: float = 0.0
        self.attempts: int = 0
        self.conversation_history = deque(maxlen=10)
        self.show_normalized: bool = False
        self.current_session_id: int = 0

class CourseManager:
    def __init__(self):
        self.enrollments = self.load_enrollments()
        self.courses = self.load_courses()
    
    def load_enrollments(self) -> dict:
        """Load enrollments from JSON file"""
        if os.path.exists(ENROLLMENT_FILE):
            try:
                with open(ENROLLMENT_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def save_enrollments(self):
        """Save enrollments to JSON file"""
        with open(ENROLLMENT_FILE, 'w') as f:
            json.dump(self.enrollments, f, indent=2)
    
    def load_courses(self) -> dict:
        """Load course data from JSON file"""
        if os.path.exists(COURSES_FILE):
            with open(COURSES_FILE, 'r') as f:
                return json.load(f)
        else:
            # Default course structure
            default_courses = {
                "en": {
                    "name": "English Course",
                    "language": "en",
                    "sessions": [
                        {
                            "id": 1,
                            "name": "Basics 1",
                            "words": ["hello", "goodbye", "thank you"],
                            "completed": False
                        },
                        {
                            "id": 2,
                            "name": "Basics 2",
                            "words": ["please", "excuse me", "sorry"],
                            "completed": False
                        }
                    ]
                },
                # Add more language courses here
            }
            # Save default courses to file
            with open(COURSES_FILE, 'w') as f:
                json.dump(default_courses, f, indent=2)
            return default_courses
        
    def get_current_session(self, user_id: str) -> Optional[dict]:
        """Get user's current session"""
        enrollment = self.get_user_enrollment(user_id)
        if not enrollment:
            return None
        
        language_code = enrollment["language_code"]
        current_session_id = enrollment["current_session"]
        
        if language_code not in self.courses:
            return None
            
        course = self.courses[language_code]
        if current_session_id >= len(course["sessions"]):
            return None
            
        return course["sessions"][current_session_id]
    
    
    def enroll_user(self, user_id: str, language_code: str) -> dict:
        """Enroll a user in a course"""
        if language_code not in self.courses:
            raise ValueError(f"No course available for language code: {language_code}")
        
        enrollment = {
            "user_id": user_id,
            "language_code": language_code,
            "enrolled_at": datetime.now().isoformat(),
            "current_session": 0,
            "completed_sessions": [],
            "progress": {
                "total_practice_sessions": 0,
                "total_words_practiced": 0,
                "average_score": 0.0,
                "last_practice": None
            }
        }
        
        self.enrollments[user_id] = enrollment
        self.save_enrollments()
        return enrollment
    
    def get_user_enrollment(self, user_id: str) -> Optional[dict]:
        """Get user's enrollment data"""
        return self.enrollments.get(user_id)
    
    def update_progress(self, user_id: str, session_id: int, score: float):
        """Update user's progress"""
        if user_id in self.enrollments:
            enrollment = self.enrollments[user_id]
            progress = enrollment["progress"]
            
            # Update statistics
            progress["total_practice_sessions"] += 1
            progress["total_words_practiced"] += 1
            progress["last_practice"] = datetime.now().isoformat()
            
            # Update average score
            old_avg = progress["average_score"]
            old_total = progress["total_practice_sessions"] - 1
            progress["average_score"] = (old_avg * old_total + score) / progress["total_practice_sessions"]
            
            # Update completed sessions if score is good enough
            if score >= 80 and session_id not in enrollment["completed_sessions"]:
                enrollment["completed_sessions"].append(session_id)
            
            self.save_enrollments()

class CustomBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.all()
        super().__init__(command_prefix='!', intents=intents)
        
        # Initialize OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.course_manager = CourseManager()
        
        # User settings storage
        self.user_settings: Dict[int, UserSettings] = {}
    
    def get_user_settings(self, user_id: int) -> UserSettings:
        if user_id not in self.user_settings:
            self.user_settings[user_id] = UserSettings()
        return self.user_settings[user_id]
    
    def get_current_session(self, user_id: str) -> Optional[dict]:
        """Get the current session for a user"""
        return self.course_manager.get_current_session(user_id)

    async def get_course_progress(self, user_id: str) -> discord.Embed:
        """Generate a progress report embed"""
        enrollment = self.course_manager.get_user_enrollment(user_id)
        if not enrollment:
            return discord.Embed(
                title="Not Enrolled",
                description="You're not enrolled in any course. Use !setlang to get started!",
                color=discord.Color.red()
            )
        
        progress = enrollment["progress"]
        language = SUPPORTED_LANGUAGES[enrollment["language_code"]]
        
        embed = discord.Embed(
            title=f"{language} Course Progress",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="Sessions Completed",
            value=f"{len(enrollment['completed_sessions'])}/{len(self.course_manager.courses[enrollment['language_code']]['sessions'])}",
            inline=True
        )
        
        embed.add_field(
            name="Total Practice Sessions",
            value=str(progress["total_practice_sessions"]),
            inline=True
        )
        
        embed.add_field(
            name="Average Score",
            value=f"{progress['average_score']:.1f}%",
            inline=True
        )
        
        if progress["last_practice"]:
            last_practice = datetime.fromisoformat(progress["last_practice"])
            embed.add_field(
                name="Last Practice",
                value=last_practice.strftime("%Y-%m-%d %H:%M"),
                inline=True
            )
        
        return embed
    
    async def generate_tts(self, text: str) -> str:
        """Generate TTS audio using OpenAI's API"""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            
            filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            response.stream_to_file(filename)
            return filename
        except Exception as e:
            print(f"Error generating TTS: {e}")
            return None

# Initialize bot
bot = CustomBot()

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.event
async def on_message(message):
    # Ignore bot's own messages
    if message.author == bot.user:
        return
    
    # Handle direct messages that aren't commands
    if isinstance(message.channel, discord.DMChannel) and not message.content.startswith('!'):
        try:
            async with message.channel.typing():
                response = bot.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": message.content}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                
                reply = response.choices[0].message.content

                if len(reply) > 2000:
                    chunks = [reply[i:i+2000] for i in range(0, len(reply), 2000)]
                    for chunk in chunks:
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(reply)
        except Exception as e:
            await message.channel.send(f"Sorry, I encountered an error: {str(e)}")
    
    await bot.process_commands(message)

# Existing commands
@bot.command()
async def ping(ctx):
    await ctx.send(f'Pong! Latency: {round(bot.latency * 1000)}ms')

@bot.command()
async def echo(ctx, *, message):
    await ctx.send(message)

@bot.command()
async def serverinfo(ctx):
    server = ctx.guild
    member_count = len(server.members)
    embed = discord.Embed(title=f"{server.name} Info", color=0x00ff00)
    embed.add_field(name="Server ID", value=server.id, inline=True)
    embed.add_field(name="Member Count", value=member_count, inline=True)
    embed.add_field(name="Created On", value=server.created_at.strftime("%b %d, %Y"), inline=True)
    await ctx.send(embed=embed)

@bot.command()
async def chat(ctx, *, message):
    try:
        async with ctx.typing():
            response = bot.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": message}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            reply = response.choices[0].message.content

            if len(reply) > 2000:
                chunks = [reply[i:i+2000] for i in range(0, len(reply), 2000)]
                for chunk in chunks:
                    await ctx.send(chunk)
            else:
                await ctx.send(reply)
    except Exception as e:
        await ctx.send(f"Sorry, I encountered an error: {str(e)}")

# New pronunciation-related commands
@bot.command()
async def languages(ctx):
    """List all supported languages"""
    embed = discord.Embed(
        title="Supported Languages",
        description="Here are all the supported languages and their codes:",
        color=discord.Color.blue()
    )
    
    lang_list = "\n".join([f"`{code}` - {name}" for code, name in SUPPORTED_LANGUAGES.items()])
    embed.add_field(name="Available Languages", value=lang_list, inline=False)
    
    await ctx.send(embed=embed)

@bot.command()
async def practice(ctx):
    """Practice the current session"""
    settings = bot.get_user_settings(ctx.author.id)
    session = bot.get_current_session(str(ctx.author.id))
    
    if not session:
        await ctx.send("Please start learning first with !start")
        return

    # Send instructions with an embed
    embed = discord.Embed(
        title=f"Practice Session: {session['name']}",
        description="Let's practice pronunciation! I'll play each word, then you can repeat it.",
        color=discord.Color.blue()
    )
    await ctx.send(embed=embed)
    
    # Practice each word
    for word in session['words']:
        # Generate and send TTS
        await ctx.send(f"📢 Listen to: **{word}**")
        audio_file = await bot.generate_tts(word)
        
        if audio_file:
            await ctx.send(file=discord.File(audio_file))
            os.remove(audio_file)
            
            # Wait for user's pronunciation
            await ctx.send("🎤 Now you try! Send a voice message with your attempt.")
            
            try:
                def check(message):
                    return (message.author == ctx.author and 
                           len(message.attachments) > 0 and 
                           message.attachments[0].filename.endswith(('.mp3', '.wav', '.ogg')))
                
                msg = await bot.wait_for('message', timeout=30.0, check=check)
                
                # Process the pronunciation
                status_msg = await ctx.send("Analyzing your pronunciation... 🎧")
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                await msg.attachments[0].save(temp_file.name)
                
                # Check pronunciation
                result = await bot.check_pronunciation(
                    temp_file.name,
                    word,
                    settings.language
                )
                
                if result:
                    # Create feedback embed
                    feedback = discord.Embed(
                        title="Pronunciation Feedback",
                        color=get_score_color(result['similarity'])
                    )
                    
                    feedback.add_field(
                        name="Your Pronunciation",
                        value=f"```{result['transcribed']}```",
                        inline=False
                    )
                    
                    feedback.add_field(
                        name="Accuracy Score",
                        value=f"{result['similarity']:.1f}%",
                        inline=True
                    )
                    
                    await status_msg.delete()
                    await ctx.send(embed=feedback)
                
                os.unlink(temp_file.name)
                
            except asyncio.TimeoutError:
                await ctx.send("⏰ Time's up! Let's move to the next word.")
                continue

@bot.command()
async def start(ctx):
    """Start the language learning process"""
    available_languages = ", ".join(SUPPORTED_LANGUAGES.values())
    embed = discord.Embed(
        title="Welcome to Language Learning!",
        description=f"Available languages: {available_languages}\n\nUse `!setlang <code>` to choose your language.",
        color=discord.Color.green()
    )
    await ctx.send(embed=embed)

@bot.command()
async def setlang(ctx, language_code: str):
    """Set the language to enroll in course"""
    language_code = language_code.lower()
    if language_code not in SUPPORTED_LANGUAGES:
        supported_codes = ", ".join(f"`{code}`" for code in SUPPORTED_LANGUAGES.keys())
        await ctx.send(f"Unsupported language code. Please use one of: {supported_codes}")
        return
        
    try:
        # Enroll user in course
        enrollment = bot.course_manager.enroll_user(str(ctx.author.id), language_code)
        
        # Create welcome embed
        embed = discord.Embed(
            title=f"Welcome to {SUPPORTED_LANGUAGES[language_code]}!",
            description="You've been enrolled in the course. Here's what you can do:",
            color=discord.Color.green()
        )
        
        embed.add_field(
            name="Start Learning",
            value="Use `!practice` to begin your first lesson",
            inline=False
        )
        
        embed.add_field(
            name="Check Progress",
            value="Use `!progress` to see your course progress",
            inline=False
        )
        
        embed.add_field(
            name="Need Help?",
            value="Use `!help` to see all available commands",
            inline=False
        )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"Error enrolling in course: {str(e)}")


async def translate_phrase(phrase: str, target_lang: str, source_lang: str, client: OpenAI) -> tuple[str, str]:
    """
    Translate phrase using GPT and return both romanized and native script versions
    """
    prompt = f"""Translate this phrase from {SUPPORTED_LANGUAGES.get(source_lang, 'English')} to {SUPPORTED_LANGUAGES[target_lang]}:

Original: "{phrase}"

Provide the translation in this exact format:
NATIVE: [translation in native script]
ROMAN: [romanized version if applicable]

For languages using Latin alphabet, both NATIVE and ROMAN should be identical.
Ensure the translation sounds natural in the target language."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional translator. Provide accurate, natural-sounding translations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        translation_text = response.choices[0].message.content
        
        # Extract native and romanized versions
        native_match = translation_text.find("NATIVE: ")
        roman_match = translation_text.find("ROMAN: ")
        
        if native_match != -1:
            native_text = translation_text[native_match:roman_match if roman_match != -1 else None].replace("NATIVE: ", "").strip()
        else:
            native_text = phrase  # Fallback to original if no translation found
            
        if roman_match != -1:
            roman_text = translation_text[roman_match:].replace("ROMAN: ", "").strip()
        else:
            roman_text = native_text  # For languages using Latin alphabet
            
        return native_text, roman_text
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return phrase, phrase  # Return original phrase if translation fails

@bot.command()
async def progress(ctx):
    """Show user's course progress"""
    embed = await bot.get_course_progress(str(ctx.author.id))
    await ctx.send(embed=embed)

@bot.command()
async def setphrase(ctx, *, phrase: str):
    """Set the target phrase for pronunciation practice with automatic translation"""
    settings = bot.get_user_settings(ctx.author.id)
    
    # If no language is set, default to English
    if not settings.language:
        settings.language = "en"
    
    try:
        async with ctx.typing():
            # Detect source language (assuming English if not specified)
            source_lang = "en"
            
            # Only translate if target language is different from source
            if settings.language != source_lang:
                native_translation, roman_translation = await translate_phrase(
                    phrase,
                    settings.language,
                    source_lang,
                    bot.client
                )
                
                settings.original_phrase = phrase
                settings.target_phrase = native_translation.lower()
                
                # Create response embed
                embed = discord.Embed(
                    title="Phrase Set for Pronunciation Practice",
                    color=discord.Color.green()
                )
                
                embed.add_field(
                    name="Original Phrase",
                    value=f"```{phrase}```",
                    inline=False
                )
                
                embed.add_field(
                    name=f"Translation ({SUPPORTED_LANGUAGES[settings.language]})",
                    value=f"```{native_translation}```",
                    inline=False
                )
                
                # Add romanization if different from native script
                if roman_translation.lower() != native_translation.lower():
                    embed.add_field(
                        name="Romanized Version",
                        value=f"```{roman_translation}```",
                        inline=False
                    )
                
                embed.add_field(
                    name="Instructions",
                    value="Use the !pronounce command to practice pronunciation",
                    inline=False
                )
                
                await ctx.send(embed=embed)
                
            else:
                # For English or same language, just set the phrase directly
                settings.original_phrase = phrase
                settings.target_phrase = phrase.lower()
                await ctx.send(f"Target phrase set to: {phrase}")
                
    except Exception as e:
        await ctx.send(f"Error setting phrase: {str(e)}")

@bot.command()
async def currentphrase(ctx):
    """Show the current target phrase and its translation"""
    settings = bot.get_user_settings(ctx.author.id)
    
    if not settings.target_phrase:
        await ctx.send("No phrase is currently set. Use !setphrase to set one.")
        return
        
    embed = discord.Embed(
        title="Current Practice Phrase",
        color=discord.Color.blue()
    )
    
    if settings.original_phrase != settings.target_phrase:
        embed.add_field(
            name="Original Phrase",
            value=f"```{settings.original_phrase}```",
            inline=False
        )
        
    embed.add_field(
        name=f"Target Phrase ({SUPPORTED_LANGUAGES[settings.language]})",
        value=f"```{settings.target_phrase}```",
        inline=False
    )
    
    await ctx.send(embed=embed)

@bot.command()
async def togglenormalized(ctx):
    """Toggle display of normalized text in pronunciation assessment"""
    settings = bot.get_user_settings(ctx.author.id)
    settings.show_normalized = not settings.show_normalized
    status = "enabled" if settings.show_normalized else "disabled"
    await ctx.send(f"Normalized text display {status}.")

@bot.command()
async def pronounce(ctx):
    """Start pronunciation assessment"""
    settings = bot.get_user_settings(ctx.author.id)
    
    if not settings.target_phrase:
        await ctx.send("Please set a target phrase first using !setphrase")
        return
        
    await ctx.send(f"Please send an audio message pronouncing the phrase in {SUPPORTED_LANGUAGES[settings.language]}.")
    
    def check(message):
        return (message.author == ctx.author and 
               len(message.attachments) > 0 and 
               message.attachments[0].filename.endswith(('.mp3', '.wav', '.ogg')))
    
    try:
        msg = await bot.wait_for('message', timeout=30.0, check=check)
        await process_audio(ctx, msg.attachments[0], settings)
    except asyncio.TimeoutError:
        await ctx.send("Timed out waiting for audio. Please try again.")

def normalize_text(text: str) -> str:
    """
    Normalize text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing extra whitespace
    4. Normalizing unicode characters
    """
    # Normalize unicode characters (e.g., converting é to e)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove all punctuation except apostrophes in contractions
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Remove standalone apostrophes and keep only those in contractions
    text = re.sub(r'\s\'|\'\s|^\'|\'$', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

async def process_audio(ctx, attachment, settings: UserSettings):
    """Process the audio using OpenAI Whisper API with enhanced feedback"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    await attachment.save(temp_file.name)
    
    status_msg = await ctx.send("Processing your audio... 🎧")
    
    try:
        # Convert the audio to wav format
        audio = AudioSegment.from_file(temp_file.name)
        audio.export(temp_file.name, format="wav")
        
        # Use OpenAI Whisper API
        with open(temp_file.name, 'rb') as audio_file:
            transcript = bot.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=settings.language
            )
            
        # Normalize both transcribed and target text
        transcribed_text = transcript.text.strip()
        normalized_transcribed = normalize_text(transcribed_text)
        normalized_target = normalize_text(settings.target_phrase)
        
        # Store original texts for display
        original_transcribed = transcribed_text
        original_target = settings.target_phrase
        
        # Calculate similarity metrics using normalized text
        levenshtein_ratio = fuzz.ratio(normalized_transcribed, normalized_target) / 100
        token_sort_ratio = fuzz.token_sort_ratio(normalized_transcribed, normalized_target) / 100
        overall_similarity = (levenshtein_ratio * 0.6 + token_sort_ratio * 0.4) * 100
        
        # Get detailed feedback using GPT
        detailed_feedback = await generate_detailed_feedback(
            original_transcribed,  # Pass original text for display
            original_target,       # Pass original text for display
            normalized_transcribed,  # Pass normalized text for analysis
            normalized_target,       # Pass normalized text for analysis
            settings.language,
            overall_similarity,
            bot.client
        )
        
        # Update statistics
        settings.attempts += 1
        settings.last_score = overall_similarity
        
        # Create response embed
        response = discord.Embed(
            title=f"Pronunciation Assessment ({SUPPORTED_LANGUAGES[settings.language]})",
            color=get_score_color(overall_similarity)
        )
        
        # Show original texts in the embed
        response.add_field(
            name="Your Pronunciation",
            value=f"```{original_transcribed}```",
            inline=False
        )
        response.add_field(
            name="Target Phrase",
            value=f"```{original_target}```",
            inline=False
        )
        
        # Optional: Show normalized versions for debugging
        if settings.show_normalized:  # Add this flag to UserSettings if needed
            response.add_field(
                name="Normalized Pronunciation",
                value=f"```{normalized_transcribed}```",
                inline=False
            )
            response.add_field(
                name="Normalized Target",
                value=f"```{normalized_target}```",
                inline=False
            )
        
        response.add_field(
            name="Accuracy Score",
            value=f"{overall_similarity:.1f}%",
            inline=True
        )
        response.add_field(
            name="Attempt #",
            value=str(settings.attempts),
            inline=True
        )
        
        # Add feedback sections
        for section in detailed_feedback:
            content = section['content']
            if len(content) > 1024:
                content = content[:1021] + "..."
            response.add_field(
                name=section['title'],
                value=content,
                inline=False
            )
        
        await status_msg.delete()
        await ctx.send(embed=response)
        
    except Exception as e:
        print(f"Error in process_audio: {str(e)}")  # For debugging
        await status_msg.edit(content=f"Error processing audio: {str(e)}")
    finally:
        # Cleanup
        import os
        os.unlink(temp_file.name)

def get_score_color(similarity: float) -> discord.Color:
    """Return color based on similarity score"""
    if similarity >= 90:
        return discord.Color.green()
    elif similarity >= 75:
        return discord.Color.blue()
    elif similarity >= 60:
        return discord.Color.gold()
    else:
        return discord.Color.red()

async def generate_detailed_feedback(
    original_transcribed: str,
    original_target: str,
    normalized_transcribed: str,
    normalized_target: str,
    language: str,
    similarity: float,
    client: OpenAI
) -> list:
    """Generate detailed feedback using GPT analysis"""
    
    prompt = f"""Analyze these two phrases as a pronunciation expert:

Original Target: "{original_target}"
Original Pronunciation: "{original_transcribed}"
Language: {SUPPORTED_LANGUAGES[language]}
Similarity Score: {similarity:.1f}%

Note: Punctuation and capitalization differences should be ignored in the assessment.
Focus on pronunciation, rhythm, and intonation differences.

Provide a concise analysis in exactly these 4 sections, using the exact headers:

SUMMARY:
[2-3 sentences about the overall pronunciation quality and main differences]

DETAILED ANALYSIS:
[Analyze specific pronunciation differences, word by word]

COMMON CHALLENGES:
[List 2-3 specific challenges related to the differences found]

IMPROVEMENT TIPS:
[Provide 2-3 concrete, actionable tips for improvement]

Keep each section brief but specific to these exact phrases."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a language pronunciation expert. Provide clear, concise feedback in exactly the requested format with the exact section headers. Focus on sound patterns and pronunciation, not spelling or punctuation"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        feedback_text = response.choices[0].message.content
        
        # Initialize feedback sections
        formatted_feedback = [{
            'title': '📊 Overall Assessment',
            'content': get_quick_assessment(similarity)
        }]
        
        # Define sections to look for
        sections = {
            'SUMMARY:': '📝 Summary',
            'DETAILED ANALYSIS:': '🔍 Detailed Analysis',
            'COMMON CHALLENGES:': '⚠️ Common Challenges',
            'IMPROVEMENT TIPS:': '💡 Improvement Tips'
        }
        
        # Extract each section
        for current_header, next_header in zip(sections.keys(), list(sections.keys())[1:] + [None]):
            if current_header in feedback_text:
                start_idx = feedback_text.find(current_header) + len(current_header)
                if next_header:
                    end_idx = feedback_text.find(next_header)
                    content = feedback_text[start_idx:end_idx].strip()
                else:
                    content = feedback_text[start_idx:].strip()
                
                if content:  # Only add if content is not empty
                    formatted_feedback.append({
                        'title': sections[current_header],
                        'content': content
                    })
        
        # Add language-specific challenges if no challenges were found
        if not any(section['title'] == '⚠️ Common Challenges' for section in formatted_feedback):
            if language in LANGUAGE_CHALLENGES:
                challenges = LANGUAGE_CHALLENGES[language]
                formatted_feedback.append({
                    'title': '⚠️ Common Challenges',
                    'content': f"Common challenges in {SUPPORTED_LANGUAGES[language]}:\n" +
                              "\n".join(f"• {challenge}" for challenge in challenges['sounds'])
                })
        
        return formatted_feedback

    except Exception as e:
        print(f"Error generating feedback: {str(e)}")  # For debugging
        return [{
            'title': '📊 Overall Assessment',
            'content': get_quick_assessment(similarity)
        }]

def get_quick_assessment(similarity: float) -> str:
    """Get a quick assessment based on similarity score"""
    if similarity >= 90:
        return ("🌟 Excellent! Your pronunciation is nearly perfect.\n"
                "Focus on maintaining this level of accuracy.")
    elif similarity >= 80:
        return ("✨ Very Good! You're close to native-like pronunciation.\n"
                "Small refinements will make it perfect.")
    elif similarity >= 70:
        return ("👍 Good Progress! Your pronunciation is clearly understandable.\n"
                "Some areas need attention for improvement.")
    elif similarity >= 60:
        return ("💪 Fair Attempt! The basic sounds are there.\n"
                "Focus on clarity and accuracy in pronunciation.")
    else:
        return ("🎯 Keep Practicing! Your effort is noticeable.\n"
                "Follow the tips below to improve significantly.")

# Language-specific common challenges
LANGUAGE_CHALLENGES = {
    'en': {
        'sounds': ['th', 'r', 'w', 'v/w', 'short/long vowels'],
        'patterns': ['stress patterns', 'reduced vowels', 'linking words'],
        'tips': [
            'Practice "th" sounds slowly (think, this)',
            'Focus on word stress in multi-syllable words',
            'Pay attention to the difference between similar sounds (e.g., ship/sheep)'
        ]
    },
    'es': {
        'sounds': ['ñ', 'rr', 'j', 'b/v'],
        'patterns': ['word stress', 'diphthongs', 'silent h'],
        'tips': [
            'Practice rolling your "r" sounds',
            'Focus on proper stress placement',
            'Practice the difference between "b" and "v"'
        ]
    },
    # Add more languages as needed
}

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Command not found!")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("You don't have permission to use this command!")
    else:
        await ctx.send(f"An error occurred: {str(error)}")

if __name__ == "__main__":
    try:
        bot.run(os.getenv('DISCORD_TOKEN'))
    except Exception as e:
        print(f"Failed to start bot: {str(e)}")