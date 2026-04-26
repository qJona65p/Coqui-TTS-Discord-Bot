import discord
from discord import app_commands

from collections import defaultdict
import configparser
import asyncio
import emoji
import json
import os
import re

import torch
from TTS.api import TTS

# -------------------    CONFIG    -------------------

config = configparser.ConfigParser()
config.read("config.cfg")

BOT_TOKEN           = config["Bot"]["token"]
IDLE_TIMEOUT        = int(config["Bot"]["idle_time"])

ALLOWED_CHANNEL_ID  = int(config["TTS"]["channel"]) if config["TTS"]["channel"].strip() else None
DEFAULT_LANGUAGE    = config["TTS"]["default_lang"]
DEFAULT_SPEAKER     = config["TTS"]["default_sp"]
MEDIA_MSG           = config["TTS"]["media_msg"]

ADMIN_IDS           = list(int(i) for i in config["Admin"]["admin_ids"].split(","))
RESTRICT_VOICES     = list(i for i in config["Admin"]["restricted_voices"].split(","))
AUTHORIZED_USERS    = list(int(i) for i in config["Admin"]["authorized_users"].split(","))

# ----------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

class TTSBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tts = None
        self._voice_clients = {}        # guild_id -> voice_client
        self.user_cfg = {}              # user_id -> speaker_name
        self.banned_users = {}

        self.load_bans() 
        self.load_user_configs()
        self.reload_replacements()

        self.queues = defaultdict(asyncio.Queue)   # guild_id -> asyncio.Queue of tasks
        self.processing = defaultdict(bool)        # guild_id -> is currently playing
        self.idle_tasks = {}

        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        print("[TTSBot] Loading XTTS-v2 model ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print(f"[TTSBot] XTTS model loaded successfully on {device}!")

        await self.tree.sync()
        print("[TTSBot] Bot ready and slash commands synced.")

    def load_user_configs(self):
        try:
            with open("user_configs.json", "r") as file:
                data = json.load(file)
                self.user_cfg = {
                    int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in data.items()
                }
                print("[TTSBot] Loaded user configs")
        except Exception as e:
            print(f"[TTSBot] Could not load user_configs.json: {e}")
    
    def dump_user_configs(self):
        with open("user_configs.json", "w") as f:
            json.dump(self.user_cfg, f, indent=4)
        print("[TTSBot] Saved user configs")

    def reload_replacements(self, interaction: discord.Interaction=None):
        try:
            with open("replacements.json", "r", encoding="utf-8") as f:
                self.replacements = json.load(f)
            print("[TTSBot] Replacements loaded")
            return True
        except Exception as e:
            self.replacements = {"emojis":{}, "symbols":{}}
            print(f"[TTSBot] Could not load replacements.json: {e}")
            return False

    def reload_config(self, interaction: discord.Interaction):
        """Reload the config.cfg file and update runtime values"""
        try:
            new_config = configparser.ConfigParser()
            new_config.read("config.cfg")

            # Update the values you actually use at runtime
            global IDLE_TIMEOUT, ALLOWED_CHANNEL_ID, DEFAULT_LANGUAGE, DEFAULT_SPEAKER, MEDIA_MSG, ADMIN_IDS, RESTRICT_VOICES, AUTHORIZED_USERS

            IDLE_TIMEOUT        = int(new_config["Bot"]["idle_time"])
            ALLOWED_CHANNEL_ID  = int(new_config["TTS"]["channel"]) if new_config["TTS"]["channel"].strip() else None
            DEFAULT_LANGUAGE    = new_config["TTS"]["default_lang"]
            DEFAULT_SPEAKER     = new_config["TTS"]["default_sp"]
            MEDIA_MSG           = new_config["TTS"]["media_msg"]
            ADMIN_IDS           = list(int(i) for i in new_config["Admin"]["admin_ids"].split(","))
            RESTRICT_VOICES     = list(i for i in new_config["Admin"]["restricted_voices"].split(","))
            AUTHORIZED_USERS    = list(int(i) for i in new_config["Admin"]["authorized_users"].split(","))

            print(f"[TTSBot] ({interaction.user.name}) Reloaded configuration")
            return True

        except Exception as e:
            print(f"[TTSBot] ({interaction.user.name}) Failed to reload configuration: {e}")
            return False

    def load_bans(self):
        """Load persistent bans from file"""
        try:
            with open("bans.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.banned_users = {int(k): v for k, v in data.items()}
            print(f"[TTSBot] Loaded {len(self.banned_users)} banned users")
        except FileNotFoundError:
            self.banned_users = {}
        except Exception as e:
            print(f"[TTSBot] Error loading bans.json: {e}")
            self.banned_users = {}

    def dump_bans(self):
        """Save bans to file"""
        try:
            with open("bans.json", "w", encoding="utf-8") as f:
                json.dump(self.banned_users, f, indent=4, ensure_ascii=False)
            print("[TTSBot] Bans saved")
        except Exception as e:
            print(f"[TTSBot] Error saving bans: {e}")

    def is_banned(self, user_id: int) -> bool:
        """Check if user is banned or timed out"""
        if user_id not in self.banned_users:
            return False
        
        ban_info = self.banned_users[user_id]
        
        if isinstance(ban_info, dict) and "until" in ban_info:
            if ban_info["until"] < asyncio.get_event_loop().time():
                # Ban expired → remove it
                del self.banned_users[user_id]
                self.dump_bans()
                return False
            return True # Temporary ban (timeout)
        return True  # Permanent ban    

    async def join_voice(self, interaction: discord.Interaction=None, message: discord.Message=None):
        if message:
            if not message.author.voice or not message.author.voice.channel:
                return None
            channel = message.author.voice.channel
            guild_id = message.guild.id

            # not join if AFK channel
            if channel.id == message.guild._afk_channel_id:
                print(f"[TTSBot] Ignored message from AFK channel: {channel.name}")
                return None

        elif interaction:
            if not interaction.user.voice or not interaction.user.voice.channel:
                await interaction.followup.send("Debes unirte a un canal de voz primero.", ephemeral=True)
                return None

            channel = interaction.user.voice.channel
            guild_id = interaction.guild.id

            # not join if AFK channel
            if channel.id == interaction.guild._afk_channel_id:
                await interaction.followup.send("No puedo unirme al canal AFK.", ephemeral=True)
                print(f"[TTSBot] Ignored message from AFK channel: {channel.name}")
                return None
        else: 
            return

        # If bot is already connected elsewhere, move it to follow the user
        if guild_id in self._voice_clients and self._voice_clients[guild_id].is_connected():
            vc = self._voice_clients[guild_id]
            if vc.channel.id != channel.id:
                try:
                    await vc.move_to(channel)
                    print(f"[TTSBot] Moved to channel {channel.name}")
                except Exception as e:
                    print(f"[Move Error] {e}")
                    return None
        else:
            try:
                vc = await channel.connect()
                self._voice_clients[guild_id] = vc
                print(f"[TTSBot] Joined voice channel: {channel.name}")
            except Exception as e:
                print(f"[Join Error] {e}")
                return None

        self.reset_idle_timer(guild_id)
        return vc

    def reset_idle_timer(self, guild_id: int):
        """Reset the idle disconnect timer"""
        # Cancel existing timer
        if guild_id in self.idle_tasks and not self.idle_tasks[guild_id].done():
            self.idle_tasks[guild_id].cancel()

        # Create new timer
        self.idle_tasks[guild_id] = asyncio.create_task(self._idle_disconnect(guild_id))
    
    async def _idle_disconnect(self, guild_id: int):
        """Background task that disconnects after IDLE_TIMEOUT seconds of inactivity"""
        try:
            await asyncio.sleep(IDLE_TIMEOUT)
            
            # If we reach here, no activity happened
            if guild_id in self._voice_clients and self._voice_clients[guild_id].is_connected():
                vc = self._voice_clients[guild_id]
                try:
                    await vc.disconnect()
                    print(f"[TTSBot] Auto-disconnected from guild {guild_id} due to inactivity")
                except Exception as e:
                    print(f"[TTSBot] Error while idle disconnecting: {e}")
                
                self._voice_clients.pop(guild_id, None)
                # Clear queue
                while not self.queues[guild_id].empty():
                    try:
                        self.queues[guild_id].get_nowait()
                        self.queues[guild_id].task_done()
                    except:
                        break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[TTSBot] Idle Timer Error: {e}")

    def preprocess_text(self, text:str):
        """Clean text for better TTS"""
        if not text:
            return ""

        # Link removal
        pattern = r'(https?://)?([a-z0-9.-]+\.[a-z]{2,})([/\w.-]*)?(\?[^#\s]*)?(#[^\s]*)?'
        text = re.sub(pattern, "Link", text)

        # Handle Discord custom emojis first (<:name:id> or <a:name:id>)
        def replace_discord_emoji(match):
            full = match.group(0)
            # Extract the name (everything between the first : and the last :)
            name_match = re.search(r':([^:]+):', full)
            if name_match:
                name = name_match.group(1)
                return f"{name.replace('_', ' ')} "
            return " "

        text = re.sub(r'<a?:[^:]+:\d+>', replace_discord_emoji, text)

        emoji_mapping = self.replacements.get("emojis", {})
        symbol_mapping = self.replacements.get("symbols", {})

        # Convert emojis to descriptions
        def replace_unicode_emoji(emoji_char, data=None):
            if emoji_char in emoji_mapping:
                return emoji_mapping[emoji_char]
            return emoji.demojize(emoji_char, language="es").replace(":", " ").replace("_", " ").strip()

        text = emoji.replace_emoji(text, replace=replace_unicode_emoji)

        # Fix common symbols / punctuation
        for old, new in symbol_mapping.items():
            text = text.replace(old, new)

        return text

    async def _play_text(self, voice_client, text: str, speaker: str, language: str, interaction=None):
        """Internal method to generate and play one message"""

        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            if not sentence.strip():
                continue

            try:
                temp_path = f"temp_tts_{voice_client.guild.id}.wav"
                
                await asyncio.to_thread(
                    self.tts.tts_to_file,
                    text=sentence.strip(),
                    speaker=speaker,
                    language=language,
                    file_path=temp_path
                )

                voice_client.play(discord.FFmpegPCMAudio(temp_path))

                # Wait until finished
                while voice_client.is_playing():
                    await asyncio.sleep(0.2)

                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                print(f"[TTS Error] Playing: {e}")
                if interaction and interaction.channel:
                    try:
                        await interaction.channel.send(f"Error generando audio: {str(e)[:100]}")
                    except:
                        pass
    
    async def process_tts_message(self, message: discord.Message):
        """Automatically speak any message sent in the monitored text channel"""
        if self.is_banned(message.author.id):
            try:
                await message.add_reaction("❌")
            except:
                pass
            return
        
        guild_id = message.guild.id

        # Get the user's current voice channel
        if not message.author.voice or not message.author.voice.channel:
            return

        # Join voice if needed
        vc = await self.join_voice(message=message)
        if not vc:
            return

        # Get speaker for this user (or default)
        default_speaker = DEFAULT_SPEAKER
        if self.tts and self.tts.speakers:
            default_speaker = self.tts.speakers[20]

        speaker = self.user_cfg.get(message.author.id, default_speaker)

        # Preprocess text + detect media
        clean_text = self.preprocess_text(message.content)

        if message.attachments:
            clean_text += f" {MEDIA_MSG}"

        # Add to queue (text = message.content)
        await self.queues[guild_id].put((clean_text, speaker, None))  # interaction=None for auto mode

        # Start queue processor
        asyncio.create_task(self.process_queue(guild_id))

        # React to the message so users know it's being processed
        try:
            await message.add_reaction("🎙️")
        except:
            pass
    
    async def process_queue(self, guild_id: int):
        """Background task that processes the queue for a guild"""
        if self.processing[guild_id]:
            return
        self.processing[guild_id] = True

        try:
            while True:
                task = await asyncio.wait_for(self.queues[guild_id].get(), timeout=30)
                text, speaker, interaction = task

                vc = self._voice_clients.get(guild_id)
                if not vc or not vc.is_connected():
                    break

                await self._play_text(vc, text, speaker, DEFAULT_LANGUAGE, interaction)

                self.queues[guild_id].task_done()

                # Small delay between messages
                await asyncio.sleep(0.3)
        except TimeoutError:
            pass
        except Exception as e:
            print(f"[TTSBot] Queue Error: Guild {guild_id}: {e}")
        finally:
            self.processing[guild_id] = False

    async def voice_autocomplete(self, interaction: discord.Interaction, current: str):
        """Autocomplete for /setvoice - shows only real speakers"""
        if not self.tts or not self.tts.speakers:
            return []
        
        # Filter speakers that match what the user is typing
        current = current.lower()
        matching = [
            app_commands.Choice(name=speaker, value=speaker)
            for speaker in self.tts.speakers
            if current in speaker.lower()
        ]
        
        # Return up to 25 choices (Discord limit)
        return matching[:25]
    
# ------------------- BOT INSTANCE -------------------
client = TTSBot()

@client.event
async def on_ready():
    print(f"[TTSBot] Logged in as {client.user}")

# -------------------   COMMANDS   -------------------
@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Only react to messages in the specific TTS channel
    if ALLOWED_CHANNEL_ID and message.channel.id != ALLOWED_CHANNEL_ID:
        return

    # Process the message for TTS
    await client.process_tts_message(message)

@client.tree.command(name="tts", description="Un tts we, que esperabas")
@app_commands.describe(text="El texto que quieres que hable")
async def tts(interaction: discord.Interaction, text: str):
    if ALLOWED_CHANNEL_ID and interaction.channel_id != ALLOWED_CHANNEL_ID:
        await interaction.response.send_message("Este comando solo se puede usan en el canal de TTS.", ephemeral=True)
        return
    
    await interaction.response.defer()

    if client.is_banned(interaction.user.id):
        await interaction.followup.send("Baneao.", ephemeral=True)
        return

    vc = await client.join_voice(interaction=interaction)
    if not vc:
        return
    
    # Get user's chosen voice or default (first speaker)
    speaker = client.user_cfg.get(interaction.user.id, client.tts.speakers[20] if client.tts.speakers else DEFAULT_SPEAKER)

    clean_text = client.preprocess_text(text)

    # Add to queue
    await client.queues[interaction.guild.id].put((clean_text, speaker, interaction))

    # Start queue processor if not running
    asyncio.create_task(client.process_queue(interaction.guild.id))

    await interaction.followup.send(f"**{speaker}** en cola ({client.queues[interaction.guild.id].qsize()} mensajes)", ephemeral=True)

@client.tree.command(name="setvoice", description="Selecciona la voz para el TTS")
@app_commands.describe(voice="Nombre de la voz (usa /voices para verlas)")
@app_commands.autocomplete(voice=client.voice_autocomplete)
async def setvoice(interaction: discord.Interaction, voice: str):
    if not client.tts or not client.tts.speakers:
        await interaction.response.send_message("El modelo aún no ha cargado las voces.", ephemeral=True)
        return
    
    if voice not in client.tts.speakers:
        await interaction.response.send_message(f"La voz **{voice}** no existe.\n", ephemeral=True)
        return

    voice_name = voice.strip()

    if voice_name in RESTRICT_VOICES and interaction.user.id != AUTHORIZED_USERS[RESTRICT_VOICES.index(voice_name)]: # Restriccion de voces
        await interaction.response.send_message(f"No autorizo.\n", ephemeral=True)
        return
    
    # Save option
    client.user_cfg[interaction.user.id] = voice_name
    client.dump_user_configs()
    await interaction.response.send_message(f"Tu voz ha sido cambiada a **{voice}**.", ephemeral=True)

@client.tree.command(name="voices", description="Muestra las voces disponibles en XTTS-v2")
async def voices(interaction: discord.Interaction):
    if not client.tts or not client.tts.speakers:
        await interaction.response.send_message("Las voces aún no están cargadas.", ephemeral=True)
        return
    
    speakers_list = "\n".join([f"• {s}" for s in client.tts.speakers[:60]])
    embed = discord.Embed(
        title="Voces disponibles (Primeras 60)",
        description=speakers_list,
        color=0x00ff00
    )
    embed.set_footer(text=f"Ejemplo: /setvoice {DEFAULT_SPEAKER}")
    await interaction.response.send_message(embed=embed, ephemeral=True)
    
@client.tree.command(name="leave", description="Desconectar el bot del canal de voz")
async def leave(interaction: discord.Interaction):
    guild_id = interaction.guild.id
    if guild_id in client._voice_clients and client._voice_clients[guild_id].is_connected():
        await client._voice_clients[guild_id].disconnect()
        del client._voice_clients[guild_id]
        # Clear queue
        while not client.queues[guild_id].empty():
            try:
                client.queues[guild_id].get_nowait()
                client.queues[guild_id].task_done()
            except:
                break
        await interaction.response.send_message("Bot desconectado del canal de voz.", ephemeral=True)
    else:
        await interaction.response.send_message("El bot no está en ningún canal de voz.", ephemeral=True)

# ----------------  ADMINS COMMANDS   ----------------

@client.tree.command(name="ban", description="Bloquear el uso del bot a un usuario")
@app_commands.describe(user="Usuario a banear")
async def ban(interaction: discord.Interaction, user: discord.User):
    if interaction.user.id not in ADMIN_IDS:
        await interaction.response.send_message("No tienes permitido el uso de este comando.", ephemeral=True)
        return

    user_id = user.id
    client.banned_users[user_id] = "permanent"
    client.dump_bans()

    await interaction.response.send_message(f"Usuario **{user}** ({user_id}) ha sido baneado.", ephemeral=True)

@client.tree.command(name="unban", description="Desbloquear el uso del bot a un usuario")
@app_commands.describe(user="Usuario a desbanear")
async def unban(interaction: discord.Interaction, user: discord.User):
    if interaction.user.id not in ADMIN_IDS:
        await interaction.response.send_message("No tienes permitido el uso de este comando.", ephemeral=True)
        return
    
    user_id = user.id
    if user_id in client.banned_users:
        del client.banned_users[user_id]
        client.dump_bans()
        await interaction.response.send_message(f"Usuario **{user}** ha sido desbaneado.", ephemeral=True)
    else:
        await interaction.response.send_message(f"El usuario no estaba baneado.", ephemeral=True)

@client.tree.command(name="timeout", description="Bloquear el uso del bot a un usuario temporalmente")
@app_commands.describe(user="Usuario", minutes="Duración en minutos (default: 3)")
async def timeout(interaction: discord.Interaction, user: discord.User, minutes: int=3):
    if interaction.user.id not in ADMIN_IDS:
        await interaction.response.send_message("No tienes permitido el uso de este comando.", ephemeral=True)
        return
    
    user_id = user.id
    until = asyncio.get_event_loop().time() + (minutes * 60)

    client.banned_users[user_id] = {"until": until, "reason": f"Timeout de {minutes} minutos"}
    client.dump_bans()

    await interaction.response.send_message(
        f"Usuario **{user}** ha sido baneado por **{minutes} minutos**.", 
        ephemeral=True
    )
    
@client.tree.command(name="reload", description="Recargar configuraciones del bot. (Admins)")
@app_commands.choices(option=[
    app_commands.Choice(name="Configs", value="Configs"),
    app_commands.Choice(name="Replacements", value="Replacements"),
])
async def reload(interaction: discord.Interaction, option: str):
    if interaction.user.id not in ADMIN_IDS:
        await interaction.response.send_message("No tienes permitido el uso de este comando.", ephemeral=True)
        return

    if option == "Configs":
        await interaction.response.defer(ephemeral=True)
        success = client.reload_config(interaction)

        if success:
            await interaction.followup.send("Configuración recargada correctamente.", ephemeral=True)
        else:
            await interaction.followup.send("Error al recargar la configuración.", ephemeral=True)
    
    elif option == "Replacements":
        await interaction.response.defer(ephemeral=True)
        success = client.reload_replacements(interaction)

        if success:
            await interaction.followup.send("Replacements recargados correctamente.", ephemeral=True)
        else:
            await interaction.followup.send("Error al recargar los replacements.", ephemeral=True)

client.run(BOT_TOKEN)