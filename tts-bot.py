import discord
from discord import app_commands
import torch
import asyncio
import os
import re
import configparser
from TTS.api import TTS
import soundfile as sf  # for safe audio loading if needed
from collections import defaultdict
import json

# -------------------    CONFIG    -------------------

config = configparser.ConfigParser()
config.read("config.cfg")

BOT_TOKEN           = config["Bot"]["token"]
IDLE_TIMEOUT        = int(config["TTS"]["idle_time"])
DEFAULT_LANGUAGE    = config["TTS"]["default_lang"]
DEFAULT_SPEAKER     = config["TTS"]["default_sp"]
ALLOWED_CHANNEL_ID  = int(config["TTS"]["channel"]) if config["TTS"]["channel"].strip() else None

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
        self.load_user_configs()

        self.queues = defaultdict(asyncio.Queue)   # guild_id -> asyncio.Queue of tasks
        self.processing = defaultdict(bool)        # guild_id -> is currently playing
        self.idle_tasks = {}

        self.tree = app_commands.CommandTree(self)

    def load_user_configs(self):
        try:
            with open("user_configs.json", "r") as file:
                data = json.load(file)
                self.user_cfg = {
                    int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in data.items()
                }
                print("[TTSBot] Loaded user configs")
        except:
            pass
    
    def dump_user_configs(self):
        with open("user_configs.json", "w") as f:
            json.dump(self.user_cfg, f, indent=4)
        print("[TTSBot] Saved user configs")

    async def setup_hook(self):
        print("[TTSBot] Loading XTTS-v2 model ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print(f"[TTSBot] XTTS model loaded successfully on {device}!")

        await self.tree.sync()
        print("[TTSBot] Bot ready and slash commands synced.")

    async def join_voice(self, interaction: discord.Interaction):
        if not interaction.user.voice or not interaction.user.voice.channel:
            await interaction.followup.send("Debes unirte a un canal de voz primero.", ephemeral=True)
            return None

        channel = interaction.user.voice.channel
        guild_id = interaction.guild.id

        if guild_id not in self._voice_clients or not self._voice_clients[guild_id].is_connected():
            vc = await channel.connect()
            self._voice_clients[guild_id] = vc
            print(f"[TTSBot] Joined voice channel: {channel.name}")
        else:
            vc = self._voice_clients[guild_id]

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
                except:
                    pass
                
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

    async def _play_text(self, voice_client, text: str, speaker: str, language: str, interaction=None):
        """Internal method to generate and play one message"""

        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            if not sentence.strip():
                continue

            try:
                temp_path = f"temp_tts_{voice_client.guild.id}.wav"

                self.tts.tts_to_file(
                    text=sentence.strip(),
                    speaker=speaker,
                    language=language,
                    file_path=temp_path
                )

                if voice_client.is_playing():
                    voice_client.stop()

                voice_client.play(discord.FFmpegPCMAudio(temp_path))

                # Wait until finished
                while voice_client.is_playing():
                    await asyncio.sleep(0.2)

                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                print(f"[TTS Error] {e}")
                if interaction and interaction.channel:
                    try:
                        await interaction.channel.send(f"Error generando audio: {str(e)[:100]}")
                    except:
                        pass
    
    async def process_tts_message(self, message: discord.Message):
        """Automatically speak any message sent in the monitored text channel"""
        guild_id = message.guild.id

        # Get the user's current voice channel
        if not message.author.voice or not message.author.voice.channel:
            return

        # Join voice if needed
        vc = await self.join_voice_from_message(message)
        if not vc:
            return

        # Get speaker for this user (or default)
        default_speaker = DEFAULT_SPEAKER
        if self.tts and self.tts.speakers:
            default_speaker = self.tts.speakers[0]

        speaker = self.user_cfg.get(message.author.id, default_speaker)

        # Add to queue (text = message.content)
        await self.queues[guild_id].put((message.content, speaker, None))  # interaction=None for auto mode

        # Start queue processor
        asyncio.create_task(self.process_queue(guild_id))

        # Optional: React to the message so users know it's being processed
        try:
            await message.add_reaction("🎙️")
        except:
            pass

    async def join_voice_from_message(self, message: discord.Message):
        """Helper to join voice from a normal message (not interaction)"""
        channel = message.author.voice.channel
        guild_id = message.guild.id

        if guild_id not in self._voice_clients or not self._voice_clients[guild_id].is_connected():
            try:
                vc = await channel.connect()
                self._voice_clients[guild_id] = vc
                print(f"[TTSBot] Auto-joined voice channel: {channel.name}")
            except Exception as e:
                print(f"[Join Error] {e}")
                return None
        else:
            vc = self._voice_clients[guild_id]

        self.reset_idle_timer(guild_id)
        return vc
    
    async def process_queue(self, guild_id: int):
        """Background task that processes the queue for a guild"""
        if self.processing[guild_id]:
            return
        self.processing[guild_id] = True

        try:
            while True:
                task = await self.queues[guild_id].get()
                text, speaker, interaction = task

                vc = self._voice_clients.get(guild_id)
                if not vc or not vc.is_connected():
                    break

                await self._play_text(vc, text, speaker, DEFAULT_LANGUAGE, interaction)

                self.queues[guild_id].task_done()

                # Small delay between messages
                await asyncio.sleep(0.3)

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

    vc = await client.join_voice(interaction)
    if not vc:
        return
    
    # Get user's chosen voice or default (first speaker)
    speaker = client.user_cfg.get(interaction.user.id, client.tts.speakers[0] if client.tts.speakers else DEFAULT_SPEAKER)

    # Add to queue
    await client.queues[interaction.guild.id].put((text, speaker, interaction))

    # Start queue processor if not running
    asyncio.create_task(client.process_queue(interaction.guild.id))

    await interaction.followup.send(f"**{speaker}** en cola ({client.queues[interaction.guild.id].qsize()} mensajes)", ephemeral=True)

@client.tree.command(name="setvoice", description="Selecciona la voz para el TTS")
@app_commands.describe(voice="Nombre de la voz (usa /voices para verlas)")
@app_commands.autocomplete(voice=client.voice_autocomplete)
async def setvoice(interaction: discord.Interaction, voice: str = DEFAULT_SPEAKER):
    if not client.tts or not client.tts.speakers:
        await interaction.response.send_message("El modelo aún no ha cargado las voces.", ephemeral=True)
        return
    
    if voice not in client.tts.speakers:
        await interaction.response.send_message(f"La voz **{voice}** no existe.\n", ephemeral=True)
        return

    # Save option
    client.user_cfg[interaction.user.id] = voice.strip()
    client.dump_user_configs()
    await interaction.response.send_message(f"Tu voz ha sido cambiada a **{voice}**.", ephemeral=True)

@client.tree.command(name="voices", description="Muestra las voces disponibles en XTTS-v2")
async def voices(interaction: discord.Interaction):
    if not client.tts or not client.tts.speakers:
        await interaction.response.send_message("Las voces aún no están cargadas.", ephemeral=True)
        return
    
    speakers_list = "\n".join([f"• {s}" for s in sorted(client.tts.speakers[:60])])
    embed = discord.Embed(
        title="Voces disponibles en XTTS-v2",
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
        await interaction.response.send_message("Bot desconectado del canal de voz.", ephemeral=True)
    else:
        await interaction.response.send_message("El bot no está en ningún canal de voz.", ephemeral=True)

client.run(BOT_TOKEN)